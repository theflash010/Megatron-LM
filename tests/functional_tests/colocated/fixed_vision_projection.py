# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Share one ``vision_projection`` initialization between both sides of the comparison (Task 6.7 ⓒ).

`vision_projection` 不在转换出来的 ckpt 里（CLIP 权重里没有它，LLaVA 第一阶段才训出来；
两侧都靠 `--allow-missing-vision-projection-checkpoint` 各自随机初始化）⇒ 不做处理的话，
两侧模型从第一步起就不是同一个模型，逐元素对照没有意义。

做法：**先跑的一侧存、后跑的一侧读**——文件不存在就把本侧的 `vision_projection` 参数存盘，
存在就读回来 `copy_` 覆盖。按"从参数名里 `vision_projection` 开始的后缀"匹配，这样共置侧的
encoder chunk（名字前面没有别的前缀）与非共置侧的 LLaVAModel 用同一份文件。

只覆盖参数值与其 **fp32 主副本**、不动优化器动量：调用点在 `setup_model_and_optimizer` 之后，
优化器此时还没有任何动量/二阶矩（冷启动、`--pretrained-checkpoint` 不加载优化器状态，见 7.4③），
但 **fp32 主副本在建优化器时就已经从当时的权重克隆出来了**（`Float16OptimizerWithFloat16Params`），
所以只写 bf16 参数是不够的：第一步 `step()` 结束时的 `_copy_main_params_to_model_params` 会用主副本
把刚覆盖的共享初始化**冲掉**，两侧从第 2 个 iteration 起就不是同一个模型了。2026-08-31 的多步对照
就是这样被污染的（非共置侧的 `vision_projection` 在第 1 步后跳回它自己的随机初始化；共置侧因为
恰好是存盘的那一侧才没露出来）。
"""
import os

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import unwrap_model

VISION_PROJECTION_PREFIX = "vision_projection"


def _vision_projection_parameters(model_chunks):
    """Yield ``(suffix_name, parameter)`` for every vision_projection parameter in the chunks."""
    for model_chunk in model_chunks:
        module = unwrap_model(model_chunk)
        for name, parameter in module.named_parameters():
            position = name.find(VISION_PROJECTION_PREFIX)
            if position >= 0:
                yield name[position:], parameter


def sync_vision_projection(model_chunks, path):
    """Save this side's vision_projection to ``path``, or load it if the file already exists.

    Returns:
        ("saved" | "loaded", number of parameters handled)
    """
    named_parameters = dict(_vision_projection_parameters(model_chunks))
    assert named_parameters, (
        "no vision_projection parameter found on this rank; this helper must be called on a rank "
        "that owns the vision projection (colocated: every rank; non-colocated: pipeline stage 0)"
    )

    action = "loaded"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in named_parameters.items()},
            path,
        )
        action = "saved"
        # 存盘的一侧也**照样读回来**：文件里是 bf16（模型参数的 dtype），而本侧的 fp32 主副本存的
        # 是未舍入的原值 ⇒ 不读回来的话两侧的主副本仍然不同（差一次 bf16 舍入），第一步的回拷会
        # 把这点差异放大成两条轨迹。走同一条 copy 路径才能保证两侧位级一致。
        # The saving side reloads as well, so both sides go through the same bf16 rounding.

    stored = torch.load(path, weights_only=False)
    missing = sorted(set(named_parameters) - set(stored))
    unexpected = sorted(set(stored) - set(named_parameters))
    assert not missing and not unexpected, (
        f"vision_projection parameter names do not match {path}: missing {missing}, "
        f"unexpected {unexpected}"
    )
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            source = shard_like(stored[name], parameter, name)
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
            # fp32 主副本必须一起覆盖，见模块 docstring：否则第一步的回拷会用建优化器时克隆的
            # 那份随机初始化冲掉这里写入的共享值。
            main_parameter = getattr(parameter, "main_param", None)
            if main_parameter is not None:
                main_source = source.to(device=main_parameter.device, dtype=main_parameter.dtype)
                assert main_source.numel() == main_parameter.numel(), (
                    f"{name}: projection parameter and main_param have different element counts: "
                    f"{main_source.numel()} vs {main_parameter.numel()}"
                )
                main_parameter.copy_(main_source.reshape_as(main_parameter))
    return "loaded", len(named_parameters)


def shard_like(source, parameter, name):
    """Slice a full (TP=1) tensor down to this rank's shard when the model is tensor parallel.

    存盘的那一侧可能是 TP=1（整块权重），读的一侧可能是 TP=4（每 rank 一片）——
    ``linear_fc1`` 是 column parallel（沿 dim 0 切）、``linear_fc2`` 是 row parallel（沿 dim 1
    切），切法由参数自带的 ``partition_dim`` 声明（mcore 在建参数时打上），因此这里不用猜。
    形状相同则原样返回。
    """
    if source.shape == parameter.shape:
        return source
    assert getattr(parameter, "tensor_model_parallel", False), (
        f"{name}: stored shape {tuple(source.shape)} != model shape {tuple(parameter.shape)}, "
        "but the parameter is not marked tensor parallel"
    )
    partition_dim = parameter.partition_dim
    tensor_parallel_size = get_tensor_model_parallel_world_size()
    tensor_parallel_rank = get_tensor_model_parallel_rank()
    assert source.shape[partition_dim] == parameter.shape[partition_dim] * tensor_parallel_size, (
        f"{name}: stored shape {tuple(source.shape)} is not "
        f"{tensor_parallel_size}x the model shape {tuple(parameter.shape)} along dim "
        f"{partition_dim}"
    )
    return source.chunk(tensor_parallel_size, dim=partition_dim)[tensor_parallel_rank].contiguous()
