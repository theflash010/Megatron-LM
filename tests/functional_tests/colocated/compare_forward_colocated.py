# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Colocated side of the module-level forward comparison (Task 6.7 ①).

在**固定数据 + dropout 0 + 共享 vision_projection 初始化**三个前置条件下跑 2 个 iteration，
把第 1 个 iteration 里每个 microbatch 在每个位置上的张量记下来落盘，供
``compare_forward_dumps.py`` 与非共置侧逐元素比对。

记录的位置（tag）：
  * ``encoder_output``——phase ① 的 ``image_embeddings``（encoder 出口，在 producer 上）；
  * ``backbone_stage_output``——每个 pipeline stage 的 ``backbone_chunk`` 返回值。**末 stage
    上它不是 logits 而是逐 token 的 loss**（传了 labels 时 GPTModel 直接算 loss），所以形状
    很小、能存原张量；
  * ``loss`` / ``num_tokens``——末 stage 每个 microbatch 的标量。

三个前置条件的落实方式：
  ⓐ 数据——``FixedMicroBatchIterator`` 从 ``dump_fixed_micro_batches.py`` 存的 ``.pt`` 读，
     本 rank 按 ``get_microbatches_for_producer`` 的顺序读自己那几个（与 phase ① 的循环一致）；
  ⓑ dropout——驱动内**硬断言** ``hidden_dropout == attention_dropout == 0``，不只靠启动脚本；
  ⓒ vision_projection——第一次前向时由全局 rank 0 存盘（文件不存在时）、barrier 后所有 rank
     统一读回，两侧共用同一个文件。第一次前向在每个 rank 上都是 phase ①，是对称点，可以放
     barrier。
"""
import os
import sys

import torch

MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from colocated_args import add_colocated_extra_args, validate_colocated_args
from colocated_train import colocated_forward_step
from fixed_micro_batch import FixedMicroBatchIterator
from fixed_vision_projection import sync_vision_projection
from forward_record import (
    ForwardRecorder,
    patch_optimizer_step_to_record,
    register_transformer_layer_hooks,
    wrap_finalize_to_record_gradients,
)
from model import model_provider
from train import llava_embedding_ranks, llava_position_embedding_ranks

from megatron.core import parallel_state as mpu
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.colocated_llava_model import ColocatedViTEncoder
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import get_microbatches_for_producer
from megatron.core.utils import get_attr_wrapped_model, get_pg_rank, get_pg_size, unwrap_model
from megatron.training import pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args

FIXED_BATCH_DIRECTORY = "/home/zn/zn_data/workspace/fixed_micro_batches"
VISION_PROJECTION_PATH = "/home/zn/zn_data/workspace/fixed_vision_projection.pt"
DUMP_DIRECTORY = os.environ.get(
    "FORWARD_DUMP_DIRECTORY", "/home/zn/zn_data/workspace/forward_dumps"
)
DISTRIBUTED_OPTIMIZER_MODE = os.environ.get("DISTRIBUTED_OPTIMIZER_MODE", "non_distopt")
# ④ 优化器步等价性（Task 9.4）：记录前 N 步的 grad norm / 裁剪系数 / fp32 主权重 / Adam 动量。
# N 步是为了让"误差是否随 Adam 二阶动量累积放大"能被看见（单步看不出来，Task 9.5）。
RECORDED_OPTIMIZER_STEPS = int(os.environ.get("RECORDED_OPTIMIZER_STEPS", "5"))
# 只有首末两步额外存小张量的原值：中间步只需要摘要判断趋势，末步用于逐元素定位。
RAW_TENSOR_OPTIMIZER_STEPS = (1, RECORDED_OPTIMIZER_STEPS)

state = {"encoder_calls": 0, "backbone_calls": 0, "ids": None, "dumped": False, "synced": False}
state["recorder"] = None
state["current_microbatch_id"] = 0
state["hooked_chunks"] = set()
state["wrapped_configs"] = set()
# 前向/loss 记第 1、2 两个 iteration（第 2 个的键带 ``/it2`` 后缀，用于诊断更新是否晚一轮生效）；
# 优化器步记 N 步（键带步号）。
state["iteration"] = 0
state["optimizer_steps"] = 0
state["model_chunks"] = []


def encoder_distributed_optimizer_instances():
    """Encoder DistOpt instance count, derived from the groups themselves rather than from args.

    每个 instance 内部由 intra 组分片，instance 之间由 inter 组 all-reduce ⇒
    instances = encoder 全 DP size / intra size。从组本身反推而不是读 args，是为了让 side 名
    直接反映**实际建出来的层次**：如果参数没被接进 `initialize_model_parallel`，这里会得到 1，
    dump 文件名就与预期的 enc2 不符，比对时立刻暴露，而不是静默跑成 enc1。
    Deriving it from the groups makes a mis-wired instance count visible in the dump file name.
    """
    intra_group = mpu.get_colocated_encoder_intra_distributed_optimizer_instance_group(
        check_initialized=False
    )
    if intra_group is None:
        return 1
    return get_pg_size(mpu.get_colocated_data_parallel_group()) // get_pg_size(intra_group)


def side_name():
    """Topology-derived side name, so the sides never overwrite each other's dumps.

    名字里带 encoder 的 DistOpt instance 数（Task 8.7）：encoder=1 与 encoder=2 是两条不同的
    通信路径（后者才会走 `param_and_grad_buffer.py:620-647` 的 inter-instance all-reduce），
    必须能在同一个 dump 目录里共存并互相对照。
    """
    return (
        f"colocated_{DISTRIBUTED_OPTIMIZER_MODE}"
        f"_enc{encoder_distributed_optimizer_instances()}"
        f"_tp{mpu.get_tensor_model_parallel_world_size()}"
        f"pp{mpu.get_pipeline_model_parallel_world_size()}"
    )


def get_recorder():
    """Create the recorder on first use (it needs an initialized job to read the TP rank)."""
    if state["recorder"] is None:
        state["recorder"] = ForwardRecorder(
            side=side_name(),
            is_recording_rank=mpu.get_tensor_model_parallel_rank() == 0,
        )
    return state["recorder"]


def next_optimizer_step_index():
    """The step about to run (finalize happens before ``step()``); None once N steps are done."""
    step_index = state["optimizer_steps"] + 1
    return step_index if step_index <= RECORDED_OPTIMIZER_STEPS else None


def hook_layers_once(model_chunk):
    """Register per-layer hooks on this chunk (encoder chunk -> ViT, backbone chunk -> decoder)."""
    module = unwrap_model(model_chunk)
    if id(module) in state["hooked_chunks"]:
        return
    state["hooked_chunks"].add(id(module))
    # 记下 chunk 本身：优化器步的记录需要按名字遍历参数，而驱动拿不到 pretrain 内部的 model 列表。
    state["model_chunks"].append(model_chunk)
    language_hooks, vision_hooks = register_transformer_layer_hooks(
        module, get_recorder(), lambda: state["current_microbatch_id"]
    )
    # ③ 反向：把梯度收尾包一层，在 finalize 之后（DDP 归约 + 按全局 token 数归一化之后）记下每个
    # 参数的 main_grad——那是优化器真正会用的值，也是"共置 encoder 在共置 dp 组（全 W）上的一次
    # 归约是否等价于非共置的一次归约"这条不变量的最终检验点。**逐步都记**（键带步号）：只看
    # grad norm 一个标量无法区分"差异弥散在所有参数上"与"集中在某个组件上"。
    wrap_finalize_to_record_gradients(
        model_chunk, get_recorder(), state["wrapped_configs"], next_optimizer_step_index
    )
    print(
        f"[rank {torch.distributed.get_rank()}] hooked {language_hooks} language layers and "
        f"{vision_hooks} vision layers on {type(module).__name__}",
        flush=True,
    )


def fixed_micro_batch_dataloaders_provider(train_val_test_num_samples):
    """Return an iterator over the stored micro batches this rank consumes, in phase ① order."""
    boundary_group = mpu.get_colocated_boundary_group()
    microbatch_ids = get_microbatches_for_producer(
        get_pg_rank(boundary_group), get_num_microbatches(), get_pg_size(boundary_group)
    )
    print(
        f"[rank {torch.distributed.get_rank()}] fixed micro batches {microbatch_ids}",
        flush=True,
    )
    return FixedMicroBatchIterator(microbatch_ids, FIXED_BATCH_DIRECTORY), None, None


fixed_micro_batch_dataloaders_provider.is_distributed = True


def sync_vision_projection_once(encoder_chunk):
    """Rank 0 writes the shared vision_projection on the first forward; every rank then loads it."""
    if state["synced"]:
        return
    if torch.distributed.get_rank() == 0 and not os.path.exists(VISION_PROJECTION_PATH):
        action, count = sync_vision_projection([encoder_chunk], VISION_PROJECTION_PATH)
        print(f"[rank 0] vision_projection {action} ({count} parameters)", flush=True)
    torch.distributed.barrier()
    action, count = sync_vision_projection([encoder_chunk], VISION_PROJECTION_PATH)
    assert action == "loaded", "the shared vision_projection file should exist by now"
    print(
        f"[rank {torch.distributed.get_rank()}] vision_projection loaded ({count} parameters)",
        flush=True,
    )
    state["synced"] = True


def recording_loss_func(original_loss_func, microbatch_id):
    """Record the per-microbatch loss and num_tokens returned by loss_func."""

    def wrapper(*call_args, **call_kwargs):
        result = original_loss_func(*call_args, **call_kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            get_recorder().record_scalar("loss", microbatch_id, result[0].item())
            get_recorder().record_scalar("num_tokens", microbatch_id, result[1].item())
        return result

    return wrapper


def instrumented_forward_step(data_iterator, model, packet=None, intra_packet=None):
    """``colocated_forward_step`` 的记录版：只旁路记录，不改变任何返回值语义。"""
    model_chunk = model[0] if isinstance(model, (list, tuple)) else model
    is_encoder_branch = isinstance(unwrap_model(model_chunk), ColocatedViTEncoder)

    # 先登记 chunk（换轮点的参数快照要用），再判断换轮：第 1 个 iteration 的快照里只有 encoder
    # chunk（backbone 还没被前向过），之后两轮两个 chunk 都在。
    hook_layers_once(model_chunk)

    if is_encoder_branch:
        sync_vision_projection_once(model_chunk)
        if state["ids"] is None:
            boundary_group = mpu.get_colocated_boundary_group()
            state["ids"] = get_microbatches_for_producer(
                get_pg_rank(boundary_group), get_num_microbatches(), get_pg_size(boundary_group)
            )
        # 每个 iteration 的第一次 encoder 调用是天然的换轮点（此刻上一个 iteration 的全部相位都
        # 已走完）。第 2 个 iteration 的前向改用 ``/it2`` 后缀继续记，用来诊断"第 1 步的更新是否
        # 晚一个 iteration 才被前向看到"；第 3 个 iteration 起停记（落盘在第 N 步之后）。
        # 同时在每个换轮点记一次 bf16 模型参数与 fp32 主副本，用于区分"回拷没生效"与"输入是旧的"。
        iteration = state["encoder_calls"] // len(state["ids"]) + 1
        if iteration != state["iteration"]:
            state["iteration"] = iteration
            # 第 1 个 iteration 不拍快照：此刻 backbone chunk 还没被前向过、没登记进来，拍出来
            # 只有 encoder 一半，会在比对里变成一堆"只在一侧出现"的键；而它的内容（ckpt + 共享
            # projection）已经被 ``param/step1``（第 1 步 lr=0，等于更新前的值）覆盖。
            if iteration > 1:
                get_recorder().record_model_parameters(state["model_chunks"], f"it{iteration}")
            if iteration == 2:
                get_recorder().set_forward_key_suffix("/it2")
            elif iteration > 2:
                get_recorder().stop_forward_recording()

    # 逐层 hook 在 forward 内部触发，所以当前 microbatch 号必须**在调用之前**设好。
    # consumer 从包里读；非首 stage 没有包，1F1B 的前向顺序就是 0,1,2,... ⇒ 用调用计数。
    if is_encoder_branch:
        state["current_microbatch_id"] = state["ids"][state["encoder_calls"] % len(state["ids"])]
    elif packet is not None:
        state["current_microbatch_id"] = int(packet.microbatch_id.item())
    else:
        state["current_microbatch_id"] = state["backbone_calls"] % get_num_microbatches()
    microbatch_id = state["current_microbatch_id"]

    result = colocated_forward_step(
        data_iterator, model, packet=packet, intra_packet=intra_packet
    )

    if is_encoder_branch:
        produced_packet, _ = result
        get_recorder().record_tensor(
            "encoder_output", microbatch_id, produced_packet.image_embeddings
        )
        # 换轮点靠这个计数判断，所以**无条件**递增（停记之后也要继续数）。
        state["encoder_calls"] += 1
        return result

    state["backbone_calls"] += 1
    output_tensor, loss_func = result
    # 末 stage 的返回值是逐 token loss（不是层输出）；非末 stage 的返回值等于本 stage 最后一层
    # 的输出，已由逐层 hook 覆盖，不重复记。
    if get_attr_wrapped_model(model_chunk, "post_process"):
        get_recorder().record_tensor("final_token_loss", microbatch_id, output_tensor)
    return output_tensor, recording_loss_func(loss_func, microbatch_id)


def record_optimizer_steps_and_dump(step_index):
    """Called after each recorded optimizer step; the last one triggers the dump."""
    state["optimizer_steps"] = step_index
    if step_index >= RECORDED_OPTIMIZER_STEPS and not state["dumped"]:
        get_recorder().dump(DUMP_DIRECTORY)
        state["dumped"] = True


if __name__ == "__main__":
    arguments = parse_and_validate_args(
        extra_args_provider=add_colocated_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    validate_colocated_args(arguments)
    assert arguments.train_iters >= RECORDED_OPTIMIZER_STEPS, (
        "this driver records the first "
        f"{RECORDED_OPTIMIZER_STEPS} optimizer steps, so --train-iters must be at least that many"
    )
    # ⓑ dropout=0：固定输入逐元素对照要求两侧使用相同的确定性 dropout 配置。
    # The element-wise comparison requires deterministic zero-dropout execution on both sides.
    arguments.hidden_dropout = 0.0
    arguments.attention_dropout = 0.0
    assert arguments.hidden_dropout == 0.0 and arguments.attention_dropout == 0.0, (
        "element-wise comparison requires dropout 0 on both sides: the two topologies consume "
        "RNG in different orders, so non-zero dropout gives different masks by construction"
    )
    full_config = pretrain_cfg_container_from_args(arguments)

    # ④ 优化器步：类方法打桩必须在 pretrain 建出优化器**之前**完成。
    patch_optimizer_step_to_record(
        get_recorder,
        lambda: state["model_chunks"],
        RECORDED_OPTIMIZER_STEPS,
        raw_tensor_steps=RAW_TENSOR_OPTIMIZER_STEPS,
        on_recorded=record_optimizer_steps_and_dump,
    )

    pretrain(
        full_config,
        fixed_micro_batch_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        instrumented_forward_step,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
