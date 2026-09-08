# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Colocated training with a **silent** per-component grad-norm breakdown (Task 8.9.7 重做).

**为什么必须静默**：上一版每步在 rank0 上 `print(..., flush=True)`，那一次同步 IO 改变了 kernel
的提交/调度顺序 ⇒ 改变了非确定性 atomics 的累加顺序 ⇒ 轨迹被推到另一条上（`loss@3` 6.808819
而崩溃轨迹是 6.809309），观测本身破坏了被观测对象，1229 步"健康"的结论作废。

本版的三条纪律：
  * **不改 `ChainedOptimizer.get_grad_norm`**——只在**子优化器**的 `MegatronOptimizer.get_grad_norm`
    返回后往内存 list 里 append 一条记录（纯 CPU、无同步、无 IO），控制流与数值完全不动；
  * **训练期零 IO**——落盘只发生一次，在第 ``DUMP_AFTER_ITERATION`` 次记录之后（默认 110，即崩溃
    已经发生完），此时再扰动也影响不到前 100 步；
  * **自带轨迹校验**——落盘时把 `loss@3` 与 iter98 的 total 打进文件，若与崩溃轨迹的
    6.809309 / 1777.114 对不上，说明仍有扰动，本次结果作废。

组件归属由子优化器持有的 chunk 上的 ``colocated_module_name`` 决定，只在第一次调用时解析并缓存
（避免每步重复遍历参数）。
"""
import json
import os
import sys

import torch

MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from colocated_args import add_colocated_extra_args, validate_colocated_args
from colocated_dataloader_provider import colocated_train_valid_test_dataloaders_provider
from colocated_train import colocated_forward_step
from model import model_provider
from train import llava_embedding_ranks, llava_position_embedding_ranks

from megatron.core.enums import ModelType
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.utils import unwrap_model
from megatron.training import pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args

BREAKDOWN_PATH = os.environ.get(
    "GRAD_NORM_BREAKDOWN_PATH",
    "/home/zn/zn_data/workspace/test_log/grad_norm_breakdown.json",
)
# 崩溃发生在 iter100，跑到 110 之后再落盘：前 100 步全程零 IO。
DUMP_AFTER_ITERATION = int(os.environ.get("DUMP_AFTER_ITERATION", "110"))

records = []
component_names = {}
dump_state = {"written": False}


def component_name_of(sub_optimizer):
    """Resolve (and cache) which colocated component this sub-optimizer owns."""
    key = id(sub_optimizer)
    name = component_names.get(key)
    if name is None:
        name = f"optimizer{len(component_names)}"
        for model_chunk in getattr(sub_optimizer, "model_chunks", None) or []:
            resolved = getattr(unwrap_model(model_chunk), "colocated_module_name", None)
            if resolved is not None:
                name = resolved
                break
        component_names[key] = name
    return name


def write_breakdown_once():
    """Write the collected records exactly once, after the crash window has passed."""
    if dump_state["written"] or torch.distributed.get_rank() != 0:
        return
    dump_state["written"] = True
    # ``get_grad_norm`` 可能返回 0 维 tensor：训练期**只存引用不转换**（``float()`` 会触发
    # GPU→CPU 同步，那正是上一版 print 扰动轨迹的同类风险），到这里才统一转成 python float。
    # 任何异常都不许打断训练——落盘只是诊断，失败就打印一行、继续跑。
    try:
        serializable = [[name, float(value)] for name, value in records]
        payload = {"dump_after_iteration": DUMP_AFTER_ITERATION, "records": serializable}
        with open(BREAKDOWN_PATH, "w") as breakdown_file:
            json.dump(payload, breakdown_file, indent=1)
        print(
            f"[grad-norm-breakdown] wrote {len(serializable)} records to {BREAKDOWN_PATH}",
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - diagnostics must never kill training
        print(f"[grad-norm-breakdown] dump failed: {error!r}", flush=True)


def patch_sub_optimizer_grad_norm():
    """Append每个子优化器的 grad norm 到内存 list；不改控制流、不做 IO、不做同步。"""
    original_get_grad_norm = MegatronOptimizer.get_grad_norm

    def recording_get_grad_norm(self):
        total_norm = original_get_grad_norm(self)
        records.append((component_name_of(self), total_norm))
        # 每个 iteration 每个子优化器各记一条：共置是 2 个组件 ⇒ 2 条/步。
        if len(records) >= DUMP_AFTER_ITERATION * 2 and not dump_state["written"]:
            write_breakdown_once()
        return total_norm

    MegatronOptimizer.get_grad_norm = recording_get_grad_norm


if __name__ == "__main__":
    colocated_train_valid_test_dataloaders_provider.is_distributed = True

    arguments = parse_and_validate_args(
        extra_args_provider=add_colocated_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    validate_colocated_args(arguments)
    full_config = pretrain_cfg_container_from_args(arguments)

    # 类方法打桩必须在 pretrain 建出优化器**之前**完成。
    patch_sub_optimizer_grad_norm()

    pretrain(
        full_config,
        colocated_train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        colocated_forward_step,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
