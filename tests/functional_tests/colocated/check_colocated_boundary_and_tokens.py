# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Single-side verification driver: boundary packet integrity + per-token accounting (Task 6.7 前置).

只在**共置一侧**运行，验证两件事（不与非共置对照）：

① **边界数据对得上**：producer 侧 phase ① 产出的包（image_embeddings / tokens / labels /
   num_image_tiles）与 consumer 侧 phase ② 收到的包，逐字段摘要必须相等，且每个 microbatch
   id 恰好被一个 producer 负责、被 consumer 消费一次。

② **per-token 分母对得上**：每个 microbatch 的 ``loss_mask.sum()`` 之和（真实 token 数），
   必须等于 backbone finalize 与 encoder finalize 各自拿到并规约之后的 num_tokens。分母偏小
   会让梯度被放大（等效步长变大），这正是共置 loss 比非共置降得快得多的首要疑点。

做法：不改生产代码，只包一层——
  * 包 ``colocated_forward_step``：encoder 分支记录产出包的摘要，consumer 分支记录收到包的
    摘要，backbone 分支把返回的 loss_func 换成记录 num_tokens 的版本；
  * 首次进入某个分支时，把该 chunk 的 ``config.finalize_model_grads_func`` 包一层，记录
    num_tokens 规约前后的值（training.py:3267 才挂上这个回调，所以只能在运行期包）。

记录在**第 2 个 iteration 的第一次 encoder 调用**处落盘：那一刻每个 rank 都恰好走完第 1 个
iteration 的全部相位，是一个所有 rank 同步到达的逻辑时刻；落盘而不是集合通信，避免在前向里
插入集合操作带来的次序风险。结果由 ``compare_colocated_boundary_dump.py`` 离线比对。
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

from megatron.core import parallel_state as mpu
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.colocated_llava_model import ColocatedViTEncoder
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import get_microbatches_for_producer
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_pg_rank, get_pg_size, unwrap_model
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args

DUMP_DIRECTORY = "/home/zn/zn_data/workspace"

records = {
    "producer_packets": {},   # microbatch id -> digest（本 rank 作为 producer 产出的包）
    "consumer_packets": {},   # microbatch id -> digest（本 rank 作为 consumer 收到的包）
    "loss_num_tokens": [],    # 末 stage 上每个 microbatch 的 loss_mask.sum()，按 microbatch 顺序
    "finalize_num_tokens": [],  # {"module":..., "before":..., "after":...}
}
state = {"encoder_calls": 0, "ids": None, "dumped": False, "wrapped_configs": set()}


def tensor_digest(tensor):
    """Deterministic, cross-rank comparable summary of one tensor."""
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sum": float(flat.double().sum().item()),
        "abs_sum": float(flat.double().abs().sum().item()),
        "first": float(flat[0].double().item()) if flat.numel() else None,
        "last": float(flat[-1].double().item()) if flat.numel() else None,
    }


def packet_digest(packet):
    """Field-wise digest of a ForwardPacket (id 字段单独取值，不参与摘要)."""
    return {
        "image_embeddings": tensor_digest(packet.image_embeddings),
        "tokens": tensor_digest(packet.tokens),
        "labels": tensor_digest(packet.labels),
        "num_image_tiles": tensor_digest(packet.num_image_tiles),
    }


def wrap_finalize_once(model_chunk):
    """Wrap this chunk's finalize callback to record num_tokens before / after the reduce.

    ``config.finalize_model_grads_func`` 是 train()（training.py:3267）在运行期挂上的，
    所以只能在第一次前向时包。per-token 模式下 finalize 会**就地**把 num_tokens 规约成
    全局值（finalize_model_grads.py:494-497），因此规约前后都要记。
    """
    config = get_model_config(model_chunk)
    if id(config) in state["wrapped_configs"]:
        return
    original = config.finalize_model_grads_func
    if original is None:
        return

    def recording_finalize(*args, **kwargs):
        # 组件名从**本次调用传入的 chunk** 上读，而不是从包装时的 chunk 上读：encoder 与
        # backbone 可能共用同一个 config 对象，那时只会包一次，但两次调用都要被正确归属。
        # Read the module name off the chunk passed to *this* call: the encoder and the
        # backbone may share one config object, so one wrapper serves both calls.
        chunks = kwargs.get("model") if "model" in kwargs else args[0]
        module_name = get_attr_wrapped_model(
            chunks[0] if isinstance(chunks, (list, tuple)) else chunks,
            "colocated_module_name",
        )
        num_tokens = kwargs.get("num_tokens")
        if num_tokens is None and len(args) > 1:
            num_tokens = args[1]
        before = float(num_tokens.item()) if torch.is_tensor(num_tokens) else num_tokens
        result = original(*args, **kwargs)
        after = float(num_tokens.item()) if torch.is_tensor(num_tokens) else num_tokens
        records["finalize_num_tokens"].append(
            {"module": module_name, "before": before, "after": after}
        )
        return result

    config.finalize_model_grads_func = recording_finalize
    state["wrapped_configs"].add(id(config))


def dump_records():
    """Write this rank's records to a json file (one file per rank)."""
    rank = torch.distributed.get_rank()
    payload = {
        "rank": rank,
        "producer_id": get_pg_rank(mpu.get_colocated_boundary_group()),
        "num_producers": get_pg_size(mpu.get_colocated_boundary_group()),
        "pipeline_rank": mpu.get_pipeline_model_parallel_rank(),
        "num_microbatches": get_num_microbatches(),
        **records,
    }
    path = os.path.join(DUMP_DIRECTORY, f"boundary_dump_rank{rank}.json")
    with open(path, "w") as dump_file:
        json.dump(payload, dump_file, indent=1)
    print(f"[rank {rank}] wrote {path}", flush=True)


def recording_loss_func(original_loss_func):
    """Record the per-microbatch num_tokens (== loss_mask.sum()) returned by loss_func."""

    def wrapper(*call_args, **call_kwargs):
        result = original_loss_func(*call_args, **call_kwargs)
        # 训练路径返回 (loss, num_tokens, reporting)；eval 的 non_loss_data 路径返回别的形状，
        # 那时不记录。The training path returns a 3-tuple; other shapes are left alone.
        if isinstance(result, tuple) and len(result) == 3:
            records["loss_num_tokens"].append(float(result[1].item()))
        return result

    return wrapper


def instrumented_forward_step(data_iterator, model, packet=None, intra_packet=None):
    """``colocated_forward_step`` 的记录版：不改变任何返回值语义，只旁路记录。"""
    model_chunk = model[0] if isinstance(model, (list, tuple)) else model
    wrap_finalize_once(model_chunk)
    is_encoder_branch = isinstance(unwrap_model(model_chunk), ColocatedViTEncoder)

    if is_encoder_branch:
        if state["ids"] is None:
            boundary_group = mpu.get_colocated_boundary_group()
            state["ids"] = get_microbatches_for_producer(
                get_pg_rank(boundary_group), get_num_microbatches(), get_pg_size(boundary_group)
            )
        # 第 2 个 iteration 的第一次 encoder 调用：所有 rank 都刚走完第 1 个 iteration 的
        # 全部相位，此刻落盘的记录完整且各 rank 同步到达。
        if state["encoder_calls"] == len(state["ids"]) and not state["dumped"]:
            dump_records()
            state["dumped"] = True

    result = colocated_forward_step(
        data_iterator, model, packet=packet, intra_packet=intra_packet
    )

    if state["dumped"]:
        return result

    if is_encoder_branch:
        produced_packet, _ = result
        # id 按 phase ① 的循环顺序取（schedule 在 forward_step_func 返回**之后**才给包打上
        # microbatch_id，这里拿不到）。若这个顺序假设不成立，producer/consumer 摘要就会对不上，
        # 由离线比对的第 ② 项直接暴露，而不会静默通过。
        # The id follows phase ①'s loop order, because the schedule stamps microbatch_id only
        # after forward_step_func returns. A wrong assumption here surfaces as a digest
        # mismatch in check ②, never as a silent pass.
        microbatch_id = state["ids"][state["encoder_calls"] % len(state["ids"])]
        records["producer_packets"][str(microbatch_id)] = packet_digest(produced_packet)
        state["encoder_calls"] += 1
        return result

    if packet is not None:
        records["consumer_packets"][str(int(packet.microbatch_id.item()))] = packet_digest(packet)
    output_tensor, loss_func = result
    return output_tensor, recording_loss_func(loss_func)


if __name__ == "__main__":
    # 入口与 colocated_train.py 完全一致，只把 forward_step_func 换成记录版。
    # --train-iters 必须 >= 2：记录在第 2 个 iteration 的第一次 encoder 调用处落盘。
    colocated_train_valid_test_dataloaders_provider.is_distributed = True

    arguments = parse_and_validate_args(
        extra_args_provider=add_colocated_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    validate_colocated_args(arguments)
    assert arguments.train_iters >= 2, (
        "this driver dumps at the start of the second iteration, so --train-iters must be >= 2"
    )
    full_config = pretrain_cfg_container_from_args(arguments)

    pretrain(
        full_config,
        colocated_train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        instrumented_forward_step,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )



