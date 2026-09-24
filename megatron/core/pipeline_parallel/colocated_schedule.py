# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Colocated encoder schedule wrapper (Task 4).

共置训练的 schedule wrapper：encoder 前传（phase ①）与统一反传（phase ④）放在
现有 backbone 1F1B 流水之外，backbone 流水复用现有 schedule（phase ②，Task 4.2）。
与其余 schedule 一致，本模块提供**顶层函数** ``forward_backward_colocated``
（配套私有辅助函数），经 ``get_forward_backward_func`` 插件接入（Task 4.5），
train_step 零改动。

- 本模块只依赖 ``megatron.core``（不依赖 examples/*）。
- **4.1 实现**：``forward_backward_colocated``（签名对齐
  ``forward_backward_pipelining_without_interleaving``，train_step 以关键字参数
  原样调用）+ phase ① ``_colocated_encoder_forward``（encoder 轮盘前传 -> 本地
  buffer）。
- 4.2/4.3/4.4 后续接入 backbone schedule、``colocated_forward_step``（含梯度
  hook）与统一 encoder 反传。

microbatch 分配（三层，见 doc.md §2.11）：总 micro batch 数 ``global_mbs = global_batch /
micro_batch_size``；每 data parallel 副本（外层）的流水 micro batch 数 = ``global_mbs /
D_outer``（即 Megatron 的 ``num_microbatches``）；副本内轮盘——producer p 负责
microbatch p, p+P, p+2P, ...，即每个 producer 处理 ``num_microbatches / P = global_mbs /
(dp * inner_dp)`` 个。
"""

import contextlib
import os
import sys
import time

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.colocated_encoder_comm import (
    BackwardPacket,
    EncoderBackboneBoundaryCommunicator,
    ForwardPacket,
    _GradRecvRequest,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_step,
    get_tensor_shapes,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import colocated_encoder_rng_tracker
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    group_colocated_model_chunks,
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)


# dual-channel-p2p 诊断探针（2026-09-22）：env COLOCATED_CPU_PROBE=1 打开后，在 backbone schedule 的
# 各相位与边界收发前后打印 CPU 侧执行位置（flush 到 stderr，带全局 rank 与时间戳），用于挂起时定位每个
# rank 的 CPU 最后走到哪一步、卡在哪个 microbatch（BEGIN 有、END 无即卡在该步）。关闭时每次调用只做一次
# 布尔判断，无其它开销，生产不受影响；与已有 NVTX 区间互补，后续可直接用 nsys 采样。
# dual-channel-p2p diagnostic probe (2026-09-22): with env COLOCATED_CPU_PROBE=1, print the CPU-side
# execution point (flushed to stderr, tagged with global rank + timestamp) around each backbone
# schedule phase and boundary send/recv, so a hang shows where each rank's CPU last got and on which
# microbatch (a BEGIN with no matching END marks the blocked step). When disabled each call is a
# single boolean check with no other cost; complements the existing NVTX ranges for later nsys use.
_COLOCATED_CPU_PROBE = os.environ.get("COLOCATED_CPU_PROBE", "0") == "1"


def _cpu_probe(message: str) -> None:
    """Print a flushed, rank-tagged CPU-execution marker to stderr when COLOCATED_CPU_PROBE=1."""
    if not _COLOCATED_CPU_PROBE:
        return
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    print(
        f"[cpu-probe rank{global_rank} {time.time():.3f}] {message}",
        file=sys.stderr,
        flush=True,
    )


@dataclass
class IntraPacket:
    """Handoff carrier between the schedule and the injected forward step (Task 4.3j).

    Carries the expanded labels/loss_mask of the backbone P2P accompaniment in both
    directions without storing them on the model (replacing the old
    ``colocated_new_labels``/``colocated_new_loss_mask`` model attributes):
    - consumer (pre_process): ``colocated_forward_step`` writes the assembled
      ``new_labels``/``new_loss_mask`` back into this carrier; the schedule reads
      them after the ``forward_step`` helper (schedules.py) returns and sends them
      down the pipeline (``_send_intra_packet``).
    - non-consumer: the schedule fills it from ``_recv_intra_packet`` and binds it into
      the forward step via ``functools.partial``; the last stage uses them to
      compute the loss.

    schedule 与注入的 forward step（``colocated_forward_step``）之间关于展开
    labels/loss_mask 的交接载体（4.3j 重构，替代原模型属性
    ``colocated_new_labels``/``colocated_new_loss_mask``，模型不再充当 schedule
    mailbox）：consumer 方向由 ``colocated_forward_step`` 写回、schedule 在
    ``forward_step`` 返回后读取；非 consumer 方向由 schedule 填充、经
    ``functools.partial`` 闭包绑定进 forward step。
    """

    labels: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None


# NVTX（nsys 时间线用）：共置 schedule 的三大相位各打一个区间——encoder 前传/反传是
# 共置独有的计算块，backbone 1F1B 内部再分 colocated-warmup/steady/cooldown。区间名统一
# 加 colocated- 前缀，与标准 schedules.py 的同名区间（"warmup"/"steady"/"cooldown"，
# schedules.py:1449-2020）区分开；nvtx_range_push/pop 对 colocated- 前缀的消息自动追加
# _rank<N> 后缀（megatron/core/utils.py），多 rank 合一份报告时区间可按 rank 区分。
# _nvtx_enabled 门控（--nvtx-ranges 才打开）保证关闭时只有函数调用开销。
# NVTX (for the nsys timeline): one range per colocated phase. The encoder forward /
# backward are colocated-only compute blocks; the backbone 1F1B loop is further split
# into colocated-warmup/steady/cooldown. The colocated- prefix distinguishes them from
# the standard schedules.py ranges of the same name; nvtx_range_push/pop append an
# automatic _rank<N> suffix to colocated- messages so per-rank ranges stay
# distinguishable when several ranks share one report.
@nvtx_decorator(message="colocated-iteration")
def forward_backward_colocated(
    *,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    **kwargs,  # adjust_tensor_shapes_fn / force_all_reduce 等（train_step 会传，4.3 起处理）
):
    """Run one colocated step: ① encoder forward -> ② backbone -> ④ encoder backward.

    共置 schedule 的顶层入口（与其余 schedule 一致）：model 必须是
    ``[encoder_chunk, backbone_chunk]``（Task 2 的 provider 返回）。签名与
    ``forward_backward_pipelining_without_interleaving``（schedules.py:2035）对齐
    （关键字参数），train_step（training.py:1902）原样调用。

    职责分工（2026-08-12 用户确认，对照 schedules.py）：本函数只做 orchestration——
    phase ① 循环调注入的 ``forward_step_func`` 的 encoder 分支拿包存 buffer；phase ②
    调 1F1B；phase ④ 直接 autograd。数据获取 / 模型 forward / loss 全在注入的
    ``forward_step_func``（``colocated_forward_step``，见 examples/multimodal/
    colocated_train.py），schedule 不接收 ``get_batch_fn`` / ``image_token_index`` /
    ``img_seq_len`` 等数据/模型参数（train_step 也不传它们，training.py:1902）。

    Args:
        forward_step_func / data_iterator / num_microbatches / seq_length /
        micro_batch_size / decoder_seq_length / forward_only / collect_non_loss_data:
            与现有 schedule 契约一致（train_step 传入）。
            forward_step_func 按 ``model[0]`` 的 chunk 类型分支：encoder（phase ①，
            本函数传 ``model=[encoder_chunk]``，返回 ``(ForwardPacket, None)``）/
            backbone（phase ②，1F1B 传 ``model=[backbone_chunk]``，返回
            ``(output, loss_func)``）。

    当前实现：phase ①（4.2：已建边界 communicator、phase ① 走 forward step 的
    encoder 分支）→ phase ②（4.3c：统一自写 1F1B 循环，PP=1 时 P2P 空转、全本地 take）
    → **phase ④（统一 encoder 反传，4.6e）**——三段都是本函数编排的**分段解耦子过程**，
    phase ② 只负责 backbone 流水本身，不承担全流程收尾。
    """
    # 按 chunk 自身声明的 ``colocated_module_name`` 取出两个组件，而不是按下标拆
    # （与训练入口构建 model 列表、串接优化器时用的是同一个函数，core/utils.py）。
    # 每个组件恰好一个 chunk 是当前最小实现的前提：encoder 的 pipeline 组是单成员组、
    # 不进 VPP 分支；backbone 的 VPP 属于 Task 8 范围，届时这里改为遍历 chunk 列表。
    # Split by each chunk's own colocated_module_name rather than by position, reusing the
    # same helper the training entry uses to build the model list and chain the optimizers.
    # Exactly one chunk per component is the current minimal implementation's premise.
    chunks_per_module = group_colocated_model_chunks(model)
    encoder_chunks = chunks_per_module["encoder"]
    backbone_chunks = chunks_per_module["language_model"]
    assert len(encoder_chunks) == 1 and len(backbone_chunks) == 1, (
        "colocated schedule needs exactly one chunk per component, got "
        f"{len(encoder_chunks)} encoder and {len(backbone_chunks)} language_model chunks "
        f"out of {len(model)} model chunks"
    )
    encoder_chunk = encoder_chunks[0]
    backbone_chunk = backbone_chunks[0]
    config = get_model_config(backbone_chunk)  # backbone config（dtype 等）

    # num_microbatches 必须是 pipeline_parallel_size 的整数倍（轮盘要求：每 producer
    # 处理 num_microbatches / P 个 microbatch）。Round-robin requires num_microbatches % P == 0.
    parallel_state.validate_colocated_num_microbatches(num_microbatches)

    # 边界通信器：两个方向隔离的独立共置边界组（dual-channel-p2p Task 2，2026-09-22）——
    # activation 组走前向激活 producer→consumer，grad 组走反向梯度 consumer→producer，各
    # 自独立 NCCL 实例 + 内部 stream，成员与 pp 组相同。4.3 起 phase ②/④ 使用。
    # Boundary communicator on the two direction-split colocated groups (independent
    # NCCL instances); used by phases ②/④ from Task 4.3 on.
    comm = EncoderBackboneBoundaryCommunicator(
        parallel_state.get_colocated_boundary_activation_group(),
        parallel_state.get_colocated_boundary_grad_group(),
        config,
    )

    # Boundary/PP warmup is NOT done here: both NCCL communicators are eagerly built
    # ONCE at init time inside initialize_model_parallel (use_colocated_encoder block),
    # which is the zero-traffic point; see the "one-time eager warmup" block there.
    # 边界/PP 预热不在这里做：两个 NCCL communicator 已在 initialize_model_parallel
    # （use_colocated_encoder 块）的初始化阶段一次性建好——那里是零流量时点；见该处的
    # "one-time eager warmup" 块。

    # Phase ①：encoder **合并**前传 -> 本地 buffer + 合并张量（优化 spec Task 1，设计 A）。
    # schedule 每个 iteration 只调一次 forward_step_func 的 encoder 分支（一次取数 +
    # 一次前传），再由 ForwardPacket.split_merged_batch 等分回逐 microbatch 的包。producer
    # 槽位取自边界组内的编号（``comm.producer_id``）而不是 pipeline rank：包的收发
    # 路由用的就是这个编号，槽位与路由必须同源，否则一旦 rank order 变化、两者不再
    # 重合，microbatch 归属与发包目标就会错配（Task 5.7）。
    # Phase ①: ONE merged encoder forward -> local buffer + the merged tensor
    # (optimization spec Task 1, design A). forward_step_func's encoder branch is called
    # ONCE per iteration (one fetch + one forward), then ForwardPacket.split_merged_batch
    # splits it back into per-microbatch packets. The producer slot comes from the
    # boundary group's own rank (``comm.producer_id``), the very id used to route the
    # packets, instead of the pipeline rank — slot and routing must share one source.
    encoder_buffers, merged_image_embeddings = _colocated_encoder_forward(
        forward_step_func,
        data_iterator,
        encoder_chunk,
        num_microbatches,
        producer_id=comm.producer_id,
        num_producers=comm.group_size,
    )
    # Phase ②：backbone 1F1B schedule（Task 4.3c，2026-08-13 用户确认统一走自写循环）。
    # 自写循环天然覆盖 PP=1：P=1 时 producer 恒 0、is_consumer=True、全部本地 take、
    # P2P 空转（P2PCommunicator 在 first=last 时全部方法安全跳过/返回 None）——退化为
    # 标准 DP，不再单独走 no-pipelining schedule（也避免"no-pipelining 不感知包"
    # 的额外包装）。它是**分段解耦的子函数**，返回 backbone 的 loss（forward_data_store，
    # 与 1F1B 一致，train_step 在 training.py:1902 直接消费）+ 本 rank 的边界梯度
    # （producer_grad_buffers，交给 phase ④）。
    # Phase ②: backbone schedule (Task 4.3c). The self-written loop also covers PP=1
    # (P=1: producer is always 0, is_consumer=True, all-local take, P2P no-ops), so no
    # separate pp-size dispatch is needed. It is a decoupled sub-phase returning the loss
    # store plus this rank's boundary grads.
    forward_data_store, producer_grad_buffers, num_tokens_for_encoder = (
        colocated_backbone_forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=[backbone_chunk],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            adjust_tensor_shapes_fn=kwargs.get("adjust_tensor_shapes_fn"),
            force_all_reduce=kwargs.get("force_all_reduce", False),
            comm=comm,
            encoder_buffers=encoder_buffers,
        )
    )

    # Phase ④：合并 encoder 反传 + encoder 自己的梯度收尾（4.6e；优化 spec Task 1 合并
    # 反传）——与 phase ① 对称的**独立一段**，由本函数（全流程编排者）调用，不塞进
    # phase ② 里面。forward_only（eval）没有梯度，跳过。per-token 模式才传 token 数
    # （与 finalize_model_grads 的调用约定一致）。合并张量由 phase ① 交回、本处转交。
    # Phase ④: the merged encoder backward plus the encoder's own grad finalize (4.6e;
    # optimization spec Task 1 merged backward) — a separate phase, symmetric with
    # phase ①, driven by this function rather than nested inside phase ②. The merged
    # tensor comes back from phase ① and is forwarded here.
    if not forward_only:
        _colocated_encoder_backward(
            encoder_chunk,
            merged_image_embeddings,
            producer_grad_buffers,
            num_tokens=(
                num_tokens_for_encoder if config.calculate_per_token_loss else None
            ),
        )

    return forward_data_store


@nvtx_decorator(message="colocated-encoder-forward")
def _colocated_encoder_forward(
    forward_step_func,
    data_iterator,
    encoder_chunk,
    num_microbatches: int,
    producer_id: int,
    num_producers: int,
) -> Tuple[Dict[int, ForwardPacket], torch.Tensor]:
    """Phase ①: ONE merged encoder forward over the rank's whole iteration, then split.

    优化 spec Task 1（设计 A）：dataloader 的 batch_size 已在 provider 侧提到合并粒度
    ``micro_batch_size * num_microbatches / num_producers``，因此本函数对
    ``forward_step_func`` 的 encoder 分支**每个 iteration 只调一次**——一次取数（1 次
    ``next()`` + 1 组 broadcast）+ 一次 ``encoder_chunk`` 前传，替代原先"逐 microbatch
    取数+前传"的循环（24 层 ViT 的 per-layer Python/eager 调度成本原本被支付了 16 次，
    而每层 GPU 只需 0.33 ms，是 encoder 相位 93.7% GPU 空闲的根因）。

    本 rank（producer p）负责的 microbatch 由 owner 表决定（``get_colocated_owned_microbatches``，
    升序，支持任意/逆序划分；dual-channel-p2p Task 3 起取代只支持轮盘的
    ``get_microbatches_for_producer``），与合并批的 batch 维一一对应（owned 升序 == batch
    顺序）。合并批由 ``ForwardPacket.split_merged_batch``（classmethod 直接吃裸张量）等分回
    逐 microbatch 的包；schedule 只负责逐包打 ``microbatch_id``（1 元素 int64 张量，
    consumer 在 ``_take_boundary_packet`` 校验）并存入 ``encoder_buffers``。

    Returns:
        (encoder_buffers, merged_image_embeddings)：

        - ``encoder_buffers[microbatch]``：逐 microbatch 的包，phase ② 边界通信按现
          契约消费（producer 发送 / consumer 本地直传），字段 layout 不变；
        - ``merged_image_embeddings``：合并张量本体 ``[img_seq_len, merged_batch,
          h_lang]``，**保留 grad_fn**——各包持有的只是它的 view，而 phase ④ 的单次
          backward 必须作用在图输出对象上（从 view 拿不回父张量，逐 view 反传又会
          16 次遍历整图），所以由本函数交还调用方、传给 phase ④。

    注意：这里**不 detach**——切断图发生在 consumer 组装时（4.6b 的
    ``_take_boundary_packet`` 对喂给 backbone 的那份做 ``detach().requires_grad_(True)``，
    backbone 反传不进入 encoder 图；本地合并图保留给 phase ④ 反传）。
    """
    # dual-channel-p2p Task 3：本 producer 拥有的 microbatch 由 owner 表决定（升序，支持任意/
    # 逆序划分），取代只支持轮盘整除的 get_microbatches_for_producer。
    # This producer's owned microbatches come from the owner table (ascending, any partition).
    microbatches = parallel_state.get_colocated_owned_microbatches(producer_id)

    # Task 5.12: the whole encoder forward runs with the ENCODER's own RNG tracker
    # installed globally, not merely under one forked named state. Two kinds of random
    # points have to be covered and a single fork covers only the first:
    #   - the ViT hidden dropout (bda, fused_bias_dropout.py:47) draws from the GLOBAL
    #     cuda RNG without touching the tracker;
    #   - the attention dropout (dot_product_attention.py:217) forks the DEFAULT state
    #     name, i.e. "model-parallel-rng", and an inner fork overrides any outer one.
    # The backbone is seeded last, so the tracker's "model-parallel-rng" holds the
    # BACKBONE's state, whose seed contains the backbone pipeline rank - different on
    # every stage. Consuming it here would give the encoder replicas different dropout
    # masks, i.e. they would stop computing the same function, and gradient reduction
    # would average inconsistent results. Swapping the tracker itself redirects every
    # fork inside the forward, default-named ones included.
    # Task 5.12：整个 encoder 前传都在**换入 encoder 自己的 RNG tracker** 下运行（合并
    # 后仍是"一次前传包住整段"），理由同上：hidden dropout 吃全局 RNG、attention dropout
    # fork 默认状态名，只有换掉 tracker 本身才能全部重定向。
    with colocated_encoder_rng_tracker():
        # forward step 的 encoder 分支：取合并批数据 + encoder_chunk(images) 只调一次，
        # 返回 (MergedEncoderBatch, None)——裸张量集合，非 ForwardPacket（合并批永不
        # 上线路，见 examples/multimodal/colocated_train.py 的 MergedEncoderBatch）。
        # 取数与前传逻辑全在 ``colocated_forward_step``，schedule 只负责拆分、打标与存储。
        merged_batch, _ = forward_step_func(data_iterator, [encoder_chunk])
        # 合并张量本体（图输出对象）——schedule 持有它，phase ④ 单次 backward 用。
        merged_image_embeddings = merged_batch.image_embeddings

    # 拆分在包定义旁（ForwardPacket.split_merged_batch，classmethod 直接吃裸张量），
    # batch 顺序 == 轮盘序列（同为升序）；切片是 view，通信器序列化时逐字段 contiguous
    # （扁平化拷贝本来就免不了），本地路径消费 view 也没问题。 #划分为micro batch packet，一个packet里面包含micro batch size个样本
    per_microbatch_packets = ForwardPacket.split_merged_batch(
        image_embeddings=merged_batch.image_embeddings,
        tokens=merged_batch.tokens,
        labels=merged_batch.labels,
        num_image_tiles=merged_batch.num_image_tiles,
        num_splits=len(microbatches),
    )

    encoder_buffers: Dict[int, ForwardPacket] = {}
    for index, microbatch in enumerate(microbatches):
        # 逐包打 microbatch id（schedule 的元数据，serialize 前必须已打标）。
        packet = per_microbatch_packets[index]
        packet.microbatch_id = torch.tensor(
            [microbatch], dtype=torch.int64, device=packet.image_embeddings.device
        )
        encoder_buffers[microbatch] = packet

    # 完整性：buffer 的键恰好是 owner 表给出的 owned 列表（升序）。不同划分下每 producer 的负载
    # 可不均（如 reverse_block），故不再断言 == num_microbatches / P。
    # Completeness: buffer keys equal this producer's owned list (per-producer counts may differ
    # under non-round-robin partitions, so no balanced-count assertion).
    assert sorted(encoder_buffers) == microbatches, (
        f"buffer keys {sorted(encoder_buffers)} != owned microbatches {microbatches}"
    )
    return encoder_buffers, merged_image_embeddings


@nvtx_decorator(message="colocated-encoder-backward")
def _colocated_encoder_backward(
    encoder_chunk,
    merged_image_embeddings: torch.Tensor,
    producer_grad_buffers: Dict[int, torch.Tensor],
    num_tokens: Optional[torch.Tensor] = None,
) -> None:
    """Phase ④: MERGED encoder backward + the encoder's own grad finalize (4.6e; Task 1).

    每个 rank 在**backbone 流水全部前传与反传结束后**统一做自己的 encoder 反传。优化
    spec Task 1 起反传也合并：把 ``num_microbatches / P`` 份边界梯度沿 dim=1 ``cat`` 回
    ``[img_seq_len, merged_batch, h_lang]``，对 phase ① 交回的**合并张量**调**一次**
    ``torch.autograd.backward``——每层 wgrad 只吃一次 batch 维合并的梯度，图只遍历一遍，
    替代原先 16 次"逐 view backward、每次遍历整图"的 launch-bound 反传。最后做 encoder
    自己的梯度收尾（复用 ``finalize_model_grads``，传 encoder 自己的 ``pg_collection``：
    DDP 在 colocated dp 组上的一次 SUM 同时完成"跨副本"与"副本内轮盘"两个数据并行维的
    求和，随后按全局 token 数归一化）。

    梯度来源对每个 rank 完全同构（consumer 用 4.6c 本地留存的梯度、producer 用 4.6d
    收到的梯度，两者都在自己的 ``producer_grad_buffers`` 里，每份形状
    ``[img_seq_len, mbs, h_lang]``，与合并张量的 dim=1 切片同形同序——顺序由两侧共用
    同一份升序轮盘序列保证）。反传对象是 phase ① 的**合并张量**（带 encoder 计算图的
    图输出对象），不是各包持有的 view，也不是 4.6b ``detach()`` 出来的切断点。

    选项 B（统一反传）而非选项 A（补货 step 立即反传）：后者会让 ViT 反传穿插进 1F1B、
    拖慢流水（doc §2.11，2026-08-12 用户定）。合并反传进一步把"逐 mb 反传"的 n/P 次
    图遍历压成 1 次。全部梯度一次性消费后即从 buffer 里移除，encoder 激活图随 backward
    释放。

    **DDP grad sync 纪律（合并后退化为最简形式）**：只有**一次** backward，它天然就是
    "最后一次"——直接在 no_sync 之外调用即触发 DDP 的桶归约（``overlap_grad_reduce`` 下
    与反传重叠），之后由收尾里的 ``finish_grad_sync`` 等它完成。原先"除最后一个 mb 外
    都进 no_sync"的进出逻辑随之消失（对裸模型（4.8 数值对照，没有 DDP）与未开
    ``overlap_grad_reduce`` 的场景同样成立：那时反传期间根本不发起桶归约）。

    Args:
        merged_image_embeddings: phase ① 交回的合并输出张量（图输出对象，保留 grad_fn）。
        num_tokens: per-token 模式（``calculate_per_token_loss=True``）下 phase ② 交回的
            **未规约**本 rank token 数；其余模式传 None。
    """
    microbatches = sorted(producer_grad_buffers)
    # cat 顺序 = phase ① 的切片顺序（轮盘序列同为升序）；总量对齐合并张量的 batch 维。
    # The cat order equals phase ①'s slice order (both round-robin ascending); the total
    # must match the merged tensor's batch dimension.
    boundary_grads = [producer_grad_buffers.pop(microbatch) for microbatch in microbatches]
    assert sum(grad.shape[1] for grad in boundary_grads) == merged_image_embeddings.shape[1], (
        f"boundary grads batch dim sum "
        f"({sum(grad.shape[1] for grad in boundary_grads)}) != merged encoder output "
        f"batch dim ({merged_image_embeddings.shape[1]})"
    )
    # 单份梯度时 torch.cat 也无害，但直接用原张量省一次拷贝。
    full_boundary_grad = (
        torch.cat(boundary_grads, dim=1) if len(boundary_grads) > 1 else boundary_grads[0]
    )
    assert full_boundary_grad.shape == merged_image_embeddings.shape, (
        f"merged boundary grad shape {tuple(full_boundary_grad.shape)} != encoder output "
        f"shape {tuple(merged_image_embeddings.shape)}"
    )
    # no_sync 不需要了：DDP 只看 backward 触发次数，不看 batch 里有几个样本——合并后
    # 只有一次 backward（16 份梯度已在图内合并），天然就是"退出 no_sync 的那一次"，
    # 在 no_sync 外调用即触发唯一一次桶归约，"区分最后一个 mb"的进出配对随之消失。
    torch.autograd.backward(merged_image_embeddings, grad_tensors=full_boundary_grad)
    del boundary_grads, full_boundary_grad

    # encoder 的梯度收尾走 ``config.finalize_model_grads_func``（与 backbone 相位同一个入口，
    # 不再函数内延迟 import）：Task 5 起 encoder 带着自己的 ``pg_collection``（每次 get_model
    # 调用只有一种拓扑），``finalize_model_grads`` 的每一段都按 encoder 自己的组判断——DDP
    # 归约走 colocated dp 组（全 W，一次 SUM 同时覆盖"跨副本"与"副本内轮盘"两个数据并行维，
    # finalize_model_grads.py:446-447，doc §2.9）；conditional embedding 与非 TP 参数两段被
    # pp/tp 单成员的门挡掉（:104、:333，后者在 encoder 支持 TP 后自动生效）；word/position
    # embedding 两段因 ``embd``/``pos_embd`` 为 None 而跳过
    # （parallel_state.build_colocated_encoder_process_groups）。
    # The encoder's grad finalize goes through config.finalize_model_grads_func, the same
    # entry point the backbone phase uses. The config comes from the encoder chunk's OWN
    # wrapper chain (the no_sync block that used to fetch it earlier is gone with the
    # merged single-backward discipline).
    config = get_model_config(encoder_chunk)
    if config.finalize_model_grads_func is not None and hasattr(
        encoder_chunk, "finish_grad_sync"
    ):
        # per-token 的分母：encoder 的 pipeline 组只有一个成员，finalize_model_grads.py:494
        # 的 broadcast 因此是空操作，全局 token 数完全由 :497 在 colocated dp 组（全 W）上的
        # 一次 SUM 得到——每个副本内只有 backbone 末 stage 的 rank 持有非零值，求和即全局
        # 总数。这依赖上游"中间 stage 不算 loss、num_tokens 恒为 0"（schedules.py:262-269），
        # 由 Task 5.8 的数值用例守（分母错 P 倍会直接反映在梯度上），不在训练主循环里加
        # 断言——那需要 .item() 同步，且属于为当前不可能发生的状态加防御。
        # The per-token divisor: the encoder's pipeline group is single-member, so that
        # broadcast is a no-op and the global token count comes solely from the SUM over the
        # colocated data-parallel group - only the backbone last stage of each replica holds a
        # non-zero value. That upstream assumption is guarded by the Task 5.8 numerical test,
        # not by a runtime assert (which would need a .item() sync every iteration).
        # NVTX：encoder 的梯度收尾（DP 归约 + per-token 归一化）是通信爆发段，单独打区间。
        # NVTX: the encoder grad finalize (DP reduce + per-token normalization) is a
        # communication burst - give it its own range.
        nvtx_range_push("colocated-encoder-grad-finalize")
        config.finalize_model_grads_func(
            [encoder_chunk],
            num_tokens,
            pg_collection=get_attr_wrapped_model(encoder_chunk, "pg_collection"),
        )
        nvtx_range_pop("colocated-encoder-grad-finalize")


def colocated_backbone_forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
    # --- colocated 特有参数（4.3b）---
    comm: Optional[EncoderBackboneBoundaryCommunicator] = None,
    encoder_buffers: Optional[Dict[int, ForwardPacket]] = None,
):
    """Run non-interleaved 1F1B schedule for the colocated backbone (phase ②).

    Task 4.3：从 ``forward_backward_pipelining_without_interleaving``（schedules.py:2035）
    复制的 1F1B 骨架（warmup/steady/cooldown、P2P 配对节奏、backward 的 input/output
    tensor 队列、deallocate、grad_sync 时机逐行对照），并做 **colocated 适配**：
    - 4.3a：去掉 MultiModule 分支（colocated backbone 是单 GPTModel），``backward_func``
      直接用 ``backward_step``；复用 ``forward_step``/``backward_step``/``get_tensor_shapes``
      等辅助函数与 ``P2PCommunicator``（backbone stage 间 P2P）。
    - 4.3b：接入**边界通信器**（``comm``）——producer（stage>0）在进入流水前**异步**
      发送 phase ① 的全部包（isend 入队不等待，wait 推迟到函数末尾统一做）；consumer
      （stage 0）在每个前传 microbatch 前 **take 包**（producer 0 本地直传
      ``encoder_buffers[microbatch]`` / producer>0 ``colocated_recv_forward(wait=True)``），
      用 ``functools.partial`` 绑定到 forward step 传给 ``colocated_forward_step``
      （packet 走闭包、**不经模型属性**）。4.4 把"全量发送"改为"先发首包 + 循环内
      按需补发"的流水化；4.5 把 consumer 收包改 prefetch 异步。

    注意：本模块顶层 import schedules.py 的辅助函数；4.7 在 ``get_forward_backward_func``
    接入 colocated 分支时必须用**函数内延迟 import**（``from .colocated_schedule import
    forward_backward_colocated``），避免模块级循环 import。

    Args:
        与 ``forward_backward_pipelining_without_interleaving``（schedules.py:2035）一致，
        另加 colocated 特有参数：
        comm: 边界通信器（EncoderBackboneBoundaryCommunicator，Task 4.2c 建）；
        encoder_buffers: phase ① 结果 ``Dict[int, ForwardPacket]``（本 rank 负责的
        microbatch 的 encoder 输出 + 文本字段）。forward_step_func 传
        colocated_forward_step 的 backbone 分支
        （``(data_iterator, model) -> (output, loss_func)``）。

    Returns:
        list: forward_data_store（与 1F1B 一致，losses 等）。
    """
    if isinstance(model, list):  # 非交错 1F1B 不会有多个模型切分
        assert len(model) == 1, (
            "non-interleaved pipeline-parallel schedule does not support model chunking"
        )
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, (
            "non-interleaved pipeline-parallel schedule does not support model chunking"
        )
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:  # 非交错 schedule 中 P2P 通信是同步的，不支持 overlap
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    tp_group, cp_group, cp_size = None, None, None

    if p2p_communicator is None and pg_collection is None:
        # Default: single-module with parallel_state groups.
        # 默认：用 parallel_state 的组构建 P2P 通信器与集合组。
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        if isinstance(pg_collection, ProcessGroupCollection):
            # Single-module: extract tp/cp groups and cp_size.
            # 单模块：从 pg_collection 提取 tp/cp 组。
            assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
            assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()
        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection, "
                f"got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")

    # Needed only when gradients are finalized in M-Core.
    # 仅当梯度在 M-Core 收尾时需要：清空 embedding 激活 buffer。
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions.
    # 禁用异步 DDP 梯度通信。
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions."""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions."""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # --- colocated 边界适配（4.3b）---
    # --- colocated boundary adaptation (4.3b) ---
    assert comm is not None, "colocated loop needs the boundary communicator (Task 4.2c)"
    assert encoder_buffers is not None, (
        "colocated loop needs the phase-① encoder_buffers (Dict[int, ForwardPacket])"
    )
    # 角色判断一律由边界通信器提供（逻辑隔离：通信器知道"我是谁"——消费者/生产者
    # 及边界组大小；schedule 不做 pp rank 硬编码，TP>1 时 consumer 判断同样成立）。
    # Role queries come from the boundary communicator (the communicator knows whether
    # this rank is the consumer, its producer id, and the boundary group size; the
    # schedule must not hardcode pipeline-parallel-rank == 0, which breaks under TP>1).
    is_consumer = comm.is_consumer()
    producer_id = comm.producer_id

    # dual-channel-p2p Task 3：本 producer 拥有的 microbatch（升序，owner 表为权威来源，支持
    # 任意/逆序划分），以及已 isend 出去的集合（前传供给按 i+x+1 增量发送，避免重发）。owner 0
    # （consumer）的 owned 本地直传、不 isend，故该集合对 consumer 恒空。
    # This producer's owned microbatches (ascending, from the owner table) and the set already
    # isent — the i+x+1 forward supply sends incrementally and never resends.
    owned_microbatches = parallel_state.get_colocated_owned_microbatches(producer_id)
    producer_sent_activations = set()

    # dual-channel-p2p Task 3/5：consumer 的边界接收就一条 one-ahead 规则——primer(mb0) 引导后，
    # 前传 mb i 前 post mb i+1 的 recv；峰值在飞接收 ~2 份（正在 take 的 + 刚 prefetch 的），天然
    # <= P+1，无需显式限窗。未匹配的 P2P recv 不占自旋 kernel、不阻塞 cudaMalloc（已实测），故提前
    # post 无死锁风险。prefetched 按 microbatch 号索引（owner 表任意划分下相邻 mb 可能同 owner）。
    # dual-channel-p2p Task 3/5: the consumer's boundary receive is a single one-ahead rule
    # (primer for mb 0, then before forwarding mb i post the recv of mb i+1); peak in-flight
    # receives ~2, naturally <= P+1. Unmatched P2P recvs never block cudaMalloc (measured), so
    # posting ahead is deadlock-free. prefetched is keyed by microbatch (owner table, any split).
    prefetched = {}

    # dual-channel-p2p Task 4：反向梯度传输的状态容器。
    # - producer_grad_buffers：microbatch → encoder 输出梯度。每个 rank 只存本 rank 自己产的
    #   microbatch（owner==producer_id 的；consumer 算出的非本地梯度立刻发走）——phase ④ 统一
    #   encoder 反传时逐个取用。
    # - producer_grad_requests：本 producer 已 POST 但未等数据的 owned grad irecv（owned mb →
    #   _GradRecvRequest）。每次自身反传后按 i-x 界增量 POST，phase④ 前由 _finish_owned_grads
    #   统一等数据落地，使梯度数据传输与 backbone cooldown/后续计算重叠。
    # - consumer_boundary_inputs：consumer 侧 FIFO，(microbatch, boundary_embeddings)；
    #   _take_boundary_packet append、backward 之后 pop(0)，容量上限 P。
    # dual-channel-p2p Task 4: backward grad transport state.
    # - producer_grad_buffers: microbatch -> encoder output grad; each rank keeps only what it owns.
    # - producer_grad_requests: owned grad irecvs POSTed but not yet waited (owned mb ->
    #   _GradRecvRequest); posted incrementally (bound i-x) and drained by _finish_owned_grads
    #   before phase ④ so the grad data transfer overlaps backbone compute.
    # - consumer_boundary_inputs: the consumer's FIFO of (microbatch, boundary_embeddings).
    producer_grad_buffers: Dict[int, torch.Tensor] = {}
    producer_grad_requests: Dict[int, _GradRecvRequest] = {}
    consumer_boundary_inputs: List[Tuple[int, torch.Tensor]] = []

    def _deallocate_encoder_output(packet: ForwardPacket) -> None:
        """Task 4.9 pseudo-deallocation — structurally INERT under the merged forward.

        优化 spec Task 1（合并前传）后，producer 各包的 ``image_embeddings`` 是合并张量
        的 **view**，本函数的前置契约（"不能是别的张量的 view"）不再成立，且伪释放对
        view 结构性失效：释放 view 不会释放 base 持有的显存，而合并张量本就要活到
        phase ④ 的单次 backward（它就是图输出对象）。因此这里**直接返回**（no-op），
        调用点保留作流程标记；``--deallocate-encoder-outputs`` 开关在合并路径下无效果，
        原"每发一个 mb 省一份 encoder 输出"的收益被合并设计取代（合并张量 1 份、
        phase ④ 反传后随图释放）。
        Task 4.9 pseudo-deallocation is structurally inert after the merged forward: each
        packet's image_embeddings is now a VIEW of the merged tensor, which violates the
        old contract (deallocate_output_tensor asserts on views) and cannot free the
        base's storage anyway — the merged tensor is the graph output and must live until
        phase ④'s single backward. Keep the call sites as flow markers; the flag has no
        effect on the merged path.
        """
        return

    # dual-channel-p2p Task 3：consumer 用一颗"引子"引导 one-ahead 预取——进 warmup 前 pre-post
    # mb 0 的 recv（仅当 mb 0 的 owner 是远端；默认轮盘下 owner(0)==0 本地零拷贝，是 no-op）。
    # 之后循环内每步"前传 mb i 前 post mb i+1 的 recv"接力。producer **不再需要**流水前的启动
    # 首包——供给统一由循环顶端的 i+x+1 规则驱动，其在 i=0 时自动等价于 warmup 无条件预热。
    # Consumer primer that bootstraps the one-ahead prefetch: pre-post mb 0's recv if it is
    # remote (a no-op under round-robin where owner(0)==0 is local). Producers need no
    # pre-pipeline startup send — supply is driven by the i+x+1 rule at the top of each step,
    # which at i=0 doubles as the warmup priming.
    if is_consumer and num_microbatches > 0:
        first_owner = parallel_state.get_colocated_microbatch_owner(0)
        if first_owner != 0:
            prefetched[0] = comm.colocated_recv_forward(
                first_owner, expected_microbatch_id=0, wait=False
            )

    def _take_boundary_packet(microbatch: int) -> ForwardPacket:
        """Consumer (stage 0): take the boundary packet of ``microbatch``.

        消费者（stage 0）前传 microbatch ``microbatch`` 前取它的边界包（encoder 输出 +
        文本字段）：producer 0 本地直传（``encoder_buffers[microbatch]``，零拷贝）、
        producer>0 用 prefetch 的 ``_ForwardRecvRequest`` 的 ``finish()``（4.3k：prefetch
        时通信器已把 request 启动好——header 已等完、数据 irecv 已入队，finish 等数据 →
        组装）。
        **返回包，由调用方用 ``functools.partial`` 绑定到 forward step**（packet 走闭包
        传给 ``colocated_forward_step``，不经模型属性——模型不背 schedule 交接状态）。

        4.6b：返回前把 image_embeddings 换成**边界切断点**
        ``detach().requires_grad_(True)``，并把 ``(microbatch, boundary_embeddings)``
        追加进 ``consumer_boundary_inputs``（4.6c 在 backward 之后 pop 取 ``.grad``）。

        NVTX：外层区间覆盖"取包 + 边界切断 + FIFO 记账"整段——内层
        ``colocated-boundary-forward-recv-finish`` 只覆盖等数据那一步，外减内即为
        本地侧的组装开销（producer 0 的本地直传路径完全没有内层区间）。
        NVTX: the outer range covers take + boundary cut + FIFO bookkeeping; the inner
        finish range covers only the data wait (absent on producer 0's local path).
        """
        nvtx_range_push("colocated-boundary-packet-take")
        # dual-channel-p2p Task 3：owner 由 owner 表决定（支持任意/逆序划分）；prefetched 按
        # microbatch 号索引（不再按 producer——逆序划分下相邻 mb 可能同 owner，按 producer 键会
        # 被后一个覆盖）。owner 0 = consumer 自己：本地零拷贝；否则取 prefetch 的 request。
        producer = parallel_state.get_colocated_microbatch_owner(microbatch)
        if producer == 0:
            _cpu_probe(f"take mb={microbatch} LOCAL(producer0)")
            packet = encoder_buffers[microbatch]  # owner 0 = 自己：本地直传（零拷贝）
        else:
            # prefetch 时通信器已启动 request（数据 irecv 已入队），take 时只需 finish() 取数据。
            _cpu_probe(f"take mb={microbatch} finish() from producer={producer} BEGIN")
            packet = prefetched.pop(microbatch).finish()
            _cpu_probe(f"take mb={microbatch} finish() from producer={producer} END")

        # 4.6b：边界切断——本地路径与网络路径**同构处理**（无分支），两个操作各服务一条路径：
        # - detach()：本地直传（producer 0）的 image_embeddings 还挂在 encoder 计算图上，
        #   不切断则 backbone 的 per-microbatch 反传会一路冲进 encoder——phase ④ 的统一
        #   encoder 反传就没了、encoder 图首次反传后即被释放（phase ④ 再来会报错）、DDP 的
        #   grad sync 时机也错乱；
        # - requires_grad_(True)：网络路径的 image_embeddings 是 irecv buffer、没有计算图，
        #   不置则 autograd 认为这个入口不需要梯度、.grad 恒为 None，什么也接不到。
        # 切断后得到的是 **leaf** 张量 → autograd 自动把梯度累积到 .grad（不需要
        # retain_grad()；对比 backward_step 对非 leaf 的 input_tensor 必须 retain_grad()
        # 后再读 .grad，schedules.py:482/512——同一套路，只是那里只覆盖了 PP 输入这一个入口，
        # image_embeddings 这第二个入口要我们自己在 schedule 里做）。
        # 用 dataclasses.replace 造**新包**而不是原地改字段：原包的 image_embeddings 是
        # phase ④ 反传的对象（带 encoder 计算图），原地改会把它顶掉。
        # 4.6b: cut the graph at the boundary — one uniform expression for both paths.
        # detach() serves the local path (producer 0's zero-copy embeddings still carry the
        # encoder graph; without cutting, the per-microbatch backbone backward would run
        # straight into the encoder and break the unified phase-④ backward), while
        # requires_grad_(True) serves the network path (an irecv buffer has no graph, so
        # autograd would not compute this entry at all). The result is a leaf tensor, so .grad
        # is accumulated automatically and no retain_grad() is needed. A new packet is built
        # with dataclasses.replace instead of mutating the field in place: the original
        # image_embeddings is what phase ④ backpropagates through.
        boundary_embeddings = packet.image_embeddings.detach().requires_grad_(True)
        # forward_only（eval）没有反传、不会有人 pop，入 FIFO 只会白占显存。
        # In forward_only (eval) nothing pops the FIFO, so skip the bookkeeping entirely.
        if not forward_only:
            consumer_boundary_inputs.append((microbatch, boundary_embeddings))
        nvtx_range_pop("colocated-boundary-packet-take")
        return replace(packet, image_embeddings=boundary_embeddings)

    # 4.6c：consumer 侧梯度派发的在飞 isend handle，**按 producer 分桶**（与 prefetched 同构）
    # dual-channel-p2p（2026-09-22）：consumer 侧边界梯度**纯 fire-and-forget**——isend(wait=False)
    # 发完即丢引用，不留任何监测（无 pending 列表、无 is_completed、无 wait）。显存安全靠 PyTorch 的
    # record_stream：ProcessGroupNCCL 默认把 P2P 张量登记到 NCCL stream，caching allocator 在该 isend
    # 完成前不复用其显存、完成后惰性回收，故引用可立即释放（前提：不设 TORCH_NCCL_AVOID_RECORD_STREAMS=1）。
    # 到达同步的责任在 producer——它在 encoder 反传前的 _finish_owned_grads 里等自己 owned 的 grad irecv
    # 落地；consumer 全程不阻塞，也不随 owner 划分/包大小变化（原实现按 producer 桶 + 发下一个前 inline
    # wait 上一个，那个 wait 在大包 rendezvous 下会真卡到对端 post irecv、reverse 下死锁，已整体删除）。
    # dual-channel-p2p (2026-09-22): the consumer's boundary-grad send is pure fire-and-forget —
    # isend(wait=False) then drop the reference, with no tracking (no pending list, no is_completed,
    # no wait). Memory is safe via PyTorch record_stream (ProcessGroupNCCL records P2P tensors on the
    # NCCL stream by default, so the caching allocator defers reuse until the isend completes). The
    # arrival barrier belongs to the producer's _finish_owned_grads before its encoder backward.

    def _dispatch_boundary_grad() -> None:
        """Consumer (stage 0): take the boundary grad of the finished backward and dispatch it.

        消费者（stage 0）在**每个 microbatch 的 backbone 反传结束后**取它的边界梯度并派发。
        为什么不能从 ``backward_func`` 的返回值拿：``backward_step`` 只返回
        ``input_tensor_grad``——对 **PP 输入激活**的梯度（schedules.py:512），而 consumer 是
        stage 0、``input_tensor`` 为 None，返回值恒为 None。image_embeddings 是
        ``colocated_forward_step`` 从包里喂进去的**第二个入口**，Megatron 不知道它存在，
        梯度不在返回值里；但反传图本身必然算到它（4.6b 已把它切成 leaf），梯度已累积在
        ``.grad`` 上，这里直接取。

        派发（与 producer 侧 ``_receive_owned_grads`` 按 owned 升序 POST irecv 的顺序匹配，因为
        pop 顺序 = 反传顺序 = microbatch 增序 = 各 owner 的 owned 升序）：
        - owner 表判 ``owner == 0``（producer 0 = 消费者自己）：本地存 ``producer_grad_buffers``；
        - 否则 ``colocated_send_backward(..., wait=False)`` 发回 owner 对应 producer（grad 组）。

        FIFO（``consumer_boundary_inputs``）而不是 ``prefetched`` 做载体的原因：``prefetched``
        按 producer 为 key，会被同一 producer 的后续 microbatch 覆盖（P=2 时 step k=2 就把
        microbatch 3 的请求写进 ``prefetched[1]``，而 microbatch 1 的反传要到 cooldown 才
        发生，张量早已被回收）；FIFO 持有**独立强引用**故安全，且 ``pop(0)`` 直接给出
        microbatch 号（不必按 steady=i / cooldown=num_microbatches_remaining+i 推导）。

        **只有 consumer 调用本函数**——``is_consumer`` 判断放在两处调用点（2026-08-26
        用户要求：写在调用处更清晰，避免让人以为 producer 也要发边界梯度）。

        NVTX：一个区间覆盖"取梯度 + fire-and-forget 派发"整段（内层的
        ``colocated-boundary-send-backward`` 由通信器提供，外减内即为本函数自身的开销）。
        NVTX: one range for the whole helper; the inner send range comes from the communicator.
        """
        nvtx_range_push("colocated-boundary-grad-dispatch")
        microbatch, boundary_embeddings = consumer_boundary_inputs.pop(0)
        _cpu_probe(f"grad-dispatch ENTER mb={microbatch}")
        boundary_grad = boundary_embeddings.grad
        assert boundary_grad is not None, (
            f"microbatch {microbatch}: the boundary image_embeddings got no grad — it must be "
            f"a leaf with requires_grad=True (4.6b) and take part in the backbone backward"
        )
        boundary_embeddings.grad = None  # 断开 .grad 属性（grad 已由 boundary_grad 接住）；boundary_embeddings 出 FIFO 后其激活存储随作用域回收 / detach the .grad attribute; the activation is freed once this leaf leaves scope
        # dual-channel-p2p Task 4：owner 由 owner 表决定（支持任意/逆序划分）。
        producer = parallel_state.get_colocated_microbatch_owner(microbatch)
        if producer == 0:
            producer_grad_buffers[microbatch] = boundary_grad  # producer 0 = 自己：本地留存
            nvtx_range_pop("colocated-boundary-grad-dispatch")
            return
        # dual-channel-p2p（2026-09-22）：fire-and-forget——异步 isend 发回 owner，发完**不留任何句柄/引用**。
        # 显存由 record_stream 兜底（见函数上方说明），到达由 producer 的 _finish_owned_grads 保证。
        # dual-channel-p2p (2026-09-22): fire-and-forget — async isend to the owner, keeping no handle
        # or reference afterwards; memory is guarded by record_stream, arrival by the producer's
        # _finish_owned_grads.
        comm.colocated_send_backward(
            BackwardPacket(grad=boundary_grad), producer=producer, wait=False
        )
        _cpu_probe(f"grad-dispatch isend mb={microbatch} -> producer={producer} DONE")
        nvtx_range_pop("colocated-boundary-grad-dispatch")

    # dual-channel-p2p Task 4：producer 的边界梯度接收——两步分离以最大化重叠：
    # _receive_owned_grads 在本 producer 每次自身反传后**只 POST（异步 irecv，不等数据）** owned
    # 中 <= i-x 的梯度；_finish_owned_grads 在 phase④ encoder 反传前**统一等数据落地**，使梯度数据
    # 传输与 backbone cooldown/后续计算重叠（consumer 侧对应 _dispatch_boundary_grad 反传后即发）。
    # dual-channel-p2p Task 4: the producer's boundary-grad receive is split in two to maximize
    # overlap — _receive_owned_grads only POSTs the async irecv (bound i-x) after each of this
    # producer's own backwards, and _finish_owned_grads waits the data just before phase ④.
    def _receive_owned_grads(backward_microbatch: int) -> None:
        """Producer: POST (async, do NOT wait data) the grad irecv of every owned microbatch
        <= ``backward_microbatch - producer_id`` not yet posted (dual-channel-p2p Task 4, grad
        recv bound i-x; x = producer_id = stage 号).

        参数与 _supply_owned_activations 对称——传入**本 producer 自身反传的 mb 序号 i**（不是预算好
        的 limit），内部算 recv_limit = i - x。1F1B 下 consumer（stage 0）对 mb <= i-x 的反传必早于
        producer x 对 mb i 的反传（doc §3），故这些 owned 梯度已被 consumer isend、irecv 的 header
        会合立即完成；数据后台传输，由 _finish_owned_grads 在 phase④ 前统一等落地（与 cooldown/
        后续计算重叠）。owned 里被 i-x 界够不到的最后几个（触发反传号 mb+x 超过 N-1）由
        _finish_owned_grads 在 schedule 结束后补 POST（那时全部梯度已就绪）。梯度包无 mb id：同一
        (consumer 0 → producer x) 对按 FIFO 到达 =
        consumer 反传升序 = 本 producer owned 升序，故按 owned 升序 POST、request 即与该 owned 对号。
        ``forward_only``（eval）无边界梯度，直接返回。
        Takes this producer's own backward microbatch (symmetric with _supply_owned_activations);
        posts async only, the data wait is deferred to _finish_owned_grads before phase ④.
        """
        if forward_only or is_consumer:
            return
        recv_limit = backward_microbatch - producer_id
        _cpu_probe(f"recv-owned-grads ENTER bwd_mb={backward_microbatch} limit={recv_limit}")
        for owned_microbatch in owned_microbatches:  # ascending
            if owned_microbatch > recv_limit:
                break
            if owned_microbatch in producer_grad_requests:
                continue
            _cpu_probe(f"recv-owned-grads post irecv mb={owned_microbatch} BEGIN")
            producer_grad_requests[owned_microbatch] = comm.colocated_recv_backward(
                producer=producer_id, wait=False
            )
            _cpu_probe(f"recv-owned-grads post irecv mb={owned_microbatch} END")

    def _finish_owned_grads() -> None:
        """Producer: POST any still-un-posted owned grad, then wait every posted grad's DATA and
        store it into producer_grad_buffers, right before phase ④ encoder backward.

        调用点：cooldown 全部结束后。此刻 consumer 已反传全部 mb、发出全部梯度，故 owned 里被
        i-x 界够不到的最后几个（触发反传号 mb+x 超过 N-1）此刻也可安全接收——先把这些剩余 owned
        补 POST，再统一等所有已 POST 的 grad 数据落地（phase④ 前唯一的数据等待点，把 grad 传输藏在
        前面的 backbone cooldown/计算里）。finish() 顺序与配对无关（匹配在 POST 时按 FIFO 已定），
        每个 request 落进它 POST 时对应的 owned mb。``forward_only``/consumer 直接返回。
        """
        if forward_only or is_consumer:
            return
        for owned_microbatch in owned_microbatches:  # POST any tail the i-x bound never reached
            if owned_microbatch not in producer_grad_requests:
                _cpu_probe(f"finish-owned-grads post tail irecv mb={owned_microbatch}")
                producer_grad_requests[owned_microbatch] = comm.colocated_recv_backward(
                    producer=producer_id, wait=False
                )
        for owned_microbatch, request in producer_grad_requests.items():
            _cpu_probe(f"finish-owned-grads WAIT data mb={owned_microbatch} BEGIN")
            producer_grad_buffers[owned_microbatch] = request.finish().grad
            _cpu_probe(f"finish-owned-grads WAIT data mb={owned_microbatch} END")
        producer_grad_requests.clear()

    def _supply_owned_activations(
        forward_microbatch: int, is_warmup: bool
    ) -> None:
        """Producer: isend owned activations with mb id <= forward_microbatch + x (steady) or
        + x + 1 (warmup) not yet sent (x = producer_id = this stage index).

        **dual-channel-p2p（2026-09-23 更正）调用位置：warmup 前所有 rank 各调一次 (0)（universal
        supply，启动兜底）+ 每个 warmup/steady step 在 take/partial 之后、forward_step 之前调用。**
        要发的 encoder 激活在 phase① 已算好、已驻留，isend 是纯 host 异步动作、不阻塞（未匹配
        isend 不占自旋 kernel、不阻塞后续 cudaMalloc，已实测）。原设计放在 step 最顶端、recv_forward
        之前以打破启动循环依赖；2026-09-23 起 backbone PP p2p 内部 cudaMalloc 的全设备同步不能被
        boundary kernel 挡住（用户定案 boundary 通信后置），故 step 内调用点后移到 take 之后，启动期
        由 warmup 前的 universal supply 覆盖。owned 升序，超过上界即停。
        **dual-channel-p2p（2026-09-23 波前分析定案）供给紧界：steady 用 ``i+x``（紧界）、
        warmup/pre-warmup 用 ``i+x+1``（提前一拍），由 ``is_warmup`` 区分。** consumer 能开始
        ``forward(i+x)`` 的前提是 PP 依赖链把 producer x 的 ``forward(i)`` 送到（chain root =
        consumer 上一拍的 forward 发送），故 steady 的 ``isend(i+x)`` 的**发射**被依赖链保证先于
        consumer 的 ``take(i+x)``——紧界就是 ``i+x``，且 consumer 的反传尾巴（PP sb/rf + B + grad
        派发）给了 producer 整拍缓冲，等待有界、无同波竞速。warmup 全是 F、step 尾巴只有一次 PP
        send，吸收不了跨 x-1 跳的 relay 延迟，``isend(i+x)`` 与 ``take(i+x)`` 同波竞速，必须
        ``+1`` 提前一拍。在 ``=1``（单硬件队列）下 steady 的 ``+1`` 会造成同波 isend/irecv 竞速并
        闭环成执行序死锁（GBS≥32 实测），故 steady 不得带 ``+1``；多连接（``=2``）下无害。旧
        "``+1`` 是紧界"的结论属于旧调度模型（供给在 step 顶端、recv 阻塞在循环体），已被本模型取代。
        Supply-bound invariant (2026-09-23 wave-front analysis): steady supplies at the tight bound
        ``i+x`` while warmup/pre-warmup supplies one wave early at ``i+x+1``, selected by
        ``is_warmup``. The PP dependency chain guarantees producer x reaches step i before
        consumer's ``forward(i+x)`` starts, so the steady ``isend(i+x)`` is always issued before
        consumer's ``take(i+x)`` — tight and race-free, with the backward tail giving the producer
        a full step of slack. Warmup is forward-only (its step tail cannot absorb the x-1 hop
        relay latency), so ``+1`` is required there. Under a single hardware queue
        (CUDA_DEVICE_MAX_CONNECTIONS=1) the same-wave send/recv race created by a steady ``+1``
        closes into an execution-order deadlock (observed at GBS>=32), so steady must not carry
        the ``+1``; with multiple connections it is harmless. The old "+1 is the tight bound"
        result belonged to the previous schedule model and is superseded.
        Call sites (2026-09-23 correction): once per rank BEFORE warmup (universal supply,
        is_warmup=True, covers startup) plus once per warmup/steady step after take/partial and
        before forward_step — moved away from the step top so the backbone PP p2p's internal
        cudaMalloc (device-wide sync) is never blocked by queued boundary kernels. Owned ascending,
        stop past the bound.
        """
        if is_consumer:
            return
        # dual-channel-p2p（2026-09-23 定案）：warmup/pre-warmup 供给 = i+x+1——提前一拍，吸收
        # warmup 纯 F 的 step 尾巴吸收不了的跨 x-1 跳 relay 延迟；steady 供给 = i+x（紧界——PP
        # 依赖链保证发射先于 consumer 的 take，反传尾巴再给一拍缓冲）。=1 下 steady 多出的 +1
        # 造成同波 isend/irecv 竞速并闭环成执行序死锁（GBS>=32 实测），故必须按相位区分。
        # dual-channel-p2p (2026-09-23 settled): warmup/pre-warmup supplies at i+x+1 — one wave
        # early, absorbing the x-1 hop relay latency the forward-only warmup step tail cannot
        # absorb; steady supplies at the tight bound i+x — the PP chain guarantees the isend
        # issues before the consumer's take, and the backward tail adds a full step of slack.
        # Under CUDA_DEVICE_MAX_CONNECTIONS=1 a steady +1 creates a same-wave send/recv race that
        # closes into an execution-order deadlock (observed at GBS>=32), hence the phase split.
        if is_warmup:
            supply_limit = forward_microbatch + producer_id + 1
        else:
            supply_limit = forward_microbatch + producer_id
        _cpu_probe(
            f"supply ENTER fwd_mb={forward_microbatch} limit={supply_limit} warmup={is_warmup}"
        )
        nvtx_range_push("colocated-boundary-activation-supply")
        for owned_microbatch in owned_microbatches:  # ascending
            if owned_microbatch > supply_limit:
                break
            if owned_microbatch in producer_sent_activations:
                continue
            _cpu_probe(f"supply isend mb={owned_microbatch} -> consumer BEGIN")
            comm.colocated_send_forward(
                encoder_buffers[owned_microbatch], producer=producer_id, wait=False
            )
            _cpu_probe(f"supply isend mb={owned_microbatch} -> consumer END")
            producer_sent_activations.add(owned_microbatch)
            _deallocate_encoder_output(encoder_buffers[owned_microbatch])  # 4.9
        nvtx_range_pop("colocated-boundary-activation-supply")

    # --- 4.3i：backbone P2P 伴随传输展开 labels/loss_mask（4.3h 定案；4.3l 起命名
    # 统一为 intra_packet 伴随传输）---
    # 在 send_forward/recv_forward 之后、两端同序（先 activation 后 labels/mask）额外小
    # 传输：consumer 组装产物 / 中间 stage 透传，last stage 消费算 loss。反传只走
    # activation 梯度（labels/mask 不参与）。shape 由 activation 推导（labels [b,s']）。
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_next_rank = dist.get_global_rank(pp_group, (pp_rank + 1) % pp_size)
    pp_prev_rank = dist.get_global_rank(pp_group, (pp_rank - 1) % pp_size)

    def _send_intra_packet(labels, loss_mask, is_pp_last_stage: bool) -> None:
        """Send the expanded labels/loss_mask alongside the activation (4.3h).

        在 send_forward 之后伴随发送 labels/loss_mask（consumer 组装产物 / 中间 stage
        透传）。张量由调用方提供（4.3j：不再从模型属性读取）——consumer 传
        ``colocated_forward_step`` 写回 intra_packet 的值，非 consumer 透传刚收到的
        同份张量。末 stage 或张量为 None（当前 stage 不参与）时跳过。
        **4.3l：与 _recv_intra_packet 对称（统一结构），同步等 isend 完成**——
        intra_packet 与 activation 是绑定的一对通信（send_forward/recv_forward 已同步
        两端到同一拍），同步 wait 不会额外阻塞；不再需要外部 handle 列表与末尾统一 wait。

        NVTX：intra_packet 走的是 **pp_group**（不是共置边界组），既不在
        ``colocated-boundary-*`` 覆盖内，也不在 p2p_communication.py 的装饰器覆盖内，
        故单独标一个区间。**不给 ``handle.wait()`` 单独标区间**：``Work::wait()`` 只是把
        「等 NCCL 完成事件」插进当前 stream 后立即返回，CPU 不阻塞（probe 实测 0.000 s），
        单标只会得到一个恒为 0 的区间；真正的等待发生在 GPU 时间线上（计算 stream 的空隙
        对齐到该 NCCL kernel 的结束），要用 ``nsys stats -r nvtx_gpu_proj_sum`` 的
        NVTX-GPU 投影看，而不是 CPU 侧区间。
        NVTX: one range for the whole helper, and deliberately none around handle.wait() -
        Work::wait() only enqueues a stream wait and returns (measured 0.000 s on the CPU),
        so a range there would always be empty; the real wait shows up on the GPU timeline
        and must be read through the NVTX-GPU projection.
        """
        if is_pp_last_stage or labels is None or loss_mask is None:
            return
        nvtx_range_push("colocated-intra-packet-send")
        intra_packet_send_handles = [
            dist.isend(labels.contiguous(), dst=pp_next_rank, group=pp_group),
            dist.isend(loss_mask.contiguous(), dst=pp_next_rank, group=pp_group),
        ]
        for handle in intra_packet_send_handles:
            handle.wait()
        nvtx_range_pop("colocated-intra-packet-send")

    def _recv_intra_packet(
        input_tensor: Optional[torch.Tensor], is_pp_first_stage: bool
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Receive the expanded labels/loss_mask alongside the activation (4.3h).

        在 recv_forward 之后伴随接收（两端同序），返回 ``(labels, loss_mask)`` 供调用方
        绑定进 intra_packet 并透传（4.3j：不再写入模型属性）。shape 由 activation
        推导：activation [s', b, h] → labels/loss_mask [b, s']。首 stage 或 activation
        为 None 时返回 ``(None, None)``。

        NVTX：与 ``_send_intra_packet`` 对称——一个区间覆盖 shape 推导 + 分配 + 提交 + 插
        stream-wait，同样不给 ``handle.wait()`` 单独标区间（理由见那里）。区间从形状推导之后
        才开始（首 stage / activation 为 None 的早返回不产生区间，避免时间线上出现大量零长度
        噪音区间）。
        NVTX: symmetric with _send_intra_packet (no separate range around the wait); the range
        starts after the early returns so non-participating stages emit no zero-length noise.
        """
        if isinstance(input_tensor, list):
            # P2PCommunicator 的 recv_forward / send_backward_recv_forward 返回值形态不统一
            #（`Union[torch.Tensor, list[torch.Tensor]]`）：recv_forward 按
            # `is_single_shape(tensor_shapes)` 决定是否解包，而 send_backward_recv_forward 按
            # `isinstance(input_tensor_grads, list)` 决定——同一个 recv_tensor_shapes 在两条
            # 路径上可能一个给张量、一个给列表。这里统一取第一个（colocated backbone 是单个
            # GPTModel，只有一份 activation）。
            # The P2P API returns either a tensor or a list depending on which method is used
            # (recv_forward unwraps by is_single_shape(tensor_shapes), whereas
            # send_backward_recv_forward unwraps by isinstance(input_tensor_grads, list)), so
            # normalize here; the colocated backbone is a single GPTModel with one activation.
            input_tensor = input_tensor[0]
        if is_pp_first_stage or input_tensor is None:
            return None, None
        # 从 activation [seq_len, batch, hidden] 取两个形状维度：
        #   seq_len = activation 第 0 维 = 展开后的序列长度（图像 token 展开后比原始
        #             文本序列长）；
        #   batch   = activation 第 1 维 = 本 micro batch 的样本数；
        # labels 与 loss_mask 形状相同，都是 [batch, seq_len]（activation 的转置对应）。
        # Extract the two shape dims from the activation [seq_len, batch, hidden]:
        #   seq_len = activation dim 0 = the expanded sequence length (longer than the
        #             original text sequence after image-token expansion);
        #   batch   = activation dim 1 = the micro-batch size;
        # labels and loss_mask share the same shape [batch, seq_len] (the transpose of
        # the activation).
        nvtx_range_push("colocated-intra-packet-recv")
        seq_len, batch = input_tensor.shape[0], input_tensor.shape[1]
        labels = torch.empty((batch, seq_len), dtype=torch.int64, device=input_tensor.device)
        loss_mask = torch.empty(
            (batch, seq_len), dtype=torch.float32, device=input_tensor.device
        )
        intra_packet_recv_handles = [
            dist.irecv(labels, src=pp_prev_rank, group=pp_group),
            dist.irecv(loss_mask, src=pp_prev_rank, group=pp_group),
        ]
        for handle in intra_packet_recv_handles:
            handle.wait()
        nvtx_range_pop("colocated-intra-packet-recv")
        return labels, loss_mask

    # Compute number of warmup microbatches.
    # warmup 阶段的 microbatch 数量 = 后续 stage 数（total_stages - current_stage - 1）。
    num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of
    # micro-batches within the maximum outstanding micro-batch backpropagations.
    # 部分激活重计算：窗口 = warmup 数 + 1（越靠后的 stage 窗口越小）。
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Single-module backward: the colocated backbone is one GPTModel.
    # 单模块 backward（colocated backbone 是单个 GPTModel，不做多模块分支）。
    backward_func = backward_step

    recv_tensor_shapes = get_tensor_shapes(  # recv 张量 shape
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes(  # send 张量 shape
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes.
    # 只有训练（非 forward_only）才需要保存 input/output 激活。
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # NVTX：backbone warmup 阶段（纯前传）。对应 schedules.py 的 "warmup" 区间。
    # NVTX: the backbone warmup phase (forward only), mirroring schedules.py's "warmup".
    _cpu_probe(
        f"SCHEDULE-START is_consumer={is_consumer} producer_id={producer_id} "
        f"is_pp_last_stage={p2p_communicator.is_pp_last_stage} owned={owned_microbatches} "
        f"num_warmup={num_warmup_microbatches} num_remaining={num_microbatches_remaining} "
        f"batch_p2p_comm={config.batch_p2p_comm} batch_p2p_sync={config.batch_p2p_sync}"
    )
    # dual-channel-p2p（2026-09-23 用户定案）：**所有 rank** 在进 warmup 前统一补一次货
    # （_supply_owned_activations 对 consumer 直接返回，是 no-op）。原先只有 last stage 在
    # pre-steady 补一次（2026-09-22 启动死锁修复），那次补货与 consumer 的 prefetch irecv 是
    # 同拍紧 race——GBS=64 fa504 下自旋 recv 窗口会与 TE 冷 cudaMalloc 的全设备同步相撞。
    # 提前到 warmup 前、每个 producer 无条件发 owned ≤ x+1，把 isend 提前到一切自旋 recv
    # 窗口之前，缩小/消除撞车窗口。
    # dual-channel-p2p (2026-09-23, user decision): EVERY rank supplies once BEFORE the warmup
    # loop (a no-op for the consumer). The old last-stage-only pre-steady supply raced with the
    # consumer's prefetch irecv in the same step; issuing owned <= x+1 unconditionally up front
    # moves every isend ahead of any spinning-recv window.
    if num_microbatches > 0:
        _cpu_probe(f"PRE-WARMUP universal supply(0) producer_id={producer_id}")
        _supply_owned_activations(0, is_warmup=True)
    nvtx_range_push("colocated-warmup")
    # Run warmup forward passes.
    # 执行 warmup 阶段的前传。
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch.
        # 判断当前 microbatch 是否完全重计算。
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        _cpu_probe(f"WARMUP step i={i} TOP")
        _cpu_probe(f"WARMUP i={i} recv_forward(PP) BEGIN")
        input_tensor = p2p_communicator.recv_forward(  # 接受前序 stage 的激活
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        _cpu_probe(f"WARMUP i={i} recv_forward(PP) END")
        # 4.3i：伴随接收 labels/loss_mask（在 recv_forward 之后、两端同序）；4.3j 起
        # _recv_intra_packet 返回张量，不再写入模型属性。
        received_labels, received_loss_mask = _recv_intra_packet(
            input_tensor, p2p_communicator.is_pp_first_stage
        )
        # 4.3b：consumer 前传 microbatch i 前 take 包（stage 0 前传 0..P-2），并用
        # ``functools.partial`` 绑定到 forward step（packet 走闭包，不经模型属性）。
        # 4.3j：intra_packet——consumer 用输出盒子（colocated_forward_step 写回组装
        # 产物），非 consumer 用输入载体（本次 recv 到的 labels/loss_mask，last stage
        # 算 loss）。
        if is_consumer:
            intra_packet = IntraPacket()
            forward_step_func = partial(
                forward_step_func,
                packet=_take_boundary_packet(i),
                intra_packet=intra_packet,
            )
        else:
            intra_packet = IntraPacket(
                labels=received_labels, loss_mask=received_loss_mask
            )
            forward_step_func = partial(forward_step_func, intra_packet=intra_packet)
        # dual-channel-p2p（2026-09-23 用户定案：boundary 通信后置）：supply 与 prefetch 从
        # step 顶端（recv_forward 之前）移到 take/partial 之后——warmup 的 recv_forward /
        # send_forward 是 backbone PP p2p，其内部 cudaMalloc 的全设备同步不能被 boundary
        # kernel 挡住。启动期仍由 warmup 前的 universal supply（owned ≤ x+1）兜底，避免
        # last stage 的 pre-steady recv 成环。
        # dual-channel-p2p (2026-09-23, user decision: boundary comm after backbone PP comm):
        # warmup supply + prefetch moved after take/partial; the pre-warmup universal supply
        # still covers the startup window.
        _supply_owned_activations(i, is_warmup=True)
        if is_consumer:
            next_microbatch = i + 1
            if next_microbatch < num_microbatches:
                next_owner = parallel_state.get_colocated_microbatch_owner(next_microbatch)
                if next_owner != 0:
                    prefetched[next_microbatch] = comm.colocated_recv_forward(
                        next_owner, expected_microbatch_id=next_microbatch, wait=False
                    )
        output_tensor, num_tokens = forward_step(  # 单个 microbatch 的前传
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        _cpu_probe(f"WARMUP i={i} forward_step END")
        # dual-channel-p2p Task 3：warmup 供给已移到 step 顶端的 _supply_owned_activations(i)
        # （recv_forward 之前）；warmup 纯前传、consumer 尚未反传，无梯度可收（①③ 不适用）。
        # dual-channel-p2p Task 3: forward supply moved to _supply_owned_activations(i) at the
        # top of the step (before recv_forward); warmup is forward-only, no grad to receive.
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)  # 发给后序 stage
        _cpu_probe(f"WARMUP i={i} send_forward(PP) DONE")
        # 4.3i：伴随发送 labels/loss_mask——consumer 用 colocated_forward_step 写回的
        # intra_packet，非 consumer 透传 recv 到的同份张量（两分支 intra_packet 内容一致）。
        _send_intra_packet(
            intra_packet.labels,
            intra_packet.loss_mask,
            p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # dual-channel-p2p（2026-09-23 更正）：原先这里有一段"仅 last stage 在 pre-steady 补货"
    # （2026-09-22 修复 reverse_block 启动死锁：last stage num_warmup==0 从没进过 warmup 循环、
    # 其首次通信是下方被提升出循环体的 pre-steady recv_forward，若持有低号 mb 会与 stage 0
    # 成环）。现已被上方"warmup 前所有 rank 统一 _supply_owned_activations(0)"取代——last stage
    # （num_warmup==0）同样在 warmup 前发齐 owned ≤ x+1，pre-steady recv 不再是它的第一次通信，
    # 该特例成为死路径，按约定删除（idempotent：producer_sent_activations 保证重复调用也无害）。
    # dual-channel-p2p (2026-09-23 correction): the former last-stage-only pre-steady supply
    # (the 2026-09-22 startup-deadlock fix) is superseded by the universal pre-warmup supply
    # above — the last stage (num_warmup==0) now also sends owned <= x+1 before warmup, so the
    # hoisted pre-steady recv is no longer its first communication. The special case became a
    # dead path and is removed (re-calls would be harmless anyway via producer_sent_activations).

    nvtx_range_pop("colocated-warmup")

    # Before running 1F1B, need to receive first forward tensor.
    # 进入 steady 前先收 steady 第一个 microbatch 的输入激活。
    # 4.3i：steady 第一个 microbatch 的伴随 labels/loss_mask——存为 pending 变量，
    # 供 steady 循环第一轮的 forward 绑定与发送（4.3j：intra_packet 输入载体）。
    pending_labels, pending_loss_mask = None, None
    if num_microbatches_remaining > 0:
        _cpu_probe("PRE-STEADY recv_forward(PP) BEGIN")
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        _cpu_probe("PRE-STEADY recv_forward(PP) END")
        pending_labels, pending_loss_mask = _recv_intra_packet(
            input_tensor, p2p_communicator.is_pp_first_stage
        )

    # NVTX：backbone steady 阶段（1F1B 主体）。对应 schedules.py 的 "steady" 区间。
    # NVTX: the backbone steady phase (the 1F1B main loop), mirroring schedules.py's "steady".
    nvtx_range_push("colocated-steady")
    # Run 1F1B in steady state.
    # 执行 steady 阶段的 1F1B。
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        # 4.4/4.5：本 step 的 microbatch 号（k = W + i）。
        # 4.4/4.5: the current microbatch of this step (k = W + i).
        current_microbatch = i + num_warmup_microbatches
        _cpu_probe(f"STEADY step i={i} mb={current_microbatch} TOP")

        # 4.3b：consumer 前传 microbatch W+i 前 take 包，并 partial 绑定到 forward step。
        # 4.3j：intra_packet——consumer 用输出盒子，非 consumer 用 pending 的输入载体。
        if is_consumer:
            intra_packet = IntraPacket()
            forward_step_func = partial(
                forward_step_func,
                packet=_take_boundary_packet(current_microbatch),
                intra_packet=intra_packet,
            )
        else:
            intra_packet = IntraPacket(
                labels=pending_labels, loss_mask=pending_loss_mask
            )
            forward_step_func = partial(forward_step_func, intra_packet=intra_packet)
        # dual-channel-p2p（2026-09-23 用户定案：boundary 通信后置到 backbone PP 通信之后）：
        # supply 与 prefetch 从 step 顶端移到 take/partial 之后——backbone PP p2p（_communicate）
        # 内部会 torch.empty/cudaMalloc，其全设备同步不能被已入队的 boundary kernel（尤其自旋
        # irecv）挡住，故本 step 的 boundary 收发尽量靠后。预取 mb k+1 仍早于下一步的 take k+1；
        # 供给界 i+x+1 只依赖 current_microbatch，位置后移不改变界。
        # dual-channel-p2p (2026-09-23, user decision: boundary comm goes AFTER backbone PP comm):
        # supply + prefetch moved from the step top to after take/partial — PP p2p allocates its
        # recv buffers internally (cudaMalloc) and its device-wide sync must not be blocked by
        # queued boundary kernels. Prefetch of k+1 still precedes next step's take of k+1; the
        # i+x+1 supply bound only depends on current_microbatch, so it is unchanged.
        # dual-channel-p2p（2026-09-23 定案：波前分析 + 实测 iter12+）：steady 供给 = 紧界 i+x
        # （PP 依赖链保证 isend(i+x) 先于 take(i+x) 发射，反传尾巴再给一拍缓冲，无同波竞速）；
        # warmup / pre-warmup 才用 i+x+1（纯 F 尾巴吸收不了 relay 延迟，需提前一拍）。=1 下 steady
        # 带 +1 的同波竞速会闭环成执行序死锁（GBS>=32 实测），故这里必须 is_warmup=False。
        # dual-channel-p2p (2026-09-23 settled: wave-front analysis + runs): steady supplies at
        # the tight bound i+x — the PP chain guarantees the isend issues before the take and the
        # backward tail adds a full step of slack (no same-wave race); warmup/pre-warmup use
        # i+x+1 (the forward-only tail cannot absorb the relay latency). Under =1 a steady +1
        # deadlocks via the same-wave race (GBS>=32 observed), hence is_warmup=False here.
        _supply_owned_activations(current_microbatch, is_warmup=False)
        if is_consumer:
            next_microbatch = current_microbatch + 1
            if next_microbatch < num_microbatches:
                next_owner = parallel_state.get_colocated_microbatch_owner(next_microbatch)
                if next_owner != 0:
                    prefetched[next_microbatch] = comm.colocated_recv_forward(
                        next_owner, expected_microbatch_id=next_microbatch, wait=False
                    )
        _cpu_probe(f"STEADY mb={current_microbatch} forward_step BEGIN")
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=current_microbatch,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        _cpu_probe(f"STEADY mb={current_microbatch} forward_step END")
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            _send_intra_packet(  # 4.3i：伴随发送 labels/loss_mask
                intra_packet.labels,
                intra_packet.loss_mask,
                p2p_communicator.is_pp_last_stage,
            )
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
                pending_labels, pending_loss_mask = _recv_intra_packet(  # 4.3i：更新下一轮
                    input_tensor, p2p_communicator.is_pp_first_stage
                )
        else:  # 训练走这里，前传+反传
            _cpu_probe(f"STEADY mb={current_microbatch} send_forward_recv_backward(PP) BEGIN")
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(  # 发激活、收梯度
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )
            _cpu_probe(f"STEADY mb={current_microbatch} send_forward_recv_backward(PP) END")
            _send_intra_packet(  # 4.3i：伴随发送 labels/loss_mask
                intra_packet.labels,
                intra_packet.loss_mask,
                p2p_communicator.is_pp_last_stage,
            )

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass (FIFO: 最早前传的最早反传).
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            _cpu_probe(f"STEADY mb={current_microbatch} backward BEGIN")
            input_tensor_grad = backward_func(  # 单个 microbatch 的反传
                input_tensor, output_tensor, output_tensor_grad, config
            )
            _cpu_probe(f"STEADY mb={current_microbatch} backward END")

            if last_iteration:
                input_tensor = None
                _cpu_probe(f"STEADY mb={current_microbatch} send_backward(PP,last) BEGIN")
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
                _cpu_probe(f"STEADY mb={current_microbatch} send_backward(PP,last) END")
            else:
                _cpu_probe(f"STEADY mb={current_microbatch} send_backward_recv_forward(PP) BEGIN")
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
                _cpu_probe(f"STEADY mb={current_microbatch} send_backward_recv_forward(PP) END")
                pending_labels, pending_loss_mask = _recv_intra_packet(  # 4.3i：下一轮 forward 的载体
                    input_tensor, p2p_communicator.is_pp_first_stage
                )
            # dual-channel-p2p（2026-09-23 用户定案：boundary 通信后置）：梯度派发/接收移到本步
            # 全部 backbone PP 通信（send_backward / send_backward_recv_forward）之后——PP p2p 的
            # cudaMalloc 全设备同步不再被 boundary kernel 挡住；grad irecv 晚 post 只会缩小自旋
            # 窗口，consumer 的 grad isend 后移不改 FIFO 顺序（仍按反传序升序发出）。
            # dual-channel-p2p (2026-09-23, user decision): boundary-grad dispatch/receive moved
            # after this step's PP send_backward ops (i-x post bound unchanged, FIFO order kept).
            if is_consumer:
                _dispatch_boundary_grad()
            else:
                _receive_owned_grads(i)

    nvtx_range_pop("colocated-steady")

    # Run cooldown backward passes.
    # 执行 cooldown 阶段的反传。
    if not forward_only:
        # NVTX：backbone cooldown 阶段（剩余反传）。对应 schedules.py 的 "cooldown" 区间。
        # NVTX: the backbone cooldown phase (remaining backwards), mirroring schedules.py's
        # "cooldown".
        nvtx_range_push("colocated-cooldown")
        for i in range(num_warmup_microbatches):
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            _cpu_probe(f"COOLDOWN i={i} recv_backward(PP) BEGIN")
            output_tensor_grad = p2p_communicator.recv_backward(  # 从后序 stage 收梯度
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )
            _cpu_probe(f"COOLDOWN i={i} recv_backward(PP) END")

            _cpu_probe(f"COOLDOWN i={i} backward BEGIN")
            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )
            _cpu_probe(f"COOLDOWN i={i} backward END")

            _cpu_probe(f"COOLDOWN i={i} send_backward(PP) BEGIN")
            p2p_communicator.send_backward(  # 把梯度发给前序 stage
                input_tensor_grad, p2p_communicator.is_pp_first_stage
            )
            _cpu_probe(f"COOLDOWN i={i} send_backward(PP) END")
            # dual-channel-p2p（2026-09-23 用户定案：boundary 通信后置）：梯度派发/接收移到本步
            # backbone PP 通信（recv_backward / send_backward）之后，理由同 steady——PP p2p 内部
            # cudaMalloc 的全设备同步不能被 boundary kernel 挡住。
            # dual-channel-p2p (2026-09-23, user decision): boundary-grad ops moved after the
            # step's PP ops (same rationale as steady).
            if is_consumer:
                _dispatch_boundary_grad()
            else:
                _receive_owned_grads(num_microbatches_remaining + i)

        nvtx_range_pop("colocated-cooldown")

        # dual-channel-p2p（2026-09-22）：consumer 侧边界梯度是 fire-and-forget，无任何在飞句柄/引用需
        # 排空（record_stream 兜显存、_finish_owned_grads 兜到达）；此处仅断言 FIFO 已空——每个前传的
        # microbatch 都恰好被反传+派发一次。
        # dual-channel-p2p (2026-09-22): the consumer's grad sends are fire-and-forget, so there is
        # nothing to drain here; just assert the FIFO emptied (every forwarded microbatch was
        # backwarded and dispatched exactly once).
        assert not consumer_boundary_inputs, (
            f"boundary grads not dispatched for microbatches "
            f"{[microbatch for microbatch, _ in consumer_boundary_inputs]}"
        )

        # dual-channel-p2p Task 4：cooldown 收尾——此刻 consumer 已反传全部 mb、发出全部梯度，由
        # _finish_owned_grads 补 POST owned 里剩余未 POST 的（> N-1-x、触发反传号 mb+x 超过 N-1、
        # 被 i-x 界够不到的最后几个），再统一等所有已 POST 的 grad 数据落地（phase④ 前唯一等待点，
        # 数据传输与前面 cooldown 计算重叠；对 consumer 是 no-op）。
        # After cooldown all grads are available; _finish_owned_grads posts any still-un-posted
        # owned (the tail the i-x bound never reached) and then waits every posted grad's data.
        _cpu_probe("COOLDOWN done; _finish_owned_grads BEGIN")
        _finish_owned_grads()
        _cpu_probe("_finish_owned_grads END (entering phase-4 encoder backward)")

        # dual-channel-p2p Task 4：边界梯度收齐——已 POST 的接收请求全部 drain，且每个 rank 的梯度
        # buffer 恰好覆盖自己产的全部 microbatch（phase ④ 统一 encoder 反传的前提）。
        # The grads are complete: all posted receives drained and every rank's buffer covers
        # exactly the microbatches it produced (the precondition of the phase-④ encoder backward).
        assert not producer_grad_requests, (
            f"boundary grad receives still posted but not finished for owned microbatches "
            f"{sorted(producer_grad_requests)}"
        )
        assert sorted(producer_grad_buffers) == sorted(encoder_buffers), (
            f"boundary grad buffer keys {sorted(producer_grad_buffers)} != owned microbatches "
            f"{sorted(encoder_buffers)}"
        )

        # Launch any remaining grad reductions.
        # 开启剩余的梯度归约。
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    # 4.6e：交回给 phase ④ 的 token 数——**必须在 backbone 的 finalize 之前克隆**：
    # finalize_model_grads 在 per-token 模式下会就地把这个张量规约成全局值
    # （broadcast + all_reduce，finalize_model_grads.py:494-497），而 encoder 需要的是
    # 未规约的原始值（它自己带 encoder 的 pg_collection 再走一遍 finalize_model_grads，
    # 在 colocated dp 组上求和得到同一个全局 token 数）。
    # 4.6e: clone the token count for phase ④ *before* the backbone finalize, which reduces
    # the tensor in place; the encoder needs the raw per-rank value.
    num_tokens_for_encoder = total_num_tokens.clone()

    if config.finalize_model_grads_func is not None and not forward_only:
        # NVTX：backbone 的梯度收尾（DP 归约 / layernorm / embedding 归约）与 encoder 的
        # 收尾分开打区间，两者是不同参数集上的两段通信。
        # NVTX: the backbone grad finalize, kept as a separate range from the encoder's -
        # they are two communication bursts over two different parameter sets.
        nvtx_range_push("colocated-backbone-grad-finalize")
        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        # 若开启 defer_embedding_wgrad_compute，这里补齐 LM Head 的 wgrad 计算。
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        # 梯度收尾：数据并行的全归约 / reduce-scatter、序列并行的 layernorm 归约、PP 的 embedding 归约。
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )
        nvtx_range_pop("colocated-backbone-grad-finalize")

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()

    # 4.6e：把边界梯度与 token 数交回给 forward_backward_colocated（全流程编排者）——
    # phase ④ 的统一 encoder 反传与梯度收尾是**独立的一段**，与 phase ① 对称，不放在本
    # 函数（phase ②）里面。
    # 4.6e: hand the boundary grads and the token count back to forward_backward_colocated
    # (the whole-flow orchestrator); phase ④ is a separate step, symmetric with phase ①.
    return forward_data_store, producer_grad_buffers, num_tokens_for_encoder
