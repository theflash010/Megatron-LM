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

    # 边界通信器：独立共置边界组（get_colocated_boundary_group()，成员与 pp 组相同
    # 但是独立 NCCL 实例）+ backbone config。4.3 起 phase ②/④ 使用。
    # Boundary communicator on the dedicated colocated group (independent NCCL
    # instance); used by phases ②/④ from Task 4.3 on.
    comm = EncoderBackboneBoundaryCommunicator(
        parallel_state.get_colocated_boundary_group(), config
    )

    # 4.6g：流水前预热边界通信——每 producer 双向各一次 1 元素交换。边界组上的
    # communicator 与 p2p transport 都是懒初始化且会合带超时，而 1F1B 里 consumer 与
    # producer 到达边界收发的时刻天然错开（最长 P-2 步），不预热会直接超时报错退出。
    # 4.6g: warm up the boundary transports before the pipeline starts; lazy per-pair
    # communicator / per-direction transport creation is a rendezvous with a timeout,
    # and the two ends reach their first boundary op several steps apart in 1F1B.
    comm.warmup_boundary_communicators()

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

    本 rank（producer p）负责的 microbatch = p, p+P, p+2P, ...（
    ``get_microbatches_for_producer``，Task 1.2）——每 producer 恰好
    ``num_microbatches / P`` 个，与合并批的 batch 维一一对应（轮盘序列升序 == batch
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
    # 轮盘分配：producer p -> microbatch p, p+P, p+2P, ...（每个 producer num_microbatches/P 个）。
    # Round-robin: producer p handles microbatches p, p+P, p+2P, ... (num_microbatches/P each).
    microbatches = parallel_state.get_microbatches_for_producer(
        producer_id, num_microbatches, num_producers
    )

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

    # 完整性：本 rank 恰好 num_microbatches / P 个 microbatch（每个 producer 负载一致）。
    # Completeness: exactly num_microbatches / P entries (balanced across producers).
    expected = num_microbatches // num_producers
    assert len(encoder_buffers) == expected, (
        f"producer {producer_id} got {len(encoder_buffers)} microbatches, expected "
        f"{expected} (num_microbatches {num_microbatches} / num_producers {num_producers})"
    )
    assert sorted(encoder_buffers) == microbatches, (
        f"buffer keys {sorted(encoder_buffers)} != round-robin microbatches {microbatches}"
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
    group_size = comm.group_size

    # 4.5b（2026-08-26 用户定案，取代 4.3b/4.3k 的"流水前全量 prefetch"）：consumer 在
    # 流水前**不做任何边界接收**，统一由循环体内的"前传 mb k 之前 post mb k+1 的 recv"
    # 一条规则覆盖首波（1..P-1）与后续包（>=P）——首波不再特例化，峰值在飞接收从 P-1
    # 份降到 1 份（每份含一个 image_embeddings 数据 buffer）。
    # 代价（已知并接受）：① producer 的首包改为异步发送后会在边界组上悬到 consumer 的
    # step p-1（最长 P-2 步），这是 4.3k 死锁的同一类 pending op——当前靠
    # batch_p2p_sync=False 已移除设备级 torch.cuda.synchronize() 这条阻塞路径；
    # ② 首波的数据搬运从"流水前已完成"变为"只有一步重叠窗口"。
    # 4.5b (2026-08-26): the consumer posts no boundary receive before the pipeline; a single
    # rule inside the loop ("before forwarding mb k, post the recv of mb k+1") now covers both
    # the first wave (1..P-1) and the later packets (>=P). Peak in-flight receives drop from
    # P-1 to 1. Accepted costs: the producers' first packet is now an async send that stays
    # pending until the consumer's step p-1 (the 4.3k class of pending op, whose deadlock path
    # via the device-wide synchronize is already removed by batch_p2p_sync=False), and the
    # first wave now only has a one-step overlap window instead of completing pre-pipeline.
    prefetched = {}

    # 4.6a：反向梯度传输的状态容器（三个，语义各自独立，见 doc §2.11"反向梯度时序"）。
    # - producer_grad_buffers：microbatch → encoder 输出梯度。**每个 rank 只存本 rank 自己
    #   产的 microbatch**（consumer 存 m%P==0 的、producer p 存 m%P==p 的）——phase ④ 统一
    #   encoder 反传时逐个取用。consumer 算出的非本地梯度立刻发走、本地不留，所以这里不会
    #   出现"别人的 microbatch"。
    # - pending_grad_requests：producer 侧在飞的梯度接收请求（microbatch → _GradRecvRequest）。
    #   补货 step 的 ① 存 handle、③ finish() 取梯度，中间夹着 forward 做重叠。
    # - consumer_boundary_inputs：consumer 侧 FIFO，(microbatch, boundary_embeddings)。
    #   _take_boundary_packet（前传节奏）append、backward 之后 pop(0)，纪律与
    #   input_tensors / output_tensors 一致，容量上限 P（stage 0 的在飞 microbatch 数）。
    # 4.6a: state containers of the backward grad transport (see doc §2.11).
    # - producer_grad_buffers: microbatch -> encoder output grad; every rank only keeps the
    #   microbatches it produced itself (the consumer ships the others out immediately), so
    #   phase ④ is symmetric on every rank.
    # - pending_grad_requests: the producer's in-flight grad receives (microbatch ->
    #   _GradRecvRequest); stored by step ① and consumed by step ③ of a restock step, with the
    #   forward in between for overlap.
    # - consumer_boundary_inputs: the consumer's FIFO of (microbatch, boundary_embeddings),
    #   appended in _take_boundary_packet and popped after the backward — same discipline as
    #   input_tensors / output_tensors, bounded by P.
    producer_grad_buffers: Dict[int, torch.Tensor] = {}
    pending_grad_requests: Dict[int, _GradRecvRequest] = {}
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

    # 4.4/4.5b：进入 backbone 流水前，每个 producer 先给 consumer 发**第 1 个包**
    #（microbatch == producer_id）作为**启动**——**异步 `wait=False`**（4.5b 用户定案，
    # 取代 4.3k 的同步等待）：consumer 已不在流水前 post irecv，同步等会直接挂死；异步
    # 发送让 sender 始终早于 receiver 行动，consumer 到 step producer_id-1 才 post 对应
    # 的 recv、随即配对。后续包（producer_id+P, ...）在循环体内按需补发（④）。
    # handle 不保留：与 `_restock_boundary_packet` 一致——包本体活在 encoder_buffers 里
    # 直到 phase ④，发送缓冲不会被提前回收；完成性由 consumer 的配对 recv 保证。
    # 4.4/4.5b: before entering the backbone pipeline each producer sends its first packet
    # (microbatch == producer_id) as a startup seed, now with wait=False (2026-08-26): the
    # consumer no longer posts any pre-pipeline irecv, so a synchronous send would hang. The
    # async send keeps the sender ahead of the receiver, which posts the matching recv at its
    # step producer_id-1. Later packets are restocked inside the loop (step ④). The handle is
    # dropped on purpose (same as _restock_boundary_packet): the packet itself stays alive in
    # encoder_buffers until phase ④, so the send buffer cannot be reclaimed early.
    if not is_consumer: #producer直接异步发送一个micro batch给consumer
        # NVTX：流水前的启动首包——它在时间线上标出"producer 何时开始喂 consumer"，是
        # 判断 backbone 流水启动是否被 encoder 相位拖住的锚点。
        # NVTX: the pre-pipeline startup packet marks when the producer starts feeding the
        # consumer - the anchor for judging whether the backbone start is held up.
        nvtx_range_push("colocated-boundary-packet-startup")
        first_microbatch = producer_id
        assert first_microbatch in encoder_buffers, (
            f"producer {producer_id} must own the first microbatch {first_microbatch}"
        )
        comm.colocated_send_forward(
            encoder_buffers[first_microbatch], producer=producer_id, wait=False
        )
        _deallocate_encoder_output(encoder_buffers[first_microbatch])
        nvtx_range_pop("colocated-boundary-packet-startup")

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
        producer = microbatch % group_size  # 轮盘映射：microbatch s 由 producer s%P 算（doc §2.2）
        if producer == 0:
            packet = encoder_buffers[microbatch]  # producer 0 = 自己：本地直传（零拷贝）
        else:
            # 4.3k：prefetch 时通信器已启动 request（数据 irecv 已入队），take 时只需
            # finish() 取数据。
            packet = prefetched[producer].finish()

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
    # ——每个 producer 至多悬 1 份（发它的下一个梯度前先 wait 掉，cooldown 结束后统一 wait
    # 剩余的），在飞上限 P-1。为什么不是全局只悬 1 份：NCCL 的 send/recv 按 **peer 对**懒创建
    # 独立 communicator 与独立 stream（key 含两端 rank，即 4.3l 诊断到的那个懒初始化现象），
    # 同组 FIFO 只约束同一对内部的顺序——跨 producer 本就独立，全局串成一条会让"派发给
    # producer 2"白等"producer 1 的发送完成"（producer 1 晚 post irecv 就卡住 stage 0）。
    # 4.6c: the consumer's in-flight grad isend handles, bucketed per producer (mirroring
    # prefetched) — at most one outstanding per producer (waited before that producer's next
    # grad, and drained after cooldown), so the bound is P-1 rather than 1. NCCL creates a
    # dedicated communicator and stream per peer pair, so same-group FIFO only orders ops
    # within one pair; a single global bucket would serialize independent producers.
    boundary_grad_send_handles: Dict[int, List] = {}

    def _dispatch_boundary_grad() -> None:
        """Consumer (stage 0): take the boundary grad of the finished backward and dispatch it.

        消费者（stage 0）在**每个 microbatch 的 backbone 反传结束后**取它的边界梯度并派发。
        为什么不能从 ``backward_func`` 的返回值拿：``backward_step`` 只返回
        ``input_tensor_grad``——对 **PP 输入激活**的梯度（schedules.py:512），而 consumer 是
        stage 0、``input_tensor`` 为 None，返回值恒为 None。image_embeddings 是
        ``colocated_forward_step`` 从包里喂进去的**第二个入口**，Megatron 不知道它存在，
        梯度不在返回值里；但反传图本身必然算到它（4.6b 已把它切成 leaf），梯度已累积在
        ``.grad`` 上，这里直接取。

        派发（与 producer 侧"补货 step 按 k-P 递增收"顺序匹配，因为 pop 顺序 = 反传顺序 =
        microbatch 增序）：
        - ``microbatch % P == 0``（producer 0 = 消费者自己）：本地存
          ``producer_grad_buffers``，不走通信；
        - 否则 ``colocated_send_backward(..., wait=False)`` 发回对应 producer。

        FIFO（``consumer_boundary_inputs``）而不是 ``prefetched`` 做载体的原因：``prefetched``
        按 producer 为 key，会被同一 producer 的后续 microbatch 覆盖（P=2 时 step k=2 就把
        microbatch 3 的请求写进 ``prefetched[1]``，而 microbatch 1 的反传要到 cooldown 才
        发生，张量早已被回收）；FIFO 持有**独立强引用**故安全，且 ``pop(0)`` 直接给出
        microbatch 号（不必按 steady=i / cooldown=num_microbatches_remaining+i 推导）。

        **只有 consumer 调用本函数**——``is_consumer`` 判断放在两处调用点（2026-08-26
        用户要求：写在调用处更清晰，避免让人以为 producer 也要发边界梯度）。

        NVTX：一个区间覆盖"取梯度 + 排空上一次 isend + 派发"整段（内层的
        ``colocated-boundary-send-backward`` 由通信器提供，外减内即为本函数自身的开销）。
        **不给排空的 ``handle.wait()`` 单独标区间**：它只是插 stream-wait、CPU 不阻塞，
        单标恒为 0；真正的等待要看 GPU 时间线（见 _send_intra_packet 的说明）。
        NVTX: one range for the whole helper (the inner send range comes from the
        communicator); deliberately none around the drain wait, which does not block the CPU.
        """
        nvtx_range_push("colocated-boundary-grad-dispatch")
        microbatch, boundary_embeddings = consumer_boundary_inputs.pop(0)
        boundary_grad = boundary_embeddings.grad
        assert boundary_grad is not None, (
            f"microbatch {microbatch}: the boundary image_embeddings got no grad — it must be "
            f"a leaf with requires_grad=True (4.6b) and take part in the backbone backward"
        )
        boundary_embeddings.grad = None  # 释放引用，接收 buffer 可回收 / release the reference
        producer = microbatch % group_size
        if producer == 0:
            producer_grad_buffers[microbatch] = boundary_grad  # producer 0 = 自己：本地留存
            nvtx_range_pop("colocated-boundary-grad-dispatch")
            return
        # 先 wait 掉**这个 producer** 上一次派发的 isend（跨 producer 独立，不互等）——那次
        # 发送已被它的补货 step 收走（早 P 步），wait 只是确认、不实际阻塞。
        # Wait this producer's previous dispatch (producers are independent); it was consumed
        # P steps ago by that producer's restock step, so the wait only confirms.
        for handle in boundary_grad_send_handles.pop(producer, []): #按 producer 分桶，只有同一producer产的micorbatch梯度再发才需要wait
            handle.wait()
        boundary_grad_send_handles[producer] = comm.colocated_send_backward(
            BackwardPacket(grad=boundary_grad), producer=producer, wait=False
        )
        nvtx_range_pop("colocated-boundary-grad-dispatch")

    # --- 4.4/4.6d：producer 的补货 step——**steady 阶段**顺序固定为 ① 收梯度 → ② forward →
    # ③ 等梯度 → ④ 补货；**warmup 阶段只有 ④**（纯前传、consumer 尚未反传，无梯度可收）。
    # 三个动作各抽一个函数，由两个循环体按各自需要调用。
    # --- 4.4/4.6d: in the steady loop the producer's restock step is fixed to ① recv grad ->
    # ② forward -> ③ wait grad -> ④ restock; the warmup loop only does ④ (forward-only, the
    # consumer has not backwarded yet, so there is no grad to receive).
    def _start_boundary_grad_recv(microbatch: int) -> None:
        """Producer step ①: post the grad irecv of ``microbatch`` (one it owns).

        生产者异步收 **microbatch ``microbatch``**（自己产的）的边界梯度：
        ``colocated_recv_backward(wait=False)`` 内部已完成"提交 shape 头 → 等头 → 分配
        buffer → post 数据 irecv"，返回的 request 已在后台传输，存 ``pending_grad_requests``。
        参数是"**要收谁的梯度**"而不是"当前 step 的 microbatch"——这样 steady 的补货 step
        （收 k-P）与 cooldown 收尾（收最后一个自己产的）用同一个接口，不必造一个不存在的
        step 号。所有权判断对两者等价（``(k-P) % P == k % P``）。

        steady 的补货 step 把它放在 forward **之前**：数据传输与紧随其后的 forward 重叠，
        且梯度此刻**已经就绪**——consumer（stage 0）对 microbatch k-P 的反传恰好比本 step
        早 1 步（stage i 做 backward x 的时刻 = stage 0 做 backward x+i 的时刻，doc §2.11），
        所以 irecv 立即配对、不阻塞。
        ``microbatch < 0`` 表示"本 rank 自己产的第一个 microbatch 没有上一个"，直接返回。

        ``forward_only``（eval）下直接返回：consumer 走的是纯前传分支、根本不会派发边界
        梯度，这里 post 的 irecv 永远没有对端，而 ``colocated_recv_backward(wait=False)``
        内部含"等 shape 头"这一次会合，会阻塞并把整个 pipeline 挂死。梯度相关的其它动作
        早已门住（consumer 的 FIFO 记账、cooldown 整块、phase ④ 的 encoder 反传、两处
        finalize_model_grads），守卫写在函数体里而不是逐个调用点，是为了让 steady 与
        cooldown 收尾两处调用一次性覆盖。
        Return immediately under ``forward_only`` (eval): the consumer takes the pure-forward
        branch and never dispatches boundary grads, so this irecv would have no counterpart and
        the header rendezvous inside the receive would block and hang the whole pipeline. The
        guard lives in the function body rather than at each call site so that both the
        steady-state call and the cooldown tail call are covered at once.
        """
        if forward_only or is_consumer or microbatch < 0 or microbatch % group_size != producer_id:
            return
        request = comm.colocated_recv_backward(producer=producer_id, wait=False)
        pending_grad_requests[microbatch] = request

    def _finish_boundary_grad_recv(microbatch: int) -> None:
        """Producer step ③: wait the grad posted by ① for ``microbatch`` and store it.

        ``finish()`` 等数据 handle（补货 step 里已后台传输了一整个 forward 的时间，通常
        早已完成）→ 把梯度存进 ``producer_grad_buffers``，phase ④ 统一 encoder 反传时取用。
        没有对应的在飞请求（① 被守卫拦掉）时直接返回。
        必须在 ④ 补货**之前**完成：补货的 isend 与这里的梯度 irecv 是**同一对 peer**
        （producer p ↔ consumer 0）→ 同一个 NCCL communicator 与 stream，未完成的 irecv
        会挡住后面 enqueue 的 isend。
        ``forward_only``（eval）下直接返回：与 ① 对称——eval 不产生边界梯度，此时字典本该
        是空的，早返回让这个不变量显式而不是依赖"pop 不到就返回"的巧合。
        Return immediately under ``forward_only``, symmetric with step ①: eval produces no
        boundary grads, so the dictionary is expected to be empty and the early return states
        that invariant explicitly instead of relying on the pop simply missing.
        """
        if forward_only:
            return
        request = pending_grad_requests.pop(microbatch, None)
        if request is None:
            return
        producer_grad_buffers[microbatch] = request.finish().grad

    def _restock_boundary_packet(microbatch: int) -> None:
        """Producer step ④: asynchronously send the next packet it owns ("consume one, add one").

        生产者在补货 step 的**最后一件事**：forward 推进到 microbatch k（k%P==producer_id，
        即"流水消耗到我负责的包"）后，异步发下一个自己负责的包（k+P）——"消费一个补一个"。
        ``wait=False`` 提前发送，与 consumer 的提前一整步 prefetch（4.5）配对。

        NVTX：外层区间额外覆盖补货的所属判断与 4.9 的伪释放——内层
        ``colocated-boundary-send-forward`` 只覆盖发送本身。
        NVTX: the outer range also covers the ownership check and the 4.9 pseudo-release.
        """
        if is_consumer or microbatch % group_size != producer_id:
            return
        nvtx_range_push("colocated-boundary-packet-restock")
        next_microbatch = microbatch + group_size
        if next_microbatch < num_microbatches:
            comm.colocated_send_forward(
                encoder_buffers[next_microbatch], producer=producer_id, wait=False
            )
            _deallocate_encoder_output(encoder_buffers[next_microbatch])  # 4.9
        nvtx_range_pop("colocated-boundary-packet-restock")

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

        input_tensor = p2p_communicator.recv_forward(  # 接受前序 stage 的激活
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        # 4.3i：伴随接收 labels/loss_mask（在 recv_forward 之后、两端同序）；4.3j 起
        # _recv_intra_packet 返回张量，不再写入模型属性。
        received_labels, received_loss_mask = _recv_intra_packet(
            input_tensor, p2p_communicator.is_pp_first_stage
        )
        # 4.5/4.5b：consumer 提前一整步 prefetch——forward microbatch i 前，异步收下一个
        # 需要的包（i+1，producer (i+1)%P）进 prefetched，让数据传输与当前 step 计算重叠。
        # 4.5b（2026-08-26）：去掉原先的 `next_microbatch >= group_size` 守卫，**首波
        #（1..P-1）也走这条同一规则**（原先首波由流水前全量 prefetch 负责，已删）——
        # warmup 的 step i 正好 post mb i+1 的 recv，与 producer 流水前的异步首包配对。
        # 唯一跳过的情况是 producer 0（本地直传、无通信）——它同时覆盖了"最后一个
        # forward"这个边界：n % P == 0（validate_colocated_num_microbatches），故
        # i+1 == n 时 next_producer == 0，无需再判越界。
        # 4.5/4.5b: the consumer prefetches one full step ahead — before forwarding microbatch
        # i it asynchronously receives the next needed packet (i+1, producer (i+1)%P) so the
        # transfer overlaps the current step's compute. 4.5b drops the former
        # `next_microbatch >= group_size` guard so the first wave (1..P-1) follows the very
        # same rule (the pre-pipeline bulk prefetch is gone); warmup step i posts the recv of
        # mb i+1, pairing with the producers' async startup sends. The only skipped case is
        # producer 0 (local hand-off), which also covers the last forward: n % P == 0, so
        # i+1 == n maps to producer 0 and no range check is needed.
        if is_consumer:
            next_microbatch = i + 1
            next_producer = next_microbatch % group_size
            if next_producer != 0:
                prefetched[next_producer] = comm.colocated_recv_forward(
                    next_producer, expected_microbatch_id=next_microbatch, wait=False
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
        # 4.4：warmup 是纯前传、consumer 还没开始反传，**没有梯度可收**（①③ 不适用），
        # 只有 producer 的补货（④）——它是前向侧的动作。
        # 4.4: warmup is forward-only and the consumer has not started backwarding yet, so
        # there is no grad to receive (①③ do not apply); only the producer's restock (④),
        # which is a forward-side action.
        _restock_boundary_packet(i)
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)  # 发给后序 stage
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

    nvtx_range_pop("colocated-warmup")

    # Before running 1F1B, need to receive first forward tensor.
    # 进入 steady 前先收 steady 第一个 microbatch 的输入激活。
    # 4.3i：steady 第一个 microbatch 的伴随 labels/loss_mask——存为 pending 变量，
    # 供 steady 循环第一轮的 forward 绑定与发送（4.3j：intra_packet 输入载体）。
    pending_labels, pending_loss_mask = None, None
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
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
        # 4.6d ①：producer 先异步收上一个自己产的 microbatch（k-P）的梯度（与 forward 重叠）。
        # 4.6d ①: the producer first posts the grad irecv of its previous microbatch (k-P).
        _start_boundary_grad_recv(current_microbatch - group_size)
        # 4.5/4.5b：consumer 提前一整步 prefetch——forward k 前，异步收下一个需要的包
        #（k+1，producer (k+1)%P）进 prefetched，与当前 step 计算重叠。4.5b（2026-08-26）
        # 与 warmup 用同一条规则（唯一守卫是"producer 0 本地直传"）；steady 里
        # k+1 >= P 天然成立，去掉原守卫不改变行为，只是两处形态统一。
        # 4.5/4.5b: same one-step-ahead prefetch as warmup — before forwarding k, post the
        # recv of k+1 (producer (k+1)%P). The only guard left is "producer 0 is local"; in the
        # steady loop k+1 >= P always holds, so dropping the former guard is behaviour-neutral
        # and merely makes both sites identical.
        if is_consumer:
            next_microbatch = current_microbatch + 1
            next_producer = next_microbatch % group_size
            if next_producer != 0:
                prefetched[next_producer] = comm.colocated_recv_forward(
                    next_producer, expected_microbatch_id=next_microbatch, wait=False
                )

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
        # 4.6d ③④：前传结束——先等 ① 的梯度落地，再补货（顺序不能换）。
        # 4.6d ③④: after the forward, wait the grad posted by ① and only then restock.
        _finish_boundary_grad_recv(current_microbatch - group_size)
        _restock_boundary_packet(current_microbatch)
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
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(  # 发激活、收梯度
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )
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

            input_tensor_grad = backward_func(  # 单个 microbatch 的反传
                input_tensor, output_tensor, output_tensor_grad, config
            )
            if is_consumer:  # 4.6c：只有 consumer 持边界输入、需要取梯度并派发给 producer
                _dispatch_boundary_grad()

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
                pending_labels, pending_loss_mask = _recv_intra_packet(  # 4.3i：下一轮 forward 的载体
                    input_tensor, p2p_communicator.is_pp_first_stage
                )

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

            output_tensor_grad = p2p_communicator.recv_backward(  # 从后序 stage 收梯度
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )
            if is_consumer:  # 4.6c：只有 consumer 持边界输入、需要取梯度并派发给 producer
                _dispatch_boundary_grad()

            p2p_communicator.send_backward(  # 把梯度发给前序 stage
                input_tensor_grad, p2p_communicator.is_pp_first_stage
            )

        nvtx_range_pop("colocated-cooldown")

        # 4.6c：cooldown 结束——wait 掉各 producer 桶里剩余的 isend，此后 consumer 侧无在飞
        # 梯度发送；并断言 FIFO 已空（每个前传的 microbatch 都恰好被反传+派发一次）。
        # 4.6c: after cooldown, drain every producer's remaining isend and assert the FIFO
        # drained (every forwarded microbatch was backwarded and dispatched exactly once).
        # NVTX：这里**不标区间**——整段只有 handle.wait()，而 Work::wait() 只插 stream-wait、
        # CPU 不阻塞（probe 实测 0.000 s），CPU 侧区间恒为 0；真正的等待在 GPU 时间线上。
        # NVTX: no range here - the block is nothing but handle.wait(), which does not block
        # the CPU, so a CPU-side range would always be empty. #这里统一 wait 生产者把梯度发给encoder自身，后续异构batch需要调整，也许会变成不 wait，来一个算一个反传
        for handles in boundary_grad_send_handles.values():
            for handle in handles:
                handle.wait()
        boundary_grad_send_handles.clear()
        assert not consumer_boundary_inputs, (
            f"boundary grads not dispatched for microbatches "
            f"{[microbatch for microbatch, _ in consumer_boundary_inputs]}"
        )

        # 4.6d：cooldown 收尾——producer 额外收最后一个自己产的 microbatch 的梯度。
        # 为什么恰好差这一个：owned microbatch m 的梯度是在"forward 推进到 m+P"的那个补货
        # step 的 ① 收的，而最后一个 owned microbatch 满足 m+P >= n（没有那一步），所以每个
        # producer 恰好剩 1 个没收。它的反传发生在 consumer 的 cooldown 里，此刻 consumer
        # 已经派发（4.6c 的 drain 与这里配对）。
        # 复用 ①③ 两个接口（参数是"收谁的梯度"，与补货 step 无关）：这里没有 forward 可以
        # 重叠，start 后立即 finish，等价于同步收。
        # 4.6d: after cooldown the producer receives the grad of the last microbatch it owns —
        # grads are otherwise received by step ① at m+P, and the last owned microbatch has
        # m+P >= n, so exactly one is left per producer. The same ①③ helpers are reused (their
        # parameter is whose grad to receive); with no forward to overlap, start is immediately
        # followed by finish, which is equivalent to a synchronous receive.
        if not is_consumer:
            last_owned_microbatch = max(encoder_buffers)
            _start_boundary_grad_recv(last_owned_microbatch)
            _finish_boundary_grad_recv(last_owned_microbatch)

        # 4.6d：边界梯度收齐——每个 rank 的梯度 buffer 恰好覆盖自己产的全部 microbatch
        # （phase ④ 统一 encoder 反传的前提），且没有遗留在飞的接收请求。
        # 4.6d: the grads are complete — every rank's buffer covers exactly the microbatches it
        # produced (the precondition of the unified phase-④ encoder backward).
        assert not pending_grad_requests, (
            f"boundary grad receives still in flight for microbatches "
            f"{sorted(pending_grad_requests)}"
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
