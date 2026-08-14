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
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.colocated_encoder_comm import (
    EncoderBackboneBoundaryCommunicator,
    ForwardPacket,
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
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import get_model_config


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

    共置 schedule（顶层函数，与其余 schedule 一致）：model 必须是
    ``[encoder_chunk, backbone_chunk]``（Task 2 的 provider 返回）。签名与
    ``forward_backward_pipelining_without_interleaving``（schedules.py:2035）对齐
    （关键字参数），train_step（training.py:1902）原样调用。

    职责分工（2026-08-12 用户确认，对照 schedules.py）：本函数（schedule）只做编排
    ——phase ① 循环调 ``forward_step_func`` 的 encoder 分支拿包存 buffer；phase ② 调
    现有 1F1B；phase ④ 直接 autograd。数据获取 / 模型 forward / loss 全在注入的
    ``forward_step_func``（业务层，见 examples/multimodal/colocated_train.py），
    schedule 不接收 ``get_batch_fn`` / ``image_token_index`` / ``img_seq_len`` 等
    数据/模型参数（train_step 也不传它们，training.py:1902）。

    Args:
        forward_step_func / data_iterator / num_microbatches / seq_length /
        micro_batch_size / decoder_seq_length / forward_only / collect_non_loss_data:
            与现有 schedule 契约一致（train_step 传入）。
            一个函数按 ``model[0]`` 的 chunk 类型分支：encoder（phase ①，本函数传
            ``model=[encoder_chunk]``，返回 ``(ForwardPacket, None)``）/ backbone
            （phase ②，1F1B 传 ``model=[backbone_chunk]``，返回 ``(output, loss_func)``）。

    当前实现：phase ①（4.2：已建边界 communicator、phase ① 走 forward step 的
    encoder 分支）→ phase ②（4.3c：统一自写 1F1B 循环，PP=1 时 P2P 空转、全本地 take）；
    phase ④（统一 encoder 反传）由 Task 4.6 接入（当前先返回 backbone 的 loss）。
    """
    assert isinstance(model, (list, tuple)) and len(model) == 2, (
        "colocated schedule needs model = [encoder_chunk, backbone_chunk], "
        f"got {type(model)} of len {len(model)}"
    )
    encoder_chunk, backbone_chunk = model
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

    # Phase ①：encoder 轮盘前传 -> 本地 buffer（本任务）。schedule 只调 forward step
    # 的 encoder 分支拿包并存储，数据/模型细节全在业务层。
    # Phase ①: encoder round-robin forward -> local buffer. The schedule only calls
    # the forward step's encoder branch and stores the returned packet.
    encoder_buffers = _colocated_encoder_forward(
        forward_step_func, data_iterator, encoder_chunk, num_microbatches
    )
    # Phase ②：backbone 1F1B 调度（Task 4.3c，2026-08-13 用户确认统一走自写循环）。
    # 自写循环天然覆盖 PP=1：P=1 时 producer 恒 0、is_consumer=True、全部本地 take、
    # P2P 空转（P2PCommunicator 在 first=last 时全部方法安全跳过/返回 None）——退化为
    # 标准 DP，不再单独走无流水调度（也避免"无流水调度不感知包"
    # 的额外包装）。Phase ④（统一 encoder 反传）由 Task 4.6 接入；当前先返回 backbone
    # 的 loss（forward_data_store，与 1F1B 一致，train_step 在 training.py:1902 直接消费）。
    # Phase ②: backbone schedule (Task 4.3c). The self-written loop also covers PP=1
    # (P=1: producer is always 0, is_consumer=True, all-local take, P2P no-ops), so no
    # separate pp-size dispatch is needed.
    return colocated_backbone_forward_backward_pipelining_without_interleaving(
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


def _colocated_encoder_forward(
    forward_step_func,
    data_iterator,
    encoder_chunk,
    num_microbatches: int,
) -> Dict[int, ForwardPacket]:
    """Phase ①: schedule loops over the forward step's encoder branch, stores packets.

    本 rank（producer p）负责的 microbatch = p, p+P, p+2P, ...（``get_microbatches_for_pipeline_stage``，
    Task 1.2）——每 producer 恰好 ``num_microbatches / P`` 个 = ``global_mbs / (dp * inner_dp)``。
    逐 microbatch 调 ``forward_step_func(data_iterator, [encoder_chunk])`` 的 **encoder
    分支**，把返回的 ``ForwardPacket``（image_embeddings 保留 grad_fn、phase ④ 统一
    反传用；文本字段一并携带）存入 buffer[microbatch]——**数据获取与 encoder 前传全在
    forward step（业务层 colocated_train.py），schedule 只负责调用与存储**。

    注意：这里**不 detach**——分离图发生在发送/组装时（4.4 对发出去的副本
    detach，backbone 反传不进入 encoder 图；本地原图保留给 phase ④ 反传）。
    """
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    # 轮盘分配：producer p -> microbatch p, p+P, p+2P, ...（每个 producer num_microbatches/P 个）。
    # Round-robin: producer p handles microbatches p, p+P, p+2P, ... (num_microbatches/P each).
    microbatches = parallel_state.get_microbatches_for_pipeline_stage(
        pipeline_parallel_rank, num_microbatches
    )

    encoder_buffers: Dict[int, ForwardPacket] = {}
    for microbatch in microbatches:
        # forward step 的 encoder 分支：取一个 micro batch 数据 + encoder_chunk(images)，
        # 返回 (ForwardPacket, None)。schedule 只拿包存 buffer，并给包打上 microbatch id
        # 字段（1 元素 int64 张量）——业务层不知道 id，id 是调度元数据；serialize 时作为
        # 第 6 个字段写入，消费者 take 校验。
        packet, _ = forward_step_func(data_iterator, [encoder_chunk])
        packet.microbatch_id = torch.tensor(
            [microbatch], dtype=torch.int64, device=packet.image_embeddings.device
        )
        encoder_buffers[microbatch] = packet

    # 完整性：本 rank 恰好 num_microbatches / P 个 microbatch（每个 producer 负载一致）。
    # Completeness: exactly num_microbatches / P entries (balanced across producers).
    expected = num_microbatches // pipeline_parallel_size
    assert len(encoder_buffers) == expected, (
        f"rank pipeline_parallel_rank={pipeline_parallel_rank} got {len(encoder_buffers)} "
        f"microbatches, expected {expected} (num_microbatches {num_microbatches} / "
        f"pipeline_parallel_size {pipeline_parallel_size})"
    )
    assert sorted(encoder_buffers) == microbatches, (
        f"buffer keys {sorted(encoder_buffers)} != round-robin microbatches {microbatches}"
    )
    return encoder_buffers


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
      用 ``functools.partial`` 绑定到 forward step 传给业务层（packet 走闭包、**不经
      模型属性**）。4.4 把"全量发送"改为 priming + replenish 流水化；4.5 把 consumer
      收包改 prefetch 异步。

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
    if config.overlap_p2p_comm:  # 非交错调度中 P2P 通信是同步的，不支持 overlap
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

    # 4.3b：**consumer 先 prefetch（每 producer 一个 header irecv 先入队）**——NCCL P2P
    # 首次连接需要两端配对：producer 的 isend 若在 consumer 的 irecv 之前提交会阻塞
    #（实测死锁：producer 卡在 isend、进不了 backbone recv，consumer 卡在 warmup
    # send_forward、到不了 take）——consumer 先入队 irecv 建立连接，producer 的 isend
    # 配对后不阻塞。4.5 将把这里的"全量 prefetch"改为逐 step prefetch（提前一整步
    # overlap）。
    # 4.3b: the consumer first prefetches one header-irecv per producer (establishes the
    # NCCL P2P connection), so the producers' isends (below) do not block.
    prefetched = {}
    if is_consumer:
        for producer in range(1, group_size):
            # n=P（4.3f 冒烟）：每 producer 只发 1 个包（microbatch = producer），prefetch
            # 时即可预设 expected microbatch id；n>P 的多包由 4.4 replenish 处理。
            prefetched[producer] = comm.colocated_recv_forward(
                producer, expected_microbatch_id=producer, wait=False
            )

    # 4.3b：producer（stage>0）在进入 backbone 流水前**异步**发送 phase ① 的全部包
    #（isend 入队不等待，wait 推迟到函数末尾统一做）——consumer 的 prefetch irecv 已先
    # 入队（连接建立），producer 的 isend 配对后不阻塞。
    # 4.4 将把这里的"全量发送"改为 priming + replenish 流水化（1 深 in-flight）。
    # 4.3b: producers (stage>0) asynchronously send all phase-① packets before entering
    # the backbone pipeline (isend enqueued without waiting; waits happen at the end).
    send_handles = []
    if not is_consumer:
        for microbatch in sorted(encoder_buffers):
            send_handles.append(
                comm.colocated_send_forward(
                    encoder_buffers[microbatch], producer=producer_id, wait=False
                )
            )

    def _take_forward_packet(microbatch: int) -> ForwardPacket:
        """Consumer (stage 0): take the forward packet of ``microbatch``.

        消费者（stage 0）前传 microbatch ``microbatch`` 前取它的前向包：producer 0 本地
        直传（``encoder_buffers[microbatch]``，零拷贝）、producer>0 用循环前 prefetch 的
        ``_ForwardRecvRequest`` 的 ``start().finish()``（header 已在 prefetch 时 irecv、
        连接已建立；start 等 header → 分配 → 发数据 irecv，finish 等数据 → 组装）。
        **返回包，由调用方用 ``functools.partial`` 绑定到 forward step**（packet 走闭包
        传给业务层，不经模型属性——模型不背调度交接状态）。
        """
        producer = microbatch % group_size  # 轮盘映射：microbatch s 由 producer s%P 算（doc §2.2）
        if producer == 0:
            return encoder_buffers[microbatch]  # producer 0 = 自己：本地直传（零拷贝）
        return prefetched[producer].start().finish()

    # --- 4.3i：backbone P2P 伴随传输 new_labels/new_loss_mask（4.3h 定案）---
    # 在 send_forward/recv_forward 之后、两端同序（先 activation 后 labels/mask）额外小
    # 传输：consumer 组装产物 / 中间 stage 透传，last stage 消费算 loss。反传只走
    # activation 梯度（labels/mask 不参与）。shape 由 activation 推导（labels [b,s']）。
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_next_rank = dist.get_global_rank(pp_group, (pp_rank + 1) % pp_size)
    pp_prev_rank = dist.get_global_rank(pp_group, (pp_rank - 1) % pp_size)
    target_send_handles = []

    def _send_targets(is_pp_last_stage: bool) -> None:
        """Send new_labels/new_loss_mask alongside the activation (4.3h).

        在 send_forward 之后伴随发送 ``colocated_new_labels``/``colocated_new_loss_mask``
        （consumer 组装产物 / 中间 stage 透传）；末 stage 或属性为 None（当前 stage 不
        参与）时跳过。isend 不等待，handle 在函数末尾统一 wait。
        """
        if is_pp_last_stage:
            return
        labels = getattr(model, "colocated_new_labels", None)
        loss_mask = getattr(model, "colocated_new_loss_mask", None)
        if labels is None or loss_mask is None:
            return
        target_send_handles.append(
            dist.isend(labels.contiguous(), dst=pp_next_rank, group=pp_group)
        )
        target_send_handles.append(
            dist.isend(loss_mask.contiguous(), dst=pp_next_rank, group=pp_group)
        )

    def _recv_targets(input_tensor: Optional[torch.Tensor], is_pp_first_stage: bool) -> None:
        """Receive new_labels/new_loss_mask alongside the activation (4.3h).

        在 recv_forward 之后伴随接收（两端同序），写入模型属性供 forward step 使用
        （last stage 算 loss）与后续透传。shape 由 activation 推导：activation
        [s', b, h] → labels/loss_mask [b, s']。
        """
        if is_pp_first_stage or input_tensor is None:
            return
        s_prime, batch = input_tensor.shape[0], input_tensor.shape[1]
        labels = torch.empty((batch, s_prime), dtype=torch.int64, device=input_tensor.device)
        loss_mask = torch.empty(
            (batch, s_prime), dtype=torch.float32, device=input_tensor.device
        )
        target_recv_handles = [
            dist.irecv(labels, src=pp_prev_rank, group=pp_group),
            dist.irecv(loss_mask, src=pp_prev_rank, group=pp_group),
        ]
        for handle in target_recv_handles:
            handle.wait()
        model.colocated_new_labels = labels
        model.colocated_new_loss_mask = loss_mask

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
        # 4.3i：伴随接收 labels/loss_mask（在 recv_forward 之后、两端同序）。
        _recv_targets(input_tensor, p2p_communicator.is_pp_first_stage)
        # 4.3b：consumer 前传 microbatch i 前 take 包（stage 0 前传 0..P-2），并用
        # ``functools.partial`` 绑定到 forward step（packet 走闭包，不经模型属性）；
        # 非 consumer 保持注入的 forward_step_func 不变。
        if is_consumer:
            forward_step_func = partial(forward_step_func, packet=_take_forward_packet(i))
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
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)  # 发给后序 stage
        _send_targets(p2p_communicator.is_pp_last_stage)  # 4.3i：伴随发送 labels/loss_mask
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # 进入 steady 前先收 steady 第一个 microbatch 的输入激活。
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        # 4.3i：伴随接收 labels/loss_mask（steady 第一个 microbatch 的）。
        _recv_targets(input_tensor, p2p_communicator.is_pp_first_stage)

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

        # 4.3b：consumer 前传 microbatch W+i 前 take 包，并 partial 绑定到 forward step。
        if is_consumer:
            forward_step_func = partial(
                forward_step_func, packet=_take_forward_packet(i + num_warmup_microbatches)
            )
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
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            _send_targets(p2p_communicator.is_pp_last_stage)  # 4.3i：伴随发送 labels/loss_mask
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
                _recv_targets(input_tensor, p2p_communicator.is_pp_first_stage)  # 4.3i
        else:  # 训练走这里，前传+反传
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(  # 发激活、收梯度
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )
            _send_targets(p2p_communicator.is_pp_last_stage)  # 4.3i：伴随发送 labels/loss_mask

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

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
                _recv_targets(input_tensor, p2p_communicator.is_pp_first_stage)  # 4.3i

    # Run cooldown backward passes.
    # 执行 cooldown 阶段的反传。
    if not forward_only:
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

            p2p_communicator.send_backward(  # 把梯度发给前序 stage
                input_tensor_grad, p2p_communicator.is_pp_first_stage
            )

        # Launch any remaining grad reductions.
        # 开启剩余的梯度归约。
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
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

    # 4.3b：phase ② 结束，等待所有异步 send 完成（encoder_buffers 下个 step 的 phase ①
    # 会覆盖，发送必须在本 step 内完成）。
    # Wait for all asynchronous boundary sends before returning (the encoder_buffers are
    # overwritten by phase ① of the next step, so the sends must have completed).
    for handle in send_handles:
        for h in handle:
            h.wait()
    # 4.3i：等待伴随传输的 labels/loss_mask isend 完成（同 pp_group，下个 step 复用前）。
    for handle in target_send_handles:
        handle.wait()

    return forward_data_store
