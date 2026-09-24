# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the colocated encoder<->backbone boundary communicator (Task 3.4).

共置边界通信器（``EncoderBackboneBoundaryCommunicator``）的单元测试，按 world_size 自适应：
- **world_size=1**（PP=1）：唯一 rank 即 producer 0 = 消费者 → 全本地直传（调用方分支），
  四个公开收发方法对 producer 0 全部断言报错，零网络；
- **world_size=N≥2**（pp=N）：rank 0 消费者（producer 0 本地分支 + producer 1..N-1
  网络收发），producer p>0 网络发送包/接收梯度——1 发 1 收配对、错误路径断言、无死锁；
- **定长布局 + 字节对齐**：dual-channel-p2p（2026-09-22）HEADER-FREE / 定长协议下每个样本
  shape 由 config 静态固定，通信器构造期算好定长 layout；测试通过显式 dim kwargs
  （image_seq_length/micro_batch_size/hidden_size/text_seq_length）构造 comm，round-trip serialize→deserialize；
  ``_ALIGN`` 8 字节对齐用 image_embeddings 60 字节（bf16 numel=30）的非对齐场景覆盖。

运行（CI 标准方式）：
    torchrun --nproc_per_node=N -m pytest tests/unit_tests/pipeline_parallel/test_colocated_encoder_comm.py
"""

import pytest
import torch
import torch.distributed as dist

import megatron.core.parallel_state as ps
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.colocated_encoder_comm import (
    BackwardPacket,
    EncoderBackboneBoundaryCommunicator,
    ForwardPacket,
    _padded_bytes,
)
from tests.unit_tests.test_utilities import Utils

DTYPE = torch.bfloat16
# 字段名单一来源：ForwardPacket.field_names（与 to_dict()/fields 一致）。
FIELDS = ForwardPacket.field_names

# dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：默认固定 dims，测试显式传给通信器
# （避免 mock get_args）。image_embeddings=(IMAGE_SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE)，
# tokens/labels=(MICRO_BATCH_SIZE, TEXT_SEQ_LENGTH)，num_image_tiles=(MICRO_BATCH_SIZE,)，microbatch_id=(1,)。
IMAGE_SEQ_LENGTH = 4
MICRO_BATCH_SIZE = 3
HIDDEN_SIZE = 8
TEXT_SEQ_LENGTH = 20


def _make_config():
    return ModelParallelConfig(pipeline_dtype=DTYPE)


def _make_packet(
    seed_base,
    image_seq_length=IMAGE_SEQ_LENGTH,
    micro_batch_size=MICRO_BATCH_SIZE,
    hidden_size=HIDDEN_SIZE,
    text_seq_length=TEXT_SEQ_LENGTH,
    microbatch_id=None,
):
    """Deterministic fixed-layout ForwardPacket (same seed -> same values on every rank).

    dual-channel-p2p（2026-09-22）：包的 shape 必须精确匹配通信器的定长 layout（否则
    deserialize 会 reshape 回定长 shape、与原包不符）。``microbatch_id`` 为 int 时转成
    1 元素 int64 张量（第 5 个字段），None 则不设。
    """
    g = torch.Generator(device="cuda").manual_seed(seed_base)
    mb_id_t = None if microbatch_id is None else torch.tensor(
        [microbatch_id], dtype=torch.int64, device="cuda"
    )
    return ForwardPacket(
        image_embeddings=torch.randn(
            (image_seq_length, micro_batch_size, hidden_size), dtype=DTYPE, device="cuda", generator=g
        ),
        tokens=torch.randint(
            0, 30000, (micro_batch_size, text_seq_length), dtype=torch.int64, device="cuda", generator=g
        ),
        labels=torch.randint(
            0, 30000, (micro_batch_size, text_seq_length), dtype=torch.int64, device="cuda", generator=g
        ),
        num_image_tiles=torch.ones(micro_batch_size, dtype=torch.int32, device="cuda"),
        microbatch_id=mb_id_t,
    )


def _make_grad(seed_base, shape=(IMAGE_SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE)):
    g = torch.Generator(device="cuda").manual_seed(seed_base)
    return torch.randn(shape, dtype=DTYPE, device="cuda", generator=g)


def _check_equal(a, b, what):
    a, b = a.to_dict(), b.to_dict()
    for k in FIELDS:
        if a[k] is None or b[k] is None:
            assert a[k] is None and b[k] is None, f"{what}.{k}: one side missing"
            continue
        assert a[k].shape == b[k].shape, f"{what}.{k} shape {tuple(a[k].shape)} != {tuple(b[k].shape)}"
        assert a[k].dtype == b[k].dtype, f"{what}.{k} dtype {a[k].dtype} != {b[k].dtype}"
        assert torch.equal(a[k], b[k]), f"{what}.{k} values differ"


def _expect_assertion(fn, what):
    with pytest.raises(AssertionError):
        fn()


def _make_comm(image_seq_length=IMAGE_SEQ_LENGTH, micro_batch_size=MICRO_BATCH_SIZE, hidden_size=HIDDEN_SIZE, text_seq_length=TEXT_SEQ_LENGTH):
    """Initialize the colocated parallel state and return the boundary communicator.

    TP=1, PP=world_size, DP=1: the single replica is the whole world, so the
    colocated boundary group spans all ranks (producer order) and the consumer
    is producer 0 = global rank 0.

    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：四个定长 dim 通过可选 kwargs 显式传入，
    覆盖 ``get_args()``（单测里 get_args 未初始化，故显式提供）。
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=Utils.world_size,
        use_colocated_encoder=True,
    )
    comm = EncoderBackboneBoundaryCommunicator(
        ps.get_colocated_boundary_activation_group(),
        ps.get_colocated_boundary_grad_group(),
        _make_config(),
        image_seq_length=image_seq_length,
        micro_batch_size=micro_batch_size,
        hidden_size=hidden_size,
        text_seq_length=text_seq_length,
    )
    assert comm.group_size == Utils.world_size
    return comm


def test_consumer_local_shortcut_and_network_roundtrip():
    """Local direct-transfer (producer 0) + network round-trip for producers > 0.

    world_size=1 -> PP=1 all-local: four public methods reject producer 0.
    world_size>=2 -> consumer (rank 0) local branch + network 1..N-1, producers
    send their packet and receive their grad; pairing correct, no deadlock.
    """
    world = Utils.world_size
    comm = _make_comm()
    p = comm.producer_id
    rank = Utils.rank
    try:
        if world == 1:
            # PP=1: the only rank is producer 0 = consumer -> all local, zero network.
            assert p == 0 and comm.is_consumer()
            pkt = _make_packet(101)
            got = ForwardPacket(*pkt.fields)
            assert set(got.to_dict()) == set(FIELDS)
            assert got.image_embeddings is pkt.image_embeddings  # zero-copy
            _expect_assertion(lambda: comm.colocated_recv_forward(0), "recv_forward(0)")
            _expect_assertion(
                lambda: comm.colocated_send_forward(pkt, producer=0), "send_forward(producer=0)"
            )
            _expect_assertion(
                lambda: comm.colocated_send_backward(BackwardPacket(grad=_make_grad(401)), 0),
                "send_backward(grad, 0)",
            )
            _expect_assertion(lambda: comm.colocated_recv_backward(), "recv_backward() (producer_id=0)")
        elif rank == 0:
            # Consumer (producer 0): local shortcut + network receive/send.
            assert p == 0 and comm.is_consumer()
            local_pkt = _make_packet(101)
            got = ForwardPacket(*local_pkt.fields)
            assert got.image_embeddings is local_pkt.image_embeddings  # zero-copy
            _expect_assertion(lambda: comm.colocated_recv_forward(0), "consumer recv_forward(0)")
            _expect_assertion(
                lambda: comm.colocated_send_forward(_make_packet(101), producer=0),
                "consumer send_forward(producer=0)",
            )
            _expect_assertion(
                lambda: comm.colocated_send_backward(BackwardPacket(grad=_make_grad(302)), 0),
                "consumer send_backward(grad, 0)",
            )
            _expect_assertion(lambda: comm.colocated_recv_backward(0), "consumer recv_backward(0)")
            # Network forward: receive producers 1..N-1 in order.
            for q in range(1, world):
                _check_equal(
                    comm.colocated_recv_forward(q),
                    _make_packet(200 + q, microbatch_id=200 + q),
                    f"recv_forward({q})",
                )
            # Network backward: send each producer's grad in order.
            for q in range(1, world):
                comm.colocated_send_backward(BackwardPacket(grad=_make_grad(300 + q)), q)
        else:
            # Producer p > 0: send its packet, then receive its grad.
            assert p == rank and not comm.is_consumer()
            _expect_assertion(lambda: comm.colocated_recv_forward(0), "producer recv_forward(0)")
            _expect_assertion(lambda: comm.colocated_recv_forward(p), "producer recv_forward(self)")
            _expect_assertion(
                lambda: comm.colocated_send_backward(BackwardPacket(grad=_make_grad(303)), p),
                "producer send_backward",
            )
            _expect_assertion(lambda: comm.colocated_recv_backward(0), "producer recv_backward(0)")
            comm.colocated_send_forward(_make_packet(200 + p, microbatch_id=200 + p), producer=p)
            grad = comm.colocated_recv_backward().grad
            assert torch.equal(grad, _make_grad(300 + p)), "producer grad mismatch"
        dist.barrier()
    finally:
        Utils.destroy_model_parallel()


def test_fixed_layout_alignment():
    """Fixed-layout byte alignment round-trip (producer 1 <-> consumer).

    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：协议不再支持变长 shape（每个样本 shape
    由 config 静态固定）。本测试用一个定长布局 image_embeddings=(3,5,2) → bf16 60 字节的
    **非 8 对齐**场景覆盖 ``_ALIGN`` 填充：60 字节的浮点字段后，int64 tokens 若无填充会落在
    byte offset 60，触发 view(dtype) 的 storage_offset RuntimeError；``_ALIGN`` 把它补到 64。
    连发两个同布局但不同取值的包，验证多包 FIFO 顺序 + round-trip 保真。
    """
    if Utils.world_size < 2:
        pytest.skip("fixed-layout network test needs world_size >= 2")
    # 定长 dims 选成让 image_embeddings 落在 60 字节（bf16 numel=3*5*2=30 -> 60B，非 8 对齐）。
    comm = _make_comm(image_seq_length=3, micro_batch_size=5, hidden_size=2, text_seq_length=7)
    dims = dict(image_seq_length=3, micro_batch_size=5, hidden_size=2, text_seq_length=7)
    rank = Utils.rank
    try:
        assert _padded_bytes(60) == 64
        pkt_a = _make_packet(501, microbatch_id=501, **dims)
        pkt_b = _make_packet(502, microbatch_id=502, **dims)
        if rank == 1:
            comm.colocated_send_forward(pkt_a, producer=1)
            comm.colocated_send_forward(pkt_b, producer=1)
        elif rank == 0:
            _check_equal(comm.colocated_recv_forward(1), pkt_a, "packet A (fixed layout)")
            _check_equal(comm.colocated_recv_forward(1), pkt_b, "packet B (odd 60B offset)")
        dist.barrier()

        ga = _make_grad(601, (3, 5, 2))
        gb = _make_grad(602, (3, 5, 2))
        if rank == 0:
            comm.colocated_send_backward(BackwardPacket(grad=ga), 1)
            comm.colocated_send_backward(BackwardPacket(grad=gb), 1)
        elif rank == 1:
            ra = comm.colocated_recv_backward().grad
            rb = comm.colocated_recv_backward().grad
            assert ra.shape == ga.shape and torch.equal(ra, _make_grad(601, (3, 5, 2))), "grad A mismatch"
            assert rb.shape == gb.shape and torch.equal(rb, _make_grad(602, (3, 5, 2))), "grad B mismatch"
        dist.barrier()
    finally:
        Utils.destroy_model_parallel()


def test_microbatch_id_validation():
    """Microbatch-id validation of a received packet (pure function, no network).

    包自带 microbatch id 的校验逻辑（无网络）：id 匹配通过、不匹配抛 AssertionError、
    期望 None 跳过。校验的是 ``ForwardPacket.microbatch_id``（第 5 个字段，
    dual-channel-p2p 2026-09-22 起随定长扁平 buffer 传，deserialize 按定长 layout
    还原）。
    """
    comm = _make_comm()
    try:
        pkt = _make_packet(601, microbatch_id=7)
        comm._check_microbatch_id(pkt, 7)  # 匹配 -> 不抛
        comm._check_microbatch_id(pkt, None)  # 期望 None -> 跳过
        with pytest.raises(AssertionError):
            comm._check_microbatch_id(pkt, 8)  # 不匹配 -> 抛
        with pytest.raises(AssertionError):
            comm._check_microbatch_id(_make_packet(602), 7)  # 包无 id（None）-> 抛
    finally:
        Utils.destroy_model_parallel()


def test_async_send_wait_false_and_mb_id():
    """Async send (wait=False) + microbatch id round-trip (producer 1 -> consumer).

    生产者用 ``wait=False`` 异步发前向包（replenish 语义：post 后继续算、稍后统一
    wait），consumer 同步收并校验包自带的 microbatch id（第 5 个字段，随定长扁平 buffer 传）。
    """
    if Utils.world_size < 2:
        pytest.skip("network test needs world_size >= 2")
    comm = _make_comm()
    rank = Utils.rank
    try:
        pkt = _make_packet(701, microbatch_id=5)
        handles = None
        if rank == 1:  # producer 1
            handles = comm.colocated_send_forward(pkt, producer=1, wait=False)
        elif rank == 0:  # consumer
            got = comm.colocated_recv_forward(1, expected_microbatch_id=5)
            _check_equal(got, pkt, "async send + microbatch id")
            assert got.microbatch_id.item() == 5, "packet must carry its own microbatch id"
        # ``dist.barrier()`` 走的是**默认组（全 rank）**，所以必须放在分支外——每个 rank 的
        # barrier 次数要一致。此前这一次 barrier 写在 rank 0/1 各自的分支里，world=2 时恰好
        # 成立，world≥4 时 rank≥2 少调一次，barrier 全体错位一拍 → 死锁（2026-08-27 修）。
        # dist.barrier() is a default-group collective, so it must be called the same number
        # of times on every rank; keeping it inside the rank 0/1 branches deadlocked at
        # world_size >= 4, where the uninvolved ranks called it once fewer.
        dist.barrier()  # consumer 已收完，producer 现在才 wait（模拟 replenish：提交后继续算）
        if handles is not None:
            for handle in handles:
                handle.wait()
        dist.barrier()
    finally:
        Utils.destroy_model_parallel()


def test_async_recv_requests():
    """Async recv requests: consumer prefetch-take, producer replenish-step grad recv.

    - consumer 用 ``colocated_recv_forward(..., wait=False)`` 异步 prefetch 前向包（通信器
      在返回前已启动数据接收），需要时 ``request.finish()`` 取数（传输与计算重叠）；
    - producer 用 ``colocated_recv_backward(wait=False)`` 异步提交梯度接收（补发
      step 的 forward 前），consumer 反传后发梯度，producer ``finish()`` 拿到。

    **enqueue 顺序约束（NCCL 同组 P2P 严格 FIFO）**：两端必须"先前向后反传"——
    producer 先 ``send_forward`` 再 ``recv_backward``，consumer 先 ``recv_forward``
    再 ``send_backward``；若两端方向交叉（producer 先收梯度、consumer 先收前向），
    同组内未完成的 irecv 会阻塞后续 P2P，导致死锁。
    """
    if Utils.world_size < 2:
        pytest.skip("network test needs world_size >= 2")
    comm = _make_comm()
    rank = Utils.rank
    try:
        if rank == 1:  # producer 1
            pkt = _make_packet(801, microbatch_id=3)
            # 先前向后反传：先补发前向包（consumer 已提交 recv，同步发即可配对），
            # 再异步提交收上一个自己发送的 micro batch 的梯度。
            comm.colocated_send_forward(pkt, producer=1)
            grad_req = comm.colocated_recv_backward(wait=False)
            grad = grad_req.finish().grad
            assert grad.shape == (IMAGE_SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE) and grad.dtype == DTYPE, "async grad recv mismatch"
        elif rank == 0:  # consumer
            pkt = _make_packet(801, microbatch_id=3)
            recv_req = comm.colocated_recv_forward(1, expected_microbatch_id=3, wait=False)  # prefetch
            got = recv_req.finish()  # take（prefetch 时已启动数据接收，finish 只取数）
            _check_equal(got, pkt, "async prefetch + take")
            assert got.microbatch_id.item() == 3, "packet must carry its own microbatch id"
            comm.colocated_send_backward(BackwardPacket(grad=_make_grad(901, (IMAGE_SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE))), 1)
        dist.barrier()
    finally:
        Utils.destroy_model_parallel()


def test_serialize_deserialize_roundtrip():
    """Pure serialize -> deserialize round-trip on the static fixed layout (no network).

    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：``serialize()`` 只返回一个扁平 uint8
    buffer（无 shape 头），``deserialize(flat, fields_layout)`` 按静态 ``(shape, dtype)``
    列表切分还原各字段。验证 5 个字段的 shape/dtype/取值 round-trip 保真 + microbatch id 保留
    （不依赖 world_size / 网络，直接打通新协议的序列化布局）。
    """
    pkt = _make_packet(1234, microbatch_id=9)
    flat = pkt.serialize()
    assert flat.dtype == torch.uint8 and flat.ndim == 1, "serialize must return one flat uint8 buffer"
    # 与通信器 __init__ 构造的 _forward_fields 同布局（顺序 == ForwardPacket.field_names）。
    fields_layout = [
        ((IMAGE_SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE), DTYPE),
        ((MICRO_BATCH_SIZE, TEXT_SEQ_LENGTH), torch.int64),
        ((MICRO_BATCH_SIZE, TEXT_SEQ_LENGTH), torch.int64),
        ((MICRO_BATCH_SIZE,), torch.int32),
        ((1,), torch.int64),
    ]
    got = ForwardPacket.deserialize(flat, fields_layout)
    _check_equal(got, pkt, "serialize/deserialize round-trip")
    assert got.microbatch_id.item() == 9, "microbatch id must survive the round-trip"
