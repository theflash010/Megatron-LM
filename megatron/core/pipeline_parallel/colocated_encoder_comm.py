# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Producer/consumer boundary communicator for colocated encoder training.

共置训练的"生产者/消费者"边界通信器：
- **生产者（producer）**：每个 encoder rank 都是一个生产者——把自己的 encoder 输出
  与本地文本数据包（``ForwardPacket``）发给消费者；
- **消费者（consumer）**：副本内的 backbone entry（backbone 首级）是唯一消费者——
  接收所有生产者的前向包，并把 encoder 输出梯度（``BackwardPacket``）发回各生产者。

轮盘调度：microbatch s 由生产者 s 计算并发给消费者；**producer 0 即消费者自己**，
其前向包与梯度**不经网络**——由调用方（wrapper）在调用处分支短路（producer 0 的
包/梯度零拷贝本地引用，不调用本类任何方法，因此不打乱网络侧 P2P 配对计数）。
本类四个公开收发方法只服务 producer > 0 的网络路径。

数据包由协议类承载（**Task 4.2e**）：
- ``ForwardPacket``：5 字段（4 个内容字段 image_embeddings/tokens/labels/
  num_image_tiles + **microbatch id 字段**，均为真实字段），
  负责 shape 头 + 扁平数据 buffer 的序列化布局（含 _ALIGN 字节对齐填充）；
  **loss_mask 不打包**——消费者在 ``colocated_backbone_get_batch`` 里从 labels
  （含 IGNORE/pad 掩码）本地重建（2026-08-13 用户确认）；
- ``BackwardPacket``：单张量（image_embeddings 梯度），与 ForwardPacket 对称。
类负责**布局**（serialize/parse_shape_header/deserialize），本类只做**通信原语**
（isend/irecv/wait/buffer 管理）——前向包先发一次 shape 头、再发一个连续扁平 buffer
（共 2 次 P2P，避免逐字段多次发送的延迟开销）；反向包同样 shape 头 + 梯度 2 次 P2P。

**重要**：本类使用**独立的共置边界通信组**（``get_colocated_boundary_group()``）——
成员与 pp_group / enc_inner_dp 组相同但是独立 NCCL 实例，不复用它们，避免与
backbone 1F1B 的 P2P 在同一组上排队（stream 串行）以及潜在的交叉死锁。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig

# shape 头每字段长度 = 1(ndim) + 3(最多 3 维，0 填充)。字段最多 3 维：
# image_embeddings [seq, num_tiles, h] 是 3D，文本字段都是 1D，无需预留 4D。
# Per-field shape-header length = 1 (ndim) + 3 (max dims, zero padded). Fields are
# at most 3D (image_embeddings [seq, num_tiles, h]); text fields are 1D.
_SHAPE_HEADER_LEN = 4
# 前向包字段数 = 5 个内容字段 + microbatch id 字段（第 6 个真实字段，与 num_image_tiles
# 同类：小张量，走同一套 header 行 / 数据 buffer / deserialize 机制）。
# Number of forward-packet fields = 5 content fields + the microbatch id field (a real
# 6th field of the same kind as num_image_tiles: a small tensor going through the same
# shape-header-row / flat-buffer / deserialize machinery).
_NUM_FORWARD_FIELDS = 5


def _dtype_itemsize(dtype: torch.dtype) -> int:
    """Element size of a dtype (bytes per element).

    一个 dtype 的元素字节数（用于计算扁平 buffer 的切分偏移）。
    """
    return torch.tensor([], dtype=dtype).element_size()


def _numel(shape) -> int:
    """Number of elements of a shape (empty shape -> 1).

    一个 shape 的元素数（空 shape 视为 1）。
    """
    n = 1
    for d in shape:
        n *= d
    return n


# Byte alignment for the flat forward-packet buffer. view(dtype) on a uint8 slice
# requires the storage offset to be divisible by the element size (max 8 for
# int64), so both sides pad every field's byte length to a multiple of _ALIGN.
# 扁平前向包的字节对齐：对 uint8 切片做 view(dtype) 要求存储偏移能被元素大小整除
# （最大 int64 = 8），因此收发两端都把每字段的字节长度填充到 _ALIGN 的整数倍。
_ALIGN = 8


def _padded_bytes(num_bytes: int, align: int = _ALIGN) -> int:
    """Round a byte count up to a multiple of ``align``.

    把字节数向上取整到 align 的整数倍（flat buffer 的字段对齐填充用，收发两端
    用同一规则计算，保证布局一致）。
    """
    return (num_bytes + align - 1) // align * align


@dataclass
class ForwardPacket:
    """Canonical forward packet from an encoder producer to the backbone consumer.

    共置训练边界的**前向数据包**：一个 encoder producer 的完整输出——image_embeddings
    （encoder 输出，浮点）+ 本地文本数据 tokens/labels/num_image_tiles（**loss_mask 不
    打包**：消费者从 labels 的 IGNORE/pad 掩码本地重建，2026-08-13 用户确认）。
    类负责数据包的**序列化布局**（shape 头 + 扁平数据 buffer 的构造与解析、字节对齐
    填充）；通信原语（isend/irecv/wait）留在通信器
    （``EncoderBackboneBoundaryCommunicator``）。

    发送：``serialize()`` 返回 ``(shape 头, 扁平 uint8 数据)``，通信器
    把两者各发一次（共 2 次 P2P）。接收：先收固定大小 shape 头（**无需提前知道
    形状**），``parse_shape_header`` 从 header 解析各字段 shape 并算出总字节数
    （据此动态分配数据 buffer），收完数据后 ``deserialize`` 把扁平 buffer 还原为各
    字段张量。
    """

    image_embeddings: torch.Tensor
    tokens: torch.Tensor
    labels: torch.Tensor
    num_image_tiles: torch.Tensor
    # Microbatch id this packet belongs to — a real 5th field (1-element int64 tensor,
    # same kind as num_image_tiles): it rides through the same header row + flat buffer
    # + deserialize machinery. The business layer builds the packet without it (it does
    # not know the id); the schedule stamps it right after the encoder branch returns,
    # and serialize() asserts it is stamped before any send.
    # 本包所属的 microbatch id——第 5 个真实字段（1 元素 int64 张量，与 num_image_tiles
    # 同类）：走同一套 header 行 + 数据 buffer + deserialize。业务层构造包时没有它
    # （业务层不知道 id），schedule 在 encoder 分支返回后立即打标，serialize() 断言
    # 发送前必已打标。
    microbatch_id: Optional[torch.Tensor] = None

    # Per-field fixed dtypes; None = the float dtype passed at (de)serialize time
    # (image_embeddings uses the communicator dtype, e.g. bf16).
    # 各字段固定 dtype；None = 序列化/反序列化时传入的 float dtype（image_embeddings
    # 用通信器 dtype，如 bf16）。
    _FIELD_DTYPES = [
        None,  # image_embeddings
        torch.int64,  # tokens
        torch.int64,  # labels
        torch.int32,  # num_image_tiles
        torch.int64,  # microbatch_id
    ]

    # Field names in canonical order — the single source of truth (matches
    # to_dict() keys, the fields property and the shape-header row order); other
    # modules/tests import this instead of re-defining their own key tuples.
    # 字段名规范顺序——单一来源（与 to_dict() 的 key、fields 属性、shape 头行序一致）；
    # 其他模块/测试统一从这里取，不再各自定义 key 元组。
    field_names = (
        "image_embeddings",
        "tokens",
        "labels",
        "num_image_tiles",
        "microbatch_id",
    )

    @property
    def fields(self) -> List[torch.Tensor]:
        """The fields in canonical order (matches the shape-header row order).
        各字段按规范顺序返回（与 shape 头行序一致，含 microbatch id）。
        """
        return [
            self.image_embeddings,
            self.tokens,
            self.labels,
            self.num_image_tiles,
            self.microbatch_id,
        ]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Dict view of the fields (local-path use / key checks).

        各字段的字典视图（producer 0 本地直传路径 / 组装 backbone 输入等需要按键访问
        字段的场景用），与字段列表一一对应。
        """
        return {
            "image_embeddings": self.image_embeddings,
            "tokens": self.tokens,
            "labels": self.labels,
            "num_image_tiles": self.num_image_tiles,
            "microbatch_id": self.microbatch_id,
        }

    def field_dtypes(self, float_dtype: torch.dtype) -> List[torch.dtype]:
        """Per-field dtypes, resolving None (image_embeddings) to ``float_dtype``.
        各字段 dtype（image_embeddings 用传入的 float dtype，其余固定）。
        """
        return [float_dtype if t is None else t for t in self._FIELD_DTYPES]

    def serialize(self, align: int = _ALIGN) -> Tuple[torch.Tensor, torch.Tensor]:
        """Serialize into (shape header, flat uint8 buffer); each sent in one P2P call.

        序列化为 ``(shape 头, 扁平 uint8 buffer)``——各一次 P2P 发送：
        - shape 头：``[_NUM_FORWARD_FIELDS, _SHAPE_HEADER_LEN=4]`` int64。header 就是
          一行一行的列表，每个字段一行 ``[ndim, d0, d1, d2]``（不足 3 维用 0 填充），
          **microbatch id 是第 5 个真实字段**（1 元素 int64 张量，行 ``[1, 1, 0, 0]``），
          与其余字段完全同构——没有特殊行、没有额外 append；
        - 扁平数据：各字段按自身 dtype 展平为 uint8 字节，每字段字节长度填充到
          ``align``（8）的整数倍后拼接——保证接收端任意切片的 ``view(dtype)`` 合法
          （存储偏移能被元素大小整除，最大 int64 = 8）。
        收发两端用同一填充规则（``_padded_bytes``），布局一致。
        """
        # 网络发送前 microbatch id 必已由 schedule 打标（业务层构造时不知 id）。
        assert self.microbatch_id is not None, (
            "microbatch_id must be stamped by the schedule before the packet is sent"
        )
        for f in self.fields:
            assert f.ndim <= _SHAPE_HEADER_LEN - 1, f"unsupported ndim {f.ndim}"
        device = self.image_embeddings.device
        # shape 头 = 各字段行（一个列表，所有字段统一构造，无特殊处理）。
        header_rows = [
            [f.ndim] + list(f.shape) + [0] * (_SHAPE_HEADER_LEN - 1 - f.ndim)
            for f in self.fields
        ]
        header = torch.tensor(header_rows, dtype=torch.int64, device=device)
        parts = []
        for f in self.fields:
            b = f.contiguous().view(torch.uint8).reshape(-1)
            pad = (-b.numel()) % align
            if pad:
                b = torch.cat([b, torch.zeros(pad, dtype=torch.uint8, device=device)])
            parts.append(b)
        flat = torch.cat(parts)
        return header, flat

    @staticmethod
    def parse_shape_header(
        header: torch.Tensor, float_dtype: torch.dtype
    ) -> Tuple[List[Tuple[int, ...]], int]:
        """Parse the shape header into (per-field shapes, total padded flat-buffer bytes).

        从已收到的 shape 头解析各字段 shape，并按"每字段字节长度填充到 _ALIGN 整数
        倍"算出扁平数据的总字节数——接收方据此**动态分配数据 buffer**（无需提前知道
        包的形状）。
        """
        dtypes = [float_dtype if t is None else t for t in ForwardPacket._FIELD_DTYPES]
        shapes = []
        total_bytes = 0
        for i in range(_NUM_FORWARD_FIELDS):
            ndim = header[i, 0].item()
            shape = tuple(header[i, 1 : 1 + ndim].tolist())
            shapes.append(shape)
            total_bytes += _padded_bytes(_numel(shape) * _dtype_itemsize(dtypes[i]))
        return shapes, total_bytes

    @staticmethod
    def deserialize(
        header: torch.Tensor, flat: torch.Tensor, float_dtype: torch.dtype
    ) -> "ForwardPacket":
        """Rebuild a ForwardPacket from a received shape header + flat data buffer.

        按 shape 头解析的字段偏移切分扁平 buffer：偏移按"填充后长度"推进（保证每
        字段起始偏移为 _ALIGN 整数倍，``view(dtype)`` 合法），切片取该字段的原始
        字节数（填充字节不进入还原的张量），再 ``view(dtype).reshape(shape)`` 还原
        各字段。microbatch id 是第 5 个字段（1 元素 int64），与其余字段一同切出。
        """
        shapes, _ = ForwardPacket.parse_shape_header(header, float_dtype)
        dtypes = [float_dtype if t is None else t for t in ForwardPacket._FIELD_DTYPES]
        fields = []
        offset = 0
        for shape, dtype in zip(shapes, dtypes):
            field_bytes = _numel(shape) * _dtype_itemsize(dtype)
            fields.append(flat[offset : offset + field_bytes].view(dtype).reshape(shape))
            offset += _padded_bytes(field_bytes)
        # 5 个字段按规范顺序切出（第 5 个即 microbatch id 张量），与 dataclass 字段顺序
        # 一致，直接整体构造。
        return ForwardPacket(*fields)


@dataclass
class BackwardPacket:
    """Backward grad packet from the consumer back to one encoder producer.

    共置训练边界的**反向梯度包**：消费者把某个 encoder producer 的 image_embeddings
    梯度（单张量）发回该 producer，与 ``ForwardPacket`` 对称。类负责序列化布局
    （shape 头 + 梯度张量）；通信原语留在通信器。梯度没有多字段拼接，因此无对齐
    填充——header 是单个 ``[_SHAPE_HEADER_LEN]`` int64 行 ``[ndim, d0, d1, d2]``，
    数据即 ``grad.contiguous()`` 本身。
    """

    grad: torch.Tensor

    def serialize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Serialize into (shape header, contiguous grad tensor); one P2P call each.

        序列化为 ``(shape 头, 连续梯度张量)``——各一次 P2P 发送。shape 头布局：
        ``header[0] = ndim``；``header[1:1+ndim] = 各维大小``（不足 3 维用 0 填充）。
        梯度 dtype 固定为通信器 dtype（``config.pipeline_dtype``，如 bf16），不随
        header 传（字段固定约定）。
        """
        grad = self.grad
        assert grad.ndim <= _SHAPE_HEADER_LEN - 1, f"unsupported ndim {grad.ndim}"
        shape = list(grad.shape)
        header = torch.tensor(
            [grad.ndim] + shape + [0] * (_SHAPE_HEADER_LEN - 1 - grad.ndim),
            dtype=torch.int64,
            device=grad.device,
        )
        return header, grad.contiguous()

    @staticmethod
    def parse_shape_header(header: torch.Tensor) -> Tuple[int, ...]:
        """Parse the grad shape header into the grad shape (ndim + dims).
        从已收到的 shape 头解析梯度形状（维度数 + 各维大小）。
        """
        ndim = header[0].item()
        return tuple(header[1 : 1 + ndim].tolist())

    @staticmethod
    def deserialize(header: torch.Tensor, buf: torch.Tensor) -> "BackwardPacket":
        """Wrap a received grad buffer, checking its shape against the header.
        包装已收到的梯度 buffer 为 BackwardPacket（校验形状与 header 一致）。
        """
        shape = BackwardPacket.parse_shape_header(header)
        assert tuple(buf.shape) == shape, f"grad shape {tuple(buf.shape)} != header {shape}"
        return BackwardPacket(grad=buf)


class _ForwardRecvRequest:
    """In-flight async receive of a forward packet (shape header already submitted).

    异步前向接收请求：构造时已提交 shape 头的 irecv（未等待），数据尚未接收。
    **两段式生命周期（start/finish）**：
    - ``start()``：等 shape 头 → 解析 → 分配数据 buffer → **异步提交数据 irecv**
      （不等待，拿到数据 handle）——放在"数据被需要之前"（consumer prefetch 时 /
      producer 补发 step 时），让数据传输与后续计算重叠；
    - ``finish()``：等数据 handle（start 之后已后台传输一整步，通常早已完成）→
      组装 ``ForwardPacket``——在真正需要数据时调用（consumer take / producer phase ④）。
    """

    def __init__(
        self,
        comm: "EncoderBackboneBoundaryCommunicator",
        header_handle,
        header: torch.Tensor,
        src_rank: int,
        producer: int,
        expected_microbatch_id: Optional[int],
    ):
        self._comm = comm
        self._header_handle = header_handle
        self._header = header
        self._src_rank = src_rank
        self.producer = producer
        self.expected_microbatch_id = expected_microbatch_id
        self._flat = None  # data buffer allocated by start(); read by finish()
        self._data_handle = None  # data irecv handle posted by start()
        self._packet = None

    def start(self) -> "_ForwardRecvRequest":
        """Start phase: wait the shape header, allocate the data buffer, post the data irecv.

        **开始阶段**：等 shape 头（prefetch 时提交、此刻几乎已到达）→ 解析 shape →
        分配数据 buffer → 异步提交数据 irecv（不等待）。返回 self 便于链式调用。
        """
        if self._data_handle is None:
            self._data_handle = self._comm._start_recv_forward(self)
        return self

    def finish(self) -> "ForwardPacket":
        """Finish phase: wait the data and assemble the ForwardPacket.

        **结束阶段**：等数据 handle（通常早已完成）→ 组装 ``ForwardPacket``（结果缓存，
        重复调用不重复收）。
        """
        if self._packet is None:
            self._packet = self._comm._finish_recv_forward(self)
        return self._packet


class _GradRecvRequest:
    """In-flight async receive of the backward grad (shape header already submitted).

    异步梯度接收请求：构造时已提交 shape 头的 irecv（未等待），数据尚未接收。
    **两段式生命周期（start/finish）**：``start()`` 等 shape 头 → 分配梯度 buffer →
    异步提交数据 irecv（producer 补发 step 的 forward 前做，与计算重叠）；``finish()``
    等数据 handle → 组装 ``BackwardPacket``（phase ④ 统一反传前做）。
    """

    def __init__(
        self,
        comm: "EncoderBackboneBoundaryCommunicator",
        header_handle,
        header: torch.Tensor,
        src_rank: int,
    ):
        self._comm = comm
        self._header_handle = header_handle
        self._header = header
        self._src_rank = src_rank
        self._buf = None  # grad buffer allocated by start(); read by finish()
        self._data_handle = None  # data irecv handle posted by start()
        self._packet = None

    def start(self) -> "_GradRecvRequest":
        """Start phase: wait the shape header, allocate the grad buffer, post the data irecv.

        **开始阶段**：等 shape 头 → 分配梯度 buffer → 异步提交数据 irecv（不等待）。
        返回 self 便于链式调用。
        """
        if self._data_handle is None:
            self._data_handle = self._comm._start_recv_grad(self)
        return self

    def finish(self) -> "BackwardPacket":
        """Finish phase: wait the data and assemble the BackwardPacket.

        **结束阶段**：等数据 handle → 组装 ``BackwardPacket``（结果缓存，重复调用
        不重复收）。
        """
        if self._packet is None:
            self._packet = self._comm._finish_recv_grad(self)
        return self._packet


class EncoderBackboneBoundaryCommunicator:
    """Boundary communicator between encoder producers and the backbone consumer.

    同一副本内 encoder 生产者（producer id > 0）与 backbone 消费者（producer id 0，
    即 backbone entry）之间的边界通信器，**四个公开收发方法只服务 producer > 0 的
    网络路径**：
    - 前向（``colocated_send_forward`` / ``colocated_recv_forward``）：生产者把
      encoder 输出 + 本地文本数据包（``ForwardPacket``）发给消费者（仅 producer > 0）；
    - 反向（``colocated_send_backward`` / ``colocated_recv_backward``）：消费者把
      encoder 输出梯度（``BackwardPacket``）发回各生产者（仅 producer > 0；文本字段
      的梯度不回传）。
    收发均有 ``wait`` 参数：``wait=True``（默认）同步阻塞；``wait=False`` 异步提交
    （send 返回 handle 列表，recv 返回 request 对象，稍后 ``wait()`` / ``start()`` /
    ``finish()``）
    ——producer replenish / consumer prefetch / 补发 step 收梯度用异步路径与计算重叠。
    前向包 shape 头含 **microbatch id** 行（消费者按序 take 时校验，防乱序错配）。
    **本地直传短路（producer 0 = 消费者自己）由调用方（wrapper）分支处理**：producer 0
    的包/梯度不经网络、零拷贝本地引用——wrapper 跳过本类方法直接使用本地 buffer
    （包就是 ``ForwardPacket``，dataclass 直接构造即可），不发起任何 P2P，因此网络侧
    配对计数不受影响。
    """

    def __init__(
        self,
        colocated_boundary_group,
        config: ModelParallelConfig,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize with the dedicated colocated boundary group.

        用独立的共置边界通信组初始化（``get_colocated_boundary_group()``：成员与
        pp_group / enc_inner_dp 组相同但是独立 NCCL 实例）。

        Args:
            colocated_boundary_group: the dedicated colocated boundary group
                (producer order); must come from ``get_colocated_boundary_group()``,
                NOT the pipeline-parallel group nor the encoder inner dp group.
                必须传 get_colocated_boundary_group() 的独立边界组。
            config: model parallel config (``pipeline_dtype`` is used for floats).
            dtype: optional override of the float dtype (image_embeddings/grad).
        """
        self.colocated_boundary_group = colocated_boundary_group
        self.config = config
        self.dtype = dtype if dtype is not None else config.pipeline_dtype

        self.world_rank = dist.get_rank()
        # Members of the replica in producer order; producer id 0 is the consumer
        # (the backbone entry, i.e. the backbone first stage).
        # 副本内成员（按生产者编号）；0 号生产者即消费者（backbone entry）。
        self.group_ranks = dist.get_process_group_ranks(colocated_boundary_group)
        self.producer_id = dist.get_group_rank(colocated_boundary_group, self.world_rank)
        self.group_size = len(self.group_ranks)
        self.consumer_global_rank = self.group_ranks[0]

    def is_consumer(self) -> bool:
        """Whether this rank is the consumer (the forward-packet receiver).

        本 rank 是否为消费者（前向数据包接收方，即 backbone entry）。
        """
        return self.producer_id == 0

    # ------------------------------------------------------------------
    # Internal helpers.
    # 内部辅助。
    # ------------------------------------------------------------------

    def _send_backward_packet(
        self, packet: BackwardPacket, dst_rank: int, wait: bool = True
    ) -> Optional[list]:
        """Send a BackwardPacket: shape header, then the grad tensor (2 P2P calls).

        发送反向梯度包（消费者→生产者，反传专用）：先发 shape 头再发梯度数据（共 2
        次 P2P），布局由 ``BackwardPacket.serialize`` 给出（见类 docstring）。

        ``wait=True``（默认）：阻塞到两次 isend 完成，返回 None。
        ``wait=False``：提交后立即返回两个 handle（consumer 反传 hook 用，
        wait 由调用方统一做）。
        """
        header, grad = packet.serialize()
        handles = [
            dist.isend(header, dst=dst_rank, group=self.colocated_boundary_group),
            dist.isend(grad, dst=dst_rank, group=self.colocated_boundary_group),
        ]
        if wait:
            for handle in handles:
                handle.wait()
            return None
        return handles

    def _recv_backward_packet(self, src_rank: int) -> BackwardPacket:
        """Receive a BackwardPacket synchronously: shape header, then the grad.

        同步接收反向梯度包（生产者侧，反传专用）：先收 shape 头确定形状，再按形状
        分配 GPU buffer 接收数据（共 2 次 P2P），经 ``BackwardPacket.deserialize`` 包装
        （校验形状与 header 一致）。返回的梯度形状与 image_embeddings 相同
        （[img_seq, num_tiles, h_lang]），dtype 为通信器 dtype。
        """
        header = torch.empty(_SHAPE_HEADER_LEN, dtype=torch.int64, device="cuda")
        req = dist.irecv(header, src=src_rank, group=self.colocated_boundary_group)
        req.wait()
        shape = BackwardPacket.parse_shape_header(header)
        buf = torch.empty(shape, dtype=self.dtype, device="cuda")
        req = dist.irecv(buf, src=src_rank, group=self.colocated_boundary_group)
        req.wait()
        return BackwardPacket.deserialize(header, buf)

    def _async_recv_grad(self, src_rank: int) -> _GradRecvRequest:
        """Asynchronously submit a receive of the backward grad: only the shape-header irecv.

        异步发起梯度接收：只提交 shape 头的 irecv（不等待），返回
        ``_GradRecvRequest``。生产者侧用它把接收提前挂到 NCCL 后台；需要梯度时先
        ``request.start()`` 再 ``request.finish()``。梯度 dtype 在完成时用通信器 dtype
        解析。
        """
        header = torch.empty(_SHAPE_HEADER_LEN, dtype=torch.int64, device="cuda")
        header_handle = dist.irecv(header, src=src_rank, group=self.colocated_boundary_group)
        return _GradRecvRequest(self, header_handle, header, src_rank)

    def _start_recv_grad(self, request: _GradRecvRequest) -> object:
        """Start phase of an async grad receive: wait header, allocate, post data irecv.

        梯度异步接收的**开始阶段**：等 shape 头 → 解析形状 → 分配梯度 buffer → **异步
        提交数据 irecv（不等待）**，返回数据 handle。数据在 start 到 finish 之间的计算
        窗口后台传输。
        """
        request._header_handle.wait()
        shape = BackwardPacket.parse_shape_header(request._header)
        request._buf = torch.empty(shape, dtype=self.dtype, device="cuda")
        return dist.irecv(
            request._buf, src=request._src_rank, group=self.colocated_boundary_group
        )

    def _finish_recv_grad(self, request: _GradRecvRequest) -> BackwardPacket:
        """Finish phase of an async grad receive: wait the data, assemble the packet.

        梯度异步接收的**结束阶段**：等数据 handle（通常早已完成）→ 经
        ``BackwardPacket.deserialize`` 组装（校验形状与 header 一致）。
        """
        request._data_handle.wait()
        return BackwardPacket.deserialize(request._header, request._buf)

    def _send_forward_packet(
        self, packet: ForwardPacket, dst_rank: int, wait: bool = True
    ) -> Optional[list]:
        """Send a ForwardPacket as (shape header, ONE flat buffer) — 2 P2P calls total.

        把前向包经 ``ForwardPacket.serialize`` 拼成 (shape 头, 一个扁平 buffer) 发送
        （共 2 次 P2P，布局与字段顺序见 ``ForwardPacket`` 类 docstring）：
        ① shape 头：``[_NUM_FORWARD_FIELDS, _SHAPE_HEADER_LEN=4]``（即 [5,4]）int64，
        每行一个字段，
        microbatch id 是其中一个字段行（消费者 take 时校验，来自包自身属性）；
        ② 扁平数据：各字段按自身 dtype 展平为 uint8 字节、字节长度填充到 _ALIGN 的
        整数倍后拼接（对齐保证接收端任意切片 ``view(dtype)`` 合法）。

        ``wait=True``（默认）：阻塞到两次 isend 完成，返回 None。
        ``wait=False``：提交后立即返回两个 handle（producer replenish 用，
        wait 由调用方在 phase ② 后统一做）。
        """
        header, flat = packet.serialize() #需要序列化因为dist通信只支持tensor数据，不支持类型对象
        handles = [
            dist.isend(header, dst=dst_rank, group=self.colocated_boundary_group),
            dist.isend(flat, dst=dst_rank, group=self.colocated_boundary_group),
        ]
        if wait:
            for handle in handles:
                handle.wait()
            return None
        return handles

    def _receive_forward_flat(
        self, header: torch.Tensor, src_rank: int, float_dtype: torch.dtype
    ) -> ForwardPacket:
        """Receive the flat data buffer after the shape header and rebuild the packet.

        shape 头已到达（前 _NUM_FORWARD_FIELDS 行是字段 shape），据此算出总字节数收
        扁平数据 buffer，再 ``ForwardPacket.deserialize`` 还原为 ForwardPacket（字段
        切分 + view(dtype) + reshape，偏移按"填充后长度"推进）。
        """
        _, total_bytes = ForwardPacket.parse_shape_header(header, float_dtype)
        flat = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
        req = dist.irecv(flat, src=src_rank, group=self.colocated_boundary_group)
        req.wait()
        return ForwardPacket.deserialize(header, flat, float_dtype)

    def _check_microbatch_id(
        self, packet: ForwardPacket, expected_microbatch_id: Optional[int]
    ) -> None:
        """Validate a received packet's microbatch id against the expected one.

        校验收到的包自带的 microbatch id（1 元素 int64 张量）与期望一致（消费者按序
        take 时用，防乱序错配）。通信两端是确定的，这里不是验证乱序达到的问题，只是在做校验id
        """
        if expected_microbatch_id is not None:
            packet_mb_id = packet.microbatch_id
            assert packet_mb_id is not None and packet_mb_id.item() == expected_microbatch_id, (
                f"out-of-order forward packet: packet microbatch id {packet_mb_id} "
                f"!= expected {expected_microbatch_id}"
            )

    def _recv_forward_packet(
        self, src_rank: int, expected_microbatch_id: Optional[int] = None
    ) -> ForwardPacket:
        """Receive a ForwardPacket synchronously: shape header, then flat buffer.

        同步收前向包（与 ``_send_forward_packet`` 对称，共 2 次 P2P）：先收 shape 头，
        再收扁平数据并还原为 ForwardPacket，校验包自带的 microbatch id。
        """
        header = torch.empty(
            (_NUM_FORWARD_FIELDS, _SHAPE_HEADER_LEN), dtype=torch.int64, device="cuda"
        )
        req = dist.irecv(header, src=src_rank, group=self.colocated_boundary_group)
        req.wait()
        packet = self._receive_forward_flat(header, src_rank, self.dtype)
        self._check_microbatch_id(packet, expected_microbatch_id)
        return packet

    def _async_recv_forward(
        self, src_rank: int, producer: int, expected_microbatch_id: Optional[int] = None
    ) -> "_ForwardRecvRequest":
        """Asynchronously submit a receive of a forward packet: only the shape-header irecv.

        异步发起前向包接收：只提交 shape 头的 irecv（不等待），返回
        ``_ForwardRecvRequest``。消费者在 prefetch 时用它把接收提前挂到 NCCL 后台；
        需要数据时先 ``request.start()``（等 shape → 分配 → 提交数据接收）再
        ``request.finish()``（取数据）。
        """
        header = torch.empty(
            (_NUM_FORWARD_FIELDS, _SHAPE_HEADER_LEN), dtype=torch.int64, device="cuda"
        )
        header_handle = dist.irecv(
            header, src=src_rank, group=self.colocated_boundary_group
        )
        return _ForwardRecvRequest(
            self, header_handle, header, src_rank, producer, expected_microbatch_id
        )

    def _start_recv_forward(self, request: "_ForwardRecvRequest") -> object:
        """Start phase of an async forward receive: wait header, allocate, post data irecv.

        前向异步接收的**开始阶段**：等 shape 头（prefetch 时已提交、此刻几乎已到达）→
        解析 shape 算出总字节数 → 分配数据 buffer → **异步提交数据 irecv（不等待）**，
        返回数据 handle。数据在 start 到 finish 之间的计算窗口后台传输。
        """
        request._header_handle.wait()
        _, total_bytes = ForwardPacket.parse_shape_header(request._header, self.dtype)
        request._flat = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
        return dist.irecv(
            request._flat, src=request._src_rank, group=self.colocated_boundary_group
        )

    def _finish_recv_forward(self, request: "_ForwardRecvRequest") -> ForwardPacket:
        """Finish phase of an async forward receive: wait the data, assemble the packet.

        前向异步接收的**结束阶段**：等数据 handle（通常早已完成）→ 还原为
        ``ForwardPacket`` 并校验其 microbatch id。
        """
        request._data_handle.wait()
        packet = ForwardPacket.deserialize(request._header, request._flat, self.dtype)
        self._check_microbatch_id(packet, request.expected_microbatch_id)
        return packet

    # ------------------------------------------------------------------
    # Forward: encoder output + text packet from a producer to the consumer.
    # 前向：encoder 输出 + 文本数据包从生产者发往消费者。
    # ------------------------------------------------------------------

    def colocated_send_forward(
        self,
        packet: ForwardPacket,
        producer: Optional[int] = None,
        wait: bool = True,
    ) -> Optional[list]:
        """Producer side (producer id > 0): send the forward packet to the consumer.

        生产者（producer id>0）：把本生产者的 ``ForwardPacket``（encoder 输出 + 本地
        文本数据 tokens/labels/num_image_tiles）发给消费者。microbatch id 是
        包的属性，``serialize`` 写入 shape 头（消费者 take 时校验）。
        ``producer`` 默认取本 rank 的 producer id；``wait=False`` 时提交后立即返回
        handle 列表（producer replenish 用，wait 由调用方统一做）。
        """
        producer = self.producer_id if producer is None else producer
        assert producer > 0, (
            f"producer 0 (the consumer) never sends the forward packet: the wrapper "
            f"uses the local packet directly (ForwardPacket) and skips this call; "
            f"got producer={producer}"
        )
        return self._send_forward_packet(packet, self.consumer_global_rank, wait=wait)

    def colocated_recv_forward(
        self,
        producer: int,
        expected_microbatch_id: Optional[int] = None,
        wait: bool = True,
    ) -> Union[ForwardPacket, _ForwardRecvRequest]:
        """Consumer side: receive the forward packet of producer ``producer``.

        消费者：接收生产者 ``producer``（>0）的完整前向数据包（``ForwardPacket``：
        encoder 输出 + 文本字段）。``wait=True``（默认）同步阻塞返回 ``ForwardPacket``；
        ``wait=False`` 异步提交（只挂 shape 头 irecv），返回 ``_ForwardRecvRequest``
        （consumer prefetch 用，需要数据时先 ``request.start()`` 再 ``request.finish()``）。
        ``expected_microbatch_id`` 与包自带的 microbatch id 校验（按序 take 时用）。
        **producer 0（消费者自己）的包由调用方在本地直接组装（直接构造
        ``ForwardPacket``），
        不走本函数**。
        """
        assert self.is_consumer(), "only the consumer receives the forward packet"
        assert 0 < producer < self.group_size, (
            f"producer 0's packet is local: the caller (the consumer itself) assembles "
            f"it from its own buffer (ForwardPacket), never via this network "
            f"receive; got producer={producer}"
        )
        if wait:
            return self._recv_forward_packet(
                self.group_ranks[producer], expected_microbatch_id
            )
        return self._async_recv_forward(
            self.group_ranks[producer], producer, expected_microbatch_id
        )

    # ------------------------------------------------------------------
    # Backward: encoder-output grad from the consumer back to a producer.
    # 反向：encoder 输出梯度从消费者发回各生产者（仅梯度一个张量）。
    # ------------------------------------------------------------------

    def colocated_send_backward(
        self, packet: BackwardPacket, producer: int, wait: bool = True
    ) -> Optional[list]:
        """Consumer side: send the encoder-output grad back to ``producer``.

        消费者：把生产者 ``producer``（仅 producer > 0）的 encoder 输出梯度
        （``BackwardPacket``，单张量）发回它。反向只需要这一个梯度张量（文本字段的
        梯度在消费者本地参与 backbone 反传，不涉及 encoder 参数，不回传）。
        ``wait=True``（默认）同步阻塞；``wait=False`` 异步提交返回 handle 列表
        （consumer 反传 hook 用，wait 由调用方统一做）。
        **producer 0（消费者自己）由调用方（wrapper）在调用处分支短路**：其梯度直接
        本地保留（反传时就已在本 rank），不调用本方法——不发起任何 P2P，网络侧配对
        计数不受影响。
        """
        assert self.is_consumer(), "only the consumer sends the backward grad"
        assert 0 < producer < self.group_size, (
            f"producer 0's grad is local: the caller (the consumer itself) keeps it "
            f"for the unified encoder backward, never sends it over the network; "
            f"got producer={producer}"
        )
        return self._send_backward_packet(packet, self.group_ranks[producer], wait=wait)

    def colocated_recv_backward(
        self, producer: Optional[int] = None, wait: bool = True
    ) -> Union[BackwardPacket, _GradRecvRequest]:
        """Producer side (producer id > 0): receive the encoder-output grad.

        生产者（producer id>0）：从消费者接收本生产者的 encoder 输出梯度
        （``BackwardPacket``，与 colocated_send_forward 对应）。``producer`` 默认取本
        rank 的 producer id。``wait=True``（默认）同步阻塞返回 ``BackwardPacket``；
        ``wait=False`` 异步提交返回 ``_GradRecvRequest``（producer 补发 step 的
        forward 前用，需要梯度时先 ``request.start()`` 再 ``request.finish()``）。
        **producer 0（消费者自己）由调用方（wrapper）在调用处分支短路**：其梯度直接
        取本地保留的（反传时已在本 rank），不调用本方法——不发起任何 P2P，网络侧
        配对计数不受影响。
        """
        producer = self.producer_id if producer is None else producer
        assert producer > 0, (
            f"producer 0 (the consumer) never receives the backward grad over the "
            f"network: the wrapper keeps the local grad and skips this call; "
            f"got producer={producer}"
        )
        if wait:
            return self._recv_backward_packet(self.consumer_global_rank)
        return self._async_recv_grad(self.consumer_global_rank)
