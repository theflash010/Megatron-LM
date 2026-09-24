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
  负责**定长扁平数据 buffer** 的序列化布局（含 _ALIGN 字节对齐填充，HEADER-FREE）；
  **loss_mask 不打包**——消费者在 ``colocated_backbone_get_batch`` 里从 labels
  （含 IGNORE/pad 掩码）本地重建（2026-08-13 用户确认）；
- ``BackwardPacket``：单张量（image_embeddings 梯度），与 ForwardPacket 对称。
类负责**布局**（serialize/deserialize），本类只做**通信原语**
（isend/irecv/wait/buffer 管理）。

**dual-channel-p2p（2026-09-22）：HEADER-FREE / 定长协议**——每个样本 shape 由 config
静态固定（image_seq_length/micro_batch_size/hidden_size/text_seq_length 全部在通信器构造时从 ``get_args()`` 求出），
所以**去掉 shape 头**：前向包只发**一个定长扁平 buffer**（1 次 P2P），反向包只发**一个
定长梯度张量**（1 次 P2P）。接收端按构造期算好的定长 layout 直接 ``irecv`` 一块定长
buffer，**立即返回、真正异步**。这修掉了此前"先收 shape 头 → ``parse_shape_header``
（``.item()/.tolist()`` 触发 CUDA 同步）"导致的阻塞式接收：那次同步会卡住 schedule
循环、把"异步"recv 变成同步等待并引发死锁。定长后收方无需等待/解析头即可 post
单个 irecv。
HEADER-FREE fixed-length protocol: every sample's shape is statically fixed by config,
so the shape header is removed. Forward sends ONE fixed-size flat_buffer buffer (1 P2P) and
backward sends ONE fixed-shape grad (1 P2P); the receiver posts a single fixed-size
irecv that returns immediately (truly async), fixing the deadlock where the old
shape-header wait + parse_shape_header (a CUDA sync via .item()/.tolist()) blocked the
schedule loop.

**重要**：本类使用**两个方向隔离的独立共置边界通信组**（dual-channel-p2p Task 2，
2026-09-22）：``get_colocated_boundary_activation_group()``（前向激活 producer→consumer）
与 ``get_colocated_boundary_grad_group()``（反向梯度 consumer→producer）——成员与
pp_group / enc_inner_dp 组相同但各为独立 NCCL 实例、各自内部 NCCL stream，不复用它们，
避免与 backbone 1F1B 的 P2P 同组串行，也让边界收发按方向分流、不交叉。
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.utils import nvtx_decorator

# dual-channel-p2p（2026-09-22）：HEADER-FREE / 定长协议——已删除 ``_SHAPE_HEADER_LEN`` 与
# ``_NUM_FORWARD_FIELDS``。原本它们服务于"每字段一行 [ndim, d0, d1, d2] 的 shape 头"，
# 而 shape 头（及其 parse 时的 ``.item()/.tolist()`` CUDA 同步）正是阻塞式接收 → 死锁的
# 根因。定长协议下每个字段的 shape/dtype/offset 在通信器构造期静态算好（见
# ``EncoderBackboneBoundaryCommunicator.__init__`` 的 layout），收方无需 header 即可 post
# 单个定长 irecv，故这两个常量彻底移除（全仓 grep 确认仅本文件内部引用）。
# dual-channel-p2p (2026-09-22): removed _SHAPE_HEADER_LEN / _NUM_FORWARD_FIELDS. They
# only served the per-field shape-header rows; the header (and its parse-time CUDA sync)
# was the deadlock root cause. The fixed-length protocol derives every field's
# shape/dtype/offset statically at communicator construction, so no header is needed.


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


# Byte alignment for the flat_buffer forward-packet buffer. view(dtype) on a uint8 slice
# requires the storage offset to be divisible by the element size (max 8 for
# int64), so both sides pad every field's byte length to a multiple of _ALIGN.
# 扁平前向包的字节对齐：对 uint8 切片做 view(dtype) 要求存储偏移能被元素大小整除
# （最大 int64 = 8），因此收发两端都把每字段的字节长度填充到 _ALIGN 的整数倍。
_ALIGN = 8


def _padded_bytes(num_bytes: int, align: int = _ALIGN) -> int:
    """Round a byte count up to a multiple of ``align``.

    把字节数向上取整到 align 的整数倍（flat_buffer buffer 的字段对齐填充用，收发两端
    用同一规则计算，保证布局一致）。
    """
    return (num_bytes + align - 1) // align * align


class MergedEncoderBatch(NamedTuple):
    """The merged encoder batch: raw fields, NOT a ForwardPacket (optimization Task 1).

    优化 spec Task 1（合并前传，设计 A）：dataloader 一次取回本 rank 整个 iteration 的
    合并批、encoder 只前传一次。这个中间量**只在本 rank 本地存在、永不上线路**（没有
    ``microbatch_id``、不可序列化），因此刻意**不**包成 ``ForwardPacket``——那会污染
    "packet = 跨边界线红单元"的语义。逐 microbatch 的 ``ForwardPacket`` 只在 schedule
    调 ``ForwardPacket.split_merged_batch`` 拆分并打标之后才诞生。

    Optimization spec Task 1 (design A): the dataloader fetches the rank's whole
    iteration in one merged batch and the encoder forwards it once. This intermediate
    exists only locally and never crosses the wire (no microbatch_id, not serializable),
    so it is deliberately NOT wrapped in a ForwardPacket — that would muddy the
    "packet = boundary wire unit" semantics. Per-microbatch ForwardPackets come into
    being only after the schedule splits via ForwardPacket.split_merged_batch and
    stamps the ids.

    字段布局与 ``ForwardPacket`` 的内容字段一一对应（batch 维是"样本数"：mbs=1 时等于
    microbatch 数）。``image_embeddings`` 保留 grad_fn（phase ④ 对整块做单次 backward）。
    Field layout mirrors ForwardPacket's content fields (the batch dim counts samples;
    with mbs=1 it equals the microbatch count). image_embeddings keeps grad_fn.
    """

    image_embeddings: torch.Tensor  # [img_seq_len, merged_batch, h_lang]（seq-first）
    tokens: torch.Tensor  # [merged_batch, L]
    # dual-channel-p2p（2026-09-22）修正：labels 与 tokens 同形 [merged_batch, L]（NOT L+1）。
    # colocated_train.py ~139 `labels = labels[:, 1:text_length+1]` + ~141 断言
    # tokens.shape == labels.shape，故 labels == tokens == (mbs, text_len)；旧注释 "L + 1" 有误。
    # Corrected: labels is [merged_batch, L], same as tokens (colocated_train.py asserts equal).
    labels: torch.Tensor  # [merged_batch, L]
    num_image_tiles: torch.Tensor  # [merged_batch]


@dataclass
class ForwardPacket:
    """Canonical forward packet from an encoder producer to the backbone consumer.

    共置训练边界的**前向数据包**：一个 encoder producer 的完整输出——image_embeddings
    （encoder 输出，浮点）+ 本地文本数据 tokens/labels/num_image_tiles（**loss_mask 不
    打包**：消费者从 labels 的 IGNORE/pad 掩码本地重建，2026-08-13 用户确认）。
    类负责数据包的**序列化布局**（扁平数据 buffer 的构造与解析、字节对齐填充）；
    通信原语（isend/irecv/wait）留在通信器
    （``EncoderBackboneBoundaryCommunicator``）。

    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：``serialize()`` 只返回**一个扁平
    uint8 buffer**（不再返回 shape 头），通信器一次 P2P 发出。接收端**不收 shape 头、不
    解析**——它按构造期静态算好的定长 layout（每字段 (shape, dtype)）分配一块定长 buffer，
    收完后 ``deserialize(flat_buffer, fields_layout)`` 按 padded offset 切分并
    ``view(dtype).reshape(shape)`` 还原各字段。这消除了旧协议里"先收 header →
    ``parse_shape_header``（``.item()/.tolist()`` CUDA 同步）"的阻塞式接收（死锁根因）。
    serialize() returns ONLY the flat_buffer uint8 buffer (no shape header); the receiver rebuilds
    via the statically-known fixed layout with deserialize.
    """

    image_embeddings: torch.Tensor
    tokens: torch.Tensor
    labels: torch.Tensor
    num_image_tiles: torch.Tensor
    # Microbatch id this packet belongs to — a real 5th field (1-element int64 tensor,
    # same kind as num_image_tiles): it rides through the same header row + flat_buffer buffer
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

    @classmethod
    def split_merged_batch(
        cls,
        image_embeddings: torch.Tensor,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        num_image_tiles: torch.Tensor,
        num_splits: int,
    ) -> List["ForwardPacket"]:
        """Split a MERGED encoder batch into per-microbatch packets along the batch dim.

        优化 spec Task 1（合并前传，设计 A）：dataloader 一次取回本 rank 整个 iteration
        的合并批、encoder 只前传一次，本方法把合并批的**裸张量**等分成逐 microbatch 的
        ``ForwardPacket``。输入刻意是裸张量而非某个"合并 packet"——合并批只在本
        rank 本地存在、永不上线路（没有 microbatch_id、不可序列化），包成 packet 会
        污染"packet = 跨边界线红单元"的语义；``ForwardPacket`` 实例只在本方法返回的
        逐 microbatch 粒度上诞生（调用方随后打 ``microbatch_id``）。切分维度由各字段
        的布局决定：

        - ``image_embeddings`` 是 seq-first ``[img_seq_len, batch, h_lang]``，沿 **dim=1**
          切——切片是 view 且**保留 grad_fn**，整批仍只有一张计算图（phase ④ 对合并
          张量做单次 backward，合并张量由 schedule 持有；从 view 拿不回父张量，这正是
          schedule 要显式持有它的原因）；
        - ``tokens``/``labels`` 是 ``[batch, L]``、``num_image_tiles`` 是 ``[batch]``，
          沿 **dim=0** 切（文本字段无梯度）。

        两条数据假设在这里运行时钉住（配置一变立刻失败，而不是静默切错位）：
        ① 合并批的 batch 维必须能被 ``num_splits`` 整除——每个 split 收
        ``batch_dim // num_splits`` 个样本（= mbs；mbs=1 时 batch 维恰等于 num_splits）；
        ② 每个样本恰好一个 image tile（无 tiling/packing；启用后需改 cumsum 变长切分）。
        切片保持 view、**不**做 ``contiguous()``：通信器序列化时逐字段
        ``contiguous()``（``serialize`` 的扁平化拷贝本来就免不了），本地路径消费 view
        也没有问题；预拷贝只会白白多一份峰值显存。

        Returns:
            ``num_splits`` 个 ``ForwardPacket``，**按 batch 顺序**排列——调用方
            （schedule）用自己的轮盘 microbatch 序列一一对应打 ``microbatch_id``。
        """
        batch_dim = image_embeddings.shape[1]
        assert batch_dim % num_splits == 0, (
            f"merged image_embeddings batch dim ({batch_dim}) must be divisible by "
            f"num_splits ({num_splits}): each microbatch owns "
            f"micro_batch_size = batch_dim / num_splits samples "
            "(dataloader merged batch size = micro_batch_size * num_microbatches / "
            "num_producers)"
        )
        assert (
            tokens.shape[0] == batch_dim
            and labels.shape[0] == batch_dim
            and num_image_tiles.numel() == batch_dim
        ), (
            f"merged text fields batch dims (tokens {tokens.shape}, labels {labels.shape}, "
            f"num_tiles {tuple(num_image_tiles.shape)}) must all be {batch_dim}"
        )
        if bool((num_image_tiles != 1).any().item()):
            raise AssertionError(
                "every sample must own exactly one image tile for the equal-split merge, "
                f"got num_tiles={num_image_tiles.tolist()}; tiling/packing configurations "
                "need a cumsum-based variable-length split instead"
            )

        # torch.chunk 在可整除时给出 num_splits 个等大 view；切片顺序即 batch 顺序。
        image_chunks = torch.chunk(image_embeddings, num_splits, dim=1)
        token_chunks = torch.chunk(tokens, num_splits, dim=0)
        label_chunks = torch.chunk(labels, num_splits, dim=0)
        tile_chunks = torch.chunk(num_image_tiles, num_splits, dim=0)
        return [
            cls(
                image_embeddings=image_chunk,
                tokens=token_chunk,
                labels=label_chunk,
                num_image_tiles=tile_chunk,
                # microbatch_id 留 None：调用方按轮盘序列打标后才能发送。
            )
            for image_chunk, token_chunk, label_chunk, tile_chunk in zip(
                image_chunks, token_chunks, label_chunks, tile_chunks
            )
        ]

    def field_dtypes(self, float_dtype: torch.dtype) -> List[torch.dtype]:
        """Per-field dtypes, resolving None (image_embeddings) to ``float_dtype``.
        各字段 dtype（image_embeddings 用传入的 float dtype，其余固定）。
        """
        return [float_dtype if t is None else t for t in self._FIELD_DTYPES]

    def serialize(self, align: int = _ALIGN) -> torch.Tensor:
        """Serialize into ONE flat_buffer uint8 buffer (HEADER-FREE); sent in one P2P call.

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：只返回**一个扁平 uint8 buffer**
        （不再构造/返回 shape 头）。各字段按自身 dtype 展平为 uint8 字节，每字段字节长度
        填充到 ``align``（8）的整数倍后 ``torch.cat`` 拼接——保证接收端任意切片的
        ``view(dtype)`` 合法（存储偏移能被元素大小整除，最大 int64 = 8）。收发两端用同一
        填充规则（``_padded_bytes``）+ 同一静态定长 layout，布局一致，接收端无需 header。
        Returns ONLY the flat_buffer buffer; the header row construction is removed.
        """
        # 网络发送前 microbatch id 必已由 schedule 打标（业务层构造时不知 id）。
        assert self.microbatch_id is not None, (
            "microbatch_id must be stamped by the schedule before the packet is sent"
        )
        device = self.image_embeddings.device
        parts = []
        for f in self.fields:
            b = f.contiguous().view(torch.uint8).reshape(-1)
            pad = (-b.numel()) % align
            if pad:
                b = torch.cat([b, torch.zeros(pad, dtype=torch.uint8, device=device)])
            parts.append(b)
        flat_buffer = torch.cat(parts)
        return flat_buffer

    @staticmethod
    def deserialize(
        flat_buffer: torch.Tensor, fields_layout: List[Tuple[Tuple[int, ...], torch.dtype]]
    ) -> "ForwardPacket":
        """Rebuild a ForwardPacket from a flat_buffer buffer using a STATIC fixed layout.

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：``fields_layout`` 是构造期静态
        算好的每字段 ``(shape, dtype)`` 有序列表（顺序即 ``ForwardPacket.field_names``），
        取代此前从 shape 头解析形状。按字段切分扁平 buffer：偏移按"填充后长度"推进
        （保证每字段起始偏移为 _ALIGN 整数倍、``view(dtype)`` 合法），切片取该字段原始
        字节数（填充字节不入张量），再 ``view(dtype).reshape(shape)`` 还原。第 5 个字段
        即 microbatch id（1 元素 int64），与其余字段一同切出，直接整体构造。
        Layout-based (no shape header): split ``flat_buffer`` at padded offsets and view each field.
        """
        fields = []
        offset = 0
        for shape, dtype in fields_layout:
            field_bytes = _numel(shape) * _dtype_itemsize(dtype)
            fields.append(flat_buffer[offset : offset + field_bytes].view(dtype).reshape(shape))
            offset += _padded_bytes(field_bytes)
        return ForwardPacket(*fields)


class MergedEncoderBatch(NamedTuple):
    """The merged encoder batch: raw fields, NOT a ForwardPacket (optimization Task 1).

    优化 spec Task 1（合并前传，设计 A）：dataloader 一次取回本 rank 整个 iteration 的
    合并批、encoder 只前传一次。这个中间量**只在本 rank 本地存在、永不上线路**（没有
    ``microbatch_id``、不可序列化），因此刻意**不**包成 ``ForwardPacket``——那会污染
    "packet = 跨边界线红单元"的语义。逐 microbatch 的 ``ForwardPacket`` 只在 schedule
    调 ``ForwardPacket.split_merged_batch`` 拆分并打标之后才诞生。

    Optimization spec Task 1 (design A): the dataloader fetches the rank's whole
    iteration in one merged batch and the encoder forwards it once. This intermediate
    exists only locally and never crosses the wire (no microbatch_id, not serializable),
    so it is deliberately NOT wrapped in a ForwardPacket — that would muddy the
    "packet = boundary wire unit" semantics. Per-microbatch ForwardPackets come into
    being only after the schedule splits via ForwardPacket.split_merged_batch and
    stamps the ids.

    字段布局与 ``ForwardPacket`` 的内容字段一一对应（batch 维是"样本数"：mbs=1 时等于
    microbatch 数）。``image_embeddings`` 保留 grad_fn（phase ④ 对整块做单次 backward）。
    Field layout mirrors ForwardPacket's content fields (the batch dim counts samples;
    with mbs=1 it equals the microbatch count). image_embeddings keeps grad_fn.
    """

    image_embeddings: torch.Tensor  # [img_seq_len, merged_batch, h_lang]（seq-first）
    tokens: torch.Tensor  # [merged_batch, L]
    # dual-channel-p2p（2026-09-22）修正：labels 与 tokens 同形 [merged_batch, L]（NOT L+1）。
    # Corrected: labels is [merged_batch, L], same as tokens (colocated_train.py asserts equal).
    labels: torch.Tensor  # [merged_batch, L]
    num_image_tiles: torch.Tensor  # [merged_batch]


@dataclass
class BackwardPacket:
    """Backward grad packet from the consumer back to one encoder producer.

    共置训练边界的**反向梯度包**：消费者把某个 encoder producer 的 image_embeddings
    梯度（单张量）发回该 producer，与 ``ForwardPacket`` 对称。类负责序列化布局
    （单张量，无对齐填充）；通信原语留在通信器。

    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：``serialize()`` 只返回
    ``grad.contiguous()``（不再构造 shape 头）。接收端按构造期静态算好的定长
    ``(_grad_shape, _grad_dtype)`` 直接分配 buffer 并 irecv，收到的 buffer **本身即梯度**
    （shape/dtype 天然正确），无需 deserialize。
    """

    grad: torch.Tensor

    def serialize(self) -> torch.Tensor:
        """Serialize into ONE contiguous grad tensor (HEADER-FREE); one P2P call.

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：只返回 ``grad.contiguous()``
        （不再构造/返回 shape 头）。梯度 shape 定长（= image_embeddings，
        ``[img_seq_len, mbs, h_lang]``）、dtype 固定为通信器 dtype（``config.pipeline_dtype``，
        如 bf16），均由接收端构造期静态获知，故不随包传。
        """
        return self.grad.contiguous()


class _ForwardRecvRequest:
    """In-flight async receive of a forward packet (single fixed-size data irecv).

    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：异步前向接收请求。构造后由通信器
    立即 ``_start()``——**直接分配定长扁平 buffer 并 post 单个数据 irecv（真正非阻塞：无
    等待、无 header、无 parse）**。此前需先收 shape 头 + ``parse_shape_header``（CUDA
    同步）才能分配 buffer，那次同步会卡住 schedule 循环 → 死锁；定长后彻底移除 header
    机制。拿到 request 时数据传输已在后台进行，调用方只需在真正需要数据时调
    ``finish()``（等数据 handle → 按 layout 组装 ``ForwardPacket``）。
    A request posts a single fixed-size data irecv (no header, no wait, no parse).
    """

    def __init__(
        self,
        comm: "EncoderBackboneBoundaryCommunicator",
        src_rank: int,
        producer: int,
        expected_microbatch_id: Optional[int],
    ):
        self._comm = comm
        self._src_rank = src_rank
        self.producer = producer
        self.expected_microbatch_id = expected_microbatch_id
        self._flat_buffer = None  # data buffer allocated by _start(); read by finish()
        self._data_handle = None  # data irecv handle posted by _start()
        self._packet = None

    # NVTX：start/finish 是异步收的两段——start 标"何时把等待插进流"，finish 标"何时
    # 真正等到数据"，两者的间隔就是传输藏进计算的程度。两个 request 类的方法同名
    # （start/finish），默认区间名（模块路径.函数名）会撞名，故必须显式 message。
    # NVTX: start/finish are the two halves of an async receive - start marks when the
    # wait is enqueued, finish marks when the data actually arrives. The two request
    # classes have same-named methods, so explicit messages are required.
    @nvtx_decorator(message="colocated-boundary-forward-recv-start")
    def _start(self) -> "_ForwardRecvRequest":
        """Allocate the fixed-size data buffer and post the single data irecv.

        dual-channel-p2p（2026-09-22）：**直接**分配定长扁平 buffer → post 单个数据
        irecv（**不等待、无 header、无 parse**，真正非阻塞）。由通信器在交出 request 前
        调用（``_async_recv_forward``），不是公开接口；返回 self 便于链式书写。
        重复调用是幂等的（``_data_handle`` 已存在则跳过）。
        Called by the communicator before the request is handed out, not by callers.
        """
        if self._data_handle is None:
            self._data_handle = self._comm._start_recv_forward(self)
        return self

    @nvtx_decorator(message="colocated-boundary-forward-recv-finish")
    def finish(self) -> "ForwardPacket":
        """Finish phase: wait the data and assemble the ForwardPacket.

        **结束阶段**：等数据 handle（通常早已完成）→ 组装 ``ForwardPacket``（结果缓存，
        重复调用不重复收）。
        """
        if self._packet is None:
            self._packet = self._comm._finish_recv_forward(self)
        return self._packet


class _GradRecvRequest:
    """In-flight async receive of the backward grad (single fixed-shape data irecv).

    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：异步梯度接收请求。构造后由通信器
    立即 ``_start()``——**直接分配定长梯度 buffer 并 post 单个数据 irecv（无等待、无
    header、无 parse，真正非阻塞）**。此前需先收 shape 头才能分配 buffer，那次同步是死锁
    根因；定长后移除 header。拿到 request 时梯度传输已在后台进行，调用方需要梯度时只调
    ``finish()``（等数据 handle → 包成 ``BackwardPacket``）。与 ``_ForwardRecvRequest`` 同构。
    Symmetric with _ForwardRecvRequest: posts a single fixed-shape data irecv, no header.
    """

    def __init__(
        self,
        comm: "EncoderBackboneBoundaryCommunicator",
        src_rank: int,
    ):
        self._comm = comm
        self._src_rank = src_rank
        self._grad_buffer = None  # grad buffer allocated by _start(); read by finish()
        self._data_handle = None  # data irecv handle posted by _start()
        self._packet = None

    @nvtx_decorator(message="colocated-boundary-grad-recv-start")
    def _start(self) -> "_GradRecvRequest":
        """Allocate the fixed-shape grad buffer and post the single data irecv.

        dual-channel-p2p（2026-09-22）：**直接**分配定长梯度 buffer → post 单个数据 irecv
        （**不等待、无 header、无 parse**，真正非阻塞）。由通信器在交出 request 前调用
        （``_async_recv_grad``），不是公开接口；幂等。
        Called by the communicator before the request is handed out, not by callers.
        """
        if self._data_handle is None:
            self._data_handle = self._comm._start_recv_grad(self)
        return self

    @nvtx_decorator(message="colocated-boundary-grad-recv-finish")
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
    （send 返回 handle 列表，稍后 ``wait()``；recv 返回**已启动**的 request 对象，
    数据传输已在后台进行，稍后 ``finish()`` 取数）
    ——producer replenish / consumer prefetch / 补发 step 收梯度用异步路径与计算重叠。
    前向包内含 **microbatch id 字段**（消费者按序 take 时校验，防乱序错配）。
    dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：收发不再传 shape 头——每个样本 shape
    由 config 静态固定，收方按构造期算好的定长 layout 直接 post 单个 irecv（立即返回、真正
    异步），消除了旧 shape 头 + parse（CUDA 同步）导致的阻塞式接收/死锁。
    **本地直传短路（producer 0 = 消费者自己）由调用方（wrapper）分支处理**：producer 0
    的包/梯度不经网络、零拷贝本地引用——wrapper 跳过本类方法直接使用本地 buffer
    （包就是 ``ForwardPacket``，dataclass 直接构造即可），不发起任何 P2P，因此网络侧
    配对计数不受影响。
    """

    def __init__(
        self,
        activation_comm_group,
        grad_comm_group,
        config: ModelParallelConfig,
        dtype: Optional[torch.dtype] = None,
        image_seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        text_seq_length: Optional[int] = None,
    ):
        """Initialize with the two direction-split colocated boundary groups.

        dual-channel-p2p Task 2（2026-09-22）：边界通信按方向拆成两个独立 NCCL 组，各自
        内部 NCCL stream，天然隔离收发方向（消除同流收发交叉死锁）：
        - ``activation_group``：只承载 producer→consumer 的前向激活包（forward）。
        - ``grad_group``：只承载 consumer→producer 的反向梯度（backward）。
        两组成员相同（同一外层 dp 副本的 pp ranks），producer 编号/consumer 由任一组推导
        （此处用 activation_group）。

        Args:
            activation_comm_group: forward-activation P2P group (producer order); must
                come from ``get_colocated_boundary_activation_group()``.
                前向激活组，须来自 get_colocated_boundary_activation_group()。
            grad_comm_group: backward-grad P2P group; must come from
                ``get_colocated_boundary_grad_group()``.
                反向梯度组，须来自 get_colocated_boundary_grad_group()。
            config: model parallel config (``pipeline_dtype`` is used for floats).
            dtype: optional override of the float dtype (image_embeddings/grad).
            image_seq_length / micro_batch_size / hidden_size / text_seq_length: optional
                explicit fixed dims that OVERRIDE ``get_args()``. Default None -> read from
                the global args. Kept optional so the schedule call site stays unchanged
                (reads get_args); unit tests pass them explicitly to avoid mocking get_args.
                可选显式定长维度，覆盖 ``get_args()``；默认 None 时从全局 args 读取。
                schedule 调用点不传（走 get_args），单元测试显式传入以免 mock get_args。
        """
        self.activation_comm_group = activation_comm_group
        self.grad_comm_group = grad_comm_group
        self.config = config
        self.dtype = dtype if dtype is not None else config.pipeline_dtype

        self.world_rank = dist.get_rank()
        # Members of the replica in producer order; producer id 0 is the consumer
        # (the backbone entry, i.e. the backbone first stage). Both groups share the
        # same membership; derive the topology from the activation group.
        # 副本内成员（按生产者编号）；0 号生产者即消费者（backbone entry）。两组成员相同，
        # 拓扑用 activation_comm_group 推导。
        self.group_ranks = dist.get_process_group_ranks(activation_comm_group)
        self.producer_id = dist.get_group_rank(activation_comm_group, self.world_rank)
        self.group_size = len(self.group_ranks)
        self.consumer_global_rank = self.group_ranks[0]
        assert dist.get_process_group_ranks(grad_comm_group) == self.group_ranks, (
            "activation_comm_group and grad_comm_group must have identical members"
        )

        # dual-channel-p2p（2026-09-22）HEADER-FREE / 定长协议：每个样本 shape 由 config 静态
        # 固定，故在此**一次性**算出各包的定长 layout，供收发两端复用——接收端据此直接分配
        # 定长 buffer 并 post 单个 irecv（立即返回、真正异步），**无需**先收 shape 头再
        # ``parse_shape_header``（``.item()/.tolist()`` 会触发 CUDA 同步、卡住 schedule 循环 →
        # 死锁）。四个维度默认从 ``get_args()`` 读取（schedule 调用点不必改签名），也可由
        # 显式 kwargs 覆盖（单测用）。
        # HEADER-FREE fixed-length: compute the statically-derived fixed packet layout once
        # here so the receiver can allocate a fixed buffer and post a single irecv (returns
        # immediately, truly async) without the shape-header wait + parse (a CUDA sync that
        # stalled the schedule loop and deadlocked). The four dims default to get_args().
        if None in (image_seq_length, micro_batch_size, hidden_size, text_seq_length):
            # 函数内 import 避免 megatron.core -> megatron.training 的模块级循环依赖。
            # Function-local import avoids a module-level core->training circular import.
            from megatron.training import get_args

            args = get_args()
            # image_seq_length = encoder_seq_length（model.py 在建模时置
            # seq_length == encoder_seq_length == num_image_embeddings；336px=576, 504px=1296）。
            image_seq_length = (
                args.encoder_seq_length if image_seq_length is None else image_seq_length
            )
            micro_batch_size = (
                args.micro_batch_size if micro_batch_size is None else micro_batch_size
            )
            hidden_size = args.hidden_size if hidden_size is None else hidden_size
            # 本协议要求 dataloader 序列长度固定（定长 layout 的前提）。
            # This protocol requires a fixed dataloader seq length.
            assert args.dataloader_seq_length is not None, (
                "colocated HEADER-FREE fixed-length protocol requires "
                "args.dataloader_seq_length to be set (a fixed dataloader seq length)"
            )
            text_seq_length = (
                args.dataloader_seq_length if text_seq_length is None else text_seq_length
            )

        # 5 个前向字段的定长 (shape, dtype)，顺序 == ForwardPacket.field_names：
        #   image_embeddings (image_seq_length, micro_batch_size, hidden_size) float(self.dtype)
        #   tokens           (micro_batch_size, text_seq_length)               int64
        #   labels           (micro_batch_size, text_seq_length)               int64  ← text_seq_length，NOT +1
        #   num_image_tiles  (micro_batch_size,)                               int32
        #   microbatch_id    (1,)                                              int64
        # The statically-derived fixed forward-field layout (order == field_names).
        # hidden_size 即语言模型（backbone）隐藏维——包里的 image_embeddings 已由 vision_projection
        # 投到语言维，故用 args.hidden_size。
        self._forward_fields: List[Tuple[Tuple[int, ...], torch.dtype]] = [
            ((image_seq_length, micro_batch_size, hidden_size), self.dtype),
            ((micro_batch_size, text_seq_length), torch.int64),
            ((micro_batch_size, text_seq_length), torch.int64),
            ((micro_batch_size,), torch.int32),
            ((1,), torch.int64),
        ]
        # 前向扁平 buffer 总字节数 = 各字段 padded 字节之和（与 serialize 的 _ALIGN 填充一致）。
        # Total flat_buffer-buffer bytes = sum of per-field padded bytes (matches serialize).
        self._forward_total_bytes = sum(
            _padded_bytes(_numel(shape) * _dtype_itemsize(dtype))
            for shape, dtype in self._forward_fields
        )
        # 反向梯度定长 shape/dtype（= image_embeddings）。
        # Fixed backward grad shape/dtype (matches image_embeddings).
        self._grad_shape: Tuple[int, ...] = (image_seq_length, micro_batch_size, hidden_size)
        self._grad_dtype: torch.dtype = self.dtype

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
        """Send a BackwardPacket: ONE grad tensor (1 P2P call, HEADER-FREE).

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：只发**一个**连续梯度张量
        （不再先发 shape 头），布局由 ``BackwardPacket.serialize`` 给出（见类 docstring）。

        ``wait=True``（默认）：阻塞到 isend 完成，返回 None。
        ``wait=False``：提交后立即返回 handle 列表（单元素，保持"handle 列表"契约；
        consumer 反传 hook 用，wait 由调用方统一做）。
        """
        grad = packet.serialize()
        handles = [dist.isend(grad, dst=dst_rank, group=self.grad_comm_group)]
        if wait:
            for handle in handles:
                handle.wait()
            return None
        return handles

    def _recv_backward_packet(self, src_rank: int) -> BackwardPacket:
        """Receive a BackwardPacket synchronously: ONE fixed-shape grad (HEADER-FREE).

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：直接按静态定长
        ``(_grad_shape, _grad_dtype)`` 分配 buffer 并 irecv 一次（1 次 P2P，**无 header、无
        parse**），收到的 buffer 本身即梯度（shape/dtype 天然正确），包成 BackwardPacket。
        梯度形状与 image_embeddings 相同（[img_seq_len, mbs, h_lang]），dtype 为通信器 dtype。
        """
        buffer = torch.empty(self._grad_shape, dtype=self._grad_dtype, device="cuda")
        recv_handle = dist.irecv(buffer, src=src_rank, group=self.grad_comm_group)
        recv_handle.wait()
        return BackwardPacket(grad=buffer)

    def _async_recv_grad(self, src_rank: int) -> _GradRecvRequest:
        """Asynchronously submit a receive of the backward grad: single data irecv.

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：直接构造 ``_GradRecvRequest`` 并
        ``_start()``——它**只**按静态定长分配梯度 buffer + post 单个数据 irecv（**无 header
        irecv、不等待、不 parse**，真正非阻塞立即返回）。生产者侧用它把接收提前挂到 NCCL
        后台，需要梯度时只调 ``request.finish()``。梯度 dtype/shape 用构造期定长 layout。
        Truly non-blocking: no header rendezvous, just one fixed-shape data irecv.
        """
        return _GradRecvRequest(self, src_rank)._start()

    def _start_recv_grad(self, request: _GradRecvRequest) -> object:
        """Allocate the fixed-shape grad buffer and post the single data irecv.

        dual-channel-p2p（2026-09-22）：梯度异步接收的**启动步骤**（由
        ``_GradRecvRequest._start()`` 调用）：按静态定长 ``(_grad_shape, _grad_dtype)`` 分配
        梯度 buffer → **异步提交单个数据 irecv（不等待、无 header、无 parse）**，返回数据
        handle。数据在此到 ``finish()`` 之间的计算窗口后台传输。
        """
        request._grad_buffer = torch.empty(
            self._grad_shape, dtype=self._grad_dtype, device="cuda"
        )
        return dist.irecv(
            request._grad_buffer, src=request._src_rank, group=self.grad_comm_group
        )

    def _finish_recv_grad(self, request: _GradRecvRequest) -> BackwardPacket:
        """Finish phase of an async grad receive: wait the data, wrap the packet.

        dual-channel-p2p（2026-09-22）HEADER-FREE：等数据 handle（通常早已完成）→ 收到的
        buffer 本身即梯度（定长分配，shape/dtype 天然正确），直接包成 ``BackwardPacket``。
        """
        request._data_handle.wait()
        return BackwardPacket(grad=request._grad_buffer)

    def _send_forward_packet(
        self, packet: ForwardPacket, dst_rank: int, wait: bool = True
    ) -> Optional[list]:
        """Send a ForwardPacket as ONE flat_buffer buffer — 1 P2P call total (HEADER-FREE).

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：把前向包经 ``ForwardPacket.serialize``
        拼成**一个**扁平 uint8 buffer 发送（1 次 P2P，不再先发 shape 头）：各字段按自身
        dtype 展平为 uint8 字节、字节长度填充到 _ALIGN 的整数倍后拼接（对齐保证接收端任意
        切片 ``view(dtype)`` 合法）。接收端按静态定长 layout 直接分配定长 buffer 收取、无需
        header。microbatch id 是包内第 5 个字段（消费者 take 时校验，来自包自身属性）。

        ``wait=True``（默认）：阻塞到 isend 完成，返回 None。
        ``wait=False``：提交后立即返回 handle 列表（单元素，保持"handle 列表"契约；
        producer replenish 用，wait 由调用方在 phase ② 后统一做）。
        """
        # 发送端按张量自身 dtype 展平，接收端按 ``self.dtype``（config.pipeline_dtype）
        # 反推浮点字段的字节数——两者不一致会让接收端所有偏移错位、解析出垃圾（例如
        # microbatch id 变成随机整数），且不报错、只在下游校验时才暴露。跨 rank 协议
        # 边界，值得断言。
        # The sender flattens each tensor by its own dtype while the receiver derives the
        # float field's byte length from self.dtype (config.pipeline_dtype); a mismatch
        # silently shifts every offset (the microbatch id decodes to garbage), so assert
        # here — this is a cross-rank protocol boundary.
        assert packet.image_embeddings.dtype == self.dtype, (
            f"forward packet image_embeddings dtype {packet.image_embeddings.dtype} != "
            f"communicator dtype {self.dtype} (config.pipeline_dtype): the receiver would "
            f"parse the flat_buffer buffer with the wrong element size"
        )
        flat_buffer = packet.serialize()  # 需要序列化因为 dist 通信只支持 tensor 数据，不支持类型对象
        handles = [dist.isend(flat_buffer, dst=dst_rank, group=self.activation_comm_group)]
        if wait:
            for handle in handles:
                handle.wait()
            return None
        return handles

    def _check_microbatch_id(
        self, packet: ForwardPacket, expected_microbatch_id: Optional[int]
    ) -> None:
        """Validate a received packet's microbatch id against the expected one.

        校验收到的包自带的 microbatch id（1 元素 int64 张量）与期望一致（消费者按序
        take 时用，防乱序错配）。通信两端是确定的，这里不是验证乱序达到的问题，只是在做校验id
        """
        if expected_microbatch_id is not None:
            packet_microbatch_id = packet.microbatch_id
            assert packet_microbatch_id is not None and packet_microbatch_id.item() == expected_microbatch_id, (
                f"out-of-order forward packet: packet microbatch id {packet_microbatch_id} "
                f"!= expected {expected_microbatch_id}"
            )

    def _recv_forward_packet(
        self, src_rank: int, expected_microbatch_id: Optional[int] = None
    ) -> ForwardPacket:
        """Receive a ForwardPacket synchronously: ONE fixed-size flat_buffer buffer (HEADER-FREE).

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：直接按静态定长 ``_forward_total_bytes``
        分配扁平 buffer 并 irecv 一次（1 次 P2P，**无 header、无 parse**），再经
        ``deserialize`` 按 ``_forward_fields`` 还原为 ForwardPacket，校验包自带的 microbatch id。
        """
        flat_buffer = torch.empty(self._forward_total_bytes, dtype=torch.uint8, device="cuda")
        recv_handle = dist.irecv(flat_buffer, src=src_rank, group=self.activation_comm_group)
        recv_handle.wait()
        packet = ForwardPacket.deserialize(flat_buffer, self._forward_fields)
        self._check_microbatch_id(packet, expected_microbatch_id)
        return packet

    def _async_recv_forward(
        self, src_rank: int, producer: int, expected_microbatch_id: Optional[int] = None
    ) -> "_ForwardRecvRequest":
        """Asynchronously submit a receive of a forward packet: single data irecv.

        dual-channel-p2p（2026-09-22）HEADER-FREE / 定长：直接构造 ``_ForwardRecvRequest``
        并 ``_start()``——它**只**按静态定长分配扁平 buffer + post 单个数据 irecv（**无
        header irecv、不等待、不 parse**，真正非阻塞立即返回）。消费者 prefetch 时用它把
        接收提前挂到 NCCL 后台，需要数据时只调 ``request.finish()``（取数据）。
        Truly non-blocking: no header rendezvous, just one fixed-size data irecv.
        """
        request = _ForwardRecvRequest(
            self, src_rank, producer, expected_microbatch_id
        )._start()
        return request

    def _start_recv_forward(self, request: "_ForwardRecvRequest") -> object:
        """Allocate the fixed-size data buffer and post the single data irecv.

        dual-channel-p2p（2026-09-22）：前向异步接收的**启动步骤**（由
        ``_ForwardRecvRequest._start()`` 调用）：按静态定长 ``_forward_total_bytes`` 分配扁平
        buffer → **异步提交单个数据 irecv（不等待、无 header、无 parse）**，返回数据 handle。
        数据在此到 ``finish()`` 之间的计算窗口后台传输。
        """
        request._flat_buffer = torch.empty(
            self._forward_total_bytes, dtype=torch.uint8, device="cuda"
        )
        data_handle = dist.irecv(
            request._flat_buffer, src=request._src_rank, group=self.activation_comm_group
        )
        return data_handle

    def _finish_recv_forward(self, request: "_ForwardRecvRequest") -> ForwardPacket:
        """Finish phase of an async forward receive: wait the data, assemble the packet.

        dual-channel-p2p（2026-09-22）HEADER-FREE：等数据 handle（通常早已完成）→ 按静态
        定长 ``_forward_fields`` 经 ``deserialize`` 还原为 ``ForwardPacket`` 并校验其
        microbatch id。
        """
        request._data_handle.wait()
        packet = ForwardPacket.deserialize(request._flat_buffer, self._forward_fields)
        self._check_microbatch_id(packet, request.expected_microbatch_id)
        return packet

    # ------------------------------------------------------------------
    # Forward: encoder output + text packet from a producer to the consumer.
    # 前向：encoder 输出 + 文本数据包从生产者发往消费者。
    # ------------------------------------------------------------------

    # NVTX：四个公开收发方法 + 预热各打一个区间（nsys 时间线上 boundary 收发的位置、
    # 与计算区间的重叠一眼可见；backbone 1F1B 的 P2P 已由 p2p_communication.py 的
    # @nvtx_decorator 覆盖，这里补的是共置边界组上的一段）。显式 message 而非默认
    # 函数路径，与 colocated_schedule.py 的 "colocated-*" 命名一致。
    # NVTX: one range per public boundary op plus the per-iteration warmup; the backbone
    # 1F1B P2P is already covered by p2p_communication.py's decorators, this covers the
    # colocated boundary group. Explicit messages keep the "colocated-*" naming.
    @nvtx_decorator(message="colocated-boundary-send-forward")
    def colocated_send_forward(
        self,
        packet: ForwardPacket,
        producer: Optional[int] = None,
        wait: bool = True,
    ) -> Optional[list]:
        """Producer side (producer id > 0): send the forward packet to the consumer.

        生产者（producer id>0）：把本生产者的 ``ForwardPacket``（encoder 输出 + 本地
        文本数据 tokens/labels/num_image_tiles）发给消费者。microbatch id 是包内第 5 个
        字段，随定长扁平 buffer 一同发送（消费者 take 时校验；HEADER-FREE，无 shape 头）。
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

    @nvtx_decorator(message="colocated-boundary-recv-forward")
    def colocated_recv_forward(
        self,
        producer: int,
        expected_microbatch_id: Optional[int] = None,
        wait: bool = True,
    ) -> Union[ForwardPacket, _ForwardRecvRequest]:
        """Consumer side: receive the forward packet of producer ``producer``.

        消费者：接收生产者 ``producer``（>0）的完整前向数据包（``ForwardPacket``：
        encoder 输出 + 文本字段）。``wait=True``（默认）同步阻塞返回 ``ForwardPacket``；
        ``wait=False`` 异步提交（HEADER-FREE：直接 post 单个定长数据 irecv、立即返回，
        无 shape 头、无 parse、真正非阻塞），返回**已启动**
        的 ``_ForwardRecvRequest``（consumer prefetch 用，需要数据时只调
        ``request.finish()``）。
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

    @nvtx_decorator(message="colocated-boundary-send-backward")
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

    @nvtx_decorator(message="colocated-boundary-recv-backward")
    def colocated_recv_backward(
        self, producer: Optional[int] = None, wait: bool = True
    ) -> Union[BackwardPacket, _GradRecvRequest]:
        """Producer side (producer id > 0): receive the encoder-output grad.

        生产者（producer id>0）：从消费者接收本生产者的 encoder 输出梯度
        （``BackwardPacket``，与 colocated_send_forward 对应）。``producer`` 默认取本
        rank 的 producer id。``wait=True``（默认）同步阻塞返回 ``BackwardPacket``；
        ``wait=False`` 异步提交返回**已启动**的 ``_GradRecvRequest``（producer 补发 step 的
        forward 前用，需要梯度时只调 ``request.finish()``）。
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
