# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the colocated schedule wrapper, phase ① (Task 4.1/4.2; Task 1 merged).

共置 schedule（顶层函数 ``forward_backward_colocated``）的 phase ①
（encoder **合并**前传 + 本地 buffer）单元测试：
- 每个 producer（边界组内槽位）负责的 microbatch = p, p+P, p+2P, ...（``get_microbatches_for_producer``），
  即每 producer 处理 ``num_microbatches / P`` 个 microbatch = ``global_mbs / (dp * inner_dp)``；
- 优化 spec Task 1（设计 A）：schedule 对 forward step 的 encoder 分支**每个 iteration
  只调一次**（一次取合并批 + 一次前传 -> ``MergedEncoderBatch``），经
  ``ForwardPacket.split_merged_batch`` 等分回逐 microbatch 的包、打 id 存 buffer——
  切片是合并张量的 view（共享 storage、保 grad_fn），合并张量本体由 schedule 返回；
- 跨 rank：所有 producer 的 microbatch 恰好覆盖 0..num_microbatches-1 一次（不重不漏）；
- 主函数 wiring（4.3f）：完整跑 phase ①+②（PP=1 全本地 / PP>1 边界配对），返回
  forward_data_store；encoder 分支每 iteration 恰一次、num_microbatches
  非 pp 倍数在校验处抛 AssertionError；
- n>P 完整节奏（4.6f）：n = 2*P 跑 phase ①→②→④，覆盖 4.4 补货、4.5 逐 step
  prefetch、4.6c/4.6d 的边界梯度往返与 4.6e 的**合并** encoder 反传（Task 1）。
"""

import pytest
import torch
import torch.distributed as dist
from functools import partial

import megatron.core.parallel_state as ps
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.colocated_encoder_comm import (
    ForwardPacket,
    MergedEncoderBatch,
)
from megatron.core.pipeline_parallel.colocated_schedule import (
    _colocated_encoder_forward,
    forward_backward_colocated,
)
from megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
    snapshot_colocated_encoder_rng_tracker,
)
from tests.unit_tests.test_utilities import Utils

_IMG_H, _IMG_W, _H_LANG = 4, 4, 8
# 边界包的浮点字段（image_embeddings）与 backbone activation 都必须是
# config.pipeline_dtype——通信器按它反推字节数（colocated_encoder_comm 的 send 断言）。
# The boundary packet's float field and the backbone activation must both be
# config.pipeline_dtype: the communicator derives byte lengths from it.
_PIPELINE_DTYPE = torch.bfloat16
# backbone activation 的形状契约：1F1B 的接收端不看发送端实际发了什么，它按
# get_tensor_shapes(seq_length, micro_batch_size, hidden_size) 分配 buffer 再 post irecv
# （schedules.py 的 recv_tensor_shapes），所以 backbone forward 的输出必须正好是
# [seq_length, micro_batch_size, hidden]——形状不符时 NCCL 的 send/recv 元素数对不上、
# recv 永远配不上，两端一起挂死（2026-08-26 flight recorder 实测）。
# The activation shape contract: the 1F1B receiver allocates its buffer from
# get_tensor_shapes(seq_length, micro_batch_size, hidden_size) rather than from whatever
# the sender produces, so the backbone forward must return exactly
# [seq_length, micro_batch_size, hidden]; a mismatch makes the NCCL recv unmatchable
# and deadlocks both ends.
_SEQ_LENGTH, _MICRO_BATCH_SIZE = 10, 1


class MockEncoder(torch.nn.Module):
    """Minimal encoder-only chunk: images [num_tiles, 3, h, w] -> [1, num_tiles, h_lang].

    最小 encoder chunk 桩：给真实 grad_fn（Linear 输出），模仿 ColocatedViTEncoder
    的 forward 契约（无图样本返回空 tensor）。
    """

    # 生产契约的一部分：schedule 用 group_colocated_model_chunks 按该属性拆分 model 列表
    # （core/utils.py），不按下标。真实类在 colocated_llava_model.py:58 声明同名属性。
    # Part of the production contract: the schedule splits the model list by this attribute
    # (group_colocated_model_chunks) rather than by position.
    colocated_module_name = "encoder"

    def __init__(self):
        super().__init__()
        # 生产契约：phase ④ 从 **encoder chunk 自己的 config** 读 no_sync_func 与
        # finalize_model_grads_func（colocated_schedule，5.6 起统一按 config 走）。桩用
        # 默认值（两者都是 None ⇒ nullcontext + 跳过梯度收尾），因为桩不裹 DDP、也没有
        # finish_grad_sync，与真实训练下由 train() 挂上回调的形态互补。
        # Production contract: phase (4) reads no_sync_func and finalize_model_grads_func
        # off the encoder chunk's OWN config. The stub keeps the defaults (both None), as it
        # is not DDP-wrapped and has no finish_grad_sync.
        self.config = ModelParallelConfig(pipeline_dtype=_PIPELINE_DTYPE)
        self.proj = torch.nn.Linear(3 * _IMG_H * _IMG_W, _H_LANG).to(_PIPELINE_DTYPE)

    def forward(self, images):
        if images.shape[0] == 0:
            return torch.tensor([], dtype=_PIPELINE_DTYPE, device=images.device).reshape(0, 0, 0)
        # 输出必须**自己持有存储**（``_base is None``）：Task 4.9 的 deallocate_output_tensor
        # 断言 "counter-productive to free a view of another tensor"。真实
        # ColocatedViTEncoder 满足契约（最后一步是 vision_projection 的 bias add，
        # grad_fn=AddBackward0，实测 _base is None）。这里要注意两个坑：
        # ① nn.Linear 作用于 3D 输入时内部 reshape→mm→view，输出是 view；
        # ② unsqueeze 产生 view，而 contiguous() 对已连续张量返回自身，去不掉 view 身份。
        # 所以保持 Linear 在 2D 上算（输出为新张量），再 unsqueeze + clone 得到拥有存储的 3D 张量。
        # The output must own its storage (_base is None), like the real encoder whose last op
        # is the projector's bias add; note that a 3D nn.Linear returns a view and that
        # contiguous() is a no-op on an already-contiguous view.
        out = self.proj(images.flatten(1).to(_PIPELINE_DTYPE))  # [num_tiles, h_lang]
        return out.unsqueeze(0).clone()  # [1, num_tiles, h_lang]


class FakeBackbone(torch.nn.Module):
    """Minimal backbone chunk for running phase ② (Task 4.3f).

    ``pre_process`` 标记 consumer（stage 0）；``set_input_tensor`` 存 P2P 激活（非
    consumer 用）；``forward`` 做线性输出（带 grad_fn，backward_step 需要）——consumer
    用 partial 绑定的 packet 的 image_embeddings，非 consumer 用 input_tensor。返回
    ``(output, loss_mask)``（数值无所谓，冒烟只验证通信配对与无死锁）。
    """

    # 同 MockEncoder：schedule 按该属性拆分 model 列表（真实类见
    # colocated_llava_model.py:240）。
    # Same as MockEncoder: the schedule splits the model list by this attribute.
    colocated_module_name = "language_model"

    def __init__(self, pre_process=True):
        super().__init__()
        self.config = ModelParallelConfig(
            pipeline_dtype=_PIPELINE_DTYPE,
            # 4.3k/4.4（2026-08-23 诊断）：跳过 _communicate 的设备级同步
            #（batch_p2p_sync）——它是旧 torch 的防御 workaround；且 unbatched 的
            # targets 传输（_send_targets/_recv_targets）在 pp_group 上会触发新的
            # NCCL 通信器 lazy-init，若两端不齐（一端卡在同步）会死锁。
            # 完成性由 _communicate 内 batch 级 req.wait() 保证。
            # 4.3k/4.4 (2026-08-23 diagnostic): skip the device-wide sync in _communicate
            # (batch_p2p_sync) — an old-torch defensive workaround; the unbatched targets
            # transport (_send_targets/_recv_targets) lazily creates a new pp_group NCCL
            # communicator, whose collective init deadlocks if the peer is stuck in the
            # device sync. Completion is still guaranteed by the batch-level req.wait().
            batch_p2p_sync=False,
        )
        # phase ② 运行需要 ModelParallelConfig 缺失的字段（在 TransformerConfig 里），
        # 补默认值（get_tensor_shapes / forward_step_calc_loss / 循环尾部等读取）。
        for _attr, _value in {
            'hidden_size': _H_LANG,
            'calculate_per_token_loss': False,
            'num_moe_experts': None,
            'mtp_num_layers': None,
            'fine_grained_activation_offloading': False,
            'cuda_graph_impl': None,
            'cuda_graph_scope': None,
            'fp32_residual_connection': False,
        }.items():
            if not hasattr(self.config, _attr):
                setattr(self.config, _attr, _value)
        self.pre_process = pre_process
        self.input_tensor = None
        self.proj = torch.nn.Linear(_H_LANG, _H_LANG).to(_PIPELINE_DTYPE)

    def set_input_tensor(self, input_tensor):
        # 与 ColocatedGPTBackbone.set_input_tensor 一致（colocated_llava_model.py:325-330）：
        # schedule 注入的 activation 可能是单元素 list（P2PCommunicator 的返回形态不统一），
        # 统一解包成张量。
        # Mirror ColocatedGPTBackbone.set_input_tensor: the schedule may inject a
        # single-element list (the P2P API's return shape is not uniform), so unwrap it.
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for colocated backbone"
        self.input_tensor = input_tensor[0]

    def forward(
        self,
        image_embeddings=None,
        input_ids=None,
        labels=None,
        loss_mask=None,
        num_image_tiles=None,
        **kwargs,
    ):
        x = image_embeddings if self.pre_process else self.input_tensor
        if x is None:
            x = torch.zeros(1, 1, _H_LANG, dtype=_PIPELINE_DTYPE, device="cuda")
        # 输出必须是 [seq_length, micro_batch_size, hidden]（见 _SEQ_LENGTH 处的形状契约）：
        # 先把输入压成 [hidden]（sum 保留到 image_embeddings / input_tensor 的梯度通路），
        # 再展开成契约形状。真实的 ColocatedGPTBackbone 天然输出该形状，这里只是补齐桩的保真度。
        # The output must be [seq_length, micro_batch_size, hidden] (see the shape contract
        # above): reduce the input to [hidden] (sum keeps the gradient path to
        # image_embeddings / input_tensor), then expand to the contracted shape. The real
        # ColocatedGPTBackbone produces this shape naturally.
        hidden = self.proj(x.reshape(-1, _H_LANG)).sum(0)
        activation = hidden.view(1, 1, _H_LANG).expand(
            _SEQ_LENGTH, _MICRO_BATCH_SIZE, _H_LANG
        ).contiguous()
        return activation, loss_mask


def _make_fake_merged_batch(seed, num_samples, seq=_SEQ_LENGTH):
    """Deterministic MERGED batch 4-tuple (colocated_encoder_get_batch contract, Task 1).

    优化 spec Task 1（设计 A）：dataloader 一次取回本 rank 的全部 micro batch，所以
    fake 数据是**合并批**——``num_samples`` 个样本沿 batch 维堆叠（tokens/labels
    ``[N, seq]``、imgs ``[N, 3, h, w]``、num_tiles ``[N]``）。与生产一致的两条数据假设
    在这里也成立：labels 已左移（fake 直接给最终形态）、每样本恰好 1 个 tile
    （``split_merged_batch`` 的等分断言会拦截非 1）。
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    imgs = torch.randn(
        (num_samples, 3, _IMG_H, _IMG_W), dtype=torch.float32, device="cuda", generator=generator
    )
    tokens = torch.randint(0, 100, (num_samples, seq), dtype=torch.int64, device="cuda", generator=generator)
    labels = torch.randint(0, 100, (num_samples, seq), dtype=torch.int64, device="cuda", generator=generator)
    num_tiles = torch.ones(num_samples, dtype=torch.int32, device="cuda")
    return (tokens, labels, imgs, num_tiles)


def _make_fake_colocated_forward_step(merged_batch, encoder, backbone, served):
    """Fake 'colocated_forward_step': (data_iterator, model, packet=None, intra_packet=None),
    branched by chunk.

    模拟 colocated_train.colocated_forward_step（4.3f，4.3j 加 intra_packet；优化 spec
    Task 1 改为**合并形态**）：
    - encoder 分支（``model=[encoder]`` list，phase ①）：**每个 iteration 只被调一次**，
      一次 ``next(data_iterator)`` 取回合并批 + ``encoder_chunk(images)`` 一次前传 ->
      ``(MergedEncoderBatch, None)``（与生产一致：裸张量集合，非 ForwardPacket）；
    - backbone 分支（``model=backbone`` 单 chunk 或 list，phase ②）：consumer
      （``pre_process``）用 **partial 绑定的 packet**（拆分后的逐 microbatch 包）跑
      FakeBackbone，并把展开 labels/loss_mask 写回 intra_packet（4.3j）；非 consumer 用
      intra_packet 闭包绑定传入的 labels/loss_mask。返回 ``(output, loss_func)``。
    """

    def fake_loss_func(loss_mask, output_tensor):
        # per-token 型：返回 (loss, num_tokens, loss_reduced)，见 forward_step_calc_loss。
        loss = output_tensor.float().sum()
        num_tokens = torch.tensor(
            output_tensor.numel(), dtype=torch.int, device=output_tensor.device
        )
        loss_reduced = {'lm loss': loss.detach().clone().view(1)}
        return loss, num_tokens, loss_reduced

    def fake_colocated_forward_step(data_iterator, model, packet=None, intra_packet=None):
        chunk = model[0] if isinstance(model, (list, tuple)) else model
        if isinstance(chunk, MockEncoder):
            tokens, labels, imgs, num_tiles = next(data_iterator)
            served.append(tokens.shape[0])
            image_embeddings = encoder(imgs)  # [1, merged_batch, h_lang]，保留 grad_fn
            return MergedEncoderBatch(
                image_embeddings=image_embeddings,
                tokens=tokens,
                labels=labels,
                num_image_tiles=num_tiles,
            ), None
        if isinstance(chunk, FakeBackbone):
            if chunk.pre_process:
                assert packet is not None, (
                    "consumer forward step needs the packet bound by the schedule (4.3b)"
                )
                output, loss_mask = chunk(
                    image_embeddings=packet.image_embeddings,
                    input_ids=packet.tokens,
                    labels=packet.labels,
                    num_image_tiles=packet.num_image_tiles,
                )
                # 4.3j：consumer 写回展开 labels/loss_mask 供伴随传输（fake 从输出
                # shape 派生，与接收端推导一致——activation [s',b,h] → [b,s']，
                # 避免 NCCL P2P 形状不匹配卡死）。与业务层一致：intra_packet 是强制
                # 契约（schedule 对 consumer 总是绑定）。
                assert intra_packet is not None, (
                    "consumer forward step needs the intra_packet bound by the schedule (4.3j)"
                )
                s_prime, batch = output.shape[0], output.shape[1]
                intra_packet.labels = torch.zeros(
                    (batch, s_prime), dtype=torch.int64, device=output.device
                )
                intra_packet.loss_mask = torch.ones(
                    (batch, s_prime), dtype=torch.float32, device=output.device
                )
            else:
                # 4.3j：非 consumer 用 intra_packet 闭包绑定传入的 labels/loss_mask
                #（伴随 recv 的结果，last stage 算 loss；中间 stage 无损失计算）。
                assert intra_packet is not None, (
                    "non-consumer forward step needs the intra_packet bound by the "
                    "schedule (4.3j)"
                )
                output, loss_mask = chunk(
                    labels=intra_packet.labels, loss_mask=intra_packet.loss_mask
                )
            return output, partial(fake_loss_func, loss_mask)
        raise TypeError(f"unexpected chunk type {type(chunk)}")

    return fake_colocated_forward_step


def _producer_identity():
    """Return this rank's (producer_id, num_producers) taken from the boundary group.

    与生产代码同源：producer 槽位是共置边界组内的编号（组内 rank），不是 pipeline rank，
    因此在任意 rank order 下都成立（Task 5.7）。
    """
    boundary_group = ps.get_colocated_boundary_group()
    producer_id = torch.distributed.get_group_rank(boundary_group, torch.distributed.get_rank())
    return producer_id, boundary_group.size()


def _my_microbatches_setup(world, num_microbatches=None):
    """Init parallel state (TP=1, PP=world, colocated) and return per-rank microbatches + batches.

    ``num_microbatches`` 默认 ``2*world``（round_robin 负载验证每 producer 2 个）；wiring
    测试传 ``world``（=P，每 producer 1 个包）做最小配对冒烟。**n/P>1 的完整节奏**
    （4.4 producer 启动+补货、4.5 consumer 逐 step prefetch、4.6d 的梯度 ①③）由
    ``test_forward_backward_colocated_multi_microbatch_per_producer`` 覆盖——n=P 时
    ``current_microbatch - group_size < 0``，这些分支全是 no-op（4.6f）。
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=world,
        use_colocated_encoder=True,
    )
    # Task 5.12: 生产里 encoder 的 RNG tracker 由 get_colocated_model 在 encoder 构建完
    # 立刻快照，这些测试直接调 forward_backward_colocated、绕过了它，所以这里等价地快照
    # 一次，否则 _colocated_encoder_forward 的 colocated_encoder_rng_tracker() 会断言失败。
    # The encoder's tracker is snapshotted by get_colocated_model in production; these
    # tests call the schedule directly, so snapshot it here to mirror that precondition.
    model_parallel_cuda_manual_seed(123)
    snapshot_colocated_encoder_rng_tracker()
    producer_id, num_producers = _producer_identity()
    num_microbatches = 2 * world if num_microbatches is None else num_microbatches
    my_microbatches = ps.get_microbatches_for_producer(
        producer_id, num_microbatches, num_producers
    )
    # 合并批：本 rank 全部 micro batch 的样本沿 batch 维堆叠（batch 顺序 == 轮盘序列），
    # 每个 microbatch 1 个样本（MBS=1）。优化 spec Task 1：dataloader 一次取回合并批。
    per_mb = [
        _make_fake_merged_batch(1000 + microbatch, 1) for microbatch in my_microbatches
    ]
    merged_batch = tuple(
        torch.cat([fields[index] for fields in per_mb], dim=0) for index in range(4)
    )
    return num_microbatches, my_microbatches, merged_batch


def test_colocated_encoder_forward_round_robin_buffer():
    """Phase ① helper: ONE merged forward, split back into round-robin packets (Task 1).

    优化 spec Task 1（设计 A）：每个 producer（边界组内槽位）用 get_microbatches_for_producer
    拿自己负责的 microbatch；schedule 对 forward step 的 encoder 分支**只调一次**（一次
    取合并批 + 一次前传），经 ``ForwardPacket.split_merged_batch`` 等分回逐 microbatch
    的包并打 id 存 buffer。断言：buffer key = 轮盘序列、5 字段齐全、值与合并批的对应
    切片一致、切片是 view 但保留 grad_fn（phase ④ 对合并张量单次 backward）、合并张量
    由本函数返回（schedule 显式持有）。
    """
    world = Utils.world_size
    num_microbatches, my_microbatches, merged_batch = _my_microbatches_setup(world)
    try:
        encoder = MockEncoder().cuda()
        served = []
        # backbone 参数传 None：本测试只调 encoder 分支（phase ①）。
        fake_forward_step = _make_fake_colocated_forward_step(
            merged_batch, encoder, None, served
        )
        producer_id, num_producers = _producer_identity()
        encoder_buffers, merged_image_embeddings = _colocated_encoder_forward(
            fake_forward_step,
            iter([merged_batch]),
            encoder,
            num_microbatches,
            producer_id=producer_id,
            num_producers=num_producers,
        )

        # --- 合并调用：encoder 分支每个 iteration 恰好一次，吃掉整个合并批 ---
        assert served == [len(my_microbatches)], (
            f"phase ① must call the encoder branch ONCE with the merged batch "
            f"({len(my_microbatches)} samples), got {served}"
        )

        # --- 分配公式：每 producer 恰好 num_microbatches / P 个，key = 轮盘序列 ---
        # Each producer handles exactly num_microbatches / P microbatches (global_mbs/(dp*inner_dp)).
        assert sorted(encoder_buffers) == my_microbatches, (
            "buffer keys must equal the round-robin microbatches"
        )
        assert len(encoder_buffers) == num_microbatches // world, (
            f"per-producer load must be num_microbatches/pp = {num_microbatches // world}, "
            f"got {len(encoder_buffers)}"
        )

        # --- 合并张量：形状 = [1, N, h_lang]、保留 grad_fn（phase ④ 单次 backward 的对象）---
        tokens_m, labels_m, imgs_m, tiles_m = merged_batch
        assert merged_image_embeddings.shape == (1, len(my_microbatches), _H_LANG), (
            f"merged image_embeddings shape {tuple(merged_image_embeddings.shape)}"
        )
        assert merged_image_embeddings.requires_grad, (
            "the merged tensor is the graph output; phase ④ backwards through it"
        )

        # --- buffer 内容：5 字段齐全、值 = 合并批对应切片、id 已打标、view 保 grad_fn ---
        tokens, labels, imgs, num_tiles = merged_batch
        for index, microbatch in enumerate(my_microbatches):
            pkt = encoder_buffers[microbatch]
            assert set(pkt.to_dict()) == set(ForwardPacket.field_names), (
                f"microbatch {microbatch} packet keys {sorted(pkt.to_dict())}"
            )
            # schedule 给包打上 microbatch id 字段（1 元素 int64 张量，serialize 时发送）。
            assert pkt.microbatch_id.item() == microbatch, (
                f"microbatch {microbatch}: schedule must stamp the packet's microbatch id"
            )
            # 切片顺序 == batch 顺序 == 轮盘序列：每个包必须等于合并张量的第 index 切片，
            # 且与合并张量**共享 storage**（view 语义——phase ④ 只 backward 合并张量一次）。
            merged_slice = merged_image_embeddings[:, index : index + 1, :]
            assert torch.equal(pkt.image_embeddings, merged_slice), (
                f"microbatch {microbatch} embeddings must equal merged slice {index}"
            )
            assert (
                pkt.image_embeddings.untyped_storage().data_ptr()
                == merged_image_embeddings.untyped_storage().data_ptr()
            ), (
                f"microbatch {microbatch}: the packet's image_embeddings must be a VIEW "
                "sharing storage with the merged tensor"
            )
            assert pkt.image_embeddings.requires_grad, (
                f"microbatch {microbatch}: the slice is a view of the merged tensor and "
                "must keep grad_fn"
            )
            assert torch.equal(pkt.tokens, tokens[index : index + 1]), (
                f"microbatch {microbatch} tokens must equal merged slice {index}"
            )
            assert torch.equal(pkt.labels, labels[index : index + 1]), (
                f"microbatch {microbatch} labels must equal merged slice {index}"
            )
            assert torch.equal(pkt.num_image_tiles, num_tiles[index : index + 1]), (
                f"microbatch {microbatch} num_tiles must equal merged slice {index}"
            )

        # --- 跨 rank：所有 producer 的 microbatch 恰好覆盖 0..n-1 一次（不重不漏） ---
        all_keys = [None] * world
        dist.all_gather_object(all_keys, sorted(encoder_buffers))
        flat = [k for keys in all_keys for k in keys]
        assert len(flat) == len(set(flat)), "microbatches must be disjoint across producers"
        assert sorted(flat) == list(range(num_microbatches)), (
            "every microbatch covered exactly once"
        )
    finally:
        Utils.destroy_model_parallel()


def test_forward_backward_colocated_wiring():
    """Main function: run phase ①+② end-to-end (Task 4.3f).

    主函数完整跑 forward_backward_colocated——phase ①（encoder 轮盘前传 + 本地 buffer）
    → phase ②（backbone 自写循环：PP=1 全本地 take / PP>1 边界通信配对）。断言：
    - 返回 forward_data_store（loss 列表，非空）；
    - forward step 的 encoder 分支恰好被调 num_microbatches/P 次；
    - num_microbatches 非 pp 倍数在校验处抛 AssertionError。
    """
    world = Utils.world_size
    # n = P（每 producer 1 个包）：4.3b 全量发送在 n/P>1 时死锁（NCCL 未配对 send 阻塞
    # 同组后续 isend），冒烟用 1 发 1 收配对（doc §2.11）。
    num_microbatches, my_microbatches, merged_batch = _my_microbatches_setup(
        world, num_microbatches=world
    )
    try:
        encoder = MockEncoder().cuda()
        # stage 0（consumer）是 backbone 的 pre_process=True；PP=1 时唯一 rank 也是。
        backbone = FakeBackbone(pre_process=(ps.get_pipeline_model_parallel_rank() == 0)).cuda()
        served = []
        fake_forward_step = _make_fake_colocated_forward_step(
            merged_batch, encoder, backbone, served
        )
        loss_store = forward_backward_colocated(
            forward_step_func=fake_forward_step,
            data_iterator=iter([merged_batch]),
            model=[encoder, backbone],
            num_microbatches=num_microbatches,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
        )
        # phase ①+② 跑通：返回 forward_data_store（list，末 stage 存 loss_reduced）。
        assert isinstance(loss_store, list), (
            f"expected forward_data_store (list), got {type(loss_store)}"
        )
        # phase ① executed: the encoder branch is called ONCE with the whole merged batch.
        assert served == [num_microbatches // world], (
            f"phase ① must call the encoder branch once with {num_microbatches // world} "
            f"merged samples, got {served}"
        )

        # num_microbatches 不是 pp_size 的整数倍 -> 校验处抛错（world=1 时恒整除，跳过）。
        if world > 1:
            with pytest.raises(AssertionError):
                forward_backward_colocated(
                    forward_step_func=fake_forward_step,
                    data_iterator=iter([merged_batch]),
                    model=[encoder, backbone],
                    num_microbatches=world + 1,  # 不是 pp_size 的整数倍
                    seq_length=_SEQ_LENGTH,
                    micro_batch_size=_MICRO_BATCH_SIZE,
                )
    finally:
        Utils.destroy_model_parallel()


def test_forward_backward_colocated_multi_microbatch_per_producer():
    """n = 2*P: exercise restock, per-step prefetch and boundary-grad round trip (Task 4.6f).

    n=P 的冒烟走不到 4.4/4.5/4.6 的核心分支（``current_microbatch - group_size < 0``
    时补货、逐 step prefetch、梯度 ①③ 全是 no-op）。本用例用 **n = 2*P**（每 producer
    2 个 microbatch）跑完整 phase ①→②→④，覆盖：
    - 4.4 producer 启动发第 1 包 + 补货 step 异步发第 2 包；
    - 4.5 consumer 流水前全量 prefetch + 循环内提前一整步 prefetch；
    - 4.6c consumer 反传后 ``.grad`` 派发（本地留存 / 按 producer 分桶 isend）；
    - 4.6d producer 补货 step 的梯度 ①③ + cooldown 补收最后一个 owned microbatch；
    - 4.6e phase ④ 统一 encoder 反传（断言 encoder 参数拿到梯度）。
    schedule 内部的两条收尾断言（``pending_grad_requests`` 已空、
    ``producer_grad_buffers`` 键 == ``encoder_buffers`` 键）会在运行中自检。
    """
    world = Utils.world_size
    num_microbatches, my_microbatches, merged_batch = _my_microbatches_setup(
        world, num_microbatches=2 * world
    )
    try:
        encoder = MockEncoder().cuda()
        backbone = FakeBackbone(pre_process=(ps.get_pipeline_model_parallel_rank() == 0)).cuda()
        served = []
        fake_forward_step = _make_fake_colocated_forward_step(
            merged_batch, encoder, backbone, served
        )
        loss_store = forward_backward_colocated(
            forward_step_func=fake_forward_step,
            data_iterator=iter([merged_batch]),
            model=[encoder, backbone],
            num_microbatches=num_microbatches,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
        )
        assert isinstance(loss_store, list), (
            f"expected forward_data_store (list), got {type(loss_store)}"
        )
        # phase ① 合并调用：encoder 分支只被调一次，合并批样本数 = 每 producer 的 mb 数。
        assert served == [num_microbatches // world], (
            f"phase ① must serve ONE merged call with {num_microbatches // world} "
            f"samples, got {served}"
        )
        # phase ④ 真的跑了：encoder 参数拿到梯度（裸模型，无 DDP —— 4.6e 跳过 no_sync/
        # finalize，直接 autograd.backward 到参数上）。
        assert encoder.proj.weight.grad is not None, (
            "phase ④ must backward the encoder with the boundary grads (4.6e)"
        )
        assert torch.isfinite(encoder.proj.weight.grad).all(), (
            "encoder grad must be finite after the phase-④ backward"
        )
        # backbone 也反传过（1F1B 的 backward_step）。
        assert backbone.proj.weight.grad is not None, (
            "backbone params must receive grads from the 1F1B backward"
        )
    finally:
        Utils.destroy_model_parallel()


def _run_multi_microbatch_once(
    deallocate_encoder_outputs: bool, deallocate_pipeline_outputs: bool = False
) -> torch.Tensor:
    """Run the n=2P case once and return the encoder's grad (Task 4.9 helper).

    权重与数据都确定性（``torch.manual_seed`` + ``_my_microbatches_setup`` 的固定种子），
    所以不同开关组合的多次运行可以逐元素比较。两个开关分别控制**边界** encoder 输出的
    伪释放（4.9）与 **backbone PP 激活**的伪释放（原框架 `deallocate_pipeline_outputs`）；
    生产入口把两者都硬编码为 True（arguments.py），所以"同时开"这个组合必须被覆盖。
    """
    world = Utils.world_size
    num_microbatches, my_microbatches, merged_batch = _my_microbatches_setup(world)
    try:
        torch.manual_seed(20260827)
        encoder = MockEncoder().cuda()
        backbone = FakeBackbone(pre_process=(ps.get_pipeline_model_parallel_rank() == 0)).cuda()
        backbone.config.deallocate_encoder_outputs = deallocate_encoder_outputs
        backbone.config.deallocate_pipeline_outputs = deallocate_pipeline_outputs
        served = []
        fake_forward_step = _make_fake_colocated_forward_step(
            merged_batch, encoder, backbone, served
        )
        forward_backward_colocated(
            forward_step_func=fake_forward_step,
            data_iterator=iter([merged_batch]),
            model=[encoder, backbone],
            num_microbatches=num_microbatches,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
        )
        assert encoder.proj.weight.grad is not None, "phase ④ 必须给 encoder 累积梯度"
        return encoder.proj.weight.grad.detach().clone()
    finally:
        Utils.destroy_model_parallel()


def test_colocated_deallocate_encoder_outputs_keeps_grads():
    """两个 deallocate 开关都不改变 encoder 梯度（Task 4.9；Task 1 后 encoder 侧失效）。

    优化 spec Task 1（合并前传）起，encoder 侧伪释放**结构性失效**：
    ``_deallocate_encoder_output`` 已改为显式 no-op——各包的 ``image_embeddings`` 是
    合并张量的 view，伪释放释放不了 base 的存储，且合并张量本就要活到 phase ④ 的
    单次 backward。本用例降级为**回归守卫**：开关组合不得改变梯度。
    backbone 侧的 ``deallocate_pipeline_outputs``（PP 激活伪释放、``backward_step``
    走 ``custom_backward``）仍然真实生效——生产入口把两个开关都硬编码为 True
    （arguments.py），该组合必须实测。
    world=1 时 P=1、没有网络发送路径，跳过。
    """
    if Utils.world_size == 1:
        pytest.skip("Task 4.9 只在有 producer 网络发送路径时生效（world_size > 1）")
    baseline_grad = _run_multi_microbatch_once(deallocate_encoder_outputs=False)
    encoder_only_grad = _run_multi_microbatch_once(deallocate_encoder_outputs=True)
    torch.testing.assert_close(
        encoder_only_grad,
        baseline_grad,
        msg="伪释放 encoder 输出后梯度发生了变化（4.9 只应释放数据、保留图）",
    )
    both_grad = _run_multi_microbatch_once(
        deallocate_encoder_outputs=True, deallocate_pipeline_outputs=True
    )
    torch.testing.assert_close(
        both_grad,
        baseline_grad,
        msg="encoder 与 backbone PP 两处伪释放同时开启后梯度发生了变化（生产默认配置）",
    )


def test_get_forward_backward_func_colocated_dispatch():
    """get_forward_backward_func returns the colocated schedule when enabled (Task 4.7).

    共置启用时 ``get_forward_backward_func`` 必须返回 ``forward_backward_colocated``
    （train_step 零改动接入）；关掉共置后回到标准分支——共置布局下 pp_size 恒 > 1，
    分支顺序错了就会被标准 1F1B 截走。
    """
    from megatron.core.pipeline_parallel.schedules import (
        forward_backward_pipelining_without_interleaving,
        get_forward_backward_func,
    )

    world = Utils.world_size
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=world,
            use_colocated_encoder=True,
        )
        assert ps.is_colocated_encoder_enabled(), (
            "colocated groups must exist after initialize_model_parallel(use_colocated_encoder)"
        )
        assert get_forward_backward_func() is forward_backward_colocated, (
            "colocated training must dispatch to forward_backward_colocated"
        )
    finally:
        Utils.destroy_model_parallel()

    # 非共置：同样的 PP 大小下回到标准 schedule（world=1 时是 no_pipelining，跳过断言）。
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=world
        )
        assert not ps.is_colocated_encoder_enabled(), (
            "colocated groups must not exist without use_colocated_encoder"
        )
        if world > 1:
            assert (
                get_forward_backward_func() is forward_backward_pipelining_without_interleaving
            ), "non-colocated PP>1 must keep dispatching to the standard 1F1B schedule"
    finally:
        Utils.destroy_model_parallel()

