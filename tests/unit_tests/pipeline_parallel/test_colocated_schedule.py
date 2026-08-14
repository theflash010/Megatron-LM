# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the colocated schedule wrapper, phase ① (Task 4.1/4.2).

共置 schedule（顶层函数 ``forward_backward_colocated``）的 phase ①
（encoder 轮盘前传 + 本地 buffer）单元测试：
- 每个 producer（pp stage）负责的 microbatch = p, p+P, p+2P, ...（``get_microbatches_for_pipeline_stage``），
  即每 producer 处理 ``num_microbatches / P`` 个 microbatch = ``global_mbs / (dp * inner_dp)``；
- phase ① 由 schedule 循环调 ``forward_step_func`` 的 **encoder 分支**（模拟
  colocated_train.colocated_forward_step：取数据 + encoder forward -> 5 字段包）拿包
  存 buffer——schedule 不接收 get_batch_fn/image_token_index/img_seq_len（4.2 职责分工）；
- buffer 每项 = 前向包 5 字段；``image_embeddings`` 保留 grad_fn（phase ④ 统一反传用，
  分离图发生在发送/组装时）；
- 跨 rank：所有 producer 的 microbatch 恰好覆盖 0..num_microbatches-1 一次（不重不漏）；
- 主函数 wiring（4.3f）：完整跑 phase ①+②（PP=1 全本地 / PP>1 边界配对），返回
  forward_data_store；forward step encoder 分支每 microbatch 恰一次、num_microbatches
  非 pp 倍数在校验处抛 AssertionError。

运行（CI 标准方式）：
    torchrun --nproc_per_node=N -m pytest tests/unit_tests/pipeline_parallel/test_colocated_schedule.py
"""

import pytest
import torch
import torch.distributed as dist
from functools import partial

import megatron.core.parallel_state as ps
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.colocated_encoder_comm import ForwardPacket
from megatron.core.pipeline_parallel.colocated_schedule import (
    _colocated_encoder_forward,
    forward_backward_colocated,
)
from tests.unit_tests.test_utilities import Utils

_IMG_H, _IMG_W, _H_LANG = 4, 4, 8


class MockEncoder(torch.nn.Module):
    """Minimal encoder-only chunk: images [num_tiles, 3, h, w] -> [1, num_tiles, h_lang].

    最小 encoder chunk 桩：给真实 grad_fn（Linear 输出），模仿 ColocatedViTEncoder
    的 forward 契约（无图样本返回空 tensor）。
    """

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3 * _IMG_H * _IMG_W, _H_LANG)

    def forward(self, images):
        if images.shape[0] == 0:
            return torch.tensor([], dtype=images.dtype, device=images.device).reshape(0, 0, 0)
        out = self.proj(images.flatten(1))  # [num_tiles, h_lang]
        return out.unsqueeze(0).contiguous()  # [1, num_tiles, h_lang]


class FakeBackbone(torch.nn.Module):
    """Minimal backbone chunk for running phase ② (Task 4.3f).

    ``pre_process`` 标记 consumer（stage 0）；``set_input_tensor`` 存 P2P 激活（非
    consumer 用）；``forward`` 做线性输出（带 grad_fn，backward_step 需要）——consumer
    用 partial 绑定的 packet 的 image_embeddings，非 consumer 用 input_tensor。返回
    ``(output, loss_mask)``（数值无所谓，冒烟只验证通信配对与无死锁）。
    """

    def __init__(self, pre_process=True):
        super().__init__()
        self.config = ModelParallelConfig(pipeline_dtype=torch.bfloat16)
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
        self.proj = torch.nn.Linear(_H_LANG, _H_LANG)

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

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
            x = torch.zeros(1, 1, _H_LANG, dtype=torch.bfloat16, device="cuda")
        return self.proj(x.reshape(-1, _H_LANG)).unsqueeze(0).contiguous(), loss_mask


def _make_fake_batch(seed, num_tiles, seq=10):
    """Deterministic batch 4-tuple (colocated_encoder_get_batch contract).

    与 ``colocated_train.colocated_encoder_get_batch`` 一致（4 元组
    tokens/labels/imgs/num_tiles）；loss_mask 由 consumer 从 labels 重建，不在 batch/包里
    （2026-08-13 用户确认）。
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    imgs = torch.randn((num_tiles, 3, _IMG_H, _IMG_W), dtype=torch.float32, device="cuda", generator=g)
    tokens = torch.randint(0, 100, (1, seq), dtype=torch.int64, device="cuda", generator=g)
    labels = torch.randint(0, 100, (1, seq), dtype=torch.int64, device="cuda", generator=g)
    num_tiles_t = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    return (tokens, labels, imgs, num_tiles_t)


def _make_fake_colocated_forward_step(batches, my_microbatches, encoder, backbone, served):
    """Fake 'colocated_forward_step': (data_iterator, model, packet=None), branched by chunk.

    模拟 colocated_train.colocated_forward_step（4.3f）：
    - encoder 分支（``model=[encoder]`` list，phase ①）：取一个 micro batch 数据 +
      ``encoder_chunk(images)`` -> 4 字段 ForwardPacket（schedule 打标 id 后存 buffer）；
    - backbone 分支（``model=backbone`` 单 chunk 或 list，phase ②）：consumer
      （``pre_process``）用 **partial 绑定的 packet** 的 image_embeddings 跑 FakeBackbone；
      非 consumer 直接 ``chunk()``（input_tensor 已 set_input_tensor）。返回
      ``(output, loss_func)``。
    """

    def fake_loss_func(loss_mask, output_tensor):
        # per-token 型：返回 (loss, num_tokens, loss_reduced)，见 forward_step_calc_loss。
        loss = output_tensor.float().sum()
        num_tokens = torch.tensor(
            output_tensor.numel(), dtype=torch.int, device=output_tensor.device
        )
        loss_reduced = {'lm loss': loss.detach().clone().view(1)}
        return loss, num_tokens, loss_reduced

    def fake_colocated_forward_step(data_iterator, model, packet=None):
        chunk = model[0] if isinstance(model, (list, tuple)) else model
        if isinstance(chunk, MockEncoder):
            microbatch = my_microbatches[len(served)]
            served.append(microbatch)
            tokens, labels, imgs, num_tiles = batches[microbatch]
            image_embeddings = encoder(imgs)
            return ForwardPacket(
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
            else:
                output, loss_mask = chunk()
            return output, partial(fake_loss_func, loss_mask)
        raise TypeError(f"unexpected chunk type {type(chunk)}")

    return fake_colocated_forward_step


def _my_microbatches_setup(world, num_microbatches=None):
    """Init parallel state (TP=1, PP=world, colocated) and return per-rank microbatches + batches.

    ``num_microbatches`` 默认 ``2*world``（round_robin 负载验证每 producer 2 个）；wiring
    测试传 ``world``（=P，每 producer 1 个包）——**4.3b 的全量异步发送在 n/P>1 时死锁**
    （NCCL 同组 P2P 中未配对的 send 会阻塞同组后续 isend，producer 连发多包时第 2 个包
    阻塞、进不了 backbone recv，见 doc §2.11"last stage labels"备注 / 4.4 replenish 解决）；
    冒烟用 n=P 验证 1 发 1 收配对与无死锁。
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=world,
        use_colocated_encoder=True,
    )
    pp_stage = ps.get_pipeline_model_parallel_rank()
    num_microbatches = 2 * world if num_microbatches is None else num_microbatches
    my_microbatches = ps.get_microbatches_for_pipeline_stage(pp_stage, num_microbatches)
    # 本 rank 的 batch（按 my_microbatches 顺序），num_tiles 随 microbatch 变化（microbatch % 3 + 1）。
    batches = {
        microbatch: _make_fake_batch(1000 + microbatch, num_tiles=microbatch % 3 + 1)
        for microbatch in my_microbatches
    }
    return num_microbatches, my_microbatches, batches


def test_colocated_encoder_forward_round_robin_buffer():
    """Phase ① helper: every producer computes its round-robin microbatches into the buffer.

    每个 producer（pp stage）用 get_microbatches_for_pipeline_stage 拿自己负责的
    microbatch；schedule 循环调 forward step 的 encoder 分支拿包存 buffer（buffer
    保留 grad_fn、覆盖全部 microbatch 一次；数据/模型细节在 forward step 里）。
    """
    world = Utils.world_size
    num_microbatches, my_microbatches, batches = _my_microbatches_setup(world)
    try:
        encoder = MockEncoder().cuda()
        served = []
        # backbone 参数传 None：本测试只调 encoder 分支（phase ①）。
        fake_forward_step = _make_fake_colocated_forward_step(
            batches, my_microbatches, encoder, None, served
        )
        encoder_buffers = _colocated_encoder_forward(
            fake_forward_step, object(), encoder, num_microbatches
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

        # --- buffer 内容：5 字段齐全、值与 batch/encoder 一致、grad_fn 保留 ---
        for microbatch in my_microbatches:
            pkt = encoder_buffers[microbatch]
            assert set(pkt.to_dict()) == set(ForwardPacket.field_names), (
                f"microbatch {microbatch} packet keys {sorted(pkt.to_dict())}"
            )
            # schedule 给包打上 microbatch id 字段（1 元素 int64 张量，serialize 时发送）。
            assert pkt.microbatch_id.item() == microbatch, (
                f"microbatch {microbatch}: schedule must stamp the packet's microbatch id"
            )
            t, l, imgs, nt = batches[microbatch]
            assert torch.equal(pkt.image_embeddings, encoder(imgs)), (
                f"microbatch {microbatch} embeddings"
            )
            assert pkt.image_embeddings.requires_grad, (
                f"microbatch {microbatch}: image_embeddings must keep grad_fn for the "
                "phase-④ backward"
            )
            assert torch.equal(pkt.tokens, t) and torch.equal(pkt.labels, l)
            assert torch.equal(pkt.num_image_tiles, nt)
        assert len(served) == num_microbatches // world, (
            "forward step encoder branch called once per owned microbatch"
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
    num_microbatches, my_microbatches, batches = _my_microbatches_setup(
        world, num_microbatches=world
    )
    try:
        encoder = MockEncoder().cuda()
        # stage 0（consumer）是 backbone 的 pre_process=True；PP=1 时唯一 rank 也是。
        backbone = FakeBackbone(pre_process=(ps.get_pipeline_model_parallel_rank() == 0)).cuda()
        served = []
        fake_forward_step = _make_fake_colocated_forward_step(
            batches, my_microbatches, encoder, backbone, served
        )
        loss_store = forward_backward_colocated(
            forward_step_func=fake_forward_step,
            data_iterator=object(),
            model=[encoder, backbone],
            num_microbatches=num_microbatches,
            seq_length=10,
            micro_batch_size=1,
        )
        # phase ①+② 跑通：返回 forward_data_store（list，末 stage 存 loss_reduced）。
        assert isinstance(loss_store, list), (
            f"expected forward_data_store (list), got {type(loss_store)}"
        )
        # phase ① executed: forward step encoder branch called once per owned microbatch.
        assert len(served) == num_microbatches // world, (
            f"phase ① must call the forward step encoder branch once per owned "
            f"microbatch, got {len(served)}"
        )

        # num_microbatches 不是 pp_size 的整数倍 -> 校验处抛错（world=1 时恒整除，跳过）。
        if world > 1:
            with pytest.raises(AssertionError):
                forward_backward_colocated(
                    forward_step_func=fake_forward_step,
                    data_iterator=object(),
                    model=[encoder, backbone],
                    num_microbatches=world + 1,  # 不是 pp_size 的整数倍
                    seq_length=10,
                    micro_batch_size=1,
                )
    finally:
        Utils.destroy_model_parallel()
