# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Task 4.8: numeric parity of the colocated schedule against the pp0-only path.

world=1 / PP=1 单进程数值对齐：同一批数据、同一份初始权重，分别跑

- 参考路径：原始 ``LLaVAModel`` + ``forward_backward_no_pipelining``；
- 共置路径：``ColocatedViTEncoder`` + ``ColocatedGPTBackbone`` +
  ``forward_backward_colocated``（phase ① 轮盘前传 → phase ② 自写 1F1B（PP=1 全本地
  直传、P2P 空转）→ phase ④ 统一 encoder 反传）。

比 loss 与两侧全部参数的梯度。**裸模型直接读 ``tensor.grad``**，不套 DDP、不依赖 dp
组——Task 5 的两层归约被隔离在外，本测试只回答一个问题：拆成两个 chunk 加自写调度，
有没有改变数学。

与 ``tests/unit_tests/models/test_colocated_llava_model.py``（Task 2.6）的区别：那里比的是
**模型**（直接调 forward/backward），这里比的是**调度**——边界切断（4.6b）、梯度派发
（4.6c）、统一反传（4.6e）与梯度累积（n>1）都在链路里。
"""

from functools import partial

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.colocated_llava_model import (
    ColocatedGPTBackbone,
    ColocatedViTEncoder,
)
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, LLaVAModel
from megatron.core.pipeline_parallel.colocated_encoder_comm import MergedEncoderBatch
from megatron.core.pipeline_parallel.colocated_schedule import forward_backward_colocated
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
    snapshot_colocated_encoder_rng_tracker,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

_IMAGE_TOKEN_INDEX = -200  # DEFAULT_IMAGE_TOKEN_INDEX
_IMAGE_SEQ_LEN = 577  # (336/14)^2 + 1 class token（drop_vision_class_token=False）
_NUM_MICROBATCHES = 2  # >1 才能覆盖梯度累积；P=1 时两个 microbatch 都归本 rank
_MICRO_BATCH_SIZE = 2
_SEQ_LENGTH = 5


def _build_configs():
    """Small language / vision / projection configs (mirrors test_colocated_llava_model.py)."""
    language_config = TransformerConfig(
        num_layers=3, hidden_size=64, num_attention_heads=4, use_cpu_initialization=False
    )
    vision_config = TransformerConfig(
        num_layers=2, hidden_size=16, num_attention_heads=2, use_cpu_initialization=False
    )
    vision_projection_config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=32,
        num_attention_heads=1,
        use_cpu_initialization=False,
    )
    language_config.language_model_type = "dummy"
    vision_config.vision_model_type = "clip"
    # 边界通信器按 config.pipeline_dtype 反推字节数（PP=1 时不发包，但构造时会读它）。
    # The boundary communicator derives byte lengths from config.pipeline_dtype.
    language_config.pipeline_dtype = torch.float32
    # 两个伪释放开关都打开，与生产入口一致（arguments.py 把 deallocate_pipeline_outputs 与
    # deallocate_encoder_outputs 都硬编码为 True）——这样这份"loss + 全部参数梯度"的对齐
    # 就同时在检验：backbone 1F1B 把真实 TE stage 的 output_tensor 换成空壳后（backward 走
    # custom_backward）梯度是否仍然一致。参考路径 forward_backward_no_pipelining 不做任何
    # 伪释放，所以两边天然构成对照。
    # Enable both pseudo-deallocation switches to match the production entry point; the
    # reference path (forward_backward_no_pipelining) never deallocates, so the comparison
    # also validates that shelling real TE stage outputs leaves gradients unchanged.
    language_config.deallocate_pipeline_outputs = True
    language_config.deallocate_encoder_outputs = True
    return language_config, vision_config, vision_projection_config


def _build_layer_specs():
    """Layer specs shared by both paths (TE submodules, same as the existing llava tests)."""
    from copy import deepcopy

    submodules = get_gpt_layer_with_transformer_engine_submodules()
    language_spec = ModuleSpec(module=TransformerLayer, submodules=submodules)
    vision_spec = ModuleSpec(module=TransformerLayer, submodules=deepcopy(submodules))
    vision_projection_spec = deepcopy(submodules.mlp.submodules)
    return language_spec, vision_spec, vision_projection_spec


def _make_microbatches():
    """Deterministic microbatches shared by both paths.

    tokens 含 image token 占位、labels 已按 IGNORE_INDEX 掩码（与 TaskEncoder 产出一致），
    每个样本 1 个 tile。position_ids / loss_mask 由两条路径各自按同一规则重建。
    """
    generator = torch.Generator(device="cuda").manual_seed(4008)
    microbatches = []
    for index in range(_NUM_MICROBATCHES):
        base_tokens = torch.tensor(
            [[101, 102, _IMAGE_TOKEN_INDEX, 103, 104], [201, _IMAGE_TOKEN_INDEX, 202, 203, 204]],
            device="cuda",
        )
        # 每个 microbatch 的文本 id 错开一位，但 **image token 占位不能动**——它要与
        # image_token_index 精确相等才会被组装替换，改了就会拿 -200 去查 embedding。
        # Shift the text ids per microbatch but never the image-token placeholder: it must
        # match image_token_index exactly or it is not replaced during assembly.
        tokens = base_tokens + torch.where(
            base_tokens == _IMAGE_TOKEN_INDEX, 0, index
        )
        labels = torch.tensor(
            [
                [11, IGNORE_INDEX, IGNORE_INDEX, 13, 14],
                [IGNORE_INDEX, IGNORE_INDEX, 22, 23, 24],
            ],
            device="cuda",
        )
        images = torch.randn(
            (_MICRO_BATCH_SIZE, 3, 336, 336), dtype=torch.float32, device="cuda",
            generator=generator,
        )
        num_image_tiles = torch.tensor([1, 1], dtype=torch.int, device="cuda")
        microbatches.append((tokens, labels, images, num_image_tiles))
    return microbatches


def _position_ids_and_loss_mask(labels):
    """Rebuild position_ids / loss_mask from labels — 两条路径用同一规则，保证可比。

    与 ``colocated_backbone_get_batch``（examples/multimodal/colocated_train.py）一致：
    loss_mask 由 labels 的 IGNORE_INDEX 掩码重建（4.3g 起 loss_mask 不随包传输）。
    """
    batch, seq_len = labels.shape
    position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=labels.device).unsqueeze(0).expand(batch, seq_len)
    )
    loss_mask = (labels != IGNORE_INDEX).float()
    return position_ids, loss_mask


def _loss_func(loss_mask, output_tensor):
    """per-token 型 loss_func（3 元组），两条路径共用。

    ``calculate_per_token_loss=False``（默认）时框架会 ``/= num_tokens`` 再
    ``/= num_microbatches``（schedules.py:296-299），两条路径经过同一段归一化，故可比。
    """
    losses = output_tensor.float()
    loss = torch.sum(losses.view(-1) * loss_mask.view(-1))
    num_tokens = loss_mask.sum().clone().detach().to(dtype=torch.int)
    return loss, num_tokens, {'lm loss': loss.detach().clone().view(1)}


def _make_reference_forward_step(microbatches):
    """forward_step_func for the pp0-only reference path (LLaVAModel)."""

    def reference_forward_step(data_iterator, model):
        tokens, labels, images, num_image_tiles = next(data_iterator)
        position_ids, loss_mask = _position_ids_and_loss_mask(labels)
        output, expanded_loss_mask = model(
            images=images,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )
        return output, partial(_loss_func, expanded_loss_mask)

    return reference_forward_step


def _make_merged_batch(microbatches):
    """Stack the per-microbatch list into ONE merged batch (optimization Task 1, design A).

    优化 spec Task 1（设计 A）：共置路径的 dataloader 一次取回整个合并批（batch 维 =
    ``num_microbatches × micro_batch_size`` 个样本），encoder 只前传一次。本 helper 把
    per-microbatch 列表沿 batch 维堆叠成**取数契约的 4 元组**
    （tokens/labels/images/num_image_tiles——与 ``colocated_encoder_get_batch`` 的返回
    一致）；num_tiles 逐样本为 1（``split_merged_batch`` 等分断言的前提）。encoder
    分支拿它构造 ``MergedEncoderBatch``（不含 images——那是 encoder 的输入）。
    """
    return tuple(
        torch.cat([fields[index] for fields in microbatches], dim=0) for index in range(4)
    )


def _make_colocated_forward_step(merged_batch, encoder, backbone):
    """forward_step_func for the colocated path (mirrors colocated_train.colocated_forward_step).

    优化 spec Task 1（设计 A）：encoder 分支（phase ①）**每个 iteration 只被调一次**，
    一次 ``next(data_iterator)`` 取回合并批 + 一次前传 -> ``(MergedEncoderBatch, None)``；
    backbone 分支（phase ②）用 schedule 绑定的 packet（拆分后的逐 microbatch 包）跑
    backbone，并把展开 labels/loss_mask 写回 ``intra_packet``（PP=1 时无人接收，但
    契约一致）。
    """
    encoder_calls = []

    def colocated_forward_step(data_iterator, model, packet=None, intra_packet=None):
        chunk = model[0] if isinstance(model, (list, tuple)) else model
        if chunk is encoder:
            tokens, labels, images, num_image_tiles = next(data_iterator)
            encoder_calls.append(tokens.shape[0])
            image_embeddings = encoder(images)
            return MergedEncoderBatch(
                image_embeddings=image_embeddings,
                tokens=tokens,
                labels=labels,
                num_image_tiles=num_image_tiles,
            ), None
        assert chunk is backbone, f"unexpected chunk {type(chunk)}"
        assert packet is not None, "the schedule must bind the packet for the consumer (4.3b)"
        position_ids, loss_mask = _position_ids_and_loss_mask(packet.labels)
        output, expanded_loss_mask, expanded_labels = chunk(
            image_embeddings=packet.image_embeddings,
            input_ids=packet.tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=packet.labels,
            loss_mask=loss_mask,
            num_image_tiles=packet.num_image_tiles,
        )
        assert intra_packet is not None, "the schedule must bind the intra_packet (4.3j)"
        intra_packet.labels = expanded_labels
        intra_packet.loss_mask = expanded_loss_mask
        return output, partial(_loss_func, expanded_loss_mask)

    return colocated_forward_step, encoder_calls


def _collect_grads(*modules):
    """name -> grad 的快照（裸模型，直接读 tensor.grad）。"""
    grads = {}
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is not None:
                grads[name] = parameter.grad.detach().clone()
    return grads


def test_colocated_schedule_matches_pp0_only():
    """Loss 与全部参数梯度必须与 pp0-only 路径一致（Task 4.8）。"""
    # 数值对齐只在单进程有意义（PP=1、无 dp 组、裸模型读 .grad）；多卡跑整个目录时跳过。
    # Parity is only meaningful in a single process; skip it in multi-GPU directory runs.
    if Utils.world_size != 1:
        pytest.skip(f"Task 4.8 parity test requires world_size == 1, got {Utils.world_size}")
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_colocated_encoder=True,
        )
        model_parallel_cuda_manual_seed(123)
        snapshot_colocated_encoder_rng_tracker()
        language_config, vision_config, vision_projection_config = _build_configs()
        language_spec, vision_spec, vision_projection_spec = _build_layer_specs()

        reference = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            pre_process=True,
            post_process=True,
            image_token_index=_IMAGE_TOKEN_INDEX,
            img_h=336,
            img_w=336,
            patch_dim=14,
        ).cuda()
        encoder = ColocatedViTEncoder(
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        ).cuda()
        backbone = ColocatedGPTBackbone(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            pre_process=True,
            post_process=True,
            image_token_index=_IMAGE_TOKEN_INDEX,
            img_seq_len=_IMAGE_SEQ_LEN,
        ).cuda()
        # 初始权重完全对齐（子模块命名两条路径一致：vision_model / vision_projection /
        # language_model），并 eval() 关掉 dropout 使两条路径确定性可比。
        encoder.vision_model.load_state_dict(reference.vision_model.state_dict())
        encoder.vision_projection.load_state_dict(reference.vision_projection.state_dict())
        backbone.language_model.load_state_dict(reference.language_model.state_dict())
        reference.eval()
        encoder.eval()
        backbone.eval()
        # 与 get_model 一致（training.py:1365/1390 给每个 chunk 打 model_type）：schedule 经
        # get_model_type 读它，裸模型手动构造时必须自己补上。LLaVA 用 encoder_or_decoder
        # （examples/multimodal/train.py:411）。
        reference.model_type = ModelType.encoder_or_decoder
        encoder.model_type = ModelType.encoder_or_decoder
        backbone.model_type = ModelType.encoder_or_decoder

        microbatches = _make_microbatches()

        # 参考路径：pp0-only（LLaVAModel + no_pipelining）。
        reference_losses = forward_backward_no_pipelining(
            forward_step_func=_make_reference_forward_step(microbatches),
            data_iterator=iter(microbatches),
            model=reference,
            num_microbatches=_NUM_MICROBATCHES,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
        )
        reference_grads = _collect_grads(reference)

        # 共置路径：phase ① → ② → ④（Task 1：encoder 分支一次吃合并批）。
        merged_batch = _make_merged_batch(microbatches)
        colocated_forward_step, encoder_calls = _make_colocated_forward_step(
            merged_batch, encoder, backbone
        )
        colocated_losses = forward_backward_colocated(
            forward_step_func=colocated_forward_step,
            data_iterator=iter([merged_batch]),
            model=[encoder, backbone],
            num_microbatches=_NUM_MICROBATCHES,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
        )
        colocated_grads = _collect_grads(encoder, backbone)

        # 优化 spec Task 1：encoder 分支整个 iteration 只被调一次（合并批样本数 =
        # num_microbatches × micro_batch_size）。
        assert len(encoder_calls) == 1, (
            f"phase ① 应只调一次 encoder（合并批），实际 {len(encoder_calls)}"
        )
        assert encoder_calls[0] == _NUM_MICROBATCHES * _MICRO_BATCH_SIZE, (
            f"合并批样本数应为 {_NUM_MICROBATCHES * _MICRO_BATCH_SIZE}，"
            f"实际 {encoder_calls[0]}"
        )
        assert len(colocated_losses) == len(reference_losses) == _NUM_MICROBATCHES, (
            f"两条路径的 forward_data_store 长度应相同："
            f"{len(colocated_losses)} vs {len(reference_losses)}"
        )
        for index, (colocated_entry, reference_entry) in enumerate(
            zip(colocated_losses, reference_losses)
        ):
            torch.testing.assert_close(
                colocated_entry['lm loss'],
                reference_entry['lm loss'],
                atol=1e-5,
                rtol=1e-4,
                msg=f"microbatch {index} 的 loss 不一致",
            )

        # 梯度：参数名两条路径同名同集合（vision_model / vision_projection / language_model）。
        assert set(colocated_grads) == set(reference_grads), (
            "两条路径拿到梯度的参数集合不同："
            f"仅共置有 {sorted(set(colocated_grads) - set(reference_grads))[:5]}，"
            f"仅参考有 {sorted(set(reference_grads) - set(colocated_grads))[:5]}"
        )
        for name in sorted(reference_grads):
            torch.testing.assert_close(
                colocated_grads[name],
                reference_grads[name],
                atol=1e-5,
                rtol=1e-4,
                msg=f"参数 {name} 的梯度不一致",
            )
    finally:
        Utils.destroy_model_parallel()
