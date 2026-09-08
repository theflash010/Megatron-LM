# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end parity tests: colocated encoder+backbone chunks vs the original LLaVAModel.

共置 encoder+backbone 两个 chunk 与原始 LLaVAModel 的端到端数值对齐测试：
用与 TaskEncoder 输出格式一致的模拟 batch（tokens 含 image token 占位、labels 已掩码、
imgs、num_tiles），走 get_batch 后处理（position_ids / loss_mask）→ encoder 前传 →
pp0 组装（_preprocess_data）→ GPTModel，对比原始 LLaVAModel 全流程的前向输出与梯度。
"""

import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, LLaVAModel
from megatron.core.models.multimodal.colocated_llava_model import (
    ColocatedGPTBackbone,
    ColocatedViTEncoder,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

IMG_TOK = -200  # image token placeholder id (DEFAULT_IMAGE_TOKEN_INDEX)
IMG_SEQ_LEN = 577  # (336/14)^2 + 1 class token, for the small CLIP test config


class TestColocatedLLaVAModel:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        language_config = TransformerConfig(
            num_layers=3, hidden_size=64, num_attention_heads=4, use_cpu_initialization=False
        )
        vision_config = TransformerConfig(
            num_layers=2, hidden_size=16, num_attention_heads=2, use_cpu_initialization=False
        )
        proj_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=False,
        )
        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = "clip"
        self.language_config = language_config
        self.vision_config = vision_config
        self.proj_config = proj_config

        sub = get_gpt_layer_with_transformer_engine_submodules()
        self.language_spec = ModuleSpec(module=TransformerLayer, submodules=sub)
        from copy import deepcopy

        self.vision_spec = ModuleSpec(module=TransformerLayer, submodules=deepcopy(sub))
        self.proj_spec = deepcopy(sub.mlp.submodules)

        self._build_models()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_models(self):
        # Reference: original LLaVAModel (PP=1 => pre/post both True).
        # 参考：原始 LLaVAModel（PP=1 => pre/post 都为 True）。
        self.ref = LLaVAModel(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=self.language_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            pre_process=True,
            post_process=True,
            image_token_index=IMG_TOK,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=self.vision_spec,
            drop_vision_class_token=False,
            vision_projection_config=self.proj_config,
            vision_projection_layer_spec=self.proj_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        ).cuda()

        # Colocated chunks; parameters aligned to the reference.
        # 共置两个 chunk；参数与参考对齐。
        self.enc = ColocatedViTEncoder(
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=self.vision_spec,
            drop_vision_class_token=False,
            vision_projection_config=self.proj_config,
            vision_projection_layer_spec=self.proj_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        ).cuda()
        self.bb = ColocatedGPTBackbone(
            language_transformer_config=self.language_config,
            language_transformer_layer_spec=self.language_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            pre_process=True,
            post_process=True,
            image_token_index=IMG_TOK,
            img_seq_len=IMG_SEQ_LEN,
        ).cuda()
        self.enc.vision_model.load_state_dict(self.ref.vision_model.state_dict())
        self.enc.vision_projection.load_state_dict(self.ref.vision_projection.state_dict())
        self.bb.language_model.load_state_dict(self.ref.language_model.state_dict())

        # eval() disables dropout so the two paths are deterministic and comparable.
        # eval() 关闭 dropout，使两条路径确定性可比。
        self.ref.eval()
        self.enc.eval()
        self.bb.eval()

    def _make_batch(self):
        """Simulate a TaskEncoder-produced batch plus get_batch post-processing.

        模拟 TaskEncoder 产出的 batch（tokens 含 image token 占位、labels 已掩码、
        imgs、num_tiles），以及 get_batch 的后续处理（position_ids、loss_mask）。
        """
        tokens = torch.tensor(
            [[101, 102, IMG_TOK, 103, 104], [201, IMG_TOK, 202, 203, 204]]
        ).cuda()
        labels = torch.tensor(
            [[11, IGNORE_INDEX, IGNORE_INDEX, 13, 14], [IGNORE_INDEX, IGNORE_INDEX, 22, 23, 24]]
        ).cuda()
        num_image_tiles = torch.tensor([1, 1], dtype=torch.int).cuda()
        imgs = torch.randn(2, 3, 336, 336, dtype=torch.float32).cuda()  # [num_tiles_total, 3, h, w]
        seq = tokens.shape[1]
        position_ids = torch.arange(seq, dtype=torch.long, device="cuda").unsqueeze(0).expand(2, seq)
        loss_mask = (labels != IGNORE_INDEX).float()
        return tokens, labels, num_image_tiles, imgs, position_ids, loss_mask

    def test_forward_matches_llava(self):
        tokens, labels, num_image_tiles, imgs, position_ids, loss_mask = self._make_batch()

        # Original LLaVAModel full pipeline.
        # 原始 LLaVAModel 全流程。
        out_ref, lm_ref = self.ref(
            images=imgs,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )

        # Colocated path: encoder forward -> pp0 assembly -> GPTModel.
        # 共置路径：encoder 前传 -> pp0 组装 -> GPTModel。
        image_embeddings = self.enc(imgs)
        out_bb, lm_bb, _ = self.bb(
            image_embeddings=image_embeddings,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )

        assert out_ref.shape == out_bb.shape, (out_ref.shape, out_bb.shape)
        assert torch.allclose(out_ref, out_bb, atol=1e-5), "per-token loss differs"
        assert torch.allclose(lm_ref, lm_bb), "assembled loss mask differs"

    def test_grads_match_llava(self):
        tokens, labels, num_image_tiles, imgs, position_ids, loss_mask = self._make_batch()

        out_ref, _ = self.ref(
            images=imgs,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )
        out_ref.sum().backward()

        image_embeddings = self.enc(imgs)
        out_bb, _, _ = self.bb(
            image_embeddings=image_embeddings,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )
        out_bb.sum().backward()

        def compare_grads(m1, m2, label):
            d1 = dict(m1.named_parameters())
            d2 = dict(m2.named_parameters())
            assert set(d1) == set(d2), (set(d1) ^ set(d2))
            for n in d1:
                g1, g2 = d1[n].grad, d2[n].grad
                assert (g1 is None) == (g2 is None), (label, n)
                if g1 is not None:
                    assert torch.allclose(g1, g2, atol=1e-5), (label, n)

        # GPTModel grads, vision grads and projection grads must all match.
        # GPT 梯度、视觉梯度、投影梯度都必须一致。
        compare_grads(self.ref.language_model, self.bb.language_model, "language_model")
        compare_grads(self.ref.vision_model, self.enc.vision_model, "vision_model")
        compare_grads(self.ref.vision_projection, self.enc.vision_projection, "vision_projection")

    def test_forward_matches_llava_multitile(self):
        """Multi-tile sample: the tile-expansion path must match LLaVA.

        多 tile 样本：tile 展开路径必须与 LLaVA 一致（一个 image token 展开为
        num_tiles * img_seq_len 个 embedding 位置）。
        """
        # Sample 0: 1 image token with 2 tiles; sample 1: 1 image token with 1 tile.
        # 样本 0：1 个 image token、2 个 tile；样本 1：1 个 image token、1 个 tile。
        tokens = torch.tensor(
            [[101, 102, IMG_TOK, 103, 104], [201, IMG_TOK, 202, 203, 204]]
        ).cuda()
        labels = torch.tensor(
            [[11, IGNORE_INDEX, IGNORE_INDEX, 13, 14], [IGNORE_INDEX, IGNORE_INDEX, 22, 23, 24]]
        ).cuda()
        num_image_tiles = torch.tensor([2, 1], dtype=torch.int).cuda()
        imgs = torch.randn(3, 3, 336, 336, dtype=torch.float32).cuda()  # 3 tiles total
        seq = tokens.shape[1]
        position_ids = torch.arange(seq, dtype=torch.long, device="cuda").unsqueeze(0).expand(2, seq)
        loss_mask = (labels != IGNORE_INDEX).float()

        out_ref, lm_ref = self.ref(
            images=imgs,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )
        image_embeddings = self.enc(imgs)
        out_bb, lm_bb, _ = self.bb(
            image_embeddings=image_embeddings,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )
        assert out_ref.shape == out_bb.shape, (out_ref.shape, out_bb.shape)
        # Expected max seq: 5 - 1 + 2 * 577 = 1158 (sample 0), padded to max.
        # 期望最大序列：5 - 1 + 2 * 577 = 1158（样本 0），padding 到最大值。
        assert out_bb.shape == (2, 1158), out_bb.shape
        assert torch.allclose(out_ref, out_bb, atol=1e-5), "multitile loss differs"
        assert torch.allclose(lm_ref, lm_bb), "multitile loss mask differs"
