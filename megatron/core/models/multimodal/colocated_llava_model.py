# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Colocated LLaVA model: encoder and backbone as two independent model chunks.

共置 LLaVA 模型：把 encoder（ViT + projection）与 backbone（GPT）拆成两个
独立的 model chunk，用于共置训练（encoder 全量共置做 inner/outer 两层 DP，
backbone 走 1F1B PP）。原 ``llava_model.py`` 保持不变，作为逻辑参考与数值基准；
两个 chunk 的子模块命名（``vision_model`` / ``vision_projection`` /
``language_model``）与 LLaVAModel 保持一致，保证 checkpoint 直接兼容。
"""

from typing import List, Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.multimodal.llava_model import (
    DEFAULT_IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    pixel_shuffle,
)
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

# Optional TE support for the THD (packed + CP) path of the pp0 assembly.
# TE 可选支持：pp0 组装的 THD（packed + CP）路径需要。
try:
    import transformer_engine_torch as tex

    HAVE_TEX = True
except ImportError:
    tex = None
    HAVE_TEX = False


class ColocatedViTEncoder(MegatronModule):
    """Encoder chunk: the full ViT + projection, replicated on every rank.

    共置训练的 encoder chunk：每个 rank 持完整 ViT + projection，不做 PP 切分。
    只负责把 ``images`` 算成 ``image_embeddings``；文本 embedding、tile_tags、
    combined_embeddings 组装都在 backbone 的 pp0 上完成（见 ColocatedGPTBackbone）。
    """

    def __init__(
        self,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        img_h: int = 336,
        img_w: int = 336,
        patch_dim: int = 14,
        pixel_shuffle: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        """Build the vision backbone and the vision projection sub-modules.

        构建视觉主干（ViT）与视觉投影两个子模块，命名与 LLaVAModel 一致。
        """
        super().__init__(config=vision_transformer_config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage

        self.vision_model = None
        self.vision_projection = None

        self._drop_vision_class_token = drop_vision_class_token
        self._pixel_shuffle = pixel_shuffle

        # Same vision-model type dispatch as LLaVAModel (llava_model.py L246-337).
        # 与 LLaVAModel 相同的视觉模型类型分派（llava_model.py L246-337）。
        class_token_len = 1
        add_class_token = True
        if vision_transformer_config.vision_model_type.startswith(
            ("clip", "siglip", "internvit")
        ):
            if vision_transformer_config.vision_model_type == "siglip":
                class_token_len = 0
                add_class_token = False
            self.vision_model = CLIPViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                img_h=img_h,
                img_w=img_w,
                class_token_len=class_token_len,
                patch_dim=patch_dim,
                model_subtype=vision_transformer_config.vision_model_type,
                add_class_token=add_class_token,
                pg_collection=self.pg_collection,
                vp_stage=self.vp_stage,
            )
        elif vision_transformer_config.vision_model_type in ("radio", "radio-g", "cradio-g"):
            class_token_len = 0
            max_img_h = 0
            max_img_w = 0
            embedder_bias = False
            ln_post_impl = None
            use_mask_token = False

            if vision_transformer_config.vision_model_type == "radio":
                class_token_len = 8
                max_img_h = 2048
                max_img_w = 2048
            elif vision_transformer_config.vision_model_type == "radio-g":
                class_token_len = 5
                max_img_h = 1792
                max_img_w = 1792
                embedder_bias = True
                from megatron.core.extensions.transformer_engine import TENorm

                ln_post_impl = TENorm
                use_mask_token = True
            elif vision_transformer_config.vision_model_type == "cradio-g":
                class_token_len = 8
                max_img_h = 2048
                max_img_w = 2048
                embedder_bias = False
                ln_post_impl = None
                use_mask_token = False

            self.vision_model = RADIOViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                ln_post_impl=ln_post_impl,
                img_h=img_h,
                img_w=img_w,
                max_img_h=max_img_h,
                max_img_w=max_img_w,
                class_token_len=class_token_len,
                patch_dim=patch_dim,
                add_class_token=add_class_token,
                embedder_bias=embedder_bias,
                use_mask_token=use_mask_token,
                pg_collection=self.pg_collection,
                vp_stage=self.vp_stage,
            )
        else:
            raise NotImplementedError(
                f"Colocated encoder does not support vision model type "
                f"'{vision_transformer_config.vision_model_type}' yet."
            )

        vision_projection_input_size = vision_transformer_config.hidden_size
        vision_projection_input_size *= 4 if pixel_shuffle else 1
        self.vision_projection = MultimodalProjector(
            vision_projection_config,
            vision_projection_layer_spec,
            vision_projection_type,
            vision_projection_input_size,
            tp_group=self.pg_collection.tp,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encoder-only forward: images -> image_embeddings.

        Computes only the vision side: vision_model -> (drop cls) ->
        (pixel shuffle) -> permute -> vision_projection. tile_tags are NOT done
        here (moved to the backbone pp0 assembly, which needs the language
        embedding). Text-only samples (empty ``images``) return an empty tensor,
        matching the LLaVAModel behavior.

        只算图像侧：vision_model →（drop cls）→（pixel shuffle）→ permute →
        vision_projection。tile_tags 不在这里做（挪到 backbone pp0 组装，因为
        需要语言模型的 embedding）。无图样本（images 为空）直接返回空 tensor，
        与 LLaVAModel 行为一致。

        Args:
            images (torch.Tensor): [num_tiles, img_h, img_w] (or [num_tiles, 3, h, w]).
        Returns:
            torch.Tensor: image_embeddings of shape [img_seq_len, num_tiles, h_lang].
        """
        if images.shape[0] == 0:
            # Text-only sample: no image tokens, return an empty embeddings tensor.
            # 无图样本：没有图像 token，返回空 embeddings tensor。
            return torch.tensor([], dtype=images.dtype, device=images.device).reshape(0, 0, 0)

        image_embeddings = self.vision_model(images)  # [num_tiles, img_seq_len, h_vision]
        if self._drop_vision_class_token:
            # Drop the vision class token(s) before the language model.
            # 丢掉视觉 class token（语言模型不需要它）。
            image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]

        if self._pixel_shuffle:
            # Patch merging: reduce spatial dims, expand channel dims.
            # patch 合并：空间缩小、通道扩大。
            image_embeddings = pixel_shuffle(
                image_embeddings
            )  # [num_tiles, img_seq_len_shuffled, h_vision_shuffled]

        # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining.
        # permute 可能产生稀疏张量，contiguous() 保证后续 P2P/组装不破坏形状语义。
        image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()  # [img_seq_len, num_tiles, h_vision]

        # Map vision output size to language model input size.
        # 把视觉输出维度映射到语言模型输入维度。
        image_embeddings = self.vision_projection(
            image_embeddings
        )  # [img_seq_len, num_tiles, h_lang]

        return image_embeddings


class ColocatedGPTBackbone(MegatronModule):
    """Backbone chunk: the GPT model with normal PP split.

    共置训练的 backbone chunk：GPTModel 按 PP 切分（pre_process / post_process
    控制 embedding 与 output 层归属）。pp0（pre_process=True）额外负责把收到的
    image_embeddings 与本地文本组装为 combined_embeddings（含 tile_tags）；
    组装逻辑在 2.4 迁入。中间/末 stage 走 GPTModel 既有路径（P2P 激活注入）。
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "learned_absolute",
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_INDEX,
        img_seq_len: Optional[int] = None,
        tile_tags: Optional[list] = None,
        tokenizer_type: str = "",
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        """Build the GPT language model sub-module (named ``language_model``).

        构建 GPT 语言模型子模块（命名 ``language_model``，与 LLaVAModel 一致），
        并保存 pp0 组装所需的常量（image_token_index / img_seq_len / tile_tags）。
        """
        super().__init__(config=language_transformer_config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage

        self.pre_process = pre_process
        self.post_process = post_process

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            rope_scaling=language_rope_scaling,
            rope_scaling_factor=language_rope_scaling_factor,
            scatter_embedding_sequence_parallel=False,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

        # Constants needed by the pp0 assembly (_preprocess_data / tile_tags).
        # pp0 组装（_preprocess_data / tile_tags）所需的常量。
        self._image_token_index = image_token_index
        self._img_seq_len = img_seq_len
        self._tile_tags = tile_tags
        self._tokenizer_type = tokenizer_type

        # Convenience attributes for the pp0 assembly, mirroring LLaVAModel.
        # pp0 组装用的便捷属性（与 LLaVAModel 一致，从 language config 派生）。
        self._language_max_sequence_length = language_max_sequence_length
        self._language_is_pipeline_parallel = (
            language_transformer_config.pipeline_model_parallel_size > 1
        )
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        if self.sequence_parallel_lm or self.context_parallel_lm > 1:
            # SP/CP require the TE DotProductAttention (checked at the config level).
            # SP/CP 需要 TE DotProductAttention（在 config 层校验）。
            self.cp_group = self.pg_collection.cp
        else:
            self.cp_group = None
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # 注：4.3j 重构后本模型不再持有 ``colocated_new_labels``/``colocated_new_loss_mask``
        # 交接属性——展开 labels/loss_mask 改由 forward 返回 3 元组，经
        # ``colocated_forward_step`` 的 ``intra_packet`` 载体与 schedule 交接（模型
        # 不再充当 schedule mailbox）。

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor (PP schedule entry point).

        Forwards the P2P-injected activation to the inner GPTModel (which routes it
        to the TransformerBlock's ``input_tensor``). For non-first stages the
        TransformerBlock ignores the ``decoder_input`` argument and reads from its
        ``input_tensor`` (transformer_block.py, ``pre_process=False``), so this
        forwarding is required.

        将 PP schedule 注入的激活转发给内部 GPTModel（最终落到 TransformerBlock 的
        ``input_tensor``）。非首 stage 时 TransformerBlock 会忽略 decoder_input 参数、
        强制读自己的 input_tensor（transformer_block.py 的 pre_process=False 分支），
        因此必须在此转发（与 LLaVAModel 的做法一致）。
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for colocated backbone"

        if not self.pre_process:
            self.language_model.set_input_tensor(input_tensor[0])

    def _apply_tile_tagging(self, image_embeddings, num_image_tiles):
        """Apply tile tagging to image embeddings (ported from LLaVAModel).

        给图像 embedding 加上 tile 标签（从 LLaVAModel 移植）。这必须在 pp0 做，
        因为 tile tag 的 embedding 来自语言模型的 embedding 层。

        Args:
            image_embeddings (torch.Tensor): [img_seq_len, num_tiles, h_language].
            num_image_tiles (torch.Tensor): Number of tiles for each image [num_images].

        Returns:
            torch.Tensor: [tile_seq_len + img_seq_len, num_tiles, h_language].
        """
        assert (
            num_image_tiles.shape[0] == 1 and len(num_image_tiles) == 1
        ), "multiple input images are not supported yet."

        num_tiles = num_image_tiles[0].item()
        tile_tags = self._tile_tags[: num_tiles - 1] + [self._tile_tags[-1]]

        # [num_tiles, tile_seq_len (=5)]
        tile_tag_input_ids = torch.tensor(
            tile_tags, dtype=torch.int64, device=num_image_tiles.device
        )

        # [tile_seq_len, num_tiles, h_language]
        tile_tag_embeds = self.language_model.embedding(tile_tag_input_ids, position_ids=None)

        # [num_tiles, dim] should be the same
        assert tile_tag_embeds.shape[1:] == image_embeddings.shape[1:]

        image_embeddings = torch.cat([tile_tag_embeds, image_embeddings])

        return image_embeddings  # [tile_seq_len + img_seq_len, num_tiles, h_language]

    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        loss_mask,
        labels,
        num_image_tiles,
    ):
        """Assemble image + text embeddings into the combined sequence (training
        path, ported from LLaVAModel._preprocess_data).

        把图像与文本 embedding 组装成 combined 序列（训练路径，从
        LLaVAModel._preprocess_data 移植）。image token 的位置（input_ids ==
        image_token_index）用 image_embeddings 填充；labels / loss_mask 同步展开。
        PP 中间 stage 直接返回 None（不做组装）。

        Returns:
            final_embedding [s', b, h]（pre_process 时）或 None，
            final_labels / final_loss_mask [b, s']（post_process 且有 labels 时）或 None。
        """
        # No pre- or postprocessing needed (PP middle chunks).
        # 中间 stage 不需要组装。
        if not self.pre_process and not self.post_process:
            return None, None, None

        img_seq_len = self._img_seq_len
        batch_size, text_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        with torch.no_grad():
            # Locate the image token positions (input_ids == image_token_index).
            # 定位 image token 位置（input_ids == image_token_index）。
            image_token_mask = input_ids == self._image_token_index
            num_images_per_sample = torch.sum(image_token_mask, dim=-1)

            # Number of tiles per sample.
            num_image_tiles_batch = num_image_tiles.split(num_images_per_sample.tolist(), dim=0)
            num_image_tiles_batch = torch.tensor(
                [x.sum() for x in num_image_tiles_batch], device=input_ids.device
            )

            # Sequence length for each sample = tiles * img_seq_len - image tokens
            # + text tokens (each image token expands to num_tiles * img_seq_len).
            # 每个样本的序列长度 = tiles * img_seq_len - image token 数 + text token 数。
            seq_lens = num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
            max_seq_len = seq_lens.max()
            # Pipeline parallel expects fixed input size. Check if we need to pad.
            # PP 期望固定输入大小，必要时 padding 到 language_max_sequence_length。
            if (
                self._language_is_pipeline_parallel
                and max_seq_len < self._language_max_sequence_length
            ):
                max_seq_len = self._language_max_sequence_length

            batch_indices, non_image_indices = torch.where(image_token_mask != True)

            # New position ids for the text tokens, shifted by the image sequence length.
            # 文本 token 的新位置 id（按图像序列长度平移）。
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            image_token_mask_lens[image_token_mask] = num_image_tiles * img_seq_len - 1
            new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
            text_position_ids = new_position_ids[batch_indices, non_image_indices]

            label_batch_indices = None  # dummy value to pass formatting
            if has_labels:
                label_text_position_ids = text_position_ids - 1
                valid_label_text_position_ids = label_text_position_ids >= 0
                label_text_position_ids = label_text_position_ids[valid_label_text_position_ids]

                label_batch_indices = batch_indices[valid_label_text_position_ids]

                label_non_image_indices = non_image_indices - 1
                valid_label_non_image_indices = label_non_image_indices >= 0
                label_non_image_indices = label_non_image_indices[valid_label_non_image_indices]

            # Create a mask for the image embedding positions.
            images_mask = torch.full(
                (batch_size, max_seq_len), True, dtype=torch.bool, device=input_ids.device
            )
            # No images in the text positions.
            images_mask[batch_indices, text_position_ids] = False
            # Samples can have different amount of images tokens.
            first_padding_idx = new_position_ids[:, -1] + 1
            images_mask[
                torch.arange(max_seq_len, device=first_padding_idx.device).repeat(batch_size, 1)
                >= first_padding_idx.unsqueeze(1)
            ] = False

        # Create the final input embedding (if this is the first language model stage).
        # 组装最终输入 embedding（pre_process 时）。
        final_embedding = None
        if self.pre_process:
            embed_dim = language_embeddings.shape[-1]
            final_embedding = torch.zeros(
                batch_size,
                max_seq_len,
                embed_dim,
                dtype=language_embeddings.dtype,
                device=language_embeddings.device,
            )

            # Put text embeddings to the text positions in the result tensor.
            # 文本位置填入语言 embedding。
            final_embedding[batch_indices, text_position_ids] = language_embeddings[
                batch_indices, non_image_indices
            ]

            # Put image embeddings to image positions.
            # 图像位置填入 image_embeddings（空图时 images_mask 全 False，不填充）。
            final_embedding[images_mask] = (
                image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()
            )

        # Create the final labels and loss mask (if labels are provided). In the
        # colocated split the consumer (pre_process, post_process=False when PP>1) must
        # also assemble the expanded labels/loss_mask: they are transported down the
        # pipeline (4.3h/4.3j accompaniment) for the last stage's loss. Gating on
        # post_process would leave them None for the consumer, and the receiver's
        # irecv in _recv_targets would hang.
        # 组装最终 labels / loss_mask（有 labels 时）。共置拆分下 consumer（pre_process，
        # PP>1 时 post_process=False）也必须组装展开的 labels/loss_mask——它们要沿流水
        # 伴随传输（4.3h/4.3j，供 last stage 算 loss）；若 gate 在 post_process，consumer
        # 会得到 None，接收端 _recv_targets 的 irecv 将挂死。
        final_labels, final_loss_mask = None, None
        if has_labels:
            final_labels = torch.full(
                (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            final_loss_mask = torch.full(
                (batch_size, max_seq_len), 0, dtype=loss_mask.dtype, device=loss_mask.device
            )

            # Put text labels and loss mask to the text positions.
            # 文本位置填入 labels / loss_mask。
            final_labels[label_batch_indices, label_text_position_ids] = labels[
                label_batch_indices, label_non_image_indices
            ]

            final_loss_mask[batch_indices, text_position_ids] = loss_mask[
                batch_indices, non_image_indices
            ]

            # For labels, pick the last label index that got dropped by the shift to left.
            label_extra_text_position_ids = seq_lens - 1
            batch_range = torch.arange(len(label_extra_text_position_ids))
            final_labels[batch_range, label_extra_text_position_ids] = labels[batch_range, -1]

            # Loss mask the image positions.
            # image 位置不计算 loss。
            final_loss_mask[images_mask] = 0

            # Loss mask last text position just before an image so that the text
            # token does not need to predict the first image token.
            # 屏蔽 image 前一个文本位置（不要求文本 token 预测第一个 image token）。
            batch_image_indices, image_indices = torch.where(image_token_mask)
            before_image_indices = image_indices - 1
            valid = before_image_indices >= 0
            valid_batch_image_indices = batch_image_indices[valid]
            valid_before_image_indices = before_image_indices[valid]
            valid_before_image_indices = new_position_ids[
                valid_batch_image_indices, valid_before_image_indices
            ]

            final_loss_mask[valid_batch_image_indices, valid_before_image_indices] = 0

        if final_embedding is not None and final_labels is not None:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        if final_embedding is not None:
            # Truncate if exceeding the language model's max sequence length.
            # 超过语言模型最大序列长度时截断。
            if final_embedding.shape[1] > self._language_max_sequence_length:
                final_embedding = final_embedding[:, : self._language_max_sequence_length]
            # Transpose to [s,b,h] only if not using CP.
            # 非 CP 时转置为 [s, b, h]（CP 的序列维在 dim=1）。
            if self.context_parallel_lm == 1:
                final_embedding = final_embedding.transpose(1, 0).contiguous()

        truncate_labels = (
            final_labels is not None and final_labels.shape[1] > self._language_max_sequence_length
        )
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        return final_embedding, final_labels, final_loss_mask

    def _process_embedding_token_parallel(
        self, combined_embeddings, new_labels, new_loss_mask, packed_seq_params
    ):
        """Shard the combined sequence for SP/CP (ported from LLaVAModel).

        对 combined 序列做 SP/CP 切分（从 LLaVAModel 移植）。

        Returns:
            combined_embeddings, new_labels, new_loss_mask, packed_seq_params（切分后）。
        """
        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

        shard_factor = seq_dim = None
        if self.pre_process:
            if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
                seq_dim = 1
            elif self.context_parallel_lm > 1:
                shard_factor = self.context_parallel_lm * 2
                seq_dim = 1
            elif self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm
                seq_dim = 0

            assert (
                combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length should be divisible by {shard_factor} for \
                Sequence/Context parallelism"
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                assert (
                    combined_embeddings.shape[seq_dim] == self._language_max_sequence_length
                ), f"TP Comm overlap either requires Vision+Text token length \
                == language_max_sequence_length"

        if self.context_parallel_lm > 1:
            batch = dict()
            if self.pre_process:
                batch["combined_embeddings"] = combined_embeddings
            if self.post_process:
                batch["new_labels"] = new_labels
                batch["new_loss_mask"] = new_loss_mask
            # Distribute sequence across CP ranks.
            # 把序列分发到 CP rank。
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                from megatron.training.utils import get_batch_on_this_cp_rank

                batch = get_batch_on_this_cp_rank(batch)
            else:
                assert HAVE_TEX and is_te_min_version(
                    "1.10.0"
                ), "Please update Transformer Engine to >= 1.10 to use \
                    Context Parallel with THD format data"
                index = tex.thd_get_partitioned_indices(
                    packed_seq_params.cu_seqlens_q_padded,
                    batch[next(iter(batch))].size(1),
                    self.cp_group.size(),
                    self.cp_group.rank(),
                )
                for key, data in batch.items():
                    batch[key] = data.index_select(1, index)

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [B, S/CP, H]
                combined_embeddings = combined_embeddings.transpose(
                    1, 0
                ).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
            if self.post_process:
                new_labels = batch["new_labels"]
                new_loss_mask = batch["new_loss_mask"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

    def forward(
        self,
        image_embeddings: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        num_image_tiles: Optional[list] = None,
        image_token_index: Optional[int] = None,
        inference_context=None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params=None,
    ):
        """Backbone forward: pp0 assembly + GPTModel (aligned to LLaVAModel's style).

        只在必要处分支（对齐 llava_model.py:799-952 的风格，2026-08-13 用户指出大分支
        写法问题）：
        - **组装只发生在 pre_process（consumer）**——image_embeddings（来自边界包）+
          本地文本组装为 combined_embeddings，展开 labels/loss_mask（new_labels/
          new_loss_mask）随返回值一并交还（4.3j 重构：不再写入模型属性，由
          ``colocated_forward_step`` 写回 intra_packet 供 schedule 做 backbone P2P
          伴随传输）；
        - **非 pre_process**：激活已 set_input_tensor（decoder_input=None 走
          input_tensor），labels/loss_mask 直接用入参（schedule 伴随 recv 后经
          intra_packet 闭包绑定传入，4.3j）——**last stage 用它算 loss**（不再是
          传入的 None）；
        - 统一的 ``language_model`` 调用与返回 ``(output, new_loss_mask, new_labels)``。

        Returns:
            (output_tensor, loss_mask, labels): output 为激活（中间 stage）或
            per-token loss/logits（末 stage）；loss_mask 为组装后的 new_loss_mask
            （consumer）或入参 loss_mask（非 consumer）；labels 为组装后的
            new_labels（consumer，供伴随传输）或入参 labels（非 consumer）。
        """
        if self.pre_process:
            # Optional per-call override of the image token id (defaults to the
            # constructor value), mirroring LLaVAModel.
            # 允许按调用覆盖 image token id（默认用构造时传入的值），与 LLaVA 一致。
            image_token_index = (
                image_token_index if image_token_index is not None else self._image_token_index
            )

            language_embeddings = None
            if input_ids is not None:
                input_ids_text = input_ids.clone()
                # Replace image tokens with a dummy id so that the language
                # embedding layer only embeds text (image positions are filled later).
                # 把 image token 替换为哑 id，让语言 embedding 只对文本做。
                input_ids_text[input_ids_text == image_token_index] = 0
                language_embeddings = self.language_model.embedding(  # [text_seq, b, h]
                    input_ids=input_ids_text, position_ids=position_ids
                )
                language_embeddings = language_embeddings.transpose(
                    1, 0
                ).contiguous()  # [b, text_seq, h]

            # Assume 1 tile per image if the number of tiles is not provided.
            # 未提供 tile 数时默认每张图 1 个 tile。
            if num_image_tiles is None and image_embeddings is not None:
                num_image_tiles = torch.ones(
                    image_embeddings.shape[1], dtype=torch.int, device=input_ids.device
                )

            # Apply tile tagging before assembly (needs language_model.embedding).
            # 组装前先应用 tile tagging（依赖语言模型的 embedding 层）。
            if self._tile_tags is not None and torch.any(input_ids == image_token_index):
                image_embeddings = self._apply_tile_tagging(image_embeddings, num_image_tiles)

            combined_embeddings, new_labels, new_loss_mask = self._preprocess_data(
                image_embeddings,
                language_embeddings,
                input_ids,
                loss_mask,
                labels,
                num_image_tiles,
            )
            if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
                combined_embeddings, new_labels, new_loss_mask, packed_seq_params = (
                    self._process_embedding_token_parallel(
                        combined_embeddings, new_labels, new_loss_mask, packed_seq_params
                    )
                )
            decoder_input = combined_embeddings
        else:
            # 非 consumer：激活已 set_input_tensor（decoder_input=None 走 input_tensor）；
            # labels/loss_mask 由 schedule 伴随 recv 后经 intra_packet 闭包绑定传入
            #（4.3j 重构：不再依赖模型属性——模型不充当 schedule 交接的 mailbox）。
            decoder_input = None
            new_labels = labels
            new_loss_mask = loss_mask

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=new_labels,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
        # 4.3j：返回三元组 (output, loss_mask, labels)——consumer（pre_process）的
        # new_labels 由 colocated_forward_step 写回 intra_packet 供伴随传输；非
        # consumer 的 new_labels 即闭包传入的 labels（last stage 算 loss）。
        return output, new_loss_mask, new_labels
