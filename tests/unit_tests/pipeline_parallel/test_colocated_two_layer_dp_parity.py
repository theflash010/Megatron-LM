# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Task 5.8: numeric parity of the two-layer colocated data parallelism.

多卡数值对齐：world = P x D_outer（TP=1、CP=1），同一份初始权重、同一个全局 batch，
分别跑

- 参考路径：**每个 rank 各自**建一份完整的 ``LLaVAModel``（单成员 pp 组 ⇒ 它自认
  PP=1、持全部层），用 ``forward_backward_no_pipelining`` **顺序跑完全部
  ``D_outer * n`` 个 microbatch**，梯度纯累加（``calculate_per_token_loss=True`` 时
  schedule 不做任何归一化），最后由本测试手动除以全局 token 总数；
- 共置路径：``ColocatedViTEncoder`` + ``ColocatedGPTBackbone``，各自套**真实
  ``DistributedDataParallel``**（encoder 的 dp_cp = 全 W 的共置 dp 组、backbone 的
  dp_cp = 常规 outer dp 组），跑 ``forward_backward_colocated``，由
  ``finalize_model_grads`` 完成归约与 per-token 归一化。

比的是**归约后的梯度**（DDP 的 ``param.main_grad``），因此本测试回答的是 Task 5 的核心
问题：两层数据并行（encoder 在全 W 上单次 SUM、backbone 在 outer 上 SUM）加 per-token
分母，得到的梯度是否等于"一个进程顺序跑完整个全局 batch"的数学定义。

与 Task 4.8（``test_colocated_schedule_parity.py``）的分工：4.8 在 world=1、裸模型上比
**调度本身**有没有改变数学；本测试在 world>=2 上比**归约**——4.8 被刻意隔离在外的那一层。
分桶对称性由 Task 5.11 的 ``test_colocated_setup_wiring.py`` 覆盖（真实
``setup_model_and_optimizer`` 路径），这里不重复。
"""

from copy import deepcopy
from functools import partial
from typing import Dict, List, Tuple

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.colocated_llava_model import (
    ColocatedGPTBackbone,
    ColocatedViTEncoder,
)
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, LLaVAModel
from megatron.core.pipeline_parallel.colocated_encoder_comm import ForwardPacket
from megatron.core.pipeline_parallel.colocated_schedule import forward_backward_colocated
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
    snapshot_colocated_encoder_rng_tracker,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.training import build_colocated_module_process_groups
from tests.unit_tests.pipeline_parallel.test_colocated_schedule_parity import (
    _build_layer_specs,
    _position_ids_and_loss_mask,
)
from tests.unit_tests.test_utilities import Utils

_IMAGE_TOKEN_INDEX = -200  # DEFAULT_IMAGE_TOKEN_INDEX
_MICRO_BATCH_SIZE = 2
_TEXT_SEQ_LENGTH = 8  # tokens 长度（含 1 个 image token 占位）
_IMG_H = _IMG_W = 28
_PATCH_DIM = 14
# (28/14)^2 = 4 个 patch + 1 个 class token（drop_vision_class_token=False）。
_IMAGE_SEQ_LEN = (_IMG_H // _PATCH_DIM) * (_IMG_W // _PATCH_DIM) + 1
# 组装后的序列长度：文本 8 - 1 个 image 占位 + 5 个 image token = 12。
# 取 language_max_sequence_length 与它**恰好相等**，这样 PP>1 的 pad-to-max 分支
# （llava_model.py:528-532）与参考路径的 pad-to-batch-max 结果一致，两条路径的序列
# 长度逐位相同、没有 padding 差异需要论证。
_COMBINED_SEQ_LENGTH = _TEXT_SEQ_LENGTH - 1 + _IMAGE_SEQ_LEN
_VOCAB_SIZE = 128
_LAYERS_PER_STAGE = 2
_MICROBATCHES_PER_PRODUCER = 2  # n = P * 该值，>1 才覆盖梯度累积
_LANGUAGE_HIDDEN_SIZE = 32
_LANGUAGE_NUM_HEADS = 4


def _build_configs(pipeline_parallel_size: int):
    """Language(PP=P) / language(PP=1, 参考) / vision / projection 四份 config。

    per-token loss 是共置的硬前提（arguments.py 的共置校验块强制），参考路径也必须开——
    否则 schedule 会替它做 ``/= num_tokens`` 与 ``/= num_microbatches``（schedules.py:296-299），
    两条路径的分母就不可比了。dropout 全关 + ``eval()``，保证两条路径逐位确定。
    """
    language_config = TransformerConfig(
        num_layers=_LAYERS_PER_STAGE * pipeline_parallel_size,
        hidden_size=_LANGUAGE_HIDDEN_SIZE,
        num_attention_heads=_LANGUAGE_NUM_HEADS,
        pipeline_model_parallel_size=pipeline_parallel_size,
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        # 必须在**构造参数**里给，不能事后赋值：``ModelParallelConfig.__post_init__``
        # （model_parallel_config.py:436-440）在构造期就检查 "pp>1 且 pipeline_dtype
        # is None"，事后赋值那行永远走不到。
        pipeline_dtype=torch.float32,
    )
    language_config.language_model_type = "dummy"
    # 与生产入口一致（arguments.py 把两个伪释放开关都硬编码为 True）。
    language_config.deallocate_pipeline_outputs = True
    language_config.deallocate_encoder_outputs = True

    # 参考路径：同样的层数/宽度，但 pipeline_model_parallel_size=1 ⇒ 它持全部层。
    reference_language_config = deepcopy(language_config)
    reference_language_config.pipeline_model_parallel_size = 1

    vision_config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
    )
    vision_config.vision_model_type = "clip"
    vision_config.pipeline_dtype = torch.float32

    vision_projection_config = TransformerConfig(
        num_layers=2,
        hidden_size=_LANGUAGE_HIDDEN_SIZE,
        ffn_hidden_size=32,
        num_attention_heads=1,
        pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
    )
    return language_config, reference_language_config, vision_config, vision_projection_config


def _make_global_microbatches(
    num_replicas: int, num_microbatches: int
) -> Dict[Tuple[int, int], tuple]:
    """全局 batch：``(replica, microbatch) -> (tokens, labels, images, num_image_tiles)``。

    每个 rank 都用同一个 CPU 生成器按同一顺序造，因此**无需通信**即逐位一致（用 CPU
    generator 而不是 cuda generator，避免各 rank 的 device RNG 状态差异）。每个样本恰好
    1 个 image token 占位、1 个 tile，组装后长度恒为 ``_COMBINED_SEQ_LENGTH``。
    """
    generator = torch.Generator().manual_seed(5008)
    microbatches: Dict[Tuple[int, int], tuple] = {}
    for replica in range(num_replicas):
        for microbatch in range(num_microbatches):
            # 文本 id 逐 (replica, microbatch) 错开，保证每份数据都不同——否则"分片错了"
            # 这类 bug 会因为数据相同而看不出来。image token 占位不能动（必须与
            # image_token_index 精确相等才会被组装替换）。
            offset = 1 + replica * num_microbatches + microbatch
            tokens = torch.full(
                (_MICRO_BATCH_SIZE, _TEXT_SEQ_LENGTH), offset, dtype=torch.long
            )
            tokens += torch.arange(_TEXT_SEQ_LENGTH, dtype=torch.long).unsqueeze(0)
            tokens = tokens % _VOCAB_SIZE
            tokens[:, 2] = _IMAGE_TOKEN_INDEX
            labels = (tokens + 1) % _VOCAB_SIZE
            # image 占位与其前一位不计 loss（与 TaskEncoder 产出的掩码语义一致）。
            labels[:, 1:3] = IGNORE_INDEX
            images = torch.randn(
                (_MICRO_BATCH_SIZE, 3, _IMG_H, _IMG_W),
                dtype=torch.float32,
                generator=generator,
            )
            num_image_tiles = torch.ones(_MICRO_BATCH_SIZE, dtype=torch.int)
            microbatches[(replica, microbatch)] = (
                tokens.cuda(),
                labels.cuda(),
                images.cuda(),
                num_image_tiles.cuda(),
            )
    return microbatches


def _loss_func(loss_mask, output_tensor):
    """per-token 型 loss_func（3 元组），两条路径共用。

    ``calculate_per_token_loss=True`` 时 schedule **不做任何归一化**（既不除 num_tokens
    也不除 num_microbatches），分母统一由 ``finalize_model_grads`` 用全局 token 数施加；
    参考路径不调 finalize，故由本测试手动施加同一个分母。
    """
    losses = output_tensor.float()
    loss = torch.sum(losses.view(-1) * loss_mask.view(-1))
    num_tokens = loss_mask.sum().clone().detach().to(dtype=torch.int)
    return loss, num_tokens, {'lm loss': loss.detach().clone().view(1)}


def _build_reference(configs, specs, pg_collection):
    """完整 LLaVAModel（持全部层）。

    ``pg_collection`` 传的是共置 collection 里 **encoder 那份**——它的 pp 组是单成员
    （Task 5.1 建的"单成员 encoder pp 组"），因此这个参考模型在 ``forward_step`` 的
    首/末 stage 判定与层偏移计算上都自认 PP=1，可以在**每个 rank 上独立跑完整个全局
    batch**。它不套 DDP、不调 finalize，只用来提供"数学定义上的参考梯度"。
    """
    _, reference_language_config, vision_config, vision_projection_config = configs
    language_spec, vision_spec, vision_projection_spec = specs
    reference = LLaVAModel(
        language_transformer_config=reference_language_config,
        language_transformer_layer_spec=language_spec,
        language_vocab_size=_VOCAB_SIZE,
        language_max_sequence_length=_COMBINED_SEQ_LENGTH,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_spec,
        drop_vision_class_token=False,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_spec,
        pre_process=True,
        post_process=True,
        image_token_index=_IMAGE_TOKEN_INDEX,
        img_h=_IMG_H,
        img_w=_IMG_W,
        patch_dim=_PATCH_DIM,
        pg_collection=pg_collection,
    ).cuda()
    reference.model_type = ModelType.encoder_or_decoder
    reference.eval()
    # 自检：参考模型必须真的是"完整模型"——层数齐全 + 有 embedding 与 output_layer。
    # 若单成员 pp 组这条路子被上游改掉、模型退化成一个 stage，这里立刻炸而不是给出
    # 一个静默错误的参考值。
    reference_parameter_names = {name for name, _ in reference.named_parameters()}
    expected_layers = _LAYERS_PER_STAGE * parallel_state.get_pipeline_model_parallel_world_size()
    for layer in range(expected_layers):
        assert any(
            name.startswith(f"language_model.decoder.layers.{layer}.")
            for name in reference_parameter_names
        ), f"参考模型缺第 {layer} 层，它不是完整模型"
    assert any(name.startswith("language_model.embedding.") for name in reference_parameter_names)
    assert any(
        name.startswith("language_model.output_layer.") for name in reference_parameter_names
    )
    return reference


def _broadcast_reference_weights(reference):
    """把 rank 0 的参考权重广播到所有 rank，消除"各 rank 初始化是否一致"这个假设。"""
    parameters = dict(reference.named_parameters())
    for name in sorted(parameters):
        torch.distributed.broadcast(parameters[name].data, src=0)


def _build_colocated_chunks(configs, specs, pg_collections):
    """共置的两个 chunk（encoder 全量、backbone 按 stage 切），命名与 LLaVAModel 一致。"""
    language_config, _, vision_config, vision_projection_config = configs
    language_spec, vision_spec, vision_projection_spec = specs
    encoder = ColocatedViTEncoder(
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_spec,
        drop_vision_class_token=False,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_spec,
        img_h=_IMG_H,
        img_w=_IMG_W,
        patch_dim=_PATCH_DIM,
        pg_collection=pg_collections["encoder"],
    ).cuda()
    backbone = ColocatedGPTBackbone(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_spec,
        language_vocab_size=_VOCAB_SIZE,
        language_max_sequence_length=_COMBINED_SEQ_LENGTH,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
        image_token_index=_IMAGE_TOKEN_INDEX,
        img_seq_len=_IMAGE_SEQ_LEN,
        pg_collection=pg_collections["language_model"],
    ).cuda()
    encoder.model_type = ModelType.encoder_or_decoder
    backbone.model_type = ModelType.encoder_or_decoder
    encoder.eval()
    backbone.eval()
    return encoder, backbone


def _reference_parameter_name(local_name: str, layer_offset: int) -> str:
    """本 stage 的参数名 -> 完整参考模型里的参数名（只有层号需要加偏移）。

    ``language_model.decoder.layers.{i}.*`` 的 ``i`` 是**本 stage 内的局部下标**，参考
    模型持全部层 ⇒ 全局下标 = 局部下标 + pp_rank * 每 stage 层数。embedding /
    final_layernorm / output_layer / vision_* 的名字两边完全一致。
    """
    prefix = "language_model.decoder.layers."
    if not local_name.startswith(prefix):
        return local_name
    local_layer, _, tail = local_name[len(prefix) :].partition(".")
    return f"{prefix}{int(local_layer) + layer_offset}.{tail}"


def _copy_reference_weights(reference, encoder, backbone, layer_offset: int) -> None:
    """把参考模型的权重逐参数拷进两个 chunk，保证初始权重完全一致。"""
    reference_parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for chunk in (encoder, backbone):
            for name, parameter in chunk.named_parameters():
                reference_name = _reference_parameter_name(name, layer_offset)
                assert reference_name in reference_parameters, (
                    f"chunk 参数 {name} 映射到 {reference_name}，但参考模型里没有这个名字"
                    f"——名字映射或模型结构不对"
                )
                source = reference_parameters[reference_name]
                assert source.shape == parameter.shape, (
                    f"{name} 与参考 {reference_name} 形状不一致：{parameter.shape} vs {source.shape}"
                )
                parameter.copy_(source)


def _run_reference(reference, pg_collection, microbatches, num_replicas, num_microbatches):
    """顺序跑完整个全局 batch，返回（累加后的梯度, 全局 token 总数）。

    ``calculate_per_token_loss=True`` ⇒ schedule 不做归一化，``.grad`` 就是各 microbatch
    梯度的纯和；分母由调用方统一施加。
    """
    ordered = [microbatches[(replica, index)] for replica in range(num_replicas)
               for index in range(num_microbatches)]
    token_counts: List[torch.Tensor] = []

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
        token_counts.append(expanded_loss_mask.sum().detach().clone())
        return output, partial(_loss_func, expanded_loss_mask)

    losses = forward_backward_no_pipelining(
        forward_step_func=reference_forward_step,
        data_iterator=iter(ordered),
        model=reference,
        num_microbatches=len(ordered),
        seq_length=_COMBINED_SEQ_LENGTH,
        micro_batch_size=_MICRO_BATCH_SIZE,
        pg_collection=pg_collection,
    )
    # 自检：参考路径必须真的算了 loss。若"单成员 pp 组 ⇒ 自认末 stage"这条不成立，
    # forward_data_store 会是空的 —— 那样后面的梯度对比就毫无意义。
    assert len(losses) == len(ordered), (
        f"参考路径只产出 {len(losses)} 条 loss，应为 {len(ordered)}——它没有被当成末 stage"
    )
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    assert grads, "参考路径没有产生任何梯度"
    total_tokens = torch.stack(token_counts).sum()
    assert total_tokens > 0, "全局 token 总数为 0，loss_mask 造错了"
    return grads, total_tokens


def _make_colocated_forward_step(encoder_module, backbone_module):
    """镜像 ``colocated_train.colocated_forward_step``（DDP 包一层，按类型认组件）。"""

    def colocated_forward_step(data_iterator, model, packet=None, intra_packet=None):
        chunk = model[0] if isinstance(model, (list, tuple)) else model
        module = chunk.module if isinstance(chunk, DistributedDataParallel) else chunk
        if module is encoder_module:
            tokens, labels, images, num_image_tiles = next(data_iterator)
            return ForwardPacket(
                image_embeddings=chunk(images),
                tokens=tokens,
                labels=labels,
                num_image_tiles=num_image_tiles,
            ), None
        assert module is backbone_module, f"unexpected chunk {type(module)}"
        if module.pre_process:
            # consumer（stage 0）：用 schedule 绑定的包组装并跑 backbone。
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
        else:
            # 非首 stage：激活已由 set_input_tensor 注入；展开后的 labels/loss_mask 来自
            # schedule 的伴随传输（4.3i/4.3j）。与 _backbone_forward 的 else 分支同形。
            assert intra_packet is not None, "非 consumer 也必须拿到 intra_packet (4.3j)"
            output, expanded_loss_mask, _ = chunk(
                labels=intra_packet.labels, loss_mask=intra_packet.loss_mask
            )
        return output, partial(_loss_func, expanded_loss_mask)

    return colocated_forward_step


def _wrap_with_ddp(chunk, pg_collection):
    """套真实 DDP。

    ``overlap_grad_reduce=False``：归约时机唯一（只在 ``finalize_model_grads`` 里的
    ``finish_grad_sync``），本测试要比的是归约**结果**而不是重叠时机；分桶对称性由
    Task 5.11 在真实 ``setup_model_and_optimizer`` 路径上覆盖。每个 chunk 用**各自
    一份** ddp_config（DDP 会就地改写 ``bucket_size``，共用一份会互相串改）。
    """
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        bucket_size=None,
    )
    return DistributedDataParallel(
        config=chunk.config, ddp_config=ddp_config, module=chunk, pg_collection=pg_collection
    )


def _my_data_iterator(microbatches, replica: int, my_microbatches: List[int]):
    """本 rank 的数据迭代器：只产出属于自己的 (replica, microbatch)，按 phase ① 的消费顺序。"""
    return iter([microbatches[(replica, index)] for index in my_microbatches])


def test_two_layer_dp_matches_sequential_global_batch():
    """两层 DP 归约 + per-token 分母，必须等于顺序跑完整个全局 batch（Task 5.8）。"""
    pipeline_parallel_size = 2
    if Utils.world_size < pipeline_parallel_size or Utils.world_size % pipeline_parallel_size != 0:
        pytest.skip(
            f"Task 5.8 needs world_size 为 {pipeline_parallel_size} 的整数倍且 >= "
            f"{pipeline_parallel_size}，实际 {Utils.world_size}"
        )
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_parallel_size,
            use_colocated_encoder=True,
        )
        model_parallel_cuda_manual_seed(123)
        snapshot_colocated_encoder_rng_tracker()

        num_microbatches = pipeline_parallel_size * _MICROBATCHES_PER_PRODUCER
        num_replicas = parallel_state.get_data_parallel_world_size()
        assert num_replicas * pipeline_parallel_size == Utils.world_size, (
            f"D_outer({num_replicas}) * P({pipeline_parallel_size}) != W({Utils.world_size})"
        )

        module_pg_collection = build_colocated_module_process_groups()
        configs = _build_configs(pipeline_parallel_size)
        language_config, _, vision_config, _ = configs
        specs = _build_layer_specs()

        # 参考模型先建、权重从 rank 0 广播，再拷进两个 chunk ⇒ 三者初始权重完全一致。
        reference = _build_reference(configs, specs, module_pg_collection["encoder"])
        _broadcast_reference_weights(reference)
        encoder, backbone = _build_colocated_chunks(configs, specs, module_pg_collection)
        layer_offset = parallel_state.get_pipeline_model_parallel_rank() * _LAYERS_PER_STAGE
        _copy_reference_weights(reference, encoder, backbone, layer_offset)

        microbatches = _make_global_microbatches(num_replicas, num_microbatches)
        reference_grads, total_tokens = _run_reference(
            reference,
            module_pg_collection["encoder"],
            microbatches,
            num_replicas,
            num_microbatches,
        )

        # 共置路径：套 DDP + 挂 finalize（train() 里那段逐 config 赋值的最小复刻；
        # 参考路径的 language config 是另一个对象、其 finalize 仍是 None，两者不串）。
        encoder_chunk = _wrap_with_ddp(encoder, module_pg_collection["encoder"])
        backbone_chunk = _wrap_with_ddp(backbone, module_pg_collection["language_model"])
        for chunk_config in (language_config, vision_config):
            chunk_config.finalize_model_grads_func = finalize_model_grads

        boundary_group = parallel_state.get_colocated_boundary_group()
        producer_id = torch.distributed.get_group_rank(
            boundary_group, torch.distributed.get_rank()
        )
        my_microbatches = parallel_state.get_microbatches_for_producer(
            producer_id, num_microbatches, boundary_group.size()
        )
        replica = parallel_state.get_data_parallel_rank()

        losses = forward_backward_colocated(
            forward_step_func=_make_colocated_forward_step(encoder, backbone),
            data_iterator=_my_data_iterator(microbatches, replica, my_microbatches),
            model=[encoder_chunk, backbone_chunk],
            num_microbatches=num_microbatches,
            seq_length=_COMBINED_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
        )
        if parallel_state.get_pipeline_model_parallel_rank() == pipeline_parallel_size - 1:
            assert len(losses) == num_microbatches, (
                f"末 stage 应产出 {num_microbatches} 条 loss，实际 {len(losses)}"
            )

        # 归约后的梯度 = 全局梯度和 / 全局 token 数。**先乘回分母再比**，让 rtol 作用在
        # 量级正常的数上（除以 token 数会把值缩小两个数量级，那时只有 atol 起作用）。
        checked = 0
        for chunk_module in (encoder, backbone):
            for name, parameter in chunk_module.named_parameters():
                if not parameter.requires_grad:
                    continue
                reference_name = _reference_parameter_name(name, layer_offset)
                assert reference_name in reference_grads, (
                    f"{name} -> {reference_name} 在参考梯度里不存在（参考路径没给它梯度）"
                )
                assert hasattr(parameter, "main_grad"), f"{name} 没有 main_grad（DDP 没生效）"
                actual = parameter.main_grad.float() * total_tokens
                torch.testing.assert_close(
                    actual,
                    reference_grads[reference_name].float(),
                    atol=1e-4,
                    rtol=1e-3,
                    msg=lambda formatted, name=name: f"参数 {name} 的归约后梯度不一致\n{formatted}",
                )
                checked += 1
        assert checked > 0, "一个参数都没比到"
    finally:
        Utils.destroy_model_parallel()
