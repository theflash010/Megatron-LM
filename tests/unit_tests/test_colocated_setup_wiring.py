# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Task 5.11: assembly self-check through the REAL setup_model_and_optimizer path.

Task 5.11：走**真实** ``setup_model_and_optimizer`` 链路的封装自检。

本文件不跑任何 iteration、不需要数据集：建出 model + optimizer 后立即停下，把
"哪个 chunk 拿到哪份 config / 哪个进程组 / 哪个优化器"这些**绑定关系**用断言钉死。
这些关系肉眼追踪极易看漏（同名的 ``config`` 在 training.py 里被复用多次、
``pg_collection`` 存在"传了但没生效"的情况），而现有的共置测试全部自己 new 模块、
绕开了 ``setup_model_and_optimizer``，因此 5.4（按组件建模）、5.5（展平优化器）、
5.6（逐份 config）三层装配代码此前一行都没被真实路径覆盖过。

模型故意用极小的 config，但**类型与生产完全一致**（``ColocatedViTEncoder`` /
``ColocatedGPTBackbone``），provider 也按生产那份的共置分支写（按 ``colocated_module``
分派、vision config 的 PP/CP 置 1、两份 config 是不同对象）。
"""

import sys
from copy import deepcopy

import pytest
import torch

from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.colocated_llava_model import (
    ColocatedGPTBackbone,
    ColocatedViTEncoder,
)
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.tensor_parallel.random import (
    colocated_encoder_rng_tracker,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    group_colocated_model_chunks,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import (
    get_representative_model_chunk,
    group_model_chunks_by_config,
    setup_model_and_optimizer,
)
from megatron.training.utils import calc_params_l2_norm
from tests.unit_tests.test_utilities import Utils

# 视觉侧固定几何：336/14 => 24x24 patch + 1 个 class token = 577，与
# tests/unit_tests/models/test_colocated_llava_model.py 保持一致。
# Fixed vision geometry, matching the existing colocated model test.
IMAGE_HEIGHT = 336
IMAGE_WIDTH = 336
PATCH_DIM = 14
IMAGE_SEQ_LEN = 577
IMAGE_TOKEN_INDEX = -200
LANGUAGE_VOCAB_SIZE = 8192
LANGUAGE_MAX_SEQUENCE_LENGTH = 4096
# 每桶元素数取得足够小，好让两个 chunk 都分出多个桶——桶对称性断言（encoder 组内
# 各 rank 桶数与每桶大小必须一致）只有在真的分了多桶时才有意义。
# Small enough that both chunks form several buckets, otherwise the bucket symmetry
# assertion below would trivially pass with a single bucket.
DDP_BUCKET_SIZE = 1024


def _pipeline_parallel_size():
    """Pick a pipeline size >= 2 that divides the world size (TP=1).

    选一个能整除 world size 且 >= 2 的 pipeline size（TP=1）。P < 2 时共置退化成纯 DP，
    encoder 组与 backbone 组重合，本文件要验证的"两个组件的组必须不同"就无从谈起。
    """
    for candidate in (2, 4, Utils.world_size):
        if candidate > 1 and Utils.world_size % candidate == 0:
            return candidate
    return 1


def _create_test_args(
    pipeline_parallel_size, calculate_per_token_loss=True, **argument_overrides
):
    """Build a minimal but VALIDATED args namespace for colocated training.

    构造一份最小但**经过 validate_args** 的 args：共置路径的前置校验（必须
    per-token loss、禁止 FSDP，arguments.py 的共置校验块）都会在这里真实生效，
    因此这份 args 本身也是被测对象的一部分。``calculate_per_token_loss=False`` 与
    ``argument_overrides`` 供反向用例在调用 ``validate_args`` 前注入非法配置。
    """
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ['test_colocated_setup_wiring.py']
    args = parse_args()
    # backbone：每个 pipeline stage 一层，保证 num_layers % pp_size == 0。
    # Backbone: one layer per pipeline stage.
    args.num_layers = pipeline_parallel_size
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = LANGUAGE_MAX_SEQUENCE_LENGTH
    args.seq_length = 64
    args.decoder_seq_length = LANGUAGE_MAX_SEQUENCE_LENGTH
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = pipeline_parallel_size
    args.context_parallel_size = 1
    args.micro_batch_size = 1
    # global batch = world_size 对任何 data_parallel_size = world_size / P 都可整除。
    # Divisible by micro_batch_size * data_parallel_size for any P.
    args.global_batch_size = Utils.world_size
    args.train_iters = 10
    args.lr = 1e-4
    args.bf16 = True
    args.use_colocated_encoder = True
    # 共置的硬前置条件（否则 encoder 的全 W 单次 SUM 与两层归约不等价）。
    # Hard precondition of the single all-W SUM on the encoder side.
    args.calculate_per_token_loss = calculate_per_token_loss
    args.use_distributed_optimizer = False
    args.num_distributed_optimizer_instances = 1
    args.colocated_encoder_num_distributed_optimizer_instances = 1
    # 显式传进程组集合时上游不允许派生 Gloo 组，共置校验块因此要求关掉它
    # （arguments.py 共置块；Gloo 组只服务 DistOpt 的旧式参数状态存取）。
    # Gloo groups cannot be derived from an explicit collection, so the colocated
    # validation block requires them to be off.
    args.use_gloo_process_groups = False
    # 开 overlap 才会真的按 bucket_size 分桶（DDP 在关闭时把 bucket_size 置 None）。
    # Bucketing only happens with overlap_grad_reduce on.
    args.overlap_grad_reduce = True
    args.ddp_bucket_size = DDP_BUCKET_SIZE
    args.ddp_num_buckets = None
    # 共置不支持评估（轮盘调度要求 microbatch 数是 P 的整数倍，而 eval 的批量口径不保证）
    # ⇒ validate_args 的共置块要求 --eval-iters 0；parse_args 的默认值是 100。
    # Colocated training refuses evaluation, and parse_args defaults eval_iters to 100.
    args.eval_iters = 0

    # Apply invalid-case overrides last so each test reaches the real validation entry point.
    # 非法用例的覆盖值最后写入，保证每条测试都经过真实 validate_args 入口。
    for argument_name, argument_value in argument_overrides.items():
        setattr(args, argument_name, argument_value)

    validate_args(args)
    # build_tokenizer=False：本文件的 provider 不用 tokenizer（vocab 用常量），
    # 也就不需要 tokenizer 文件。set_global_variables 仍会建好 timers 等全局量。
    # No tokenizer needed: the provider uses a literal vocab size.
    set_global_variables(args, False)
    return args


def _build_component_configs():
    """Build the vision / projection / language configs the way production does.

    按生产那份 provider 的做法造三份 config：从 args 得到 base，再 deepcopy 后逐字段
    覆盖（examples/multimodal/model.py:91-196 与 config.py 的 get_vision_model_config
    同款做法——**显式覆盖每一个被改动的派生字段**，因为 dataclass 的 __post_init__ 只在
    构造时跑一次，事后改 hidden_size 不会重算 kv_channels / ffn_hidden_size）。
    关键点：vision 与 language 是**两个不同的对象**，且 vision 侧 PP/CP 都为 1。
    """
    args = get_args()
    base_config = core_transformer_config_from_args(args)
    base_config.calculate_per_token_loss = True

    language_config = deepcopy(base_config)
    language_config.language_model_type = "dummy"

    vision_config = deepcopy(base_config)
    vision_config.vision_model_type = "clip"
    vision_config.num_layers = 2
    vision_config.num_attention_heads = 2
    vision_config.hidden_size = 16
    vision_config.ffn_hidden_size = 64
    vision_config.kv_channels = 8
    vision_config.num_query_groups = 2
    vision_config.gated_linear_unit = False
    vision_config.add_bias_linear = True
    vision_config.add_qkv_bias = True
    vision_config.normalization = "LayerNorm"
    # ViT 不做 PP / CP / SP（生产同款，model.py:165-192）——这正是 encoder chunk 必须
    # 自带一份 config 的原因。The ViT is neither pipeline- nor context-parallel.
    vision_config.pipeline_model_parallel_size = 1
    vision_config.context_parallel_size = 1
    vision_config.sequence_parallel = False
    vision_config.first_pipeline_num_layers = None
    vision_config.last_pipeline_num_layers = None

    projection_config = deepcopy(base_config)
    projection_config.language_model_type = "dummy"
    # 投影层的 hidden_size 是**输出**维度，即语言模型的 hidden（config.py:343）。
    # The projection's hidden_size is its OUTPUT size, i.e. the language hidden size.
    projection_config.hidden_size = language_config.hidden_size
    projection_config.ffn_hidden_size = 32
    projection_config.gated_linear_unit = False
    projection_config.add_bias_linear = False
    projection_config.bias_activation_fusion = False
    projection_config.pipeline_model_parallel_size = 1
    projection_config.context_parallel_size = 1
    projection_config.sequence_parallel = False

    return vision_config, projection_config, language_config


def _colocated_model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
    colocated_module=None,
):
    """Build exactly ONE component per call, mirroring the production provider.

    每次调用只构建一个组件，与生产 provider 的共置分支同构
    （examples/multimodal/model.py:204-293）：按 ``colocated_module`` 分派、把本次调用
    收到的 ``pg_collection`` 透传给组件（ViT block 会从这份集合读 pp/tp）。
    这里刻意不读 tokenizer / 数据集相关 args，vocab 与 image token 用常量。
    """
    assert colocated_module in ("encoder", "language_model"), (
        "colocated model provider must be asked for exactly one component, got "
        f"{colocated_module}"
    )
    vision_config, projection_config, language_config = _build_component_configs()

    submodules = get_gpt_layer_with_transformer_engine_submodules()
    language_layer_spec = ModuleSpec(module=TransformerLayer, submodules=submodules)

    if colocated_module == "encoder":
        vision_layer_spec = ModuleSpec(module=TransformerLayer, submodules=deepcopy(submodules))
        projection_layer_spec = deepcopy(submodules).mlp.submodules
        return ColocatedViTEncoder(
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=projection_config,
            vision_projection_layer_spec=projection_layer_spec,
            img_h=IMAGE_HEIGHT,
            img_w=IMAGE_WIDTH,
            patch_dim=PATCH_DIM,
            pg_collection=pg_collection,
        )

    return ColocatedGPTBackbone(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_layer_spec,
        language_vocab_size=LANGUAGE_VOCAB_SIZE,
        language_max_sequence_length=LANGUAGE_MAX_SEQUENCE_LENGTH,
        pre_process=pre_process,
        post_process=post_process,
        image_token_index=IMAGE_TOKEN_INDEX,
        img_seq_len=IMAGE_SEQ_LEN,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


def test_colocated_setup_binds_each_component_to_its_own_config_groups_and_optimizer():
    """Build through setup_model_and_optimizer and assert every binding.

    走真实 ``setup_model_and_optimizer`` 建出 model + optimizer，然后逐项断言绑定关系。
    分段对应 tasks 5.11 的 ①-⑤：config 身份 / DDP 进程组 / ddp_config 隔离 / 桶对称 /
    优化器结构与参数覆盖。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(
            f"world_size {Utils.world_size} not suitable for a pipeline size >= 2; "
            "with P == 1 the encoder and backbone reduction domains coincide"
        )

    args = _create_test_args(pipeline_parallel_size)
    set_args(args)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        use_colocated_encoder=True,
    )
    model_parallel_cuda_manual_seed(123)

    try:
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            _colocated_model_provider, ModelType.encoder_or_decoder
        )

        # --- ① 组件拆分：两个 chunk，顺序 encoder 在前（checkpoint 的事实约定） ---
        # Component split: exactly two chunks, encoder first.
        assert len(model) == 2, f"expected one chunk per component, got {len(model)}"
        module_names = [
            get_attr_wrapped_model(model_chunk, "colocated_module_name") for model_chunk in model
        ]
        assert module_names == ["encoder", "language_model"], module_names
        chunks_per_module = group_colocated_model_chunks(model)
        encoder_chunk = chunks_per_module["encoder"][0]
        backbone_chunk = chunks_per_module["language_model"][0]
        assert encoder_chunk is model[0]
        assert backbone_chunk is model[1]
        # 走的是 Megatron DDP 路径（共置只支持这一条，arguments.py 已断言）。
        # The Megatron DDP path is the only supported one for colocated training.
        assert isinstance(encoder_chunk, DDP)
        assert isinstance(backbone_chunk, DDP)

        # --- ② config 身份：两份不同对象，各自的并行度描述自己的组件 ---
        # Config identity: two distinct objects, each describing its own component.
        encoder_config = get_model_config(encoder_chunk)
        backbone_config = get_model_config(backbone_chunk)
        assert encoder_config is not backbone_config, (
            "the two components must not share one config object, otherwise the runtime "
            "callbacks written in train() would land on the wrong component"
        )
        assert encoder_config.pipeline_model_parallel_size == 1
        assert encoder_config.context_parallel_size == 1
        assert backbone_config.pipeline_model_parallel_size == pipeline_parallel_size
        # per-token loss 是共置的硬前置（单次全 W SUM 才等价于两层归约），两份都必须为真。
        # Per-token loss is the hard precondition of the single all-W SUM.
        assert encoder_config.calculate_per_token_loss
        assert backbone_config.calculate_per_token_loss
        # train() 写运行期回调时按 config 身份分组，这里断言分组结果正好是"一份 config
        # 对应一个组件的 chunk"——那段循环的正确性就建立在这个前提上（5.6）。
        # The runtime-callback loop in train() groups chunks by config identity; assert the
        # grouping is exactly one config per component.
        chunks_by_config = {
            id(chunk_config): config_model_chunks
            for chunk_config, config_model_chunks in group_model_chunks_by_config(model)
        }
        assert len(chunks_by_config) == 2, (
            "colocated training must expose two distinct configs so that the runtime "
            f"callbacks are written twice, got {len(chunks_by_config)}"
        )
        assert chunks_by_config[id(encoder_config)] == [encoder_chunk]
        assert chunks_by_config[id(backbone_config)] == [backbone_chunk]
        # 5.7：代表性读取必须落在 backbone chunk 上。
        # The representative chunk must be the backbone one.
        assert get_representative_model_chunk(model) is backbone_chunk

        # --- ③ DDP 拿到的进程组：encoder 归约域 = 全 W，backbone = 作业常规 dp-cp ---
        # The groups DDP actually got: the encoder reduces over all W ranks.
        encoder_data_parallel_ranks = torch.distributed.get_process_group_ranks(
            encoder_chunk.dp_cp_group
        )
        backbone_data_parallel_ranks = torch.distributed.get_process_group_ranks(
            backbone_chunk.dp_cp_group
        )
        assert sorted(encoder_data_parallel_ranks) == list(range(Utils.world_size))
        assert sorted(encoder_data_parallel_ranks) == sorted(
            mpu.get_colocated_data_parallel_global_ranks()
        )
        assert sorted(backbone_data_parallel_ranks) == sorted(
            torch.distributed.get_process_group_ranks(
                mpu.get_data_parallel_group(with_context_parallel=True)
            )
        )
        # P >= 2 ⇒ 两个归约域必须真的不同（若相同说明 pg_collection 没生效）。
        # With P >= 2 the two reduction domains must genuinely differ.
        assert set(encoder_data_parallel_ranks) != set(backbone_data_parallel_ranks)
        # encoder 的 pp 组是单成员组：这是"分桶布局全组一致"的来源（5.1/5.4）。
        # The encoder's pipeline group has a single member, which is what makes its
        # bucket layout identical on every rank.
        assert encoder_chunk.pp_group.size() == 1
        assert backbone_chunk.pp_group.size() == pipeline_parallel_size

        # --- ③b ddp_config 隔离：两次 get_model 各建一份，不能是同一个对象 ---
        # Each get_model call builds its own ddp_config; they must not be the same object.
        assert encoder_chunk.ddp_config is not backbone_chunk.ddp_config
        assert encoder_chunk.ddp_config.bucket_size == DDP_BUCKET_SIZE
        assert backbone_chunk.ddp_config.bucket_size == DDP_BUCKET_SIZE
        assert encoder_chunk.ddp_config.num_distributed_optimizer_instances == 1

        # --- ④ encoder chunk 的分桶必须在共置组内各 rank 完全一致 ---
        # 梯度归约是逐 bucket 发起的集合操作，桶数或桶大小不一致会直接死锁（不是精度
        # 问题）。这里 all_gather_object 在 rank 分支之外无条件执行。
        # The gradient reduction is one collective per bucket, so an asymmetric bucket
        # layout deadlocks. The all_gather_object below runs unconditionally on every rank.
        encoder_bucket_sizes = [
            (bucket.numel_unpadded, bucket.grad_data.numel())
            for bucket_group in encoder_chunk.bucket_groups
            for bucket in bucket_group.buckets
        ]
        assert len(encoder_bucket_sizes) > 1, (
            "the encoder was expected to form several buckets with bucket_size "
            f"{DDP_BUCKET_SIZE}, got {len(encoder_bucket_sizes)} - the symmetry check "
            "below would be vacuous"
        )
        gathered_bucket_sizes = [None] * encoder_chunk.dp_cp_group.size()
        torch.distributed.all_gather_object(
            gathered_bucket_sizes, encoder_bucket_sizes, group=encoder_chunk.dp_cp_group
        )
        assert all(
            other_bucket_sizes == encoder_bucket_sizes
            for other_bucket_sizes in gathered_bucket_sizes
        ), f"encoder bucket layout differs across the colocated group: {gathered_bucket_sizes}"

        # --- ⑤ 优化器：一层 ChainedOptimizer、按组件各一个子优化器、梯度统计组不同 ---
        # One flat ChainedOptimizer, one sub-optimizer per component, different
        # gradient-statistics groups.
        assert isinstance(optimizer, ChainedOptimizer)
        assert len(optimizer.chained_optimizers) == 2, (
            "expected exactly one sub-optimizer per component, got "
            f"{len(optimizer.chained_optimizers)}"
        )
        for sub_optimizer in optimizer.chained_optimizers:
            assert not isinstance(sub_optimizer, ChainedOptimizer), (
                "nested ChainedOptimizer breaks _synchronize_steps / save_parameter_state / "
                "_split_state_dict (see get_colocated_optimizer)"
            )
        # 子优化器顺序 = model 列表顺序（_split_state_dict 给 model{i} 编号时依赖它）。
        # 这里不能用 ``optimizer.model_chunks`` 来验证：该属性只有 DistributedOptimizer
        # 会被传入（optimizer/__init__.py:668），非分布式的
        # Float16OptimizerWithFloat16Params 没有，故 ChainedOptimizer 聚合出来是空列表。
        # 因此按**参数**认组件：每个子优化器覆盖的可训练参数必须恰好是该组件那一份。
        # The sub-optimizer order must match the model list. It cannot be checked via
        # ``optimizer.model_chunks``: only DistributedOptimizer receives that argument, so
        # for the dense path the aggregated list is empty. Identify components by parameters.
        encoder_optimizer, backbone_optimizer = optimizer.chained_optimizers

        def _trainable_parameter_shapes(model_chunk):
            return sorted(
                tuple(parameter.shape)
                for parameter in model_chunk.parameters()
                if parameter.requires_grad
            )

        def _optimizer_parameter_shapes(sub_optimizer):
            # bf16 路径下 param_groups 里是 fp32 主参数副本，形状与模型参数一一对应。
            # On the bf16 path these are the fp32 main copies, shaped like the model params.
            return sorted(
                tuple(parameter.shape)
                for param_group in sub_optimizer.param_groups
                for parameter in param_group['params']
            )

        assert _optimizer_parameter_shapes(encoder_optimizer) == _trainable_parameter_shapes(
            encoder_chunk
        ), "the first sub-optimizer must cover exactly the encoder chunk"
        assert _optimizer_parameter_shapes(backbone_optimizer) == _trainable_parameter_shapes(
            backbone_chunk
        ), "the second sub-optimizer must cover exactly the backbone chunk"
        # encoder 参数在 pp 维上是副本 ⇒ 它的梯度统计组只能是 tp 组本身（size 1），
        # 否则范数平方会被加 P 次、encoder 被过度裁剪（5.5 的撤回理由）。
        # The encoder is replicated along the pipeline dimension, so its gradient-statistics
        # group must be the tp group alone; otherwise its squared norm is summed P times.
        assert encoder_optimizer.get_grad_stats_parallel_group().size() == 1
        assert (
            backbone_optimizer.get_grad_stats_parallel_group().size() == pipeline_parallel_size
        )


        # 参数覆盖：优化器的 param_groups 必须覆盖两个 chunk 的全部可训练参数。
        # Parameter coverage across both chunks.
        expected_trainable_numel = sum(
            parameter.numel()
            for model_chunk in model
            for parameter in model_chunk.parameters()
            if parameter.requires_grad
        )
        optimizer_numel = sum(
            parameter.numel()
            for param_group in optimizer.param_groups
            for parameter in param_group['params']
        )
        assert optimizer_numel == expected_trainable_numel, (
            f"optimizer covers {optimizer_numel} elements but the two chunks hold "
            f"{expected_trainable_numel} trainable ones"
        )
        assert opt_param_scheduler is not None

        # --- ⑥ Task 5.12: encoder 的 RNG tracker 已快照，且全 rank 一致 ---
        # After a real setup_model_and_optimizer the encoder's tracker must be usable,
        # and the stream it hands out must be IDENTICAL on every rank: the encoder is a
        # full replica, so all copies have to consume the same random numbers or they
        # stop computing the same function. The seed comes from the encoder's own
        # coordinates (pp single-member => 0), so it does not shift with the backbone's
        # pipeline rank. Note this is the DEFAULT-name fork, i.e. exactly the stream the
        # attention dropout uses (dot_product_attention.py:217) - on the live tracker it
        # would be the backbone's, whose seed does contain the backbone pipeline rank.
        # 走真实装配后，encoder 的 tracker 必须可用，且它交出的流在**每个 rank 上完全一致**：
        # encoder 是完整副本，所有副本必须消费同一批随机数，否则就不再计算同一个函数。种子
        # 按 encoder 自身坐标算（pp 单成员组 => 0），不随 backbone 的 pipeline rank 变化。
        # 注意这里用的是**默认名** fork，也就是 attention dropout 实际使用的那条流
        # （dot_product_attention.py:217）——在活动 tracker 上它会是 backbone 的那条，而
        # backbone 的种子确实含 backbone pipeline rank。
        with colocated_encoder_rng_tracker():
            with get_cuda_rng_tracker().fork():
                encoder_rng_sample = torch.randn(64, device="cuda", dtype=torch.float32)
        gathered_samples = [None] * Utils.world_size
        torch.distributed.all_gather_object(
            gathered_samples, encoder_rng_sample.cpu(), group=torch.distributed.group.WORLD
        )
        reference_sample = gathered_samples[0]
        assert all(
            torch.equal(sample, reference_sample) for sample in gathered_samples[1:]
        ), "the colocated encoder RNG stream must be identical across all replicas"
    finally:
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()


def test_colocated_distributed_optimizer_uses_each_components_instance_groups():
    """Build both components with DistOpt and assert their independent instance layouts.

    两个组件共同开启 DistOpt，但分别使用自己的 instance 数与通信组。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2 or Utils.world_size < 2:
        pytest.skip(f"需要 world_size >= 2 且 P >= 2，当前 world_size={Utils.world_size}")

    encoder_num_distributed_optimizer_instances = 2
    args = _create_test_args(
        pipeline_parallel_size,
        use_distributed_optimizer=True,
        ckpt_format="torch_dist",
        colocated_encoder_num_distributed_optimizer_instances=(
            encoder_num_distributed_optimizer_instances
        ),
    )
    set_args(args)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        use_colocated_encoder=True,
        colocated_encoder_num_distributed_optimizer_instances=(
            args.colocated_encoder_num_distributed_optimizer_instances
        ),
    )
    model_parallel_cuda_manual_seed(123)

    try:
        model, optimizer, _ = setup_model_and_optimizer(
            _colocated_model_provider, ModelType.encoder_or_decoder
        )
        chunks_per_module = group_colocated_model_chunks(model)
        encoder_chunk = chunks_per_module["encoder"][0]
        backbone_chunk = chunks_per_module["language_model"][0]

        assert encoder_chunk.ddp_config.use_distributed_optimizer
        assert backbone_chunk.ddp_config.use_distributed_optimizer
        assert (
            encoder_chunk.ddp_config.num_distributed_optimizer_instances
            == encoder_num_distributed_optimizer_instances
        )
        assert (
            backbone_chunk.ddp_config.num_distributed_optimizer_instances
            == args.num_distributed_optimizer_instances
            == 1
        )

        encoder_data_parallel_ranks = mpu.get_colocated_data_parallel_global_ranks()
        encoder_intra_group_size = (
            len(encoder_data_parallel_ranks)
            // encoder_num_distributed_optimizer_instances
        )
        caller_index = encoder_data_parallel_ranks.index(Utils.rank)
        encoder_instance_index = caller_index // encoder_intra_group_size
        encoder_shard_index = caller_index % encoder_intra_group_size
        expected_encoder_intra_ranks = encoder_data_parallel_ranks[
            encoder_instance_index
            * encoder_intra_group_size : (encoder_instance_index + 1)
            * encoder_intra_group_size
        ]
        expected_encoder_inter_ranks = encoder_data_parallel_ranks[
            encoder_shard_index::encoder_intra_group_size
        ]
        assert (
            mpu.get_colocated_encoder_intra_distributed_optimizer_instance_global_ranks()
            == expected_encoder_intra_ranks
        )
        assert (
            mpu.get_colocated_encoder_inter_distributed_optimizer_instance_global_ranks()
            == expected_encoder_inter_ranks
        )
        assert torch.distributed.get_process_group_ranks(
            encoder_chunk.intra_dp_cp_group
        ) == expected_encoder_intra_ranks
        assert torch.distributed.get_process_group_ranks(
            encoder_chunk.inter_dist_opt_group
        ) == expected_encoder_inter_ranks
        assert backbone_chunk.intra_dp_cp_group is backbone_chunk.dp_cp_group
        assert getattr(backbone_chunk, "inter_dist_opt_group", None) is None

        assert isinstance(optimizer, ChainedOptimizer)
        assert len(optimizer.chained_optimizers) == 2
        encoder_optimizer, backbone_optimizer = optimizer.chained_optimizers
        assert isinstance(encoder_optimizer, DistributedOptimizer)
        assert isinstance(backbone_optimizer, DistributedOptimizer)
        assert (
            encoder_optimizer.data_parallel_group
            is mpu.get_colocated_encoder_intra_distributed_optimizer_instance_group()
        )
        assert backbone_optimizer.data_parallel_group is backbone_chunk.intra_dp_cp_group
        assert encoder_optimizer.data_parallel_group is not backbone_optimizer.data_parallel_group
    finally:
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()


def _encoder_shard_layout(encoder_chunk, intra_group_size):
    """Describe the encoder's buffer layout in a rank-independent, comparable form.

    把 encoder 的 buffer 布局写成**不含 rank 身份**的描述：参数顺序、每个 buffer 的
    padding 前后元素数、每个桶的分片边界、以及每个参数在桶内的 (start, end)。
    ``param_to_index`` 正是生产给 ``main_grad`` 建视图用的那份映射
    （param_and_grad_buffer.py:117-120），所以它一致才说明"同名参数在每个 rank 的同一
    位置"；``shard_buffer`` 也直接用生产那一个，避免测试自己算一套等分逻辑。
    """
    names_by_parameter = {
        parameter: name for name, parameter in encoder_chunk.module.named_parameters()
    }
    layout = {
        "parameter_order": [
            name
            for name, parameter in encoder_chunk.module.named_parameters()
            if parameter.requires_grad
        ],
        "buffers": [],
    }
    for buffer in list(encoder_chunk.buffers) + list(encoder_chunk.expert_parallel_buffers):
        buckets = []
        for bucket in buffer.buckets:
            buckets.append(
                {
                    "numel_unpadded": bucket.numel_unpadded,
                    "grad_numel": bucket.grad_data.numel(),
                    "shard_numels": [
                        shard.numel()
                        for shard in shard_buffer(bucket.grad_data, intra_group_size)
                    ],
                    "param_to_index": sorted(
                        (names_by_parameter[parameter], tuple(bucket.param_to_index[parameter]))
                        for parameter in bucket.params_list
                    ),
                }
            )
        layout["buffers"].append(
            {"numel": buffer.numel, "numel_unpadded": buffer.numel_unpadded, "buckets": buckets}
        )
    return layout


def test_colocated_encoder_distributed_optimizer_shard_layout_is_identical_on_every_rank():
    """Task 8.5: the encoder's parameter order, bucket padding and shard offsets must match.

    encoder 是完整副本，DistOpt 又把每个 bucket 按 intra 组大小等分 ⇒ **参数顺序、桶的
    padding 后大小、以及每个参数在桶内的 (start, end)** 必须在共置 DP 组的每个 rank 上
    完全一致。不一致不是精度问题而是硬故障：reduce-scatter 与 all-gather 的收发长度由这份
    布局决定，rank 之间错一位就会归约到别的参数上（且不会报错）。因此这里比较的是**整份
    布局描述**，而不只是桶数。
    多 instance 配置（intra=2、inter=2）下同样必须一致：分片是在 intra 组内等分，两个
    instance 的边界互为镜像，布局描述本身不含 rank 身份。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2 or Utils.world_size < 4:
        pytest.skip(
            f"需要 world_size >= 4 且 P >= 2 才能让 encoder DP 组容纳 2 个 instance，"
            f"当前 world_size={Utils.world_size}"
        )

    encoder_num_distributed_optimizer_instances = 2
    args = _create_test_args(
        pipeline_parallel_size,
        use_distributed_optimizer=True,
        ckpt_format="torch_dist",
        colocated_encoder_num_distributed_optimizer_instances=(
            encoder_num_distributed_optimizer_instances
        ),
    )
    set_args(args)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        use_colocated_encoder=True,
        colocated_encoder_num_distributed_optimizer_instances=(
            args.colocated_encoder_num_distributed_optimizer_instances
        ),
    )
    model_parallel_cuda_manual_seed(123)

    try:
        model, optimizer, _ = setup_model_and_optimizer(
            _colocated_model_provider, ModelType.encoder_or_decoder
        )
        encoder_chunk = group_colocated_model_chunks(model)["encoder"][0]
        encoder_optimizer = optimizer.chained_optimizers[0]
        intra_group_size = encoder_chunk.intra_dp_cp_group.size()
        assert intra_group_size == (
            Utils.world_size // encoder_num_distributed_optimizer_instances
        ), "the encoder intra-instance group must hold DP_size / instances ranks"

        layout = _encoder_shard_layout(encoder_chunk, intra_group_size)
        # 集合操作在任何 rank 分支之外无条件执行。
        # The collective runs unconditionally on every rank.
        gathered_layouts = [None] * Utils.world_size
        torch.distributed.all_gather_object(
            gathered_layouts, layout, group=mpu.get_colocated_data_parallel_group()
        )
        for other_rank, other_layout in enumerate(gathered_layouts):
            assert other_layout == layout, (
                "encoder shard layout differs between this rank and rank "
                f"{other_rank}: {other_layout} vs {layout}"
            )

        # 主参数分片：**不能**要求所有 rank 等大。DistOpt 的主参数覆盖本 rank 那段 buffer
        # 区间里的**真实参数**，末尾那段 padding 不属于任何参数 ⇒ 持有尾部分片的 rank 会少
        # padding 那么多个元素（实测 world=4、encoder instances=2 下是 13984 / 13856）。
        # 真正该钉住的不变量有两条：
        #   ⑴ **instance 之间逐位对称**——两个 instance 的第 i 个分片必须一样大，否则 inter
        #      组的 all-reduce 在两侧长度不同，直接是死锁或错位；
        #   ⑵ **一个 instance 的分片之和 == encoder 全部可训练参数量**——既不漏也不重复。
        # The main-parameter shards are NOT all equal: the trailing shard omits the buffer
        # padding. What must hold is instance-to-instance symmetry and exact coverage.
        main_parameter_numel = sum(
            parameter.numel()
            for param_group in encoder_optimizer.param_groups
            for parameter in param_group['params']
        )
        gathered_numels = [None] * Utils.world_size
        torch.distributed.all_gather_object(
            gathered_numels, main_parameter_numel, group=mpu.get_colocated_data_parallel_group()
        )
        shards_per_instance = [
            gathered_numels[start : start + intra_group_size]
            for start in range(0, Utils.world_size, intra_group_size)
        ]
        assert len(shards_per_instance) == encoder_num_distributed_optimizer_instances
        for instance_index, instance_shards in enumerate(shards_per_instance):
            assert instance_shards == shards_per_instance[0], (
                "the encoder shard sizes differ between DistOpt instances: instance "
                f"{instance_index} holds {instance_shards}, instance 0 holds "
                f"{shards_per_instance[0]}"
            )
        trainable_parameter_numel = sum(
            parameter.numel()
            for parameter in encoder_chunk.module.parameters()
            if parameter.requires_grad
        )
        assert sum(shards_per_instance[0]) == trainable_parameter_numel, (
            f"one instance covers {sum(shards_per_instance[0])} elements but the encoder holds "
            f"{trainable_parameter_numel} trainable ones"
        )
    finally:
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()


def test_colocated_requires_per_token_loss():
    """Reject non-summed encoder gradients during startup validation.

    关掉 --calculate-per-token-loss 必须在启动校验就报错（Task 5.8 的反向用例）。

    这条前提承重：encoder 的 DDP 拿到的是横跨全部 W 个 rank 的数据并行组，一次归约
    同时覆盖 inner 与 outer 两维，只有在归约为**纯 SUM** 时才与"先 inner 求和再 outer
    归约"等价（per-token 模式下 gradient_scaling_factor == 1.0，
    distributed_data_parallel.py:169-174）。否则 DDP 按 1/(D_outer*P) 缩放、比正确的
    1/D_outer 小 P 倍，而且**静默算错**——所以必须有用例钉住这条报错还在。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(f"需要 P >= 2，当前 world_size={Utils.world_size}")
    try:
        with pytest.raises(AssertionError, match="calculate-per-token-loss"):
            _create_test_args(pipeline_parallel_size, calculate_per_token_loss=False)
    finally:
        destroy_global_vars()
        destroy_num_microbatches_calculator()


def test_colocated_requires_microbatch_count_divisible_by_pipeline_size():
    """Report the complete round-robin constraint before model construction.

    在启动校验期报告完整的轮盘整除约束，错误中必须包含实际批量与可接受倍数。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(f"需要 P >= 2，当前 world_size={Utils.world_size}")

    data_parallel_size = Utils.world_size // pipeline_parallel_size
    try:
        with pytest.raises(AssertionError) as error:
            _create_test_args(
                pipeline_parallel_size,
                global_batch_size=data_parallel_size,
            )
        message = str(error.value)
        assert "number of microbatches to be a multiple" in message
        assert f"pipeline model parallel size ({pipeline_parallel_size})" in message
        assert f"--global-batch-size {data_parallel_size}" in message
        assert f"pick a global batch size that is a multiple of {Utils.world_size}" in message
    finally:
        destroy_global_vars()
        destroy_num_microbatches_calculator()


@pytest.mark.parametrize(
    ("argument_overrides", "error_message"),
    (
        ({"create_all_gather_group": True}, "create-all-gather-group"),
        ({"use_gloo_process_groups": True}, "disable-gloo-process-groups"),
        ({"eval_iters": 1}, "does not support evaluation"),
        ({"full_validation": True}, "does not support evaluation"),
        ({"online_evaluation_config": object()}, "online-evaluation-config"),
        ({"rampup_batch_size": [1, 1, 1]}, "changing batch size"),
        ({"decrease_batch_size_if_needed": True}, "decrease-batch-size-if-needed"),
    ),
)
def test_colocated_rejects_unsupported_training_modes(argument_overrides, error_message):
    """Keep unsupported modes from silently bypassing the component-specific wiring.

    不支持的训练模式必须在启动期报出共置专属错误，不能静默绕过按组件接线。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(f"需要 P >= 2，当前 world_size={Utils.world_size}")
    try:
        with pytest.raises(AssertionError, match=error_message):
            _create_test_args(pipeline_parallel_size, **argument_overrides)
    finally:
        destroy_global_vars()
        destroy_num_microbatches_calculator()


def test_training_config_enables_colocated_encoder_output_deallocation():
    """Exercise the args-to-config policy instead of adding a redundant CLI flag.

    验证训练入口自动开启 encoder 输出伪释放，而不是为该内部策略重复增加 CLI 参数。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(f"需要 P >= 2，当前 world_size={Utils.world_size}")
    try:
        args = _create_test_args(pipeline_parallel_size)
        config = core_transformer_config_from_args(args)
        assert config.deallocate_encoder_outputs
    finally:
        destroy_global_vars()
        destroy_num_microbatches_calculator()


def test_colocated_encoder_tensor_model_parallel_group_is_isolated():
    """encoder 的 tp 组是独立实例、成员与作业 tp 组相同，且被 encoder 的进程组集合采用。

    Task 6.5：encoder 的张量并行不再隐式借用 mpu 的组。当前只支持"encoder tp 并行度 ==
    作业 tp 并行度"（边界包按 tp 槽位一一对应发送），所以**成员必须相同**；但对象必须
    不同，否则 encoder 的通信仍然发生在 backbone 的通信子上，取数广播与 backbone 的
    张量并行集合操作会互相排队。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(f"需要 P >= 2，当前 world_size={Utils.world_size}")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        use_colocated_encoder=True,
    )
    try:
        encoder_tensor_group = mpu.get_colocated_encoder_tensor_model_parallel_group()
        job_tensor_group = mpu.get_tensor_model_parallel_group()
        assert encoder_tensor_group is not job_tensor_group, (
            "the colocated encoder tensor model parallel group must be its own NCCL "
            "instance, not the job's group object"
        )
        assert torch.distributed.get_process_group_ranks(
            encoder_tensor_group
        ) == torch.distributed.get_process_group_ranks(job_tensor_group)
        assert (
            mpu.get_colocated_encoder_tensor_model_parallel_global_ranks()
            == torch.distributed.get_process_group_ranks(job_tensor_group)
        )

        # encoder 的进程组集合里 tp 与 mp 都必须指向它：mp 是梯度范数归约的域，
        # encoder 的 pp 组单成员 ⇒ tp × pp 就是 tp 组本身。
        pg_collection = mpu.build_colocated_encoder_process_groups()
        assert pg_collection.tp is encoder_tensor_group
        assert pg_collection.mp is encoder_tensor_group
    finally:
        Utils.destroy_model_parallel()


def _component_sum_of_squares(model_chunk):
    """Independently sum one chunk's squared parameters, mirroring the norm's own filters.

    按 ``calc_params_l2_norm`` 的取值口径独立算一遍平方和：跳过张量并行副本与 shared
    参数，bf16 下取 fp32 主副本 ``main_param``。用 float64 累加，使参考值与被测实现
    （fp32 的 ``multi_tensor_l2norm``）之间的差异只来自累加精度，而不是口径不同。
    """
    total = torch.zeros((1,), dtype=torch.float64, device='cuda')
    for parameter in model_chunk.parameters():
        if not param_is_not_tensor_parallel_duplicate(parameter):
            continue
        if not param_is_not_shared(parameter):
            continue
        parameter_data = getattr(parameter, 'main_param', None)
        if parameter_data is None:
            parameter_data = parameter.data
        total += parameter_data.detach().double().pow(2).sum()
    return total


def test_colocated_params_l2_norm_counts_the_encoder_once():
    """Task 6.4: the logged parameter norm must reduce each component over its own group.

    共置下 encoder 在 pipeline 维上是副本，若沿作业的 tp x pp 组求平方和就会把它算 P
    次，日志里的 params norm 因此偏大（只影响观测，不影响更新——这个量不参与裁剪）。
    本用例独立算出期望值
        $$\\sqrt{\\|w^{enc}\\|^2+\\sum_s \\|w^{bb}_s\\|^2}$$
    并同时算出改动前的偏大值
        $$\\sqrt{P\\|w^{enc}\\|^2+\\sum_s \\|w^{bb}_s\\|^2}$$
    ，既断言等于前者，也断言不等于后者——后一条保证用例真的有分辨力。
    """
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(
            f"world_size {Utils.world_size} not suitable for a pipeline size >= 2; "
            "with P == 1 the encoder is not replicated along the pipeline dimension and "
            "the over-counting this test targets cannot happen"
        )

    args = _create_test_args(pipeline_parallel_size)
    set_args(args)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        use_colocated_encoder=True,
    )
    model_parallel_cuda_manual_seed(123)

    try:
        model, _, _ = setup_model_and_optimizer(
            _colocated_model_provider, ModelType.encoder_or_decoder
        )
        chunks_per_module = group_colocated_model_chunks(model)
        encoder_chunk = chunks_per_module["encoder"][0]
        backbone_chunk = chunks_per_module["language_model"][0]

        # 前提：两个组件的归约域必须真的不同，否则本用例退化成恒真。
        # Precondition: the two reduction domains must genuinely differ.
        encoder_reduce_ranks = torch.distributed.get_process_group_ranks(
            mpu.get_colocated_encoder_tensor_model_parallel_group()
        )
        backbone_reduce_ranks = torch.distributed.get_process_group_ranks(
            mpu.get_model_parallel_group()
        )
        assert encoder_reduce_ranks != backbone_reduce_ranks
        assert len(backbone_reduce_ranks) == pipeline_parallel_size

        encoder_sum_of_squares = _component_sum_of_squares(encoder_chunk)
        backbone_sum_of_squares = _component_sum_of_squares(backbone_chunk)

        # encoder 在每个 rank 上都是完整副本：先钉住"副本真的一致"，否则期望值本身没有
        # 意义。all_gather 无条件执行、不放在 rank 分支里。
        # The encoder replicas must be identical, otherwise the expectation below is
        # meaningless. The collective runs unconditionally on every rank.
        gathered_encoder_sums = [
            torch.zeros_like(encoder_sum_of_squares) for _ in range(Utils.world_size)
        ]
        torch.distributed.all_gather(gathered_encoder_sums, encoder_sum_of_squares.contiguous())
        assert all(
            torch.allclose(other_sum, gathered_encoder_sums[0])
            for other_sum in gathered_encoder_sums
        ), (
            "the encoder replicas differ across the colocated group, so the whole "
            f"replica assumption is broken: {[s.item() for s in gathered_encoder_sums]}"
        )

        backbone_total_sum_of_squares = backbone_sum_of_squares.clone()
        torch.distributed.all_reduce(
            backbone_total_sum_of_squares,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_model_parallel_group(),
        )

        encoder_total_sum_of_squares = gathered_encoder_sums[0]
        expected_norm = (encoder_total_sum_of_squares + backbone_total_sum_of_squares).sqrt().item()
        over_counted_norm = (
            (
                encoder_total_sum_of_squares * pipeline_parallel_size
                + backbone_total_sum_of_squares
            )
            .sqrt()
            .item()
        )
        assert expected_norm < over_counted_norm, (
            "the encoder holds no parameters with a non-zero norm, so this test cannot "
            "tell the correct value from the over-counted one"
        )

        norm = calc_params_l2_norm(model)
        assert norm == pytest.approx(expected_norm, rel=1e-5)
        assert norm != pytest.approx(over_counted_norm, rel=1e-3)
    finally:
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()










