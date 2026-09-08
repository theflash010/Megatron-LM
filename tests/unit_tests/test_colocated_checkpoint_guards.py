# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Task 7.8 negative cases: the colocated checkpoint key guards must FAIL loudly.

Task 7.8 的负例：共置 checkpoint 的键校验必须**响亮失败**。

7.8 的正例已由真实 16GB 产物验证过（tests/functional_tests/colocated/
load_colocated_checkpoint.py），但那条路径只能证明"对的 checkpoint 能加载"。本文件补的是
反面：三类坏 checkpoint 必须抛错，而不是被上游那两处静默降级吞掉——
  * ``if 'model%d' % i not in state_dict: continue``（checkpointing.py:2005-2007）
    把缺失的 chunk 当成"空 stage"直接跳过；
  * ``load_model_state_dict``（:1893-1901）在 strict 加载抛异常时改用 ``strict=False``
    重试、只 print 一行，于是键名写错的参数会停在随机初始化上继续训练。
两者都不报错、只静默算错，因此 ``_check_colocated_chunk_keys`` 是唯一的拦截点。

模型用 5.11 那套 fixture 真实建出来（``ColocatedViTEncoder`` / ``ColocatedGPTBackbone``），
所以断言比对的是**真实键名**：将来谁改了某个子模块的命名，这里会跟着红。
"""

import pytest

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import get_attr_wrapped_model
from megatron.training.checkpointing import _check_colocated_chunk_keys
from megatron.training.global_vars import destroy_global_vars, set_args
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_colocated_setup_wiring import (
    _colocated_model_provider,
    _create_test_args,
    _pipeline_parallel_size,
)
from tests.unit_tests.test_utilities import Utils


def _build_colocated_model(pipeline_parallel_size):
    """Build the two-chunk colocated model through the real setup path."""
    args = _create_test_args(pipeline_parallel_size)
    set_args(args)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        use_colocated_encoder=True,
    )
    model_parallel_cuda_manual_seed(123)
    model, _, _ = setup_model_and_optimizer(_colocated_model_provider, ModelType.encoder_or_decoder)
    return model


def _valid_state_dict(model):
    """Produce the state dict a correct colocated checkpoint would carry.

    键取自 chunk 自己的 ``state_dict()``，元数据按 combine_colocated_checkpoints.py 的写法
    给出（``colocated_chunk_modules`` = 下标到组件名的映射）。
    """
    state_dict = {
        'colocated_chunk_modules': [
            get_attr_wrapped_model(model_chunk, 'colocated_module_name') for model_chunk in model
        ]
    }
    for chunk_index, model_chunk in enumerate(model):
        state_dict['model%d' % chunk_index] = dict(
            get_attr_wrapped_model(model_chunk, 'state_dict')()
        )
    return state_dict


def test_colocated_chunk_key_guards_reject_broken_checkpoints():
    """One model build, six assertions: one positive and five distinct failures."""
    pipeline_parallel_size = _pipeline_parallel_size()
    if pipeline_parallel_size < 2:
        pytest.skip(
            f"world_size {Utils.world_size} not suitable for a pipeline size >= 2; "
            "with P == 1 there is no pipeline dimension to replicate the encoder over"
        )

    model = _build_colocated_model(pipeline_parallel_size)

    try:
        assert mpu.is_colocated_encoder_enabled()
        assert len(model) == 2

        # --- ① 正例：真实键 + 正确元数据 ⇒ 不抛错 ---
        # Positive case: real keys plus correct metadata must pass.
        _check_colocated_chunk_keys(model, _valid_state_dict(model))

        # --- ② 缺元数据：无法按身份核对 chunk 顺序 ⇒ 必须抛错 ---
        # Missing metadata: the chunk order cannot be verified by identity.
        state_dict = _valid_state_dict(model)
        del state_dict['colocated_chunk_modules']
        with pytest.raises(AssertionError, match='colocated_chunk_modules'):
            _check_colocated_chunk_keys(model, state_dict)

        # --- ③ 顺序颠倒：encoder 的权重会被灌进 backbone chunk ⇒ 必须抛错 ---
        # Reversed order would load the encoder weights into the backbone chunk.
        state_dict = _valid_state_dict(model)
        state_dict['colocated_chunk_modules'] = list(
            reversed(state_dict['colocated_chunk_modules'])
        )
        with pytest.raises(AssertionError, match='chunk order'):
            _check_colocated_chunk_keys(model, state_dict)

        # --- ④ 只有单个 ``model`` 键（非共置 ckpt 的形状）⇒ 必须抛错 ---
        # A single ``model`` key is the NON-colocated layout; it must not be accepted.
        state_dict = _valid_state_dict(model)
        state_dict['model'] = state_dict.pop('model0')
        del state_dict['model1']
        with pytest.raises(AssertionError, match='missing'):
            _check_colocated_chunk_keys(model, state_dict)

        # --- ⑤ 少给 backbone 那个 chunk ⇒ 上游会静默 continue，这里必须抛错 ---
        # Upstream would silently ``continue``; every colocated chunk is non-empty.
        state_dict = _valid_state_dict(model)
        del state_dict['model1']
        with pytest.raises(AssertionError, match='missing'):
            _check_colocated_chunk_keys(model, state_dict)

        # --- ⑥ 键名前缀写错：上游会降级成 strict=False、参数停在随机初始化 ⇒ 必须抛错 ---
        # A wrong prefix makes upstream fall back to strict=False and leave the
        # parameters at their random initialization.
        state_dict = _valid_state_dict(model)
        state_dict['model0'] = {
            name.replace('vision_model.', 'vit.', 1): value
            for name, value in state_dict['model0'].items()
        }
        with pytest.raises(AssertionError, match='does not match'):
            _check_colocated_chunk_keys(model, state_dict)
    finally:
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
