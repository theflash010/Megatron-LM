# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests of the colocated encoder arguments (Task 6.5).

校验 ``examples/multimodal/colocated_args.py`` 的默认值填充与三条断言。纯参数校验，
不需要分布式环境：``validate_colocated_args`` 只读写 args 本身。
"""

import os
import sys
from argparse import ArgumentParser, Namespace

import pytest

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "examples",
        "multimodal",
    )
)

from colocated_args import add_colocated_extra_args, validate_colocated_args  # noqa: E402


def _args(**overrides):
    args = Namespace(
        use_colocated_encoder=True,
        world_size=8,
        tensor_model_parallel_size=2,
        colocated_encoder_tensor_model_parallel_size=None,
        colocated_encoder_pipeline_model_parallel_size=1,
        colocated_encoder_context_parallel_size=1,
        colocated_encoder_num_distributed_optimizer_instances=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_encoder_tensor_parallel_size_defaults_to_the_job_size():
    args = _args()
    validate_colocated_args(args)
    assert args.colocated_encoder_tensor_model_parallel_size == 2


def test_encoder_distributed_optimizer_instances_argument_is_registered():
    parser = add_colocated_extra_args(ArgumentParser())
    args = parser.parse_args(
        [
            "--use-colocated-encoder",
            "--language-model-type",
            "mistral",
            "--tokenizer-prompt-format",
            "mistral",
            "--colocated-encoder-num-distributed-optimizer-instances",
            "2",
        ]
    )
    assert args.colocated_encoder_num_distributed_optimizer_instances == 2


def test_non_colocated_run_is_left_untouched():
    """非共置入口不该被这些字段影响：默认值也不填、断言也不跑。"""
    args = _args(use_colocated_encoder=False, colocated_encoder_context_parallel_size=4)
    validate_colocated_args(args)
    assert args.colocated_encoder_tensor_model_parallel_size is None


def test_encoder_tensor_parallel_size_must_equal_the_job_size():
    args = _args(colocated_encoder_tensor_model_parallel_size=1)
    with pytest.raises(AssertionError, match="must equal the job's tensor"):
        validate_colocated_args(args)


def test_encoder_pipeline_parallel_size_must_be_one():
    args = _args(colocated_encoder_pipeline_model_parallel_size=2)
    with pytest.raises(AssertionError, match="pipeline model parallel size must be 1"):
        validate_colocated_args(args)


def test_encoder_context_parallel_size_must_be_one():
    args = _args(colocated_encoder_context_parallel_size=2)
    with pytest.raises(AssertionError, match="context parallel size must be 1"):
        validate_colocated_args(args)


def test_encoder_distributed_optimizer_instances_must_be_positive():
    args = _args(colocated_encoder_num_distributed_optimizer_instances=0)
    with pytest.raises(AssertionError, match="instances must be greater than 0"):
        validate_colocated_args(args)


def test_encoder_data_parallel_size_must_be_divisible_by_optimizer_instances():
    args = _args(colocated_encoder_num_distributed_optimizer_instances=3)
    with pytest.raises(AssertionError, match="data parallel size .* must be divisible"):
        validate_colocated_args(args)


def test_encoder_distributed_optimizer_instances_are_independent_from_backbone():
    args = _args(
        use_distributed_optimizer=True,
        tensor_model_parallel_size=1,
        colocated_encoder_tensor_model_parallel_size=1,
        num_distributed_optimizer_instances=1,
        colocated_encoder_num_distributed_optimizer_instances=2,
    )
    validate_colocated_args(args)
    assert args.use_distributed_optimizer
    assert args.colocated_encoder_num_distributed_optimizer_instances == 2
    assert args.num_distributed_optimizer_instances == 1
