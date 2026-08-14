# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the encoder inner data-parallel group (colocated encoder training).

encoder 内部数据并行组（共置训练）的单元测试。
"""

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank
world_size = Utils.world_size


def _pp_size():
    """Pick a pipeline size >= 2 that divides the world size (TP=1).

    选一个能整除 world size 且 >= 2 的 pipeline size（TP=1）。
    """
    for cand in (2, 4, world_size):
        if cand > 1 and world_size % cand == 0:
            return cand
    return 1


def test_encoder_inner_dp_group_requires_flag():
    """Without use_colocated_encoder the getter must raise.

    未开启 use_colocated_encoder 时 getter 必须抛异常。
    """
    pp = _pp_size()
    if pp < 2:
        pytest.skip(f"world_size {world_size} not suitable for a pipeline size >= 2")
    Utils.initialize_model_parallel(pipeline_model_parallel_size=pp)
    # Without use_colocated_encoder the getter must raise.
    # 未开启 use_colocated_encoder 时 getter 必须抛 AssertionError。
    with pytest.raises(AssertionError):
        ps.get_encoder_inner_data_parallel_group()
    Utils.destroy_model_parallel()
    assert ps._ENCODER_INNER_DATA_PARALLEL_GROUP is None


def test_encoder_inner_dp_group_initialized():
    """With the flag, the inner group is created and matches the pp group.

    开启 flag 后 inner 组被创建，且成员与 pp 组一致、外部 dp 组不受影响。
    """
    pp = _pp_size()
    if pp < 2:
        pytest.skip(f"world_size {world_size} not suitable for a pipeline size >= 2")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        use_colocated_encoder=True,
    )

    inner_group = ps.get_encoder_inner_data_parallel_group()
    assert inner_group is not None
    # Inner group contains one rank per pipeline stage (the whole replica).
    # inner 组包含每个 pipeline stage 一个 rank（即整个外层副本）。
    assert inner_group.size() == ps.get_pipeline_model_parallel_world_size() == pp
    # Inner rank equals the pipeline stage index.
    # inner 组内编号等于 pipeline stage 序号。
    assert ps.get_encoder_inner_data_parallel_rank() == ps.get_pipeline_model_parallel_rank()
    # Inner group members coincide with the pipeline-parallel group members.
    # inner 组成员与 pipeline-parallel 组成员一致。
    assert sorted(ps.get_encoder_inner_data_parallel_global_ranks()) == sorted(
        torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    )

    # The regular (outer) data-parallel group is unaffected.
    # 常规（外层）dp 组不受影响。
    assert ps.get_data_parallel_group().size() == world_size // pp
    # Tensor parallel group is still intact.
    # tp 组依然正常。
    assert ps.get_tensor_model_parallel_group().size() == 1

    Utils.destroy_model_parallel()
    assert ps._ENCODER_INNER_DATA_PARALLEL_GROUP is None
    assert ps._ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS is None


def test_round_robin_microbatch_mapping():
    """Round-robin microbatch mapping and validation.

    轮盘式 microbatch 映射与参数校验。
    """
    pp = _pp_size()
    if pp < 2:
        pytest.skip(f"world_size {world_size} not suitable for a pipeline size >= 2")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        use_colocated_encoder=True,
    )

    stage = ps.get_pipeline_model_parallel_rank()
    # num_microbatches == pp_size: stage s handles exactly microbatch s.
    # num_microbatches == pp_size 时，stage s 恰好负责 microbatch s。
    assert ps.get_microbatches_for_pipeline_stage(stage, num_microbatches=pp) == [stage]
    # num_microbatches == 2 * pp_size: stage s handles s and s + pp_size.
    # num_microbatches == 2 * pp_size 时，stage s 负责 s 和 s + pp_size。
    expected = [stage, stage + pp]
    assert ps.get_microbatches_for_pipeline_stage(stage, num_microbatches=2 * pp) == expected
    # All stages together cover every microbatch exactly once.
    # 所有 stage 合起来恰好覆盖每个 microbatch 一次。
    all_mbs = sorted(
        mb for s in range(pp) for mb in ps.get_microbatches_for_pipeline_stage(s, num_microbatches=2 * pp)
    )
    assert all_mbs == list(range(2 * pp))

    # num_microbatches not a multiple of pp_size raises.
    # num_microbatches 不是 pp_size 的整数倍时抛异常。
    with pytest.raises(AssertionError):
        ps.get_microbatches_for_pipeline_stage(stage, num_microbatches=pp + 1)
    with pytest.raises(AssertionError):
        ps.validate_colocated_num_microbatches(pp + 1)
    # num_microbatches == 0 is invalid as well.
    # num_microbatches == 0 同样是非法输入。
    with pytest.raises(AssertionError):
        ps.validate_colocated_num_microbatches(0)

    Utils.destroy_model_parallel()


def test_pipeline_parallel_groups_unaffected_by_flag():
    """Enabling the colocated flag must not alter pre-existing groups.

    开启共置 flag 不得改变既有通信组。
    """
    pp = _pp_size()
    if pp < 2:
        pytest.skip(f"world_size {world_size} not suitable for a pipeline size >= 2")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        use_colocated_encoder=True,
    )

    pp_group = ps.get_pipeline_model_parallel_group()
    assert pp_group.size() == pp
    assert len(torch.distributed.get_process_group_ranks(pp_group)) == pp

    Utils.destroy_model_parallel()
