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

# The two rank orders Megatron can be launched with: the default and the one
# --use-tp-pp-dp-mapping selects (initialize.py:355). Colocated correctness must not
# depend on the order (Task 5.7), so the structural cases run under both.
# 两种 rank order：默认的与 --use-tp-pp-dp-mapping 选中的（initialize.py:355 的映射）。
# 共置的正确性不得依赖 order（Task 5.7），因此结构性用例在两种 order 下各跑一遍。
RANK_ORDERS = ("tp-cp-ep-dp-pp", "tp-cp-ep-pp-dp")


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


@pytest.mark.parametrize("order", RANK_ORDERS)
def test_encoder_inner_dp_group_initialized(order):
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
        order=order,
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


@pytest.mark.parametrize("order", RANK_ORDERS)
def test_round_robin_microbatch_mapping(order):
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
        order=order,
    )

    # producer 槽位取自边界组内的编号（与生产代码同源，见 colocated_schedule phase ①）。
    # The producer slot is the boundary group's own rank, as in production.
    boundary_group = ps.get_colocated_boundary_group()
    producer = torch.distributed.get_group_rank(boundary_group, torch.distributed.get_rank())
    # num_microbatches == num_producers: producer p handles exactly microbatch p.
    # num_microbatches == num_producers 时，producer p 恰好负责 microbatch p。
    assert ps.get_microbatches_for_producer(producer, pp, pp) == [producer]
    # num_microbatches == 2 * num_producers: producer p handles p and p + num_producers.
    # num_microbatches == 2 * num_producers 时，producer p 负责 p 和 p + num_producers。
    assert ps.get_microbatches_for_producer(producer, 2 * pp, pp) == [producer, producer + pp]
    # All producers together cover every microbatch exactly once.
    # 所有 producer 合起来恰好覆盖每个 microbatch 一次。
    all_mbs = sorted(
        mb for p in range(pp) for mb in ps.get_microbatches_for_producer(p, 2 * pp, pp)
    )
    assert all_mbs == list(range(2 * pp))

    # num_microbatches not a multiple of the producer count raises.
    # num_microbatches 不是 producer 数的整数倍时抛异常。
    with pytest.raises(AssertionError):
        ps.get_microbatches_for_producer(producer, pp + 1, pp)
    # An out-of-range producer id is rejected as well.
    # 越界的 producer 编号同样被拒绝。
    with pytest.raises(AssertionError):
        ps.get_microbatches_for_producer(pp, 2 * pp, pp)
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


def test_colocated_data_parallel_group_requires_flag():
    """Without use_colocated_encoder the colocated dp getter must raise.

    未开启 use_colocated_encoder 时共置 dp 组的 getter 必须抛异常。
    """
    pp = _pp_size()
    if pp < 2:
        pytest.skip(f"world_size {world_size} not suitable for a pipeline size >= 2")
    Utils.initialize_model_parallel(pipeline_model_parallel_size=pp)
    with pytest.raises(AssertionError):
        ps.get_colocated_data_parallel_group()
    with pytest.raises(AssertionError):
        ps.get_colocated_data_parallel_global_ranks()
    Utils.destroy_model_parallel()
    assert ps._COLOCATED_DATA_PARALLEL_GROUP is None


@pytest.mark.parametrize("order", RANK_ORDERS)
def test_colocated_data_parallel_group_spans_all_ranks(order):
    """The colocated dp group is the whole encoder-replica set (all W ranks).

    共置 dp 组等于 encoder 副本的全体（全部 W 个 rank），且与 inner / outer / 边界组
    的成员关系符合 inner x outer = W 的分解。
    """
    pp = _pp_size()
    if pp < 2:
        pytest.skip(f"world_size {world_size} not suitable for a pipeline size >= 2")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        use_colocated_encoder=True,
        order=order,
    )

    colocated_group = ps.get_colocated_data_parallel_group()
    assert colocated_group is not None
    # With TP=1 the colocated dp group spans every rank in the job.
    # TP=1 时共置 dp 组覆盖作业内所有 rank。
    assert colocated_group.size() == world_size
    assert sorted(ps.get_colocated_data_parallel_global_ranks()) == list(range(world_size))
    assert sorted(torch.distributed.get_process_group_ranks(colocated_group)) == list(
        range(world_size)
    )

    # inner (P) x outer (D_outer) == W: the colocated group is a valid product of
    # the two dimensions the distributed-optimizer variant will shard over.
    # inner (P) x outer (D_outer) == W：共置组正是分布式优化器变体要切分的两个维度之积。
    inner_size = ps.get_encoder_inner_data_parallel_group().size()
    outer_size = ps.get_data_parallel_group(with_context_parallel=True).size()
    assert inner_size == pp
    assert inner_size * outer_size == colocated_group.size()

    # A separate NCCL instance from the boundary group, which keeps its own members.
    # 与边界组是不同的 NCCL 实例，边界组成员不变。
    assert colocated_group is not ps.get_colocated_boundary_group()
    assert ps.get_colocated_boundary_group().size() == pp

    # The boundary group's member ORDER is load-bearing and must follow the pipeline
    # stages: the communicator treats member 0 as the consumer (the backbone entry) and
    # phase (1) uses the member index as the producer slot. Both hold under either rank
    # order because the group is built from the rank generator's 'pp' lists (Task 5.7).
    # 边界组的**成员顺序**是承重的，必须与 pipeline stage 一致：communicator 把成员 0 当
    # 消费者（backbone entry），phase ① 又拿组内编号当 producer 槽位。两者在任意 rank
    # order 下都成立，因为该组由 rank generator 的 'pp' 列表建出（Task 5.7）。
    boundary_group = ps.get_colocated_boundary_group()
    assert ps.get_colocated_boundary_global_ranks() == torch.distributed.get_process_group_ranks(
        ps.get_pipeline_model_parallel_group()
    )
    assert torch.distributed.get_group_rank(
        boundary_group, torch.distributed.get_rank()
    ) == ps.get_pipeline_model_parallel_rank()

    Utils.destroy_model_parallel()
    assert ps._COLOCATED_DATA_PARALLEL_GROUP is None
    assert ps._COLOCATED_DATA_PARALLEL_GLOBAL_RANKS is None


@pytest.mark.parametrize("order", RANK_ORDERS)
def test_build_colocated_encoder_process_groups(order):
    """The encoder collection overrides only the data-parallel dimension.

    encoder 的进程组集合只覆盖数据并行维度，其余字段沿用默认。
    """
    pp = _pp_size()
    if pp < 2:
        pytest.skip(f"world_size {world_size} not suitable for a pipeline size >= 2")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        use_colocated_encoder=True,
        order=order,
    )

    pg_collection = ps.build_colocated_encoder_process_groups()
    # dp / dp_cp are the colocated group; DDP reduces encoder grads over it.
    # dp / dp_cp 为共置组，encoder 梯度在其上归约。
    assert pg_collection.dp is ps.get_colocated_data_parallel_group()
    assert pg_collection.dp_cp is ps.get_colocated_data_parallel_group()
    # With one encoder optimizer instance, its intra group is the full encoder DP
    # group and no inter-instance group exists. These values come from the encoder's
    # own instance configuration, not from the language model hierarchy.
    # encoder 只有一个优化器实例时，intra 就是完整的 encoder DP 组，且不存在 inter
    # 实例组；这些值来自 encoder 自己的实例配置，不复用 language model 的层次。
    assert pg_collection.intra_dp_cp is ps.get_colocated_data_parallel_group()
    assert pg_collection.intra_dist_opt is ps.get_colocated_data_parallel_group()
    assert pg_collection.inter_dist_opt is None
    # embd / pos_embd must be the single-member encoder pipeline group, NOT None: the
    # encoder has no pipeline and no tied embeddings, and a single-member group is how
    # "nothing to sync" is expressed (finalize_model_grads.py:229 gates on size > 1).
    # None would be read as "not supplied" and fall back to the job-wide embedding group
    # (finalize_model_grads.py:189-193), which has two members under PP>1 and then trips
    # `assert pp_group is None` on the first / last backbone stage (Task 5.8).
    # embd / pos_embd 必须是单成员 encoder pipeline 组而**不是** None：encoder 无 PP、
    # 无共享 embedding，而"单成员组"就是"没什么要同步"的表达（finalize_model_grads.py:229
    # 按 size > 1 放行）。置 None 会被当成"没传"、退回取全作业 embedding 组
    # （finalize_model_grads.py:189-193），PP>1 时它有两个成员，首/末 backbone stage 上
    # 紧接着的 `assert pp_group is None` 就会炸（Task 5.8）。
    assert pg_collection.embd is ps.get_colocated_encoder_pipeline_model_parallel_group()
    assert pg_collection.pos_embd is ps.get_colocated_encoder_pipeline_model_parallel_group()
    # pp / cp describe the encoder itself: a single-member group, so every member
    # of the colocated dp group reports pipeline rank 0.
    # pp / cp 描述 encoder 自身：单成员组，故共置 dp 组内每个成员的 pipeline rank 都是 0。
    encoder_pipeline_group = ps.get_colocated_encoder_pipeline_model_parallel_group()
    assert pg_collection.pp is encoder_pipeline_group
    assert pg_collection.cp is encoder_pipeline_group
    assert encoder_pipeline_group.size() == 1
    assert encoder_pipeline_group.rank() == 0
    assert torch.distributed.get_process_group_ranks(encoder_pipeline_group) == [rank]
    # It must NOT be the backbone pipeline group, whose members report different
    # ranks and would derive inconsistent bucket layouts.
    # 它绝不能是 backbone 的 pipeline 组——那个组各成员报出的 rank 不同，会推导出不一致的分桶布局。
    assert encoder_pipeline_group is not ps.get_pipeline_model_parallel_group()
    # The encoder tensor-parallel group has the same members as the job group but
    # uses its own communicator, so no encoder collection field points at backbone.
    # encoder TP 组与作业 TP 组成员相同但 communicator 独立，确保 encoder collection
    # 不指向任何 backbone 通信组。
    assert pg_collection.tp is ps.get_colocated_encoder_tensor_model_parallel_group()
    assert pg_collection.tp is not ps.get_tensor_model_parallel_group()

    Utils.destroy_model_parallel()
    assert ps._COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP is None


@pytest.mark.parametrize("order", RANK_ORDERS)
def test_colocated_encoder_distributed_optimizer_instance_groups(order):
    """The encoder derives an independent hierarchy from its own instance count.

    encoder 按自己的实例数从共置 DP 域推导独立层次，不复用 language model 的组。
    """
    pp = _pp_size()
    if pp < 2 or world_size < 2:
        pytest.skip(f"world_size {world_size} does not support two optimizer instances")

    encoder_num_distributed_optimizer_instances = 2
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp,
            use_colocated_encoder=True,
            colocated_encoder_num_distributed_optimizer_instances=(
                encoder_num_distributed_optimizer_instances
            ),
            order=order,
        )

        encoder_data_parallel_ranks = ps.get_colocated_data_parallel_global_ranks()
        encoder_intra_group_size = (
            len(encoder_data_parallel_ranks)
            // encoder_num_distributed_optimizer_instances
        )
        caller_index = encoder_data_parallel_ranks.index(rank)
        encoder_instance_index = caller_index // encoder_intra_group_size
        encoder_shard_index = caller_index % encoder_intra_group_size
        expected_intra_ranks = encoder_data_parallel_ranks[
            encoder_instance_index
            * encoder_intra_group_size : (encoder_instance_index + 1)
            * encoder_intra_group_size
        ]
        expected_inter_ranks = encoder_data_parallel_ranks[
            encoder_shard_index::encoder_intra_group_size
        ]

        assert (
            ps.get_colocated_encoder_intra_distributed_optimizer_instance_global_ranks()
            == expected_intra_ranks
        )
        assert (
            ps.get_colocated_encoder_inter_distributed_optimizer_instance_global_ranks()
            == expected_inter_ranks
        )

        pg_collection = ps.build_colocated_encoder_process_groups()
        assert torch.distributed.get_process_group_ranks(
            pg_collection.intra_dist_opt
        ) == expected_intra_ranks
        assert torch.distributed.get_process_group_ranks(
            pg_collection.inter_dist_opt
        ) == expected_inter_ranks
        assert pg_collection.intra_dist_opt is not ps.get_intra_distributed_optimizer_instance_group()
        assert pg_collection.inter_dist_opt is not ps.get_inter_distributed_optimizer_instance_group(
            check_initialized=False
        )
    finally:
        Utils.destroy_model_parallel()
