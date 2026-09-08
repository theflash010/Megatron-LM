# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import logging
import os
import warnings
from datetime import timedelta
from math import log2
from typing import Callable, List, Optional

import numpy as np
import torch

from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

from .utils import GlobalMemoryBuffer, is_torch_min_version

logger = logging.getLogger(__name__)

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra-, pipeline, and expert) that the current rank belongs to.
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None

### Expert-related parallel states
# Naming convention:
# _EXPERT prefix in group name means it's used for expert layer in MoE models.
# _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
# _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Expert tensor and model combined parallel group
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
# Expert tensor, model, pipeline combined parallel group
_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None
# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
# Parallel state values changed on the fly
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_TENSOR_PARALLEL_RANK = None
### End of expert related parallel states

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_RANK = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each expert model parallel group to ease calculation of
# the first local rank in the expert model parallel group
_EXPERT_MODEL_PARALLEL_RANKS = None

# A list of global ranks for each model parallel group to ease calculation of
# the first local rank in the model parallel group
_MODEL_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
# Hierarchical context parallel groups
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = None
# Hybrid context parallel groups
_HYBRID_DP_CP_GROUPS = {}

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# Partial Data parallel group information with context parallel combined.
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = None
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None

# combined parallel group of TP and CP
_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Paralel group of all GPUs in a distributed optimizer instance
_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

# Encoder inner data parallel group for colocated encoder training:
# the P ranks inside one outer data-parallel replica (one rank per pipeline
# stage; with TP=1 these are exactly the members of one pipeline-parallel group).
# Used for the inner-layer gradient all-reduce of the colocated encoder before
# the regular (outer) data-parallel all-reduce.
# 共置训练中 encoder 的内部数据并行组：一个外层 dp 副本内的 P 个 rank
# （每个 pipeline stage 一个；TP=1 时恰好等于一个 pipeline-parallel 组）。
# 用于 encoder 梯度的第一层（inner）all-reduce，先于常规（外层）dp all-reduce。
_ENCODER_INNER_DATA_PARALLEL_GROUP = None
_ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS = None

# Distributed optimizer instance groups derived from the colocated encoder's own
# data-parallel domain and instance-count argument. These are independent of the
# language model's groups even when a particular topology gives them equal members.
# 根据共置 encoder 自身的数据并行域和实例数参数构造的分布式优化器实例组。即使某种
# 拓扑下成员恰好相同，它们也与 language model 的对应通信组保持独立。
_COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None
_COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS = None
_COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None
_COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS = None

# Colocated boundary communication group: same members as the encoder inner dp
# group (the ranks of one outer dp replica) but a SEPARATE NCCL instance, used
# exclusively for the encoder->backbone-entry boundary P2P (forward packet and
# backward grad). Kept independent from the pipeline-parallel group and the
# encoder inner dp group to avoid serializing boundary P2P with the 1F1B P2P on
# the same NCCL group.
# 共置边界通信组：成员与 encoder inner dp 组相同（一个外层 dp 副本内的 rank），
# 但是独立的 NCCL 实例，专用于 encoder→backbone entry 的边界 P2P（前向数据包与
# 反向梯度）。与 pp_group / enc_inner_dp 组保持独立，避免边界 P2P 与 1F1B P2P
# 在同一组上串行。
_COLOCATED_BOUNDARY_GROUP = None
_COLOCATED_BOUNDARY_GLOBAL_RANKS = None

# Colocated data parallel group for colocated encoder training: ALL W ranks that
# hold a replica of the encoder, i.e. the cartesian product of the outer
# data-parallel dimension and the encoder inner dimension (D_outer x P). The
# encoder chunk hands this group to DDP as its dp/dp_cp group, so that a single
# gradient reduction covers both the inner and the outer dimension at once
# (valid because per-token loss makes the reduction a pure SUM).
# 共置训练中 encoder 的数据并行组：持有 encoder 副本的**全部 W 个 rank**，即外层
# 数据并行维度与 encoder 内部维度的笛卡尔积（D_outer x P）。encoder chunk 把该组
# 作为 dp/dp_cp 组交给 DDP，于是一次梯度归约同时覆盖 inner 与 outer 两个维度
# （成立前提是 per-token loss 使归约退化为纯 SUM）。
_COLOCATED_DATA_PARALLEL_GROUP = None
_COLOCATED_DATA_PARALLEL_GLOBAL_RANKS = None

# Colocated encoder pipeline (and context) parallel group: the encoder is NOT
# pipeline parallel under colocated training (every rank holds a full replica,
# vision_config.pipeline_model_parallel_size == 1) and does not use context
# parallelism either, so its pipeline/context group is the caller rank alone.
# Handing this group to the encoder chunk's DDP instead of the backbone's
# pipeline group is what keeps the encoder's communication description honest:
# every member of the colocated data-parallel group then reports pipeline rank 0
# and derives the same bucket layout, whereas the backbone pipeline group would
# report a different rank per member and split the buckets inconsistently.
# 共置 encoder 的 pipeline（兼 context）并行组：共置训练下 encoder 不做 PP（每个
# rank 持完整副本，vision_config.pipeline_model_parallel_size == 1），也不做 CP，
# 因此它的 pipeline/context 组就是当前 rank 自己。把这个组（而不是 backbone 的
# pipeline 组）交给 encoder chunk 的 DDP，才是对 encoder 通信拓扑的忠实描述：
# 共置数据并行组内每个成员报出的 pipeline rank 都是 0、推导出同样的分桶布局；
# 若用 backbone 的 pipeline 组，各成员报出的 rank 不同、分桶会不一致。
_COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP = None

# Tensor model parallel group of the colocated encoder: same members as the job's
# regular tensor model parallel group (the encoder tensor parallel size is an
# explicit argument and must equal the job's, see validate_colocated_args), but a
# SEPARATE NCCL instance, so that the encoder's tensor-parallel collectives and its
# data broadcast are described by - and issued on - the encoder's own communicator
# instead of borrowing the backbone's. Same "same members, separate instance"
# device as the colocated boundary group.
# 共置 encoder 的张量并行组：成员与作业常规 tp 组相同（encoder tp 并行度是显式参数且
# 必须等于作业的，见 validate_colocated_args），但是**独立 NCCL 实例**，让 encoder 的
# 张量并行集合操作与取数广播都发生在 encoder 自己的通信子上，而不是借用 backbone 的。
# 与共置边界组同样是"同成员、独立实例"的手法。
_COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP = None
_COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


# List of all process groups
# Used for updating the timeout for all process groups
# None represents the default process group
_global_process_group_list = None


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations
    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        # When fields in nccl_options.config are not specified, NCCL applies default settings.
        # The default values for Hopper GPUs are as follows:
        # cga_cluster_size = 4, max_ctas = 32, min_ctas = 1
        # Default values may differ between GPU generations and NCCL versions.
        nccl_options = torch.distributed.ProcessGroupNCCL.Options(
            is_high_priority_stream=nccl_comm_cfgs[pg_name].get("is_high_priority_stream", False)
        )
        if "cga_cluster_size" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name]["cga_cluster_size"]
        if "max_ctas" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name]["max_ctas"]
        if "min_ctas" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name]["min_ctas"]
        if "net_name" in nccl_comm_cfgs[pg_name]:
            nccl_options.config.net_name = nccl_comm_cfgs[pg_name]["net_name"]
            # verify net_name value
            if nccl_options.config.net_name.lower() not in ["ib", "socket"]:
                raise RuntimeError(
                    f"net_name ({nccl_options.config.net_name}) is not supported."
                    f"Accepted values: 'IB' or 'socket'."
                )
        return nccl_options
    else:
        return None


def update_pg_timeout(
    timeout: timedelta, pg: Optional[torch._C._distributed_c10d.ProcessGroup] = None
):
    """Update the timeout for all process groups or a specific process group.
       Synchronize the process groups before updating the timeout.
    Args:
        timeout(datetime.timedelta): The timeout to set for the process group(s)
        pg(Optional[torch._C._distributed_c10d.ProcessGroup], default=None):
            The process group to update the timeout for.
            If None, all process groups are updated.
    """
    if hasattr(torch.distributed.distributed_c10d, "_set_pg_timeout"):
        torch.distributed.barrier(pg)
        torch.cuda.synchronize()
        try:
            if pg is None:
                global _global_process_group_list
                for group in _global_process_group_list:
                    torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
            else:
                torch.distributed.distributed_c10d._set_pg_timeout(timeout, pg)
        except Exception as e:
            logger.error(f"Error updating pg timeout: {e}")
            logger.error(f"Process group: {pg}")
            logger.error(f"Timeout: {timeout}")
            logger.error(f"Global process group list: {_global_process_group_list}")
            raise e


def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """Creates a ProcessGroup."""
    kwargs = {
        "ranks": ranks,
        "timeout": timeout,
        "backend": backend,
        "pg_options": pg_options,
        "use_local_synchronization": use_local_synchronization,
        "group_desc": group_desc,
    }
    if not is_torch_min_version("2.4.0"):
        kwargs.pop("group_desc")
        if timeout is None:
            # Old version (e.g. v2.1.2) sets default_pg_timeout as default value to timeout
            # in function signature, then check tiemout value type.
            # New version sets None as default value to timeout in function signature. If value
            # is None, torch will give value according to the backend, then check type.
            # So need to unset timeout here if caller doesn't set value. Otherwise there is
            # type error.
            kwargs.pop("timeout")
    group = torch.distributed.new_group(**kwargs)
    global _global_process_group_list
    if _global_process_group_list is None:
        # None stands for the default process group
        _global_process_group_list = [None]
    if torch.distributed.get_rank() in ranks:
        _global_process_group_list.append(group)
    return group


def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will be used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


def create_hierarchical_groups(
    rank,
    ranks,
    hierarchical_group_sizes,
    create_gloo_process_groups=False,
    pg_options=None,
    timeout=None,
    group_desc=None,
):
    """Create hierarchical groups for a set of ranks.
    Taking a group size of 16 as example, so we have a total of 16 GPUs denoted by g0 ... g15.
    If the hierarchical group sizes are [2,2,4], we use 2 GPUs in the first and second level
    of sub-groups, and 4 GPUs in the last level of sub groups. The present function will
    create 8 level-1 sub-groups, 8 level-2 sub-groups and 4 level-3 sub-groups as:
        8 level-1 sub-groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 level-2 sub-groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        4 level-3 sub-groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    """

    if not HAVE_EINOPS:
        raise ImportError("einops is not installed. Please install it with `pip install einops`.")

    hierarchical_groups = []
    hierarchical_groups_gloo = []
    if not isinstance(pg_options, list):
        pg_options = [pg_options] * len(hierarchical_group_sizes)
    for level in range(len(hierarchical_group_sizes)):
        rearranged_ranks = einops.rearrange(
            np.array(ranks),
            "(l s u) -> (l u) s",
            u=int(np.prod(hierarchical_group_sizes[:level])),
            s=hierarchical_group_sizes[level],
            l=int(np.prod(hierarchical_group_sizes[level + 1 :])),
        ).tolist()
        for sub_ranks in rearranged_ranks:
            sub_group = create_group(
                sub_ranks,
                timeout=timeout,
                pg_options=pg_options[level],
                group_desc=f"HIERARCHICAL_{group_desc}_L{level}",
            )
            if create_gloo_process_groups:
                sub_group_gloo = create_group(
                    sub_ranks,
                    timeout=timeout,
                    backend="gloo",
                    pg_options=pg_options[level],
                    group_desc=f"HIERARCHICAL_{group_desc}_GLOO_L{level}",
                )
            else:
                sub_group_gloo = None
            if rank in sub_ranks:
                hierarchical_groups.append(sub_group)
                hierarchical_groups_gloo.append(sub_group_gloo)
    assert rank not in ranks or len(hierarchical_groups) == len(hierarchical_group_sizes)
    assert rank not in ranks or len(hierarchical_groups_gloo) == len(hierarchical_group_sizes)
    return hierarchical_groups, hierarchical_groups_gloo


def create_hybrid_dp_cp_groups(rank, ranks, pg_options):
    """
    Creates groups required for hybrid DPxCP.
    Creates a new group for every power of 2 up to the number of DPxCP ranks.
    Returns a dictionary indexed by group size.
    """
    hybrid_dp_cp_groups = {}
    # Generate group for every power of 2 up to the number of CP ranks
    # We limit the allowed group sizes in order to avoid excessive overhead.
    group_sizes = [2**i for i in range(int(log2(len(ranks))))][1:]
    for group_size in group_sizes:
        for i in range(0, len(ranks), group_size):
            group = create_group(
                ranks[i : i + group_size],
                pg_options=pg_options,
                group_desc=f"HYBRID_DP_CP_GROUP_{group_size}",
            )
            if rank in ranks[i : i + group_size]:
                assert (
                    group_size not in hybrid_dp_cp_groups
                ), f"Rank {rank} appears in multiple Hybrid DP CP groups of size {group_size}"
                hybrid_dp_cp_groups[group_size] = group
    return hybrid_dp_cp_groups


class RankGenerator(object):
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(
        self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0
    ) -> None:
        assert (
            ep == 1 or cp == 1
        ), "Both EP and CP > 1 in not allow in one rank generator. \
            CP is only included in default RankGenerator, and EP only in expert RankGenerator."

        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't"
                    f"specified the order ({self.order})."
                )
            elif name not in order:
                order = order + "-" + name

        self.order = order
        self.ordered_size = []

        for token in order.split("-"):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        """Create a mask for the specified tokens based on the given order.

        Args:
            order (str): The order of parallelism types (e.g., 'tp-dp-pp').
            token (str): The specific parallelism types to include in the mask,
                         separated by hyphens (e.g., 'tp-dp').
        """
        ordered_token = order.split("-")
        token_list = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks


def default_embedding_ranks(pp_ranks):
    """Return the default ranks that constitute the stages on which the word embeddings live.
    For most models, these are the first and last pipeline stages."""
    if len(pp_ranks) == 1:
        return [pp_ranks[0]]
    else:
        return [pp_ranks[0], pp_ranks[-1]]


def default_position_embedding_ranks(pp_ranks):
    """Return the default ranks that constitute the stages on which the position embeddings live.
    For most models, this is only the first pipeline stage."""
    return [pp_ranks[0]]


def overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, key_value_pair):
    """Overwrite the nccl_comm_cfgs for the given pg_name with the given key_value_pair."""
    if pg_name not in nccl_comm_cfgs:
        nccl_comm_cfgs[pg_name] = {}
    nccl_comm_cfgs[pg_name][key_value_pair[0]] = key_value_pair[1]


# pylint: disable=C0301
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    hybrid_context_parallel: bool = False,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    create_gloo_process_groups: bool = True,
    high_priority_stream_groups: Optional[List[str]] = None,
    sharp_enabled_group: Optional[str] = None,
    rank_offset: int = 0,
    local_world_size: Optional[int] = None,
    use_colocated_encoder: bool = False,
    colocated_encoder_tensor_model_parallel_size: Optional[int] = None,
    colocated_encoder_num_distributed_optimizer_instances: int = 1,
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_comm_backend (str, optional):
            The backend to use for pipeline parallel communication.
            If None, the default backend will be used.

        use_sharp (bool, default = False): #是否开启SHARP通信优化
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        num_distributed_optimizer_instances (int, default = 1):
            The number of distributed optimizer replicas across the data-
            parallel domain. #针对ZeRO-1/2的逻辑，数值代表优化器状态切成几分

        colocated_encoder_num_distributed_optimizer_instances (int, default = 1):
            The independently configured number of distributed optimizer replicas
            across the colocated encoder's own data-parallel domain.

        expert_tensor_parallel_size (int, default = tp_size):
            The number of GPUs to split individual tensors of expert.

        nccl_communicator_config_path (str, default = None): #NCCL通信优化配置文件路径
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

        get_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group and returns
            those ranks that should have embeddings.

        get_position_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group, and returns
            those ranks that should have position embeddings.

        create_gloo_process_groups (bool, default = True):
            Create Gloo process groups if set to True. If set to False, Gloo process groups are
            not created and calls to get Gloo process groups will result in assertion errors.

        high_priority_stream_groups (List[str], default = None): #高优先级通信组的名单，后续会对这些通信组加上高优先级属性
            Specify which communicator groups should use high priority streams during creation.
            Assigning high priority to communication streams ensures that communication kernels
            are scheduled with higher priority, minimizing the exposed communication when it is
            overlapped with other computation kernels.
            Example: initialize_parallel_groups(..., high_priority_stream_groups=['dp_cp','ep_dp'])

        sharp_enabled_group (str, default = None):
            Specify which communicator group should use SHARP communication.
            This option is only valid when use_sharp is True.
            By default (None), it is enabled from dp group.
            Available options (choose one): [dp, dp_replica]

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # NCCL restricts IB SHARP usage to a single communicator group—the first one created
    # with NCCL_COLLNET_ENABLE=1. After this group is created, NCCL_COLLNET_ENABLE must be
    # set to 0 for subsequent groups.
    if "NCCL_COLLNET_ENABLE" in os.environ: #删除 NCCL_COLLNET_ENABLE，先清理SHARP环境变量，后续要用到再设置
        del os.environ["NCCL_COLLNET_ENABLE"]

    if use_sharp: #如果使用SHARP技术
        if sharp_enabled_group is None: #如果没有指定使用SHARP的组，默认用于DP组
            # By default, SHARP is enabled from dp group.
            sharp_enabled_group = "dp"
        else:#否则使用指定的组
            # Currently, only dp and dp_replica groups are supported for SHARP.
            assert sharp_enabled_group in ["dp", "dp_replica"], "Invalid sharp_enabled_group"
            if sharp_enabled_group == "dp_replica":
                assert (
                    num_distributed_optimizer_instances > 1
                ), "dp_replica group requires num_distributed_optimizer_instances > 1"
    else:
        assert (
            sharp_enabled_group is None
        ), "sharp_enabled_group is only valid when use_sharp is True"

    if get_embedding_ranks is None: #确定get_embedding_ranks函数
        get_embedding_ranks = default_embedding_ranks

    if get_position_embedding_ranks is None: #确定get_position_embedding_ranks函数
        get_position_embedding_ranks = default_position_embedding_ranks

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = (
        local_world_size if local_world_size is not None else torch.distributed.get_world_size()
    )

    model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size #一份完整模型副本需要多少张卡

    if world_size % model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")

    data_parallel_size: int = world_size // model_size #确定DP并行度

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    rank = torch.distributed.get_rank() #获取全局rank

    nccl_comm_cfgs = {} #读取NCCL通信组配置
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    # Set is_high_priority_stream flag to the nccl_comm_cfgs if it is in high_priority_stream_groups
    high_priority_stream_groups = high_priority_stream_groups or [] #设置通信组的stream为高优先级
    for pg_name in high_priority_stream_groups:
        overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, ("is_high_priority_stream", True))

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=rank_offset,
    )#Dense部分（attention）的RankGenerator

    # Build expert rank generator
    if expert_tensor_parallel_size is None: #如果未指定专家张量并行度，那么专家张量并行度等于模型张量并行度
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    ) #一份完整 MoE expert 模型副本需要多少张 GPU
    expert_data_parallel_size = world_size // expert_tensor_model_pipeline_parallel_size #算出 expert 的数据并行度
    if world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    # TODO: support expert specific ordering
    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=rank_offset,
    )#MoE部分的RankGenerator

    assert (
        order.endswith("pp")
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ), "When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same"

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks(
        "pp"
    ), f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}"

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"

    assert (
        data_parallel_size * context_parallel_size
    ) % num_distributed_optimizer_instances == 0, (
        "Data parallel size should be divisible by partial DistOpt shard factor"
    )
    intra_partial_data_parallel_size = ( #结果就是每个 optimizer shard 有多少 rank 共同维护同一份 shard。
        data_parallel_size * context_parallel_size #因为 CP 组的 rank 在数据维度上也参与 DP 通信，所以 DP 域要乘上 CP 度来覆盖所有参与 DP 通信的 rank
    ) // num_distributed_optimizer_instances #num_distributed_optimizer_instances：optimizer state 切几份

    # Set NCCL_COLLNET_ENABLE to 1 to enable SHARP for the dp group.
    if sharp_enabled_group == "dp": #为 DP group 开启 SHARP
        os.environ["NCCL_COLLNET_ENABLE"] = "1"

    # In case of using SHARP, the dp-cp group requires to use NCCL COLLNET feature.
    # Due to the hardware limitation, only the initially created communication group
    # is eligible for using the NCCL COLLNET feature.
    # Therefore, dp-cp group, which potentially requires SHARP-enablement,
    # need to be created before all the other groups
    for ranks_with_cp in decoder_rank_generator.get_ranks('dp-cp'):
        group_with_cp = create_group(#创建dp-cp通信组，backend复用default_pg的backend
            ranks_with_cp,
            timeout=timeout,
            pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP_WITH_CP",
        )
        if create_gloo_process_groups: #是否创建备份gloo
            group_with_cp_gloo = create_group(
                ranks_with_cp,
                timeout=timeout,
                backend="gloo",
                group_desc="DATA_PARALLEL_GROUP_WITH_CP_GLOO",
            )
        else:
            group_with_cp_gloo = None
        if rank in ranks_with_cp: #如果在这个组里，设置单例
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        if num_distributed_optimizer_instances > 1: #如果有num_distributed_optimizer_instances，优化器状态切分（ZeRO-1/2）
            # Create groups for intra-partial DP domain
            for i in range(num_distributed_optimizer_instances):
                intra_partial_dp_ranks_with_cp = ranks_with_cp[
                    (i * intra_partial_data_parallel_size) : (
                        (i + 1) * intra_partial_data_parallel_size
                    )
                ]
                intra_partial_dp_group_with_cp = create_group(
                    intra_partial_dp_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options("intra_dp_cp", nccl_comm_cfgs),
                    group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP",
                )#创建intra_partial_dp_group_with_cp通信组
                if create_gloo_process_groups:
                    intra_partial_dp_group_with_cp_gloo = create_group(
                        intra_partial_dp_ranks_with_cp,
                        timeout=timeout,
                        backend="gloo",
                        group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO",
                    )
                else:
                    intra_partial_dp_group_with_cp_gloo = None
                if rank in intra_partial_dp_ranks_with_cp: #如果在这个组里，设置单例
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = intra_partial_dp_group_with_cp
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = (
                        intra_partial_dp_group_with_cp_gloo
                    )
        else: #如果没有 num_distributed_optimizer_instances，直接复用 dp-cp 组
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = _DATA_PARALLEL_GROUP_WITH_CP
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = _DATA_PARALLEL_GROUP_WITH_CP_GLOO

    # Apply SHARP to the dp group.
    if sharp_enabled_group == "dp":
        if rank == 0:
            logger.info( #打印 SHARP 的硬件限制：QM1 交换机最多 8 个 SHARP group，QM2 最多 256 个。如果 DP group 数量超过这个上限，自动 fallback 到普通 all-reduce。
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        # PyTorch is performing lazy initialization of the communicator group.
        # Therefore, we need to perform a nccl call to ensure that the communicator group is created.
        torch.distributed.barrier( #强制 dp-cp group 的 NCCL communicator 真正初始化
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        torch.cuda.synchronize()
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to the dp group.
        if "NCCL_COLLNET_ENABLE" in os.environ: #清理环境变量，SHARP只能有一个通信组
            del os.environ["NCCL_COLLNET_ENABLE"]

    if hybrid_context_parallel: #如果使用混合上下文并行
        global _HYBRID_DP_CP_GROUPS
        for ranks_with_cp in decoder_rank_generator.get_ranks('dp-cp'):
            assert (
                len(ranks_with_cp) % 2 == 0
            ), "Hybrid context parallel requires an even number of ranks"
            _HYBRID_DP_CP_GROUPS.update(#创建多个不同CP size的通信组，并设置为单例
                create_hybrid_dp_cp_groups(
                    rank, ranks_with_cp, get_nccl_options("dp_cp", nccl_comm_cfgs)
                )
            )
        # TODO: Are gloo groups needed for hybrid cp?

    for ranks in decoder_rank_generator.get_ranks('dp'): #创建dense部分的DP通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("dp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP",
        )
        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks, timeout=timeout, backend="gloo", group_desc="DATA_PARALLEL_GROUP_GLOO"
            )
        else:
            group_gloo = None
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GROUP_GLOO = group_gloo
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('cp'): #创建dense部分的CP通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("cp", nccl_comm_cfgs),
            group_desc="CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
        if hierarchical_context_parallel_sizes: #如果有层次化CP结构（可以优化通信），就创建CP组内的子通信组
            assert np.prod(hierarchical_context_parallel_sizes) == context_parallel_size
            global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
            hierarchical_groups, _ = create_hierarchical_groups(
                rank,
                ranks,
                hierarchical_context_parallel_sizes,
                create_gloo_process_groups=False,
                pg_options=get_nccl_options("hcp", nccl_comm_cfgs),
                timeout=timeout,
                group_desc="CONTEXT_PARALLEL_GROUP",
            )
            if rank in ranks:
                _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = hierarchical_groups

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_GLOBAL_RANKS
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp-pp'): #创建tp-pp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("mp", nccl_comm_cfgs),
            group_desc="MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
            _MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp'): #创建tp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp", nccl_comm_cfgs),
            group_desc="TENSOR_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), "pipeline model parallel group is already initialized"
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, "position embedding group is already initialized"
    if pipeline_model_parallel_comm_backend == "ucc":
        # The UCC backend provides two key benefits:
        # 1) Achieves better bandwidth utilization than NCCL when using InfiniBand links.
        # 2) Does not use GPU SM resources (Zero-SM), mitigating performance interference
        #    with overlapping compute kernels.

        # The UCC backend is recommended in the following cases:
        # 1) When the exposed pipeline-parallel (PP) communications are significant.
        #    - E.g., Pipeline parallelism with very less gradient accumulation steps.
        #    - It may provide better performance due to improved bandwidth utilization.
        # 2) When the critical-path pipeline stage has substantial PP-communication overlap.
        #    - E.g., Uneven pipeline parallelism.
        #    - It may provide better performance due to zero SM resource usage.
        if "CUDA_DEVICE_MAX_CONNECTIONS" in os.environ:
            # UCC backend requires CUDA_DEVICE_MAX_CONNECTIONS variable to be larger than 1,
            # to gurantee the overlapped UCC communications. If this environment variable is set to 1,
            # all the UCC communication will be serialized.
            assert (
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] != "1"
            ), "UCC-backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1"

        # Setting up required environment variables for ucc backend
        #
        # "TORCH_UCC_BLOCKING_WAIT=none" allows non-blocking waits of the communiction handle
        # "UCC_EC_CUDA_STREAM_TASK_MODE" controls how CUDA execution engines (EC)
        # schedule tasks on CUDA streams.
        # "UCX_TLS" controls transport layer selection
        # "NSYS_UCP_COMM_PARAMS=1" enables capturing ucx tracing in nsys profiling
        # "UCX_RNDV_THRESH" controls threshold threshold for switching between
        # eager and rendezvous (RNDV) communication protocols.
        # "UCX_NET_DEVICES" select which network interfaces UCX should use.
        # "UCC_CL_BASIC_TLS" controls which Transport Layers are used by
        # the Basic Collective libraray

        os.environ["TORCH_UCC_BLOCKING_WAIT"] = (
            os.environ["TORCH_UCC_BLOCKING_WAIT"]
            if "TORCH_UCC_BLOCKING_WAIT" in os.environ
            else "none"
        )
        os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"] = (
            os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"]
            if "UCC_EC_CUDA_STREAM_TASK_MODE" in os.environ
            else "driver"
        )
        os.environ["UCX_TLS"] = (
            os.environ["UCX_TLS"] if "UCX_TLS" in os.environ else "ib,cuda_copy"
        )  # cuda_ipc (i.e., NVLink-enablement) will be later supported
        os.environ["NSYS_UCP_COMM_PARAMS"] = "1"
        os.environ["UCX_RNDV_THRESH"] = "0"
        os.environ["UCX_NET_DEVICES"] = "all"
        os.environ["UCC_CL_BASIC_TLS"] = "^sharp,nccl"

    for ranks in decoder_rank_generator.get_ranks('pp'): #创建pp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP",
        )
        assert (
            pipeline_model_parallel_comm_backend == None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ), f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported'

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP is None:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                _PIPELINE_GLOBAL_RANKS.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP = [_PIPELINE_MODEL_PARALLEL_GROUP, group]
                _PIPELINE_GLOBAL_RANKS = [_PIPELINE_GLOBAL_RANKS, ranks]

        embedding_ranks = get_embedding_ranks(ranks) #选出具有embedding参数的rank，一般是第一个和最后一个rank
        group = create_group( #创建embedding通信组
            embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options("embd", nccl_comm_cfgs),
            group_desc="EMBEDDING_GROUP",
        )
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        position_embedding_ranks = get_position_embedding_ranks(ranks) #选出具有position embedding参数的rank
        group = create_group( #创建position embedding通信组
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options("pos_embd", nccl_comm_cfgs),
            group_desc="POSITION_EMBEDDING_GROUP",
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the encoder inner data-parallel groups (colocated encoder training).
    # Each encoder inner dp group contains the P ranks inside one outer
    # data-parallel replica (one rank per pipeline stage). With TP=1 these are
    # exactly the members of one pipeline-parallel group, so we derive the rank
    # lists from the pipeline-parallel groups. Used for the inner-layer gradient
    # all-reduce of the colocated encoder; the outer layer reuses the regular
    # data-parallel groups. Only created when use_colocated_encoder=True.
    # 构建共置训练用的 encoder 内部数据并行组：每个组包含一个外层 dp 副本内的
    # P 个 rank（每个 pipeline stage 一个）。TP=1 时其成员恰好等于一个
    # pipeline-parallel 组，因此直接复用 pp 组的 rank 列表来创建独立组；
    # 用于 encoder 梯度的第一层（inner）all-reduce，外层梯度仍复用常规 dp 组。
    # 仅在 use_colocated_encoder=True 时创建。
    global _ENCODER_INNER_DATA_PARALLEL_GROUP
    global _ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS
    if use_colocated_encoder:
        assert _ENCODER_INNER_DATA_PARALLEL_GROUP is None, (
            "encoder inner data parallel group is already initialized"
        )
        # Each pipeline-parallel group is one outer replica: reuse its rank list.
        # 每个 pipeline-parallel 组就是一个外层副本，直接复用其 rank 列表。
        for inner_ranks in decoder_rank_generator.get_ranks('pp'):
            group = create_group(
                inner_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("dp", nccl_comm_cfgs),
                group_desc="ENCODER_INNER_DATA_PARALLEL_GROUP",
            )
            if rank in inner_ranks:
                _ENCODER_INNER_DATA_PARALLEL_GROUP = group
                _ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS = inner_ranks

        # Build the colocated boundary communication groups. Same members as the
        # encoder inner dp groups (one outer replica), but a SEPARATE NCCL
        # instance, used only for the encoder->backbone-entry boundary P2P.
        # 构建共置边界通信组：成员与 enc_inner_dp 组相同（一个外层副本），但是独立
        # NCCL 实例，专用于 encoder→backbone entry 的边界 P2P。
        global _COLOCATED_BOUNDARY_GROUP
        global _COLOCATED_BOUNDARY_GLOBAL_RANKS
        assert _COLOCATED_BOUNDARY_GROUP is None, (
            "colocated boundary communication group is already initialized"
        )
        for boundary_ranks in decoder_rank_generator.get_ranks('pp'):
            group = create_group(
                boundary_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("dp", nccl_comm_cfgs),
                group_desc="COLOCATED_BOUNDARY_GROUP",
            )
            if rank in boundary_ranks:
                _COLOCATED_BOUNDARY_GROUP = group
                _COLOCATED_BOUNDARY_GLOBAL_RANKS = boundary_ranks

        # Build the colocated data-parallel groups: all W ranks holding an encoder
        # replica (the outer dp-cp dimension times the inner pipeline dimension).
        # The rank lists come from the rank generator's 'dp-cp-pp' token, so the
        # membership is derived from the group layout instead of an index formula
        # and stays correct under any rank order.
        # This group is the encoder's complete data-parallel domain. Its distributed
        # optimizer intra/inter hierarchy is derived independently below from this rank
        # list and the encoder-specific instance-count argument.
        # 构建共置数据并行组：持有 encoder 副本的全部 W 个 rank（外层 dp-cp 维度 ×
        # 内部 pipeline 维度）。rank 列表取自 rank generator 的 'dp-cp-pp' token，
        # 成员关系由组布局推导而非下标算式，因此在任意 rank order 下都正确。该组是
        # encoder 完整的数据并行域；其 DistOpt intra/inter 层次在下方根据这份 rank
        # 列表与 encoder 专属实例数独立推导。
        global _COLOCATED_DATA_PARALLEL_GROUP
        global _COLOCATED_DATA_PARALLEL_GLOBAL_RANKS
        assert _COLOCATED_DATA_PARALLEL_GROUP is None, (
            "colocated data parallel group is already initialized"
        )
        for colocated_data_parallel_ranks in decoder_rank_generator.get_ranks('dp-cp-pp'):
            group = create_group(
                colocated_data_parallel_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs),
                group_desc="COLOCATED_DATA_PARALLEL_GROUP",
            )
            if rank in colocated_data_parallel_ranks:
                _COLOCATED_DATA_PARALLEL_GROUP = group
                _COLOCATED_DATA_PARALLEL_GLOBAL_RANKS = colocated_data_parallel_ranks

            # Derive the encoder's distributed optimizer hierarchy from the encoder's own
            # data-parallel ranks and external instance-count argument. Do not reuse the
            # language model hierarchy: the two components can choose different counts.
            # 根据 encoder 自己的数据并行 rank 与外部实例数参数推导分布式优化器层次，不能
            # 复用 language model 的层次，因为两个组件可以选择不同的实例数。
            assert colocated_encoder_num_distributed_optimizer_instances > 0, (
                "colocated encoder distributed optimizer instances must be greater than 0"
            )
            assert (
                len(colocated_data_parallel_ranks)
                % colocated_encoder_num_distributed_optimizer_instances
                == 0
            ), (
                "colocated encoder data parallel size "
                f"({len(colocated_data_parallel_ranks)}) must be divisible by its number of "
                "distributed optimizer instances "
                f"({colocated_encoder_num_distributed_optimizer_instances})"
            )
            global _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
            global _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS
            global _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
            global _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS
            if colocated_encoder_num_distributed_optimizer_instances == 1:
                if rank in colocated_data_parallel_ranks:
                    _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = group
                    _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS = (
                        colocated_data_parallel_ranks
                    )
            else:
                colocated_encoder_intra_distributed_optimizer_instance_size = (
                    len(colocated_data_parallel_ranks)
                    // colocated_encoder_num_distributed_optimizer_instances
                )
                hierarchical_groups, _ = create_hierarchical_groups(
                    rank,
                    colocated_data_parallel_ranks,
                    [
                        colocated_encoder_intra_distributed_optimizer_instance_size,
                        colocated_encoder_num_distributed_optimizer_instances,
                    ],
                    pg_options=[
                        get_nccl_options("colocated_encoder_intra_dist_opt", nccl_comm_cfgs),
                        get_nccl_options("colocated_encoder_inter_dist_opt", nccl_comm_cfgs),
                    ],
                    timeout=timeout,
                    group_desc="COLOCATED_ENCODER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP",
                )
                if rank in colocated_data_parallel_ranks:
                    (
                        _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP,
                        _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP,
                    ) = hierarchical_groups
                    _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS = (
                        torch.distributed.get_process_group_ranks(
                            _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
                        )
                    )
                    _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS = (
                        torch.distributed.get_process_group_ranks(
                            _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
                        )
                    )

        # Build the colocated encoder pipeline (and context) parallel groups: one
        # single-member group per rank, because the encoder is replicated in full
        # on every rank and is neither pipeline- nor context-parallel. Group
        # creation is collective, so every rank walks the same list of groups.
        # 构建共置 encoder 的 pipeline（兼 context）并行组：每个 rank 一个单成员组，
        # 因为 encoder 在每个 rank 上都是完整副本，既不做 PP 也不做 CP。建组是集合
        # 操作，因此所有 rank 都要遍历同一份组列表。
        global _COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP
        assert _COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP is None, (
            "colocated encoder pipeline model parallel group is already initialized"
        )
        for encoder_pipeline_rank in range(torch.distributed.get_world_size()):
            group = create_group(
                [encoder_pipeline_rank],
                timeout=timeout,
                pg_options=get_nccl_options("pp", nccl_comm_cfgs),
                group_desc="COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP",
            )
            if rank == encoder_pipeline_rank:
                _COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP = group

        # Build the colocated encoder tensor model parallel groups. The encoder's
        # tensor parallel size is an explicit argument and must equal the job's
        # (the boundary packet pairs the tensor-parallel slot of the same index on
        # the producer and the consumer), so the member lists are the job's 'tp'
        # rank lists; the point of building them again is to give the encoder its
        # OWN communicator instead of borrowing the backbone's.
        # 构建共置 encoder 的张量并行组：encoder 的 tp 并行度是显式参数且必须等于作业的
        # （边界包按 tp 槽位一一对应发送），所以成员列表就取作业的 'tp' rank 列表；重新
        # 建一遍的意义在于让 encoder 拥有**自己的**通信子，而不是借用 backbone 的。
        global _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP
        global _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
        assert _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP is None, (
            "colocated encoder tensor model parallel group is already initialized"
        )
        if colocated_encoder_tensor_model_parallel_size is None:
            colocated_encoder_tensor_model_parallel_size = tensor_model_parallel_size
        assert colocated_encoder_tensor_model_parallel_size == tensor_model_parallel_size, (
            "colocated encoder tensor model parallel size "
            f"({colocated_encoder_tensor_model_parallel_size}) must equal the job's tensor "
            f"model parallel size ({tensor_model_parallel_size})"
        )
        for encoder_tensor_ranks in decoder_rank_generator.get_ranks('tp'):
            group = create_group(
                encoder_tensor_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("tp", nccl_comm_cfgs),
                group_desc="COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP",
            )
            if rank in encoder_tensor_ranks:
                _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP = group
                _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = encoder_tensor_ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp-dp-cp'): #创建tp-dp-cp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_dp_cp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in decoder_rank_generator.get_ranks('tp-dp'): #创建tp-dp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_dp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_DATA_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP = group

    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_CONTEXT_PARALLEL_GROUP is None
    ), 'Tensor + context parallel group is already initialized'
    for ranks in decoder_rank_generator.get_ranks('tp-cp'): #创建tp-cp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_cp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

    ### Expert-related parallel groups initialization
    # Build the expert model parallel group
    global _EXPERT_MODEL_PARALLEL_GROUP, _EXPERT_MODEL_PARALLEL_RANKS
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('ep'): #创建ep通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep", nccl_comm_cfgs),
            group_desc="EXPERT_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group
            _EXPERT_MODEL_PARALLEL_RANKS = ranks

    # Build the expert tensor parallel group
    global _EXPERT_TENSOR_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_PARALLEL_GROUP is None
    ), 'Expert tensor model parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('tp'): #创建etp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_tp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is None
    ), 'Expert tensor + model parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep'): #创建tp-ep通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_ep_mp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = group

    # Build the expert+tensor+pipeline parallel groups
    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is None
    ), 'The expert_tensor_model_pipeline parallel group is already initialized'
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep-pp'): #创建tp-ep-pp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_ep_pp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = group

    # Build the expert data parallel group
    global _EXPERT_DATA_PARALLEL_GROUP
    assert _EXPERT_DATA_PARALLEL_GROUP is None, "Expert data group is already initialized"
    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    assert _EXPERT_DATA_PARALLEL_GROUP_GLOO is None, "Expert data group-gloo is already initialized"
    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    assert (
        _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is None
    ), "Intra partial expert data group is already initialized"
    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    assert (
        _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is None
    ), "Intra partial expert data group-gloo is already initialized"
    global _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    assert (
        _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is None
    ), "Inter partial expert data group is already initialized"

    assert (
        expert_data_parallel_size % num_distributed_optimizer_instances == 0
    ), "Expert data parallel size should be divisible by partial DistOpt shard factor"
    intra_partial_expert_data_parallel_size = (
        expert_data_parallel_size // num_distributed_optimizer_instances
    )

    for ranks in expert_decoder_rank_generator.get_ranks('dp'): #创建edp通信组
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_dp", nccl_comm_cfgs),
            group_desc="EXPERT_DATA_PARALLEL_GROUP",
        )
        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks, backend="gloo", group_desc="EXPERT_DATA_PARALLEL_GROUP_GLOO"
            )
        else:
            group_gloo = None
        if rank in ranks:
            _EXPERT_DATA_PARALLEL_GROUP = group
            _EXPERT_DATA_PARALLEL_GROUP_GLOO = group_gloo

        if num_distributed_optimizer_instances > 1: ##如果有num_distributed_optimizer_instances，优化器状态切分（ZeRO-1/2）
            # Create groups for Partial DistOpt, one for intra-partial DP domain
            # Another for inter-partial DP domain

            # Set NCCL_COLLNET_ENABLE to 1 to enable SHARP for the dp_replica group.
            if sharp_enabled_group == "dp_replica":
                os.environ["NCCL_COLLNET_ENABLE"] = "1"
            hierarchical_groups, hierarchical_groups_gloo = create_hierarchical_groups( #创建层次化edp通信组
                rank,
                ranks,
                [intra_partial_expert_data_parallel_size, num_distributed_optimizer_instances],
                create_gloo_process_groups=create_gloo_process_groups,
                pg_options=[
                    get_nccl_options("intra_ep_dp", nccl_comm_cfgs),
                    get_nccl_options("inter_ep_dp", nccl_comm_cfgs),
                ],
                timeout=timeout,
                group_desc="EXPERT_DATA_PARALLEL_GROUP",
            )
            if rank in ranks:
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = hierarchical_groups[0] #赋值intra
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = hierarchical_groups_gloo[0]
                _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = hierarchical_groups[1] #赋值inter

            if sharp_enabled_group == "dp_replica":
                # PyTorch is performing lazy initialization of the communicator group.
                # Therefore, we need to perform a nccl call to ensure that the communicator group is created.
                if _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None:  #强制直接初始化，避免lazy初始化
                    torch.distributed.barrier(
                        group=_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP,
                        device_ids=[torch.cuda.current_device()],
                    )
                    torch.cuda.synchronize()
                # Set NCCL_COLLNET_ENABLE to 0 to restrict SHARP application to the dp_replica group.
                if "NCCL_COLLNET_ENABLE" in os.environ:
                    del os.environ["NCCL_COLLNET_ENABLE"]
        else: #如果没有num_distributed_optimizer_instances，即优化器状态不切分（ZeRO-0），则edp通信组层次化处理退化为单层
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = _EXPERT_DATA_PARALLEL_GROUP
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = _EXPERT_DATA_PARALLEL_GROUP_GLOO
    ### End of expert related parallel groups initialization

    # build the intra distributed optimizer instance group
    global _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    assert (
        _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is None
    ), "Intra distributed optimizer instance group is already initialized"

    model_parallel_group_id = 0
    intra_dist_opt_ranks = []
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep-pp'): #构建单个 optimizer shard 内所有与 expert 计算相关的通信组
        model_parallel_group_id += 1
        intra_dist_opt_ranks.extend(ranks)
        if model_parallel_group_id % intra_partial_expert_data_parallel_size == 0:
            intra_dist_opt_instance_group = create_group(
                intra_dist_opt_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("intra_dist_opt_instance", nccl_comm_cfgs),
                group_desc="INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP",
            )
            if rank in intra_dist_opt_ranks:
                _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = intra_dist_opt_instance_group
            intra_dist_opt_ranks = []

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer() #初始化全局内存缓冲区


def create_all_gather_groups(for_expert_parallelism=False, timeout=None, nccl_comm_cfgs=None):
    """
    Helper function to create all-gather process groups for AG/RS overlap.

    Creates separate communicators with the same ranks as data parallel groups
    to enable overlapping all-gather operations with reduce-scatter operations.

    Args:
        for_expert_parallelism (bool): If True, also creates AG group for expert parameters.
        timeout (timedelta): Timeout for distributed collectives.
        nccl_comm_cfgs (dict): NCCL communicator configurations.

    Returns:
        tuple: (dp_cp_ag_group, expt_dp_ag_group) where expt_dp_ag_group is None
               if for_expert_parallelism=False.

    Example:
        # After initialize_model_parallel():
        dp_cp_ag, expt_dp_ag = parallel_state.create_all_gather_groups(
            for_expert_parallelism=True
        )

        # Add to ProcessGroupCollection:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp_ag = dp_cp_ag
        pg_collection.expt_dp_ag = expt_dp_ag
    """
    if not is_initialized():
        raise RuntimeError(
            "create_all_gather_groups() requires parallel state to be initialized. "
            "Call initialize_model_parallel() first."
        )

    rank = torch.distributed.get_rank()
    pp_size = get_pipeline_model_parallel_world_size()
    cp_size = get_context_parallel_world_size()
    tp_size = get_tensor_model_parallel_world_size()
    ep_size = get_expert_model_parallel_world_size()
    dp_size = get_data_parallel_world_size()

    # Create regular DP all-gather group
    dp_cp_ag_group = None
    decoder_rank_gen = RankGenerator(
        tp=tp_size, ep=1, dp=dp_size, pp=pp_size, cp=cp_size, order='tp-cp-ep-dp-pp', rank_offset=0
    )

    for ranks_with_cp in decoder_rank_gen.get_ranks('dp-cp'):#创建一个和dp-cp一样的通信组，只不过这个用于ag，通信-通信重叠
        group_with_cp_ag = create_group(
            ranks_with_cp,
            timeout=timeout,
            pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs or {}),
            group_desc='DATA_PARALLEL_GROUP_WITH_CP_AG',
        )
        if rank in ranks_with_cp:
            dp_cp_ag_group = group_with_cp_ag

    # Create expert DP all-gather group if requested
    expt_dp_ag_group = None
    if for_expert_parallelism and ep_size > 1:
        expert_tp_size = get_expert_tensor_parallel_world_size()
        expert_dp_size = get_expert_data_parallel_world_size()

        expert_rank_gen = RankGenerator(
            tp=expert_tp_size,
            ep=ep_size,
            dp=expert_dp_size,
            pp=pp_size,
            cp=1,
            order='tp-cp-ep-dp-pp',
            rank_offset=0,
        )

        for expert_dp_ranks in expert_rank_gen.get_ranks('dp'):##创建一个和edp一样的通信组，只不过这个用于ag
            expert_dp_ag = create_group(
                expert_dp_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("ep_dp", nccl_comm_cfgs or {}),
                group_desc='EXPERT_DATA_PARALLEL_GROUP_AG',
            )
            if rank in expert_dp_ranks:
                expt_dp_ag_group = expert_dp_ag

    return dp_cp_ag_group, expt_dp_ag_group


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def model_parallel_is_initialized():
    """Check if model- and data-parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group(check_initialized=True):
    """Get the model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group(check_initialized=True):
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _PIPELINE_MODEL_PARALLEL_GROUP is not None
        ), "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_encoder_inner_data_parallel_group(check_initialized=True):
    """Get the encoder inner data-parallel group (colocated encoder training).

    The encoder inner dp group contains the P ranks inside one outer
    data-parallel replica (one rank per pipeline stage; with TP=1 these are
    exactly the members of one pipeline-parallel group). It is used for the
    inner-layer gradient all-reduce of the colocated encoder, which runs before
    the regular (outer) data-parallel all-reduce.

    获取 encoder 内部数据并行组（共置训练）：包含一个外层 dp 副本内的 P 个
    rank（每个 pipeline stage 一个；TP=1 时即 pipeline-parallel 组的成员）。
    用于 encoder 梯度的第一层（inner）all-reduce，先于外层 dp all-reduce 执行。

    Only available when ``initialize_model_parallel(use_colocated_encoder=True)``.
    仅在 ``initialize_model_parallel(use_colocated_encoder=True)`` 时可用。
    """
    if check_initialized:
        assert _ENCODER_INNER_DATA_PARALLEL_GROUP is not None, (
            "encoder inner data parallel group is not initialized"
        )
    return _ENCODER_INNER_DATA_PARALLEL_GROUP


def get_encoder_inner_data_parallel_rank():
    """Get the caller rank's index within its encoder inner dp group.

    Under the colocated layout this equals the caller's pipeline stage index
    (s), which is also the microbatch slot assigned to this rank by the
    round-robin encoder schedule (microbatch s when num_microbatches == pp_size).

    返回当前 rank 在其 encoder inner dp 组内的编号：共置布局下它等于当前
    rank 的 pipeline stage 序号（s），也就是轮盘式 encoder 调度分配给该
    rank 的 microbatch 槽位（当 num_microbatches == pp_size 时为 microbatch s）。
    """
    group = get_encoder_inner_data_parallel_group()
    return torch.distributed.get_group_rank(group, torch.distributed.get_rank())


def get_encoder_inner_data_parallel_global_ranks():
    """Get all global ranks of the encoder inner dp group the caller belongs to.

    返回当前 rank 所属 encoder inner dp 组的全部全局 rank 列表。
    """
    assert _ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS is not None, (
        "encoder inner data parallel global ranks are not initialized"
    )
    return _ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS


def get_colocated_encoder_intra_distributed_optimizer_instance_group(check_initialized=True):
    """Get the colocated encoder's intra distributed optimizer instance group.

    获取共置 encoder 自己的分布式优化器实例内通信组。
    """
    if check_initialized:
        assert _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is not None, (
            "colocated encoder intra distributed optimizer instance group is not initialized"
        )
    return _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP


def get_colocated_encoder_intra_distributed_optimizer_instance_global_ranks():
    """Get global ranks in the colocated encoder's intra optimizer instance group.

    获取共置 encoder 分布式优化器实例内通信组的全部全局 rank。
    """
    assert _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS is not None, (
        "colocated encoder intra distributed optimizer instance ranks are not initialized"
    )
    return _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS


def get_colocated_encoder_inter_distributed_optimizer_instance_group(check_initialized=True):
    """Get the group spanning the colocated encoder's optimizer instances.

    获取横跨共置 encoder 各分布式优化器实例的通信组。
    """
    if check_initialized:
        assert _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is not None, (
            "colocated encoder inter distributed optimizer instance group is not initialized"
        )
    return _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP


def get_colocated_encoder_inter_distributed_optimizer_instance_global_ranks():
    """Get global ranks in the colocated encoder's inter optimizer instance group.

    获取横跨共置 encoder 各分布式优化器实例通信组的全部全局 rank。
    """
    assert _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS is not None, (
        "colocated encoder inter distributed optimizer instance ranks are not initialized"
    )
    return _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS


def get_colocated_boundary_group(check_initialized=True):
    """Get the colocated boundary communication group (colocated encoder training).

    The group contains the P ranks of one outer dp replica (same members as the
    encoder inner dp group) but is a SEPARATE NCCL instance used exclusively for
    the encoder->backbone-entry boundary P2P (forward packet and backward grad).
    It stays independent from the pipeline-parallel group to avoid serializing
    boundary P2P with the 1F1B P2P on the same NCCL group.

    获取共置边界通信组（共置训练）：成员为一个外层 dp 副本内的 P 个 rank（与
    encoder inner dp 组相同），但是独立的 NCCL 实例，专用于 encoder→backbone
    entry 的边界 P2P。与 pp_group 独立，避免边界 P2P 与 1F1B P2P 同组串行。

    Only available when ``initialize_model_parallel(use_colocated_encoder=True)``.
    仅在 ``initialize_model_parallel(use_colocated_encoder=True)`` 时可用。
    """
    if check_initialized:
        assert _COLOCATED_BOUNDARY_GROUP is not None, (
            "colocated boundary communication group is not initialized"
        )
    return _COLOCATED_BOUNDARY_GROUP


def is_colocated_encoder_enabled():
    """Whether colocated encoder training is enabled in this process.

    True once ``initialize_model_parallel(use_colocated_encoder=True)`` has built the
    colocated groups. This is the predicate the schedule dispatcher
    (``get_forward_backward_func``) uses to pick ``forward_backward_colocated``:
    megatron.core must not read megatron.training's args, so the parallel state that
    the flag already produced is the single source of truth.

    本进程是否启用共置 encoder 训练：``initialize_model_parallel(use_colocated_encoder=True)``
    建好共置组后为 True。schedule 分发（``get_forward_backward_func``）用它选
    ``forward_backward_colocated``——megatron.core 不读 megatron.training 的 args，
    以该开关已经落地的并行状态为单一来源。
    """
    return _COLOCATED_BOUNDARY_GROUP is not None


def get_colocated_boundary_global_ranks():
    """Get all global ranks of the colocated boundary group the caller belongs to.

    返回当前 rank 所属共置边界通信组的全部全局 rank 列表。
    """
    assert _COLOCATED_BOUNDARY_GLOBAL_RANKS is not None, (
        "colocated boundary communication global ranks are not initialized"
    )
    return _COLOCATED_BOUNDARY_GLOBAL_RANKS


def get_colocated_data_parallel_group(check_initialized=True):
    """Get the colocated data-parallel group (colocated encoder training).

    The group contains ALL W ranks that hold an encoder replica, i.e. the outer
    data-parallel dimension times the encoder inner dimension (D_outer x P). The
    encoder chunk gives this group to DDP as its dp/dp_cp group, so one gradient
    reduction covers both dimensions at once. This is only equivalent to the
    two-step (inner SUM, then outer reduction) formulation because per-token loss
    makes the data-parallel reduction a pure SUM.

    获取共置数据并行组（共置训练）：包含持有 encoder 副本的全部 W 个 rank，即外层
    数据并行维度 × encoder 内部维度（D_outer x P）。encoder chunk 把它作为
    dp/dp_cp 组交给 DDP，一次归约同时覆盖两个维度；与"先 inner SUM 再 outer 归约"
    等价的前提是 per-token loss 使数据并行归约为纯 SUM。

    Only available when ``initialize_model_parallel(use_colocated_encoder=True)``.
    仅在 ``initialize_model_parallel(use_colocated_encoder=True)`` 时可用。
    """
    if check_initialized:
        assert _COLOCATED_DATA_PARALLEL_GROUP is not None, (
            "colocated data parallel group is not initialized"
        )
    return _COLOCATED_DATA_PARALLEL_GROUP


def get_colocated_data_parallel_global_ranks():
    """Get all global ranks of the colocated data-parallel group the caller belongs to.

    返回当前 rank 所属共置数据并行组的全部全局 rank 列表。
    """
    assert _COLOCATED_DATA_PARALLEL_GLOBAL_RANKS is not None, (
        "colocated data parallel global ranks are not initialized"
    )
    return _COLOCATED_DATA_PARALLEL_GLOBAL_RANKS


def get_colocated_encoder_pipeline_model_parallel_group(check_initialized=True):
    """Get the colocated encoder pipeline group (colocated encoder training).

    The group holds the caller rank alone: the encoder is replicated in full on
    every rank, so it is neither pipeline- nor context-parallel. It is handed to
    the encoder chunk as both its pipeline and its context group.

    获取共置 encoder 的 pipeline 组（共置训练）：组内只有当前 rank——encoder 在每个
    rank 上都是完整副本，既不做 PP 也不做 CP。它同时作为 encoder chunk 的 pipeline
    组与 context 组。

    Only available when ``initialize_model_parallel(use_colocated_encoder=True)``.
    仅在 ``initialize_model_parallel(use_colocated_encoder=True)`` 时可用。
    """
    if check_initialized:
        assert _COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP is not None, (
            "colocated encoder pipeline model parallel group is not initialized"
        )
    return _COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP


def get_colocated_encoder_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group of the colocated encoder.

    获取共置 encoder 的张量并行组：成员与作业常规 tp 组相同，但是独立 NCCL 实例。
    encoder 侧的一切张量并行行为（参数切分、取数广播）都应该走这个组。

    Only available when ``initialize_model_parallel(use_colocated_encoder=True)``.
    仅在 ``initialize_model_parallel(use_colocated_encoder=True)`` 时可用。
    """
    if check_initialized:
        assert _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP is not None, (
            "colocated encoder tensor model parallel group is not initialized"
        )
    return _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP


def get_colocated_encoder_tensor_model_parallel_global_ranks():
    """Get the global ranks of the colocated encoder tensor model parallel group.

    获取共置 encoder 张量并行组的全局 rank 列表。
    """
    assert _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None, (
        "colocated encoder tensor model parallel group is not initialized"
    )
    return _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS


def build_colocated_encoder_process_groups():
    """Build the process group collection of the colocated encoder model chunk.

    The collection describes the ENCODER's own topology, not the backbone's,
    because the encoder's communication is fully separate from the backbone's:
      - ``dp`` / ``dp_cp``: the colocated data-parallel group (all W ranks), the
        group DDP reduces the encoder gradients over;
      - ``pp`` / ``cp``: the single-member colocated encoder pipeline group, since
        every rank holds a full encoder replica (no pipeline, no context
        parallelism). This also makes every member of the colocated
        data-parallel group report pipeline rank 0 and therefore derive the same
        bucket layout - handing over the backbone's pipeline group instead would
        make the members disagree on the number of buckets and deadlock the
        per-bucket gradient reduction (distributed_data_parallel.py:102-107);
      - ``intra_dp_cp`` / ``intra_dist_opt``: the encoder's own intra optimizer
        instance group, derived from its data-parallel ranks and instance count;
      - ``inter_dist_opt``: the encoder's own inter-instance group, or ``None``
        when the encoder has one optimizer instance;
      - ``embd`` / ``pos_embd``: the single-member colocated encoder pipeline
        group as well. Those two groups exist only to keep tied word / position
        embeddings in sync between the first and the last pipeline stage, so they
        are meaningless for a component that has no pipeline and no tied
        embeddings, and a single-member group is exactly how "nothing to sync"
        is expressed: ``_allreduce_embedding_grad`` gates on
        ``get_pg_size(embd_group) > 1`` plus membership
        (finalize_model_grads.py:229-230), so both sections are skipped.
        Note that ``None`` does NOT work here even though ``get_pg_size(None)``
        returns 1: ``_allreduce_word_embedding_grads`` treats a ``None`` group as
        "not supplied" and falls back to the JOB-WIDE embedding group
        (finalize_model_grads.py:189-193), which has two members under PP>1 and
        then trips ``assert pp_group is None`` because the encoder collection did
        supply a pipeline group. Inheriting the backbone's embedding group would
        be wrong for the same reason the fallback is: on the first / last
        backbone stage it would walk into the ViT chunk looking for
        ``pre_process`` and ``share_embeddings_and_output_weights``, neither of
        which ``ColocatedViTEncoder`` has;
      - ``mp``: the encoder's tensor-parallel group alone. ``mp`` is the group the
        optimizer all-reduces the gradient norm over (optimizer.py:181-199 ->
        clip_grads.py:133-137), so it must cover exactly the ranks the parameters
        are PARTITIONED over and none of the ranks holding replicas. The encoder
        is replicated across the pipeline dimension, so the job's regular
        tp x pp group would sum the same squared norm P times and over-clip the
        encoder gradients; with a single-member pipeline group the encoder's
        tp x pp product IS its tensor-parallel group;
      - ``tp``: the colocated encoder tensor model parallel group - same members as
        the job's (the encoder tensor parallel size is an explicit argument and must
        equal the job's, see validate_colocated_args) but the encoder's own
        communicator, so that no part of the encoder's description points at a
        backbone group.
    The intra/inter pair is only consumed once
    ``ddp_config.num_distributed_optimizer_instances > 1``, so attaching it here
    is inert for the current (non distributed optimizer) path and lets that
    variant be a configuration change instead of a process-group change.

    构建共置 encoder model chunk 的进程组集合：描述的是 **encoder 自身**的拓扑而
    非 backbone 的，因为 encoder 的通信与 backbone 完全分开——``dp``/``dp_cp`` 为
    共置数据并行组（全部 W 个 rank，encoder 梯度就在其上归约）；``pp``/``cp`` 为
    单成员的共置 encoder pipeline 组，因为每个 rank 都持有完整 encoder 副本（既无
    PP 也无 CP），这同时让共置数据并行组内每个成员报出的 pipeline rank 都是 0、
    推导出相同的分桶布局——若改传 backbone 的 pipeline 组，成员间桶数不一致会让
    逐 bucket 的梯度归约死锁（distributed_data_parallel.py:102-107）；
    ``intra_dp_cp``/``intra_dist_opt`` 为根据 encoder 自己的数据并行 rank 和实例数
    推导出的实例内组；``inter_dist_opt`` 为 encoder 自己的跨实例组，单实例时为 None。
    intra/inter 这一对只在
    ``ddp_config.num_distributed_optimizer_instances > 1`` 时才被消费，因此在当前
    （非分布式优化器）路径下挂了不生效，但能让该变体只改配置、不动进程组。
    ``embd``/``pos_embd`` 同样取那个单成员 pipeline 组：这两个组的唯一用途是让首尾
    pipeline stage 上共享的 word/position embedding 保持同步，对"无 PP、无共享 embedding"
    的组件毫无意义，而"单成员组"正是"没什么要同步"的表达方式——``_allreduce_embedding_grad``
    按 ``get_pg_size(embd_group) > 1`` 与成员身份放行（finalize_model_grads.py:229-230），
    单成员 ⇒ 两段都跳过。注意**不能置 ``None``**（虽然 ``get_pg_size(None)`` 返回 1）：
    ``_allreduce_word_embedding_grads`` 把 ``None`` 当成"没传"，会退回去取**全作业**的
    embedding 组（finalize_model_grads.py:189-193），PP>1 时它有两个成员，紧接着的
    ``assert pp_group is None`` 就会炸——因为 encoder collection 确实传了 pipeline 组。
    沿用 backbone 的 embedding 组错在同一处：首/末 backbone stage 上会走进去、在 ViT
    chunk 上取 ``pre_process`` 与 ``share_embeddings_and_output_weights``，而
    ``ColocatedViTEncoder`` 两者都没有。
    ``mp`` 单独取 encoder 的张量并行组：``mp`` 是优化器做梯度范数 all_reduce 的组
    （optimizer.py:181-199 → clip_grads.py:133-137），它必须恰好覆盖参数被**切分**
    的 rank、不能覆盖持有副本的 rank；encoder 在 pipeline 维上是副本，若用作业常规
    的 tp × pp 组会把同一份范数平方加 P 次、过度裁剪 encoder 梯度。pp 组只有一个
    成员时，encoder 的 tp × pp 就等于它的张量并行组。
    ``tp`` 取共置 encoder 张量并行组：成员与作业 tp 组相同（encoder tp 并行度是显式
    参数且必须等于作业的，见 validate_colocated_args），但是 encoder 自己的通信子，
    这样 encoder 的通信描述里不再有任何一项指向 backbone 的组。

    Only available when ``initialize_model_parallel(use_colocated_encoder=True)``.
    仅在 ``initialize_model_parallel(use_colocated_encoder=True)`` 时可用。
    """
    from megatron.core.process_groups_config import ProcessGroupCollection

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp = get_colocated_data_parallel_group()
    pg_collection.dp_cp = get_colocated_data_parallel_group()
    pg_collection.pp = get_colocated_encoder_pipeline_model_parallel_group()
    pg_collection.cp = get_colocated_encoder_pipeline_model_parallel_group()
    pg_collection.intra_dp_cp = (
        get_colocated_encoder_intra_distributed_optimizer_instance_group()
    )
    pg_collection.intra_dist_opt = (
        get_colocated_encoder_intra_distributed_optimizer_instance_group()
    )
    pg_collection.inter_dist_opt = (
        get_colocated_encoder_inter_distributed_optimizer_instance_group(
            check_initialized=False
        )
    )
    pg_collection.embd = get_colocated_encoder_pipeline_model_parallel_group()
    pg_collection.pos_embd = get_colocated_encoder_pipeline_model_parallel_group()
    pg_collection.tp = get_colocated_encoder_tensor_model_parallel_group()
    pg_collection.mp = get_colocated_encoder_tensor_model_parallel_group()
    # Task 5.12: the encoder has no MoE experts, so its expert-parallel groups are
    # single-member (each rank its own), exactly like pp/cp/embd/pos_embd - the
    # "nothing to shard" expression. They must be the ENCODER's own communicator:
    # ``_set_random_seed`` computes expert-parallel-rng from ``ep``/``expt_tp`` and
    # would otherwise fall back to the backbone's EP/ETP groups, tying the encoder's
    # RNG stream to the backbone topology. A single-member group yields rank 0 on
    # every rank, so the encoder's expert seed is identical across all replicas.
    # Task 5.12：encoder 没有 MoE 专家，其专家并行组取单成员（每 rank 各自一个），
    # 与 pp/cp/embd/pos_embd 同为"无切分"的表达。它们必须是 encoder **自己的**通信子：
    # ``_set_random_seed`` 用 ``ep``/``expt_tp`` 算 expert-parallel-rng，若不覆盖会回落
    # 到 backbone 的 EP/ETP 组，把 encoder 的随机流绑到 backbone 拓扑上。单成员组在
    # 每个 rank 上 rank 都是 0，因此 encoder 的 expert 种子在所有副本间一致。
    pg_collection.ep = get_colocated_encoder_pipeline_model_parallel_group()
    pg_collection.expt_tp = get_colocated_encoder_pipeline_model_parallel_group()
    pg_collection.tp_ep = get_colocated_encoder_pipeline_model_parallel_group()
    pg_collection.tp_ep_pp = get_colocated_encoder_pipeline_model_parallel_group()
    return pg_collection


def validate_colocated_num_microbatches(num_microbatches):
    """Validate num_microbatches for round-robin colocated encoder scheduling.

    Under the round-robin schedule, the microbatches are distributed across the
    pipeline stages (microbatch s is computed by stage s), so num_microbatches
    must be a multiple of the pipeline model parallel size.

    校验轮盘式共置 encoder 调度对 num_microbatches 的要求：轮盘调度把
    microbatch 按 pipeline stage 分发（microbatch s 由 stage s 计算），因此
    num_microbatches 必须是 pipeline model parallel size 的整数倍，且为正数。
    """
    pp_size = get_pipeline_model_parallel_world_size()
    assert num_microbatches > 0, f"num_microbatches must be positive, got {num_microbatches}"
    assert num_microbatches % pp_size == 0, (
        f"num_microbatches ({num_microbatches}) must be a multiple of the pipeline "
        f"model parallel size ({pp_size}) for round-robin colocated encoder scheduling"
    )


def get_microbatches_for_producer(producer_id, num_microbatches, num_producers):
    """Return the microbatch indices assigned to one encoder producer under the
    round-robin colocated encoder schedule.

    With num_microbatches == k * num_producers, producer p computes microbatches
    p, p + num_producers, p + 2 * num_producers, ... (k of them). The producer id is a
    slot index inside the boundary group (the group's own rank), not a global rank, so
    the mapping is independent of the job's rank order (Task 5.7).

    返回轮盘式共置 encoder 调度下，指定 encoder producer 负责的 microbatch 索引：
    num_microbatches == k * num_producers 时，producer p 计算 microbatch
    p, p + num_producers, p + 2 * num_producers, ...（共 k 个）。producer 编号是边界组
    内的槽位（组内 rank）而非全局 rank，因此该映射与作业的 rank order 无关（Task 5.7）。
    """
    assert num_producers > 0, f"num_producers must be positive, got {num_producers}"
    assert 0 <= producer_id < num_producers, (
        f"producer_id ({producer_id}) must be in [0, num_producers ({num_producers}))"
    )
    assert num_microbatches > 0, f"num_microbatches must be positive, got {num_microbatches}"
    assert num_microbatches % num_producers == 0, (
        f"num_microbatches ({num_microbatches}) must be a multiple of the number of encoder "
        f"producers ({num_producers}) for round-robin colocated encoder scheduling"
    )
    return list(range(producer_id, num_microbatches, num_producers))


def get_data_parallel_group(with_context_parallel=False, partial_data_parallel=False):
    """Get the data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if partial_data_parallel:
            assert (
                _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), "Intra partial data parallel group is not initialized"
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), "data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
        assert partial_data_parallel == False, "Partial DP for Optimizer needs to include CP"
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo(with_context_parallel=False, partial_data_parallel=False):
    """Get the Gloo data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if partial_data_parallel:
            assert (
                _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
            ), "Intra partial data parallel group is not initialized"
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), "data parallel group-gloo with context parallel combined is not initialized"
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, "data parallel group-gloo is not initialized"
        assert partial_data_parallel == False, "Partial DP for Optimizer needs to include CP"
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context-parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_hierarchical_context_parallel_groups(check_initialized=True):
    """Get the inner ring of context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS is not None
    return _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS


def get_hybrid_data_context_parallel_groups(check_initialized=True, group_size=None):
    """Get the hybrid context parallel groups the caller rank belongs to."""
    # If the group size is the same as the entire DPxCP group, return the original group
    if get_data_parallel_world_size(with_context_parallel=True) == group_size:
        if check_initialized:
            assert _DATA_PARALLEL_GROUP_WITH_CP is not None
        return _DATA_PARALLEL_GROUP_WITH_CP
    if check_initialized:
        assert _HYBRID_DP_CP_GROUPS is not None
    return _HYBRID_DP_CP_GROUPS[group_size]


def get_embedding_group(check_initialized=True):
    """Get the embedding group the caller rank belongs to."""
    if check_initialized:
        assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP


def get_position_embedding_group(check_initialized=True):
    """Get the position embedding group the caller rank belongs to."""
    if check_initialized:
        assert _POSITION_EMBEDDING_GROUP is not None, "position embedding group is not initialized"
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        if not tp_only_amax_red:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
        else:
            assert (
                _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    else:
        if not tp_only_amax_red:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_AND_DATA_PARALLEL_GROUP
        else:
            assert (
                _TENSOR_MODEL_PARALLEL_GROUP is not None
            ), "FP8 amax reduction group is not initialized"
            return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(check_initialized=True, with_context_parallel=False):
    """Get the tensor- and data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if check_initialized:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        if check_initialized:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
            ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_context_parallel_group(check_initialized=True):
    """Get the tensor- and context-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None
        ), "tensor and context parallel group is not initialized"
    return _TENSOR_AND_CONTEXT_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor-model-parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline-model-parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline-model-parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return get_tensor_model_parallel_group().size()


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline-model-parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return get_pipeline_model_parallel_group().size()


def set_tensor_model_parallel_rank(rank):
    """Set tensor-model-parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline-model-parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    """Return caller's rank for the tensor-model-parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return get_tensor_model_parallel_group().rank()


def get_pipeline_model_parallel_rank():
    """Return caller's rank for the pipeline-model-parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"

        if vp_stage != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=True, vp_stage=None):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"

        if vp_stage != (get_virtual_pipeline_model_parallel_world_size() - 1):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=True, vp_stage=None):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if _EMBEDDING_GLOBAL_RANKS is None:
        return False
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return _POSITION_EMBEDDING_GLOBAL_RANKS is not None and rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    warnings.warn(
        "set_virtual_pipeline_model_parallel_rank in global scope is deprecated. "
        "Pass vp_stage explicitly to is_pipeline_first_stage, is_pipeline_last_stage, etc.",
        DeprecationWarning,
    )
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    assert _MODEL_PARALLEL_GLOBAL_RANKS is not None, "Model parallel group is not initialized"
    return _MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first stage in the current rank's pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last stage in the current rank's pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that precedes the caller in the pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False, partial_data_parallel=False):
    """Return world size for the data parallel group."""
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    if _MPU_DATA_PARALLEL_WORLD_SIZE is not None:
        return _MPU_DATA_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_data_parallel_group(
            with_context_parallel=with_context_parallel, partial_data_parallel=partial_data_parallel
        ).size()
    else:
        return 0


def set_data_parallel_rank(rank):
    """Return world size for the data parallel group."""
    global _MPU_DATA_PARALLEL_RANK
    _MPU_DATA_PARALLEL_RANK = rank


def get_data_parallel_rank(with_context_parallel=False, partial_data_parallel=False):
    """Return caller's rank in the data-parallel group."""
    global _MPU_DATA_PARALLEL_RANK
    if _MPU_DATA_PARALLEL_RANK is not None:
        return _MPU_DATA_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_data_parallel_group(
            with_context_parallel=with_context_parallel, partial_data_parallel=partial_data_parallel
        ).rank()
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_context_parallel_group().size()
    else:
        return 0


def get_context_parallel_rank():
    """Return caller's rank in the context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_context_parallel_group().rank()
    else:
        return 0


def get_tensor_and_context_parallel_world_size():
    """Return world size for the tensor and context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_tensor_and_context_parallel_group().size()
    else:
        return 0


def get_tensor_and_context_parallel_rank():
    """Return caller's rank in the joint tensor-model-parallel and context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_tensor_and_context_parallel_group().rank()
    else:
        return 0


### Expert-related parallel states functions
def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_MODEL_PARALLEL_GROUP is not None
        ), "expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the expert model parallel group."""
    assert (
        _EXPERT_MODEL_PARALLEL_RANKS is not None
    ), "Expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_RANKS[0]


def get_expert_model_parallel_world_size():
    """Return world size for the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_model_parallel_group().size()
    else:
        return 0


def set_expert_model_parallel_world_size(world_size):
    """Sets the expert-model-parallel world size."""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_model_parallel_group().rank()
    else:
        return 0


def set_expert_model_parallel_rank(rank):
    """Set expert-model-parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def get_expert_tensor_parallel_group(check_initialized=True):
    """Get the expert-tensor-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_PARALLEL_GROUP is not None
        ), "Expert tensor parallel group is not initialized"
    return _EXPERT_TENSOR_PARALLEL_GROUP


def get_expert_tensor_parallel_world_size():
    """Return world size for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    # Use tensor parallel group world size for backward compability otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    else:
        return get_expert_tensor_parallel_group().size()


def set_expert_tensor_parallel_world_size(world_size):
    "Set expert tensor model parallel size"
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = world_size


def get_expert_tensor_parallel_rank():
    """Return my rank for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if _MPU_EXPERT_TENSOR_PARALLEL_RANK is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_RANK
    # Use tensor parallel group rank for backward compability otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    else:
        return get_expert_tensor_parallel_group().rank()


def set_expert_tensor_parallel_rank(rank):
    "Set expert tensor model parallel rank"
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = rank


def get_expert_tensor_and_model_parallel_group(check_initialized=True):
    """Get the expert-tensor and expert-model group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None
        ), "Expert tensor and model parallel group is not initialized"
    return _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP


def get_expert_tensor_and_model_parallel_world_size():
    """Return world size for the expert model parallel group times expert tensor parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = get_expert_tensor_and_model_parallel_group().size()
        return world_size
    else:
        return 0


def get_expert_tensor_and_model_parallel_rank():
    """Return caller's rank in the joint tensor- and expert-model-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_tensor_and_model_parallel_group().rank()
    else:
        return 0


def get_expert_tensor_model_pipeline_parallel_group(check_initialized=True):
    """Get expert tensor-model-pipeline parallel group."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is not None
        ), "Expert tensor-model-pipeline parallel group is not initialized"
    return _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP


def get_expert_data_parallel_group(check_initialized=True, partial_expert_data_parallel=False):
    """Get expert data parallel group."""
    if partial_expert_data_parallel:
        if check_initialized:
            assert (
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None
            ), "Intra partial expert data parallel group is not initialized"
        return _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    else:
        if check_initialized:
            assert (
                _EXPERT_DATA_PARALLEL_GROUP is not None
            ), "Expert data parallel group is not initialized"
        return _EXPERT_DATA_PARALLEL_GROUP


def get_expert_data_parallel_group_gloo(partial_expert_data_parallel=False):
    """Get expert data parallel group-gloo."""
    if partial_expert_data_parallel:
        assert (
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        ), "Intra partial expert data parallel group-gloo is not initialized"
        return _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    else:
        assert (
            _EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        ), "Expert data parallel group-gloo is not initialized"
        return _EXPERT_DATA_PARALLEL_GROUP_GLOO


def get_expert_data_parallel_rank(partial_expert_data_parallel=False):
    """Return caller's rank in the expert data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_data_parallel_group(
            partial_expert_data_parallel=partial_expert_data_parallel
        ).rank()
    else:
        return 0


def get_expert_data_parallel_world_size(partial_expert_data_parallel=False):
    """Return world size for the expert data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_data_parallel_group(
            partial_expert_data_parallel=partial_expert_data_parallel
        ).size()
    else:
        return 0


def get_intra_distributed_optimizer_instance_group(check_initialized=True):
    """Get the group of all GPUs in a distributed optimizer instance."""
    if check_initialized:
        assert (
            _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is not None
        ), "Intra distributed optimizer instance group is not initialized"
    return _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP


def get_inter_distributed_optimizer_instance_group(check_initialized=True):
    """Get the group spanning the different distributed optimizer instances.
    Attention and MLP/Expert share same inter-instance group, so only built
    inter_partial_expert_data_parallel_group, and return it at here.
    """
    if check_initialized:
        assert _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None, (
            "Attention and MLP/Expert share same inter distributed optimize instance group, "
            "which has not been initialized"
        )
    return _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP


### End of expert-related functions region


def _set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, "global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def get_all_ranks():
    """Get caller's rank in tensor-model-parallel, data-parallel, context-parallel,
    pipeline-model-parallel and expert-model-parallel groups."""
    ranks = [
        get_tensor_model_parallel_rank(),
        get_data_parallel_rank(),
        get_context_parallel_rank(),
        get_pipeline_model_parallel_rank(),
        get_expert_model_parallel_rank(),
    ]
    return "_".join(map(lambda x: str(x or 0), ranks))


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None

    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None

    global _ENCODER_INNER_DATA_PARALLEL_GROUP
    _ENCODER_INNER_DATA_PARALLEL_GROUP = None

    global _ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS
    _ENCODER_INNER_DATA_PARALLEL_GLOBAL_RANKS = None

    global _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

    global _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS
    _COLOCATED_ENCODER_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS = None

    global _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

    global _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS
    _COLOCATED_ENCODER_INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GLOBAL_RANKS = None

    global _COLOCATED_BOUNDARY_GROUP
    _COLOCATED_BOUNDARY_GROUP = None

    global _COLOCATED_BOUNDARY_GLOBAL_RANKS
    _COLOCATED_BOUNDARY_GLOBAL_RANKS = None

    global _COLOCATED_DATA_PARALLEL_GROUP
    _COLOCATED_DATA_PARALLEL_GROUP = None

    global _COLOCATED_DATA_PARALLEL_GLOBAL_RANKS
    _COLOCATED_DATA_PARALLEL_GLOBAL_RANKS = None

    global _COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP
    _COLOCATED_ENCODER_PIPELINE_MODEL_PARALLEL_GROUP = None

    global _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP
    _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GROUP = None

    global _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    _COLOCATED_ENCODER_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

    global _DATA_PARALLEL_GROUP_WITH_CP

    _DATA_PARALLEL_GROUP_WITH_CP = None

    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None

    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None

    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None

    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None

    global _POSITION_EMBEDDING_GLOBAL_RANKS
    _POSITION_EMBEDDING_GLOBAL_RANKS = None

    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None

    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    _TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None

    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None

    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

    global _DATA_PARALLEL_GROUP_GLOO
    if (
        _DATA_PARALLEL_GROUP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(_DATA_PARALLEL_GROUP_GLOO, None)
        is not None
    ):
        torch.distributed.destroy_process_group(_DATA_PARALLEL_GROUP_GLOO)
    _DATA_PARALLEL_GROUP_GLOO = None

    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    if (
        _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO, None
        )
        is not None
    ):
        torch.distributed.destroy_process_group(_DATA_PARALLEL_GROUP_WITH_CP_GLOO)
    _DATA_PARALLEL_GROUP_WITH_CP_GLOO = None

    # Destroy parallel state related to expert parallelism.
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None

    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None

    global _EXPERT_TENSOR_PARALLEL_GROUP
    _EXPERT_TENSOR_PARALLEL_GROUP = None

    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None

    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = None

    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None

    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None

    global _EXPERT_DATA_PARALLEL_GROUP
    _EXPERT_DATA_PARALLEL_GROUP = None

    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    if (
        _EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(
            _EXPERT_DATA_PARALLEL_GROUP_GLOO, None
        )
        is not None
    ):
        torch.distributed.destroy_process_group(_EXPERT_DATA_PARALLEL_GROUP_GLOO)
    _EXPERT_DATA_PARALLEL_GROUP_GLOO = None

    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None

    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    if (
        _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
        and torch.distributed.distributed_c10d._world.pg_map.get(
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO, None
        )
        is not None
    ):
        torch.distributed.destroy_process_group(_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO)
    _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = None

    global _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
    # End of expert parallelism destroy.

    global _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

    global _global_process_group_list
    _global_process_group_list = None

    SymmetricMemoryManager.destroy()
