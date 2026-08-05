# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


class CommRole(Enum):
    """Communication role for ranks in bridge communication.

    SENDER: Leader tp-cp rank within each DP replica of source grid. #src模块中的leader rank，负责将激活值发送到dest模块的leader rank
            Sends data to destination grid receivers.
    RECEIVER: Leader tp-cp rank within each DP replica of destination grid. #dest模块中的leader rank，负责接收src模块的leader rank发送的激活值
              Receives data from source grid senders.
    MEMBER: Non-leader ranks within DP replicas. #不参与模块间通信，只会进行模块内的broadcast来获取leader rank获取到的激活值
            Participate in broadcasts from their local leader.
    """

    SENDER = "SENDER"
    RECEIVER = "RECEIVER"
    MEMBER = "MEMBER"


@dataclass
class RankCommInfo:
    """Explicit communication plan for a single rank."""

    role: CommRole = CommRole.MEMBER
    send_to_ranks: List[int] = field(default_factory=list)
    recv_from_ranks: List[int] = field(default_factory=list)


class BridgeCommunicator:
    """Pipeline Communicator between two modules with different(TP/DP/PP/CP). #要求src和dest的dp数量必须要一方能整除另一方

    BridgeCommunicator:
    - Initialize the communicator between a pair of source and destination grids
    - Build a communication schedule for each rank
    - Provide public methods: send_forward, recv_forward, send_forward_recv_backward,
      send_backward_recv_forward to be used by the pipeline schedule.
    """

    # Cache broadcast PGs to avoid creating duplicate NCCL communicators for identical rank sets.
    _broadcast_pg_cache: Dict[str, "torch.distributed.ProcessGroup"] = {}

    @classmethod
    def destroy_broadcast_pgs(cls):
        """Destroy all cached broadcast process groups."""
        for pg in cls._broadcast_pg_cache.values():
            if pg is not None:
                dist.destroy_process_group(pg)
        cls._broadcast_pg_cache.clear()

    def __init__(
        self,
        src_grid: HyperCommGrid,
        dest_grid: HyperCommGrid,
        dim_mapping: Optional[Dict[str, int]] = None,
        comm_dtype: Optional[torch.dtype] = None, #通信数据类型（fp32/bf16这些）
        src_module_name: Optional[str] = None,
        dest_module_name: Optional[str] = None,
        tensor_ndim: int = 3,
    ):
        """Initialize the bridge communicator between source and destination grids.

        CP is not supported yet. Will be added in follow up PR.

        Args:
            src_grid: Source HyperCommGrid
            dest_grid: Destination HyperCommGrid
            dim_mapping: Dictionary mapping logical dimensions to tensor axes.
                        Expected keys: 's' (sequence), 'b' (batch), 'h' (hidden).
                        Defaults to {'s': 1, 'b': 0, 'h': 2} if None.
            tensor_ndim: Number of dimensions in tensors communicated through this
                        bridge. For 3D tensors (e.g. [S, B, H]), fan-in/fan-out
                        operates on dim_mapping['b']. For 2D tensors (e.g. [B*S, H]
                        where batch is folded into dim 0), fan-in/fan-out operates
                        on dim 0. Default: 3.
        """
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.src_module_name = src_module_name
        self.dest_module_name = dest_module_name
        self.comm_dtype = comm_dtype

        assert tensor_ndim in (2, 3), f"tensor_ndim must be 2 or 3, got {tensor_ndim}"
        self.tensor_ndim = tensor_ndim

        # TODO (ykarnati, pthombre) - CP support will be added in follow up PR.
        if 'cp' in self.src_grid.dim_names: #目前不支持cp
            assert self.src_grid.shape[self.src_grid.dim_names.index('cp')] == 1, (
                f"Source grid CP size must be 1, got "
                f"{self.src_grid.shape[self.src_grid.dim_names.index('cp')]}"
            )

        if 'cp' in self.dest_grid.dim_names:
            assert self.dest_grid.shape[self.dest_grid.dim_names.index('cp')] == 1, (
                f"Destination grid CP size must be 1, got "
                f"{self.dest_grid.shape[self.dest_grid.dim_names.index('cp')]}"
            )

        self.current_rank = dist.get_rank() #自身的rank
        self.comm_map: Dict[int, RankCommInfo] = {}
        if dim_mapping is None:
            self.dim_mapping = {'s': 1, 'b': 0, 'h': 2}
        else:
            assert set(dim_mapping.keys()) == {
                's',
                'b',
                'h',
            }, f"dim_mapping must have keys 's', 'b', 'h', got {set(dim_mapping.keys())}"
            assert all(
                v in {0, 1, 2} for v in dim_mapping.values()
            ), f"dim_mapping values must be 0, 1, or 2, got {list(dim_mapping.values())}"
            self.dim_mapping = dim_mapping

        self.src_grid_broadcast_pg = None
        self.dest_grid_broadcast_pg = None

        src_grid_broadcast_ranks_list = self.get_boundary_pp_stage_ranks(self.src_grid, is_src=True) #获取src模块的边界rank集合的列表，注意是列表的列表，每个元素代表一个dp副本里的边界rank集合，一共dp size个列表元素
        dest_grid_broadcast_ranks_list = self.get_boundary_pp_stage_ranks(self.dest_grid, is_src=False) #获取dest模块的边界rank集合的列表，注意是列表的列表，每个元素代表一个dp副本里的边界rank集合，一共dp size个列表元素

        self.src_grid_broadcast_ranks = [] #src模块所有边界rank的broadcast通信pg
        if src_grid_broadcast_ranks_list:
            self.src_grid_broadcast_pg = self._get_or_create_broadcast_pg(
                src_grid_broadcast_ranks_list
            )#获取当前rank所属的边界broadcast通信pg（src模块）
            self.src_grid_broadcast_ranks = next(
                (ranks for ranks in src_grid_broadcast_ranks_list if self.current_rank in ranks), []
            ) #获取当前rank所属的边界broadcast通信组的（src模块），如果当前rank不是src模块的边界rank，则保存空列表

        self.dest_grid_broadcast_ranks = [] #dest模块所有边界rank的broadcast通信pg
        if dest_grid_broadcast_ranks_list:
            self.dest_grid_broadcast_pg = self._get_or_create_broadcast_pg(
                dest_grid_broadcast_ranks_list
            ) #获取当前rank所属的边界broadcast通信pg（dest模块）
            self.dest_grid_broadcast_ranks = next(
                (ranks for ranks in dest_grid_broadcast_ranks_list if self.current_rank in ranks),
                [],
            ) #获取当前rank所属的边界broadcast通信组的（dest模块），如果当前rank不是dest模块的边界rank，则保存空列表

        self.src_tp_leaders, self.src_local_leader_rank = self.get_leader_rank( #获取src模块所有dp副本的leader rank和当前rank所在的dp副本的leader rank
            self.src_grid, is_src=True
        )
        self.dest_tp_leaders, self.dest_local_leader_rank = self.get_leader_rank( #获取dest模块所有dp副本的leader rank和当前rank所在的dp副本的leader rank
            self.dest_grid, is_src=False
        )

        log_msg = (
            f"[Rank {self.current_rank}] "
            f"srcLeader={self.src_local_leader_rank} "
            f"destLeader={self.dest_local_leader_rank} "
            f"srcBroadcastGrpRanks={self.src_grid_broadcast_ranks} "
            f"destBroadcastGrpRanks={self.dest_grid_broadcast_ranks}"
        )
        logging.info(log_msg)

        self.build_comm_map(self.src_tp_leaders, self.dest_tp_leaders) #确定每个rank的角色（MEMBER/SENDER/RECEIVER），还有每个SENDER rank需要向哪些RECEIVER rank发送数据，RECEIVER rank接受哪些SENDER rank发送的数据
        dist.barrier() #当前 rank 执行到这里后必须等待，直到默认 process group 中的所有 rank 都执行到这里，所有 rank 才能继续往下运行。

    @property
    def _batch_dim(self) -> int:
        """Get the tensor dimension used for fan-in/fan-out (cat/split).

        For 3D tensors (e.g. [S, B, H]), this is dim_mapping['b'].
        For 2D tensors (e.g. [B*S, H] where batch is folded into the first
        dimension), this is 0.
        """
        if self.tensor_ndim == 2:
            return 0
        return self.dim_mapping['b']

    @classmethod
    def _get_or_create_broadcast_pg(cls, ranks_list: List[List[int]]):
        """Get or create a broadcast PG, caching to avoid duplicate NCCL communicators.""" #获取或者创建执行broadcast通信的多个pg，按照ranks_list分组构建
        cache_key = str(sorted([tuple(r) for r in ranks_list]))
        if cache_key not in cls._broadcast_pg_cache:
            pg, _ = dist.new_subgroups_by_enumeration(ranks_list, backend='nccl') #创建多个 subgroup，返回当前 rank 所属的 group
            cls._broadcast_pg_cache[cache_key] = pg
        return cls._broadcast_pg_cache[cache_key]

    def get_leader_rank(self, grid: HyperCommGrid, is_src: bool) -> List[int]:
        """Get the leader rank for a given grid and direction. #获取src/dest模块的leader rank，每个dp副本有一个leader rank（leader rank是通信代表）

        We elect leader rank for each dp replica, the first tp-cp rank in the group
        in the last pp stage (for src grid) or first pp stage (for dest grid) is the leader.
        """
        leader_ranks = []
        local_leader_rank = None
        # grid.gen_rank_enum(["tp", "cp", "pp"]) # vary tp & cp, but same dp
        # returns a list of sublists, each sublist is a group of ranks
        # that have different tp & cp & pp, same dp
        per_dp_replica_ranks = grid._gen_rank_enum([x for x in grid.dim_names if x != "dp"]) #通信网格中，每个dp副本的rank集合列表
        if is_src: #如果是src模块
            # Add rank from last pp stage #添加最后一个pp阶段的rank
            ranks = []
            for group in per_dp_replica_ranks: #遍历每一个dp副本的rank集合
                if self.current_rank in group:
                    assert (
                        local_leader_rank is None
                    ), "only one local leader rank is allowed per dp replica"
                    local_leader_rank = group[-1] #当前rank所在dp副本的leader rank
                ranks.append(group[-1]) #收集最后一个rank作为这个dp副本的leader rank，最后一个rank肯定是最好一个pp阶段的rank
            leader_ranks.extend(ranks) #将所有dp副本的leader rank收集到一起，形成一个列表
        else: #如果是dest模块
            # Add rank from first pp stage #添加第一个pp阶段的rank
            ranks = []
            for group in per_dp_replica_ranks: #遍历每一个dp副本的rank集合
                if self.current_rank in group:
                    assert (
                        local_leader_rank is None
                    ), "only one local leader rank is allowed per dp replica"
                    local_leader_rank = group[0] #当前rank所在dp副本的leader rank
                ranks.append(group[0]) #收集第一个rank作为这个dp副本的leader rank，第一个rank肯定是第一个pp阶段的rank
            leader_ranks.extend(ranks) #将所有dp副本的leader rank收集到一起，形成一个列表
        return leader_ranks, local_leader_rank

    def get_boundary_pp_stage_ranks(self, grid: HyperCommGrid, is_src: bool):
        """Get TP-CP ranks at boundary PP stage for each DP replica.

        Returns ranks at the last PP stage (if src) or first PP stage (if dest)
        for each DP dimension, ordered by DP dimension.
        """

        # Get tp-cp rank enumeration (each list has same dp and pp, different tp and cp)
        tpcp_rank_lists = grid._gen_rank_enum(['tp', 'cp']) #获取通信网格中的tp-cp rank组合列表
        pp_size = grid.shape[grid.dim_names.index('pp')] #获取通信网格中的pp维度的大小

        # Determine boundary pp stage
        boundary_pp_stage = pp_size - 1 if is_src else 0 #确定边界pp阶段，如果是src模块，则边界pp阶段是最后一个pp阶段，否则是第一个pp阶段

        boundary_pp_stage_ranks = []

        for rank_list in tpcp_rank_lists: #遍历tp-cp rank组合列表
            # We can check any rank in the list since they all have the same pp coordinate
            if not rank_list:
                continue
            sample_rank = rank_list[0] #随便取一个rank来算pp rank
            # Calculate rank coordinates
            rank_coords = []
            temp_rank = sample_rank - grid.rank_offset

            # Extract coordinates in the original dimension order
            for dim_size in grid.shape: #通过除法余数计算这个rank在各个并行维度上的坐标
                rank_coords.append(temp_rank % dim_size)
                temp_rank //= dim_size

            pp_coord = rank_coords[grid.dim_names.index('pp')] #获取这个rank在pp维度上的坐标

            if pp_coord == boundary_pp_stage: #如果这个rank是边界pp阶段上的rank，则表明这个tp-cp rank集合都在边界pp阶段
                # This rank list is at the boundary pp stage, add all ranks from this list
                boundary_pp_stage_ranks.append(rank_list) #将这个tp-cp rank集合添加到边界pp阶段rank列表中

        return boundary_pp_stage_ranks

    def is_current_rank_in_grid(self, grid: HyperCommGrid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= self.current_rank < (grid.rank_offset + grid.size)

    def build_comm_map(self, src_tp_leaders: List[int], dest_tp_leaders: List[int]):
        """Get src/dest tp leaders and populate comm_map for each rank.

        This method analyzes the source and destination grids to determine
        which ranks need to send/receive data and builds the communication
        schedule accordingly.
        """
        # Ensure that the number of leaders can be evenly divided
        src_count = len(src_tp_leaders) #获取src模块所有dp副本的leader rank数量，本质上就是src模块的dp副本数量，因为每个dp副本一个leader rank
        dest_count = len(dest_tp_leaders) #获取dest模块所有dp副本的leader rank数量，本质上就是dest模块的dp副本数量，因为每个dp副本一个leader rank

        if src_count % dest_count != 0 and dest_count % src_count != 0: #如果src模块和dest模块的dp副本数量必须一方被另一方整除，否则不好分配通信拓扑（目前会报错）
            raise ValueError(
                f"Source TP leaders count ({src_count}) and destination TP leaders count "
                f"({dest_count}) must be evenly divisible. One must be a multiple of the other."
            )
        # Get all ranks in source and destination grids
        src_all_ranks = list(
            range(self.src_grid.rank_offset, self.src_grid.rank_offset + self.src_grid.size)
        ) #获取src模块所有rank的列表
        dest_all_ranks = list(
            range(self.dest_grid.rank_offset, self.dest_grid.rank_offset + self.dest_grid.size)
        ) #获取dest模块所有rank的列表

        all_ranks = src_all_ranks + dest_all_ranks #将src模块和dest模块所有rank的列表拼接在一起

        # Initialize all ranks as MEMBER by default
        for rank in all_ranks: #默认所有rank都是MEMBER角色
            self.comm_map[rank] = RankCommInfo(role=CommRole.MEMBER)

        scale_factor = int(src_count / dest_count) #计算src模块和dest模块的dp副本数量的比值
        if scale_factor > 1: #如果src模块的dp副本数量是dest模块的dp副本数量的倍数，则发生fan-in（缩放入），即一个dest模块的dp副本接受多个src模块的dp副本的激活值
            # Fan-in: multiple source leaders send to fewer destination leaders
            for i, dest_rank in enumerate(dest_tp_leaders): #遍历dest模块的leader rank
                # Each destination rank receives from scale_factor source ranks
                src_ranks = src_tp_leaders[i * scale_factor : (i + 1) * scale_factor] #获取当前dest模块的leader rank对应src模块的多个leader rank，范围是从i*scale_factor到(i+1)*scale_factor

                # Set up senders
                for src_rank in src_ranks: #设置这些src模块的leader rank为SENDER角色，表明它们需要向dest模块的当前leader rank发送激活值
                    self.comm_map[src_rank] = RankCommInfo(
                        role=CommRole.SENDER, send_to_ranks=[dest_rank]
                    )

                # Set up receiver #设置这些dest模块的leader rank为RECEIVER角色，表明它们需要接受这些src模块的leader rank发送的激活值
                self.comm_map[dest_rank] = RankCommInfo(
                    role=CommRole.RECEIVER, recv_from_ranks=src_ranks
                )
        else: #如果src模块的dp副本数量是dest模块的dp副本数量的除数，则发生fan-out（缩放出），即一个src模块的dp副本把激活值切分发给多个dest模块的dp副本
            # Fan-out: fewer source leaders send to more destination leaders
            scale_factor = int(dest_count / src_count) #计算dest模块和src模块的dp副本数量的比值
            for i, src_rank in enumerate(src_tp_leaders): #遍历src模块的leader rank
                # Each source rank sends to scale_factor destination ranks
                dest_ranks = dest_tp_leaders[i * scale_factor : (i + 1) * scale_factor] #获取当前src模块的leader rank对应dest模块的多个leader rank，范围是从i*scale_factor到(i+1)*scale_factor

                # Set up sender
                self.comm_map[src_rank] = RankCommInfo( #设置这个src模块的leader rank为SENDER角色，表明它们需要向dest模块的多个leader rank发送激活值
                    role=CommRole.SENDER, send_to_ranks=dest_ranks
                )

                # Set up receivers
                for dest_rank in dest_ranks: #设置这些dest模块的leader rank为RECEIVER角色，表明它们需要接受这个src模块的leader rank发送的激活值
                    self.comm_map[dest_rank] = RankCommInfo(
                        role=CommRole.RECEIVER, recv_from_ranks=[src_rank]
                    )

    def send_forward(self, tensor_to_send: torch.Tensor):
        """Send forward activation tensor.

        Args:
            tensor_to_send: The tensor to send to the destination grid
        """
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"[Bridge Communicator] [send_forward] Rank {self.current_rank} "
                "is not in the source grid."
            )

        rank_info = self.comm_map.get(self.current_rank) #获取当前rank的通信信息
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.SENDER: #如果当前rank是SENDER角色，则向dest模块的多个leader rank发送激活值
            # Send splits to destination ranks
            num_sends = len(rank_info.send_to_ranks)
            if num_sends > 0:
                tensor_splits = self._split_tensor_at_batch_dim(tensor_to_send, num_sends) #将tensor按照batch维度分割成num_splits个tensor，用于后续的分发
                self._communicate_shapes(tensor_to_send_next=tensor_splits[0]) #先进行shape通信，确保所有rank都能收到正确的tensor形状信息，这里传入的tensor是均分之后的
                for dest_rank, tensor_split in zip(rank_info.send_to_ranks, tensor_splits): #遍历要发送到的rank，需要发送的张量
                    logging.debug(
                        f"[Bridge Comunicator] [send_forward] Rank {self.current_rank} "
                        f"send to rank {dest_rank}"
                    )
                    dist.send(tensor_split, dst=dest_rank) #执行同步P2P通信操作

    def recv_forward(self) -> torch.Tensor:
        """Receive forward activation tensor.

        Args:
            tensor_shape: Expected tensor shape (None if using shape communication)

        Returns:
            torch.Tensor: The received activation tensor
        """
        # receive forward only gets called on the dest grid #只有dest模块的rank才能执行接受激活值的操作
        if not self.is_current_rank_in_grid(self.dest_grid): #如果当前rank不在dest模块，则报错
            raise ValueError(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                "is not in the destination grid."
            )

        rank_info = self.comm_map.get(self.current_rank) #获取当前rank的通信信息
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"
        logging.debug(
            f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
            f"[src - {self.src_module_name}] [dest - {self.dest_module_name}] "
            f"rank_info: {rank_info}"
        )
        if rank_info.role == CommRole.RECEIVER: #如果当前rank是RECEIVER角色，则接受来自src模块的多个leader rank的激活值
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            # p2p call to receive the tensor
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(recv_prev=True) #先进行shape通信，确保所有rank都能收到正确的tensor形状信息
            logging.debug(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )
            received_tensors_list = []
            for src_rank, shape in zip(rank_info.recv_from_ranks, recv_forward_shapes): #遍历所有自身对应的SENDER rank，还有对应的tensor形状
                tensor_to_recv = torch.empty( #创建空张量来接受数据
                    shape,
                    device=torch.cuda.current_device(),
                    dtype=self.comm_dtype,
                    requires_grad=True,
                )
                dist.recv(tensor_to_recv, src=src_rank) #同步P2P通信操作，接受数据
                logging.debug(
                    f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                    f"received tensor from src rank {src_rank} "
                    f"shape {tensor_to_recv.shape} sum {tensor_to_recv.sum()}"
                )
                received_tensors_list.append(tensor_to_recv) #将接受到的张量添加到列表中
            aggregated_tensor = torch.cat(received_tensors_list, dim=self._batch_dim) #将所有接受到的张量按照batch维度拼接起来
            logging.debug(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                f"broadcasting tensor {aggregated_tensor.shape} sum {aggregated_tensor.sum()}"
            )

            # Step 1: broadcast its shape so receivers can allocate
            shape_tensor = torch.tensor( #创建张量记录聚合后的tensor形状，用于后续广播
                aggregated_tensor.shape, device=aggregated_tensor.device, dtype=torch.int64
            )
            dist.broadcast(shape_tensor, src=self.current_rank, group=self.dest_grid_broadcast_pg) #先broadcast让每个参与通信的rank获得接受到的tensor形状信息

            # Step 2: broadcast the actual tensor
            dist.broadcast( #broadcast通信，使得每个参与通信的rank都能获得接受到的tensor数据，通信组为self.dest_grid_broadcast_pg
                aggregated_tensor, src=self.current_rank, group=self.dest_grid_broadcast_pg
            )

            return aggregated_tensor #返回聚合后的张量

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.dest_grid_broadcast_ranks
        ): #如果当前rank是MEMBER角色，而且是dest模块的边界rank，则接受dest模块的leader rank广播的激活值
            # Non-leader rank - participate in broadcast
            shape_tensor = torch.empty( #创建张量记录接受到的tensor形状，用于后续广播
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast( #broadcast通信，使得每个参与通信的rank都能获得接受到的tensor形状信息，通信组为self.dest_grid_broadcast_pg
                shape_tensor, src=self.dest_local_leader_rank, group=self.dest_grid_broadcast_pg
            )

            received_shape = tuple(shape_tensor.tolist())
            received_tensor = torch.empty( #创建空张量来接受数据
                received_shape,
                device=torch.cuda.current_device(),
                dtype=self.comm_dtype,
                requires_grad=True,
            )

            # Receive the full tensor via broadcast
            dist.broadcast( #broadcast通信，使得每个参与通信的rank都能获得接受到的tensor数据，通信组为self.dest_grid_broadcast_pg
                received_tensor, src=self.dest_local_leader_rank, group=self.dest_grid_broadcast_pg
            )

            logging.debug(
                f"[Bridge Communicator] [receive_forward] Rank {self.current_rank} "
                f"received tensor via broadcast, shape {received_tensor.shape}"
            )
            return received_tensor #返回接受到的张量

    def send_backward(self, grad_tensor: torch.Tensor):
        """Send backward gradient tensor.

        Note: Gradient senders are activation 'RECEIVERS'

        Args:
            grad_tensor: The gradient tensor to send back
        """
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"[Bridge Communicator] [send_backward] Rank {self.current_rank} "
                "is not in the destination grid."
            )

        rank_info = self.comm_map.get(self.current_rank) #获取当前rank的通信信息
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.RECEIVER: #如果当前rank是RECEIVER角色，因为梯度传播只有RECEIVER会发起
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            # Send gradients back to source ranks
            num_receives = len(rank_info.recv_from_ranks) #获取接受的rank数量
            tensor_splits = self._split_tensor_at_batch_dim(grad_tensor, num_receives) #将梯度张量按照batch维度均匀分割成多个子张量
            self._communicate_shapes(tensor_to_send_prev=tensor_splits[0]) #先进行shape通信
            if num_receives > 0:
                for src_rank, tensor_split in zip(rank_info.recv_from_ranks, tensor_splits): #遍历所有接受的rank和对应的子张量
                    # Send the gradient split back to the source rank
                    logging.debug(
                        f"[Bridge Communicator] [send_backward] Rank {self.current_rank} "
                        f"sending gradient to src rank {src_rank} "
                        f"shape {tensor_split.shape} sum {tensor_split.sum()}"
                    )
                    dist.send(tensor_split, dst=src_rank) #同步P2P通信，将子张量发送回对应的rank

    def recv_backward(self) -> torch.Tensor:
        """Receive backward gradient tensor.

        Note: Gradient receivers are activation 'SENDERS'

        Args:
            tensor_shape: Expected gradient tensor shape

        Returns:
            torch.Tensor: The received gradient tensor
        """
        # receive backward only gets called on the src grid
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                "is not in the source grid."
            )

        rank_info = self.comm_map.get(self.current_rank) #获取当前rank的通信信息
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.SENDER: #如果是SENDER，那就进行接受，然后broadcast给同模块的边界rank
            assert (
                self.current_rank == self.src_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(recv_next=True) #获取接受的梯度张量形状
            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )
            # Receive gradient tensors from destination ranks
            received_gradients_list = []
            for dest_rank, grad_shape in zip(rank_info.send_to_ranks, recv_grad_shapes): #遍历所有发起者rank和对应的梯度张量形状
                # The destination rank that we sent to will send us gradients back
                grad_tensor = torch.empty( #创建空张量来接受梯度数据
                    grad_shape, device=torch.cuda.current_device(), dtype=self.comm_dtype
                )
                dist.recv(grad_tensor, src=dest_rank) #同步P2P通信，接受到发起者rank发送的梯度张量
                logging.debug(
                    f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                    f"received gradient from dest rank {dest_rank} "
                    f"shape {grad_tensor.shape} sum {grad_tensor.sum()}"
                )
                received_gradients_list.append(grad_tensor) #将接受到的梯度张量添加到列表中

            # Concatenate received gradients
            aggregated_gradient = torch.cat(received_gradients_list, dim=self._batch_dim) #将接受到的梯度张量按照batch维度拼接成一个完整的张量
            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"agg grad shape {aggregated_gradient.shape} sum {aggregated_gradient.sum()}"
            )

            shape_tensor = torch.tensor( #创建张量来接受形状信息
                aggregated_gradient.shape, device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast(shape_tensor, src=self.current_rank, group=self.src_grid_broadcast_pg) #广播形状信息

            # Scatter the tensors to all ranks in the group
            dist.broadcast( #广播完整的梯度张量
                aggregated_gradient, src=self.current_rank, group=self.src_grid_broadcast_pg
            )
            return aggregated_gradient #返回完整的梯度张量

        elif (
            rank_info.role == CommRole.MEMBER and self.current_rank in self.src_grid_broadcast_ranks
        ):#如果是MEMBER角色，而且是src模块的边界rank，那么它也需要接受并广播完整的梯度张量
            # Non-leader rank - participate in gather for gradients
            # Receive broadcasted tensor shape from leader rank
            shape_tensor = torch.empty( #创建张量来接受形状信息
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast( #广播形状信息
                shape_tensor, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )

            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"received shape tensor {shape_tensor}"
            )
            received_shape = tuple(shape_tensor.tolist())
            received_gradient = torch.empty( #创建空张量来接受梯度数据
                received_shape, device=torch.cuda.current_device(), dtype=self.comm_dtype
            )

            dist.broadcast( #广播完整的梯度张量
                received_gradient, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )
            logging.debug(
                f"[Bridge Communicator] [receive_backward] Rank {self.current_rank} "
                f"received gradient from scatter operation, shape {received_gradient.shape}"
            )
            return received_gradient #返回完整的梯度张量

    def send_forward_recv_backward(
        self, input_tensor: torch.Tensor, grad_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Combined operation: send forward activation and receive backward gradient. #发送前传激活值+接受反传梯度

        Args:
            input_tensor: The tensor to send forward
            grad_shape: Expected gradient tensor shape

        Returns:
            torch.Tensor: The received gradient tensor
        """
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the source grid. "
                "send_forward_recv_backward is only allowed on src grid"
            )

        rank_info = self.comm_map.get(self.current_rank) #获取当前rank的通信信息
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"
        logging.debug(
            f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
            f"[src - {self.src_module_name}] [dest - {self.dest_module_name}] "
            f"rank_info: {rank_info}"
        )
        if rank_info.role == CommRole.SENDER: #如果是SENDER，那就进行接受，然后broadcast给同模块的边界rank
            assert (
                self.current_rank == self.src_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"

            num_sends = len(rank_info.send_to_ranks) #获取接受者数量
            activation_splits = self._split_tensor_at_batch_dim(input_tensor, num_sends) #将激活值张量按照batch维度分割成num_sends份
            # Communicate shapes for both directions (send forward, receive backward)
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes( #通信形状信息，既包括前传，也包括后传
                tensor_to_send_next=activation_splits[0], recv_next=True
            )
            logging.debug(
                f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )

            # Prepare simultaneous send/receive operations
            if num_sends > 0:
                # Prepare gradient receive tensors
                received_gradients_list = []
                for i, recv_grad_shape in enumerate(recv_grad_shapes): #创建空张量来接受梯度数据
                    grad_tensor = torch.empty(
                        recv_grad_shape, device=torch.cuda.current_device(), dtype=self.comm_dtype
                    )
                    received_gradients_list.append(grad_tensor) #将空张量添加到列表中

                # Create batch P2P operations for simultaneous send/receive
                ops = []
                for dest_rank, activation_split, grad_tensor in zip( #遍历每个接受者
                    rank_info.send_to_ranks, activation_splits, received_gradients_list
                ):
                    # Send activation
                    ops.append( #添加P2P操作到ops列表中
                        torch.distributed.P2POp(
                            torch.distributed.isend, activation_split, dest_rank
                        )
                    )
                    # Receive gradient
                    ops.append( #添加P2P操作到ops列表中
                        torch.distributed.P2POp(torch.distributed.irecv, grad_tensor, dest_rank)
                    )

                logging.debug(
                    f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                    f"executing {len(ops)} simultaneous P2P operations"
                )
                reqs = torch.distributed.batch_isend_irecv(ops) #执行ops列表中的P2P操作
                for req in reqs: #等待所有P2P操作完成
                    req.wait()

                # Concatenate received gradients
                aggregated_gradient = torch.cat(received_gradients_list, dim=self._batch_dim) #将接受到的梯度张量按照batch维度拼接起来
                logging.debug(
                    f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                    f"agg grad shape {aggregated_gradient.shape} sum {aggregated_gradient.sum()}"
                )
                # Broadcast tensor shape to all ranks in scatter_pg
                tensor_shape_to_broadcast = aggregated_gradient.shape #需要广播的张量形状
                shape_tensor = torch.tensor(#创建张量来接受形状信息
                    tensor_shape_to_broadcast, device=torch.cuda.current_device(), dtype=torch.int64
                )
                dist.broadcast( #广播张量形状
                    shape_tensor, src=self.current_rank, group=self.src_grid_broadcast_pg
                )

                # Broadcast the tensors to all ranks in the group
                dist.broadcast( #广播张量
                    aggregated_gradient, src=self.current_rank, group=self.src_grid_broadcast_pg
                )

                return aggregated_gradient #返回完整的梯度张量

        elif (
            rank_info.role == CommRole.MEMBER and self.current_rank in self.src_grid_broadcast_ranks
        ): #如果是MEMBER，而且当前rank是src模块的边界rank，那么它需要broadcast来接受完整的梯度张量
            # participate in both gather for gradients
            # Receive gradient from leader using broadcast
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            ) #创建张量来接受形状信息
            dist.broadcast( #广播张量形状
                shape_tensor, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )

            # Use the received shape to create tensor for broadcast
            received_shape = tuple(shape_tensor.tolist())
            received_gradient = torch.empty( #创建张量来接受广播的张量
                received_shape, device=torch.cuda.current_device(), dtype=self.comm_dtype
            )
            dist.broadcast( #广播张量
                received_gradient, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )
            logging.debug(
                f"[Bridge Communicator] [send_forward_recv_backward] Rank {self.current_rank} "
                f"received gradient from broadcast, shape {received_gradient.shape}"
            )
            return received_gradient

    def send_backward_recv_forward(
        self, grad_tensor: torch.Tensor, forward_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Combined operation: send backward gradient and receive forward activation. #发送反向梯度和接受前向激活值张量

        Args:
            grad_tensor: The gradient tensor to send backward
            forward_shape: Expected forward tensor shape

        Returns:
            torch.Tensor: The received activation tensor
        """
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the destination grid. "
                "send_backward_recv_forward is only allowed on dest grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == CommRole.RECEIVER: #如果是RECEIVER角色
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"

            num_receives = len(rank_info.recv_from_ranks)
            gradient_splits = self._split_tensor_at_batch_dim(grad_tensor, num_receives) #将梯度张量按照batch维度分割成多个子张量
            # Communicate shapes for both directions (send backward, receive forward)
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(
                tensor_to_send_prev=gradient_splits[0], recv_prev=True
            )#获取前向和后向张量的形状信息
            logging.debug(
                f"[Bridge Communicator] [send_backward_recv_backward] Rank {self.current_rank} "
                f"received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )

            # Prepare simultaneous send/receive operations
            if num_receives > 0:
                # Prepare activation receive tensors
                received_activations_list = []
                for i, recv_forward_shape in enumerate(recv_forward_shapes): #对每个前向张量形状，创建一个空张量来接受接受到的前向张量
                    activation_tensor = torch.empty(
                        recv_forward_shape,
                        device=torch.cuda.current_device(),
                        dtype=self.comm_dtype,
                        requires_grad=True,
                    )
                    received_activations_list.append(activation_tensor) #将空张量添加到列表中

                # Create batch P2P operations for simultaneous send/receive
                ops = []
                for src_rank, gradient_split, activation_tensor in zip( #遍历自身对应的src rank
                    rank_info.recv_from_ranks, gradient_splits, received_activations_list
                ):
                    # Send gradient
                    ops.append( #添加反向梯度张量的异步发送操作
                        torch.distributed.P2POp(torch.distributed.isend, gradient_split, src_rank)
                    )

                    # Receive activation
                    ops.append( #添加前向张量张量的异步接收操作
                        torch.distributed.P2POp(
                            torch.distributed.irecv, activation_tensor, src_rank
                        )
                    )

                # Execute all operations simultaneously
                logging.debug(
                    f"[Bridge Communicator] [send_backward_recv_backward] Rank {self.current_rank} "
                    f"executing {len(ops)} simultaneous P2P operations"
                )
                reqs = torch.distributed.batch_isend_irecv(ops) #执行所有通信操作
                for req in reqs: #等待所有通信操作完成
                    req.wait()

                # Concatenate received activations
                aggregated_activation = torch.cat(received_activations_list, dim=self._batch_dim) #将接收到的前向张量张量按照batch维度拼接成一个完整的前向张量张量
                logging.debug(
                    f"[Bridge Communicator] [send_backward_recv_forward] Rank {self.current_rank} "
                    f"agg act shape {aggregated_activation.shape} sum {aggregated_activation.sum()}"
                )

                # Broadcast tensor shape to all ranks in scatter_pg
                tensor_shape_to_scatter = aggregated_activation.shape
                shape_tensor = torch.tensor(
                    tensor_shape_to_scatter, device=torch.cuda.current_device(), dtype=torch.int64
                )
                dist.broadcast(
                    shape_tensor, src=self.current_rank, group=self.dest_grid_broadcast_pg
                )

                # Scatter the tensors to all ranks in the group
                dist.broadcast(
                    aggregated_activation, src=self.current_rank, group=self.dest_grid_broadcast_pg
                )
                return aggregated_activation

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.dest_grid_broadcast_ranks
        ):
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast(
                shape_tensor, src=self.dest_local_leader_rank, group=self.dest_grid_broadcast_pg
            )

            # Use the received shape to create tensor for scatter operation
            received_shape = tuple(shape_tensor.tolist())
            received_activation = torch.empty(
                received_shape,
                device=torch.cuda.current_device(),
                dtype=self.comm_dtype,
                requires_grad=True,
            )
            dist.broadcast(
                received_activation,
                src=self.dest_local_leader_rank,
                group=self.dest_grid_broadcast_pg,
            )
            logging.debug(
                f"[Bridge Communicator] [send_backward_recv_backward] Rank {self.current_rank}  "
                f"received activation from scatter operation, shape {received_activation.shape}"
            )
            return received_activation

    def _communicate_shapes(
        self,
        tensor_to_send_next: Optional[torch.Tensor] = None,
        recv_next: bool = False,
        recv_prev: bool = False,
        tensor_to_send_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
        """Communicate tensor shapes between sender and receiver ranks in the bridge.

        This is used to communicate tensor shapes before actual tensor communication
        when dealing with variable sequence lengths or dynamic shapes.

        Args:
            tensor_to_send_next: The tensor to send to the next rank (None if not sending)
            tensor_to_send_prev: The tensor to send to the previous rank (None if not sending)
            recv_next: Whether to receive from the next rank (None if not receiving)
            recv_prev: Whether to receive from the previous rank (None if not receiving)

        Returns:
            Tuple containing: #返回 List[Tuple[int, ...]]，就是因为当前 rank 可能从多个对端 rank 接收 tensor，因此会收到多个 shape。
            - List of forward shapes that will be received (empty if not a receiver)
            - List of gradient shapes that will be received (empty if not expecting gradients)
        """
        rank_info = self.comm_map.get(self.current_rank) #获取当前 rank 的角色信息
        if not rank_info or rank_info.role == CommRole.MEMBER: #如果是MEMBER，不进行跨模块通信，直接返回
            return [], []

        recv_forward_shapes = []
        recv_grad_shapes = []
        logging.debug(
            f"[Bridge Communicator] [communicate_shapes] Rank {self.current_rank} "
            f"is a {rank_info.role} and is running the shape communication"
        )
        # Collect all P2P operations for batch execution
        ops = []
        recv_forward_shape_tensors = []
        recv_grad_shape_tensors = []

        if rank_info.role == CommRole.SENDER:
            # Prepare send operations for forward shapes
            if tensor_to_send_next is not None:
                send_shape = tensor_to_send_next.shape #获取要发送的 tensor 的 shape，这里的tensor传入这个函数之前已经是被均匀切分的（对于fan-out情况）
                send_shape_tensor = torch.tensor(#转换为tensor
                    send_shape, device=torch.cuda.current_device(), dtype=torch.int64
                )
                # Add send operations for each destination
                for dest_rank in rank_info.send_to_ranks: #对每一个dest_rank，都进行一次isend操作
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, send_shape_tensor, dest_rank
                        )
                    )

            # If expecting gradients back, prepare receive operations
            if recv_next: #如果需要从dest_rank接收梯度 tensor，那么就对每一个dest_rank都进行一次irecv操作
                for dest_rank in rank_info.send_to_ranks: #遍历每一个dest_rank，都进行一次irecv操作
                    grad_shape_tensor = torch.empty( #创建一个空的tensor来接收梯度 shape
                        (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
                    )
                    recv_grad_shape_tensors.append(grad_shape_tensor) #将空的tensor添加到列表中，用于后续提取接收到的shape
                    ops.append( #将irecv操作添加到ops中
                        torch.distributed.P2POp(
                            torch.distributed.irecv, grad_shape_tensor, dest_rank
                        )
                    )

        elif rank_info.role == CommRole.RECEIVER: #对于RECEIVER
            # Prepare receive operations for forward shapes
            if recv_prev: #如果需要从prev_rank接收tensor，那么就对每一个prev_rank都进行一次irecv操作
                for src_rank in rank_info.recv_from_ranks:
                    forward_shape_tensor = torch.empty( #创建一个空的tensor来接收前向 shape
                        (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
                    )
                    recv_forward_shape_tensors.append(forward_shape_tensor) #将空的tensor添加到列表中，用于后续提取接收到的shape
                    ops.append( #将irecv操作添加到ops中
                        torch.distributed.P2POp(
                            torch.distributed.irecv, forward_shape_tensor, src_rank
                        )
                    )

            # If we need to send gradient shapes back, prepare send operations
            if tensor_to_send_prev is not None: #如果需要向prev_rank发送tensor，那么就对每一个prev_rank都进行一次isend操作
                grad_shape = tensor_to_send_prev.shape #获取要发送的 tensor 的 shape，同样的，这里的tensor在传入这个函数之前已经被均匀切分（对于fan-out情况）
                grad_shape_tensor = torch.tensor(#转换为tensor
                    grad_shape, device=torch.cuda.current_device(), dtype=torch.int64
                )

                for src_rank in rank_info.recv_from_ranks: #遍历每一个src_rank，都进行一次isend操作
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, grad_shape_tensor, src_rank
                        )
                    )

        # Execute all operations in a single batch
        if ops: #如果ops不为空，就批量执行P2P通信操作，同步等待
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Extract shapes from received tensors
        for forward_shape_tensor in recv_forward_shape_tensors:
            shape = forward_shape_tensor.tolist()
            recv_forward_shapes.append(tuple(shape)) #把 GPU tensor 转成 Python list，再转为tuple，添加到列表中

        for grad_shape_tensor in recv_grad_shape_tensors:
            shape = grad_shape_tensor.tolist()
            recv_grad_shapes.append(tuple(shape)) #把 GPU tensor 转成 Python list，再转为tuple，添加到列表中

        return recv_forward_shapes, recv_grad_shapes #返回接收到的前向shape和梯度shape

    def _split_tensor_at_batch_dim(
        self, aggregated_tensor: torch.Tensor, num_splits: int
    ) -> List[torch.Tensor]:
        """Split an aggregated tensor into multiple tensors at the batch dimension. #将tensor按照batch维度分割成num_splits个tensor，用于后续的分发

        Args:
            aggregated_tensor: The tensor to split
            num_splits: The number of splits to create

        Returns:
            List of tensors split at the batch dimension
        """
        if num_splits <= 0:
            raise ValueError(f"num_splits must be positive, got {num_splits}")

        splits = torch.tensor_split(aggregated_tensor, num_splits, dim=self._batch_dim) #按照batch维度分割tensor
        # PyTorch p2p requires the tensors to be contiguous #保证张量是连续的（P2P通信需要）
        return [split.contiguous() for split in splits] #返回张量列表
