# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

# Types
Shape = Union[List[int], torch.Size]


@dataclass
class RankModuleInfo:
    """Information about a rank in a module. #记录当前rank在某个模块中的信息

    Attributes:
        pp_rank: The stage index of the current rank within the module's pipeline.
        pp_size: The total number of pipeline stages (ranks) in the module.
        p2p_communicator: Intra-module point-to-point communicator.
        bridge_comms_as_src_module: Bridge communicators for outgoing connections
            from this module to downstream modules. One module may have multiple
            bridge communicators if it has multiple outgoing connections.
        bridge_comms_as_dest_module: Bridge communicators for incoming connections
            to this module from upstream modules. One module may have multiple
            bridge communicators if it has multiple incoming connections.
        is_source_stage: True if this rank is at the absolute first stage in the
            overall model (no incoming connections).
        is_terminal_stage: True if this rank is at the absolute last stage in the
            overall model (no outgoing connections).
    """

    pp_rank: int
    pp_size: int
    p2p_communicator: Optional[P2PCommunicator] #该模块内的p2p communicator
    bridge_comms_as_src_module: Optional[List[BridgeCommunicator]] #该模块内作为src边界通信rank的bridge communicators
    bridge_comms_as_dest_module: Optional[List[BridgeCommunicator]] #该模块内作为dest边界通信rank的bridge communicators
    is_source_stage: Optional[bool] = True
    is_terminal_stage: Optional[bool] = True


def _prepare_tensor_for_comm(
    tensor: Union[torch.Tensor, List[torch.Tensor], None]
) -> Union[torch.Tensor, List[torch.Tensor], None]:
    """Prepare tensor for P2P communication by expanding to 3D if needed.

    Only used for intra-module P2P paths. Bridge communicators handle 2D/3D
    tensors natively via tensor_ndim and do not need this adapter.

    P2P communicators expect 3D tensors. 2D tensors are unsqueezed by adding
    a singleton last dimension, and _restore_tensor_from_comm will squeeze it back. 3D
    tensors are passed through unchanged.

    Note: 3D tensors with a singleton last dimension (shape [a, b, 1]) are not supported
    because _restore_tensor_from_comm cannot distinguish them from unsqueezed 2D tensors.

    Args:
        tensor: Input tensor (2D or 3D), list of tensors, or None.

    Returns:
        3D tensor (with singleton last dim if input was 2D), list of 3D tensors, or None.
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [_prepare_tensor_for_comm(t) for t in tensor]
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 2:
            return tensor.unsqueeze(-1)
        assert tensor.ndim != 3 or tensor.shape[-1] != 1, (
            f"3D tensor with singleton last dim {tuple(tensor.shape)} is ambiguous for "
            "multimodule comm. Cannot distinguish from an unsqueezed 2D tensor on the "
            "receiving rank. Use a 2D tensor or a 3D tensor with last_dim > 1."
        )
    return tensor


def _restore_tensor_from_comm(
    tensor: Union[torch.Tensor, List[torch.Tensor], None]
) -> Union[torch.Tensor, List[torch.Tensor], None]:
    """Restore tensor shape after P2P communication by squeezing singleton dim. #_prepare_tensor_for_comm / _restore_tensor_from_comm 是模块内 P2P 路径的 2D↔3D 形状适配器：发送前 unsqueeze 补成 3D 以满足 P2P 的 3D 约定，接收后 squeeze 还原；并显式禁止真正的 3D 单例末维张量以避免歧义。

    Only used for intra-module P2P paths. Bridge communicators handle 2D/3D
    tensors natively via tensor_ndim and do not need this adapter.

    Removes the extra dimension added by _prepare_tensor_for_comm if it was singleton.
    Handles both single tensors and lists of tensors (for VPP).

    Args:
        tensor: Input tensor (3D with singleton last dim), list of tensors, or None.

    Returns:
        2D tensor (if last dim was singleton), list of tensors, or None.
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [_restore_tensor_from_comm(t) for t in tensor]
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 3 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


class MultiModulePipelineCommunicator:
    """Communicator for a multi-module pipeline.""" #这里的假设是每个rank负责多个模块，但是每个rank会做完所有模块的一个micro batch的前传才会统一进行每个模块需要做的通信（也就是每个 rank 每步把自己拥有的所有模块前向（和反向）整体算完再统一收发）

    def __init__(
        self,
        module_to_grid_map: Dict[str, HyperCommGrid],
        topology: Dict[str, List[str]],
        config: ModelParallelConfig,
        dim_mapping: Dict[str, List[int]] = None,
        module_output_ndim: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the MultiModulePipelineCommunicator.

        Args:
            module_to_grid_map (dict): A dictionary mapping module names to HyperCommGrids. #模块名到该模块专属 HyperCommGrid（通信网格） 的映射
                Example:
                    module_to_grid_map = {
                        'image_encoder': image_encoder_grid,
                        'audio_encoder': audio_encoder_grid,
                        'llm': llm_grid,
                        'generator': generator_grid
                    }
            topology (dict): A dictionary mapping module names to lists of outgoing modules. #topology 描述模块间的数据流，代表每个模块输出给哪个模块
                Example:
                    topology = {
                        'image_encoder': ['llm'],
                        'audio_encoder': ['llm'],
                        'llm': ['generator'],
                        'generator': []
                    }
            config (ModelParallelConfig): A ModelParallelConfig object.
            dim_mapping (Dict[str, List[int]]): Dimension mapping for sequence, batch, hidden.
                Example:
                    dim_mapping = {'s': 0, 'h': 2, 'b': 1}
                Default: None
            module_output_ndim (Dict[str, int]): Number of dimensions for each module's #描述每个模块输出给下游模块的 Tensor 有几个维度
                output tensor. Used by bridge communicators for cross-module fan-in/fan-out.
                Modules producing 2D tensors [B*S, H] (e.g. vision encoders) should be 2.
                Modules not listed default to 3.
                Example:
                    module_output_ndim = {'image_encoder': 2, 'llm': 3}
                Default: None (all modules assumed 3D)
        """
        self.module_to_grid_map = module_to_grid_map
        self.topology = topology
        self.config = config
        self.dim_mapping = dim_mapping
        self.module_output_ndim = module_output_ndim or {}
        self.current_rank = dist.get_rank() #自身的rank

        # Build bridge communicators for all modules
        self.bridge_comms = []
        self._build_bridge_comms() #构建所有跨模块的P2P通信器对象 BridgeCommunicator

        self.rank_module_map = {} #当前rank和所在模块作为key，映射到对应的RankModuleInfo对象
        self._build_rank_module_info_map()

    def _build_bridge_comms(self):
        """Construct and store BridgeCommunicator objects that describe the outgoing
        communication relationships for all of the modules.
        """
        for src_module_name, src_grid in self.module_to_grid_map.items(): #遍历每个模块，模块名称+模块的通信网格·
            for dest_module_name in self.topology[src_module_name]: #遍历当前模块的下游模块
                dest_grid = self.module_to_grid_map[dest_module_name] #下游模块的通信网格
                bridge_comm = BridgeCommunicator( #构建跨模块的P2P通信器对象BridgeCommunicator
                    src_grid=src_grid,
                    dest_grid=dest_grid,
                    dim_mapping=self.dim_mapping,
                    comm_dtype=self.config.pipeline_dtype,
                    src_module_name=src_module_name,
                    dest_module_name=dest_module_name,
                    tensor_ndim=self.module_output_ndim.get(src_module_name, 3),
                )
                self.bridge_comms.append(bridge_comm) #将构建好的 BridgeCommunicator 添加到列表中

    @property
    def is_pp_first_stage(self):
        """Return True if the current rank has the absolute first stage in the overall model.

        The absolute first stage is defined as:
        1. The current rank must be in the first PP stage (pp_rank == 0) of some module
        2. That module must be a source module (no incoming connections in topology)
        """
        for module_name, rank_module_info in self.rank_module_map.items():
            # Check if this rank is at the first PP stage of this module
            if rank_module_info.pp_rank == 0:
                # Check if this module is a source module (no incoming connections)
                if self._is_source_module(module_name):
                    return True
        return False

    @property
    def is_pp_last_stage(self):
        """Return True if the current rank has the absolute last stage in the overall model.

        The absolute last stage is defined as:
        1. The current rank must be in the last PP stage of some module
        2. That module must be a sink module (no outgoing connections in topology)
        """
        for module_name, rank_module_info in self.rank_module_map.items():
            # Check if this rank is at the last PP stage of this module
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1:
                # Check if this module is a sink module (no outgoing connections)
                if self._is_sink_module(module_name):
                    return True
        return False

    def _is_source_module(self, module_name: str) -> bool:
        """Check if a module is a source module (has no incoming connections)."""
        # A module is a source if no other module lists it as a destination
        for src_module, dest_modules in self.topology.items():
            if module_name in dest_modules:
                return False
        return True

    def _is_sink_module(self, module_name: str) -> bool:
        """Check if a module is a sink module (has no outgoing connections)."""
        return len(self.topology.get(module_name, [])) == 0

    def is_current_rank_in_grid(self, grid: HyperCommGrid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= self.current_rank < grid.rank_offset + grid.size

    @property
    def total_stages(self) -> int:
        """Return total number of pipeline stages across all modules.

        Computes the longest path through the module DAG weighted by each
        module's pipeline-parallel size.

        Returns:
            int: Total pipeline stages.
        """
        return self.compute_total_pipeline_stages(self.topology, self.module_to_grid_map)

    @property
    def current_stage(self) -> int:
        """Return current pipeline stage index (0-indexed) within the multi-module pipeline.

        Returns:
            int: Current stage index.
        """
        total = self.total_stages

        if self.rank_module_map:
            # Take the first module this rank belongs to
            # TODO: ykarnati - improve this logic.
            module_name = next(iter(self.rank_module_map.keys()))
            stage = (
                self.compute_total_pipeline_stages(
                    self.topology,
                    self.module_to_grid_map,
                    rank=self.current_rank,
                    module_name=module_name,
                )
                - 1
            )  # Convert from 1-indexed to 0-indexed
        else:
            stage = 0

        assert stage < total, f"current_stage: {stage} must be less than total_stages: {total}"
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"current_stage: {stage} total_stages: {total} "
            f"num_warmup_microbatches: {total - stage - 1}"
        )
        return stage

    def _build_rank_module_info_map(self):
        """For each module in the current rank, initialize the P2P communicator #每个rank可能负责多个模块，在不同的模块里扮演不同的角色
        and build the bridge communicator info for the module.
        Each rank may hold multiple modules when colocated.
        """
        for module_name, module_grid in self.module_to_grid_map.items(): #遍历每个模块及其对应的HyperCommGrid对象
            if self.is_current_rank_in_grid(module_grid):
                # Initialize P2P communicator
                pp_group = module_grid.get_pg('pp')
                p2p_comm = P2PCommunicator(pp_group, self.config) #创建模块内部的P2PCommunicator对象
                pp_size = dist.get_world_size(pp_group)
                rank_in_pp_group = dist.get_group_rank(pp_group, self.current_rank) #获取当前rank在该模块内部P2P通信组中的相对rank
                pp_rank = rank_in_pp_group % pp_size #计算当前 rank 在该模块 PP 流水线中的位置 pp_rank

                bridge_comms_as_dest_module = []
                bridge_comms_as_src_module = []
                # If first stage, check if the module has any incoming modules
                # If so, initialize bridge communicator
                if pp_rank == 0: #如果是该模块的首stage rank
                    for bridge_comm in self.bridge_comms: #遍历每个BridgeCommunicator对象
                        if (
                            bridge_comm.is_current_rank_in_grid(bridge_comm.dest_grid) #这个BridgeCommunicator的dest模块里面有没有当前rank
                            and bridge_comm.dest_module_name == module_name #判断该模块是否是这个BridgeCommunicator的dest模块
                        ):
                            bridge_comms_as_dest_module.append(bridge_comm) #将这个BridgeCommunicator添加到当前rank在该模块的的as_dest列表中
                # If last stage, check if the module has any outgoing modules
                # If so, initialize bridge communicator
                if pp_rank == pp_size - 1: #如果是该模块的尾stage rank
                    for bridge_comm in self.bridge_comms: #遍历每个BridgeCommunicator对象
                        if (
                            bridge_comm.is_current_rank_in_grid(bridge_comm.src_grid) #这个BridgeCommunicator的src模块里面有没有当前rank
                            and bridge_comm.src_module_name == module_name #判断该模块是否是这个BridgeCommunicator的src模块
                        ):
                            bridge_comms_as_src_module.append(bridge_comm) #将这个BridgeCommunicator添加到当前rank在该模块的as_src列表中
                # Build RankModuleInfo for the module
                rank_module_info = RankModuleInfo( #构建该rank在该模块的信息
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    p2p_communicator=p2p_comm,
                    bridge_comms_as_dest_module=bridge_comms_as_dest_module,
                    bridge_comms_as_src_module=bridge_comms_as_src_module,
                )
                self.rank_module_map[module_name] = rank_module_info #按照模块名存储该rank在该模块的信息

    def recv_forward(
        self, tensor_shape: Optional[Shape] = None, is_first_stage: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Receive forward activation tensor.

        Args:
            tensor_shape: Expected activation tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[receive_forward] tensors_shape: {tensor_shape}, is_first_stage: {is_first_stage}"
        )
        input_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items(): #遍历当前rank负责的各个模块

            if rank_module_info.pp_rank == 0: #如果是该模块的首stage rank
                # If first stage, and has incoming modules, receive forward activation
                # from incoming modules.
                for bridge_comm in rank_module_info.bridge_comms_as_dest_module: #在这个模块中rank作为dest rank，遍历作为dest rank需要参与的BridgeCommunicator
                    received_tensor = bridge_comm.recv_forward() #调用BridgeCommunicator的recv_forward方法接收前传激活值数据
                    input_dict[bridge_comm.src_module_name] = received_tensor #将接收到的前传激活值数据存储到字典中，key是发送这个激活值数据的模块名
            else: #如果是该模块的非首stage rank，那只需要进行模块内的通信
                # If not first stage, receive forward activation tensor from P2P communicator.
                # P2P hardcodes 3D shape buffers, so use adapter for 2D tensors.
                received_tensor = rank_module_info.p2p_communicator.recv_forward(
                    tensor_shapes=tensor_shape, is_first_stage=False
                )#调用P2PCommunicator的recv_forward方法接收前传激活值数据
                input_dict[module_name] = _restore_tensor_from_comm(received_tensor) #将数据进行还原并存储到字典中，key是当前模块名（因为是模块内部通信）
        return input_dict #返回一个字典，key是生产者模块名，value是该rank从生产者模块接收的前传激活值数据

    def send_forward(self, output_dict: Dict[str, torch.Tensor], is_last_stage: bool = False):
        """Send forward activation tensor.

        Args:
            output_dict: A dictionary mapping module names to tensors.
        """
        for module_name, rank_module_info in self.rank_module_map.items(): #遍历当前rank负责的各个模块
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1: #如果是该模块的尾stage rank
                # If last stage, and has outgoing modules, send forward activation
                # by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_src_module: #在这个模块中rank作为src rank，遍历作为src rank需要参与的BridgeCommunicator
                    bridge_comm.send_forward(output_dict[module_name]) #调用BridgeCommunicator的send_forward方法发送前传激活值数据
            else: #如果是该模块的非尾stage rank，那只需要进行模块内的通信
                # If not last stage, send forward activation by using P2P communicator.
                tensor_to_send = _prepare_tensor_for_comm(output_dict[module_name]) #将数据适配到P2P通信可以接受的形状
                rank_module_info.p2p_communicator.send_forward(tensor_to_send, is_last_stage=False) #调用P2PCommunicator的send_forward方法发送前传激活值数据

    def send_forward_recv_backward(
        self,
        output_dict: Dict[str, torch.Tensor],
        tensor_shape: Optional[Shape] = None,
        is_last_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Send forward activation tensor and receive backward activation tensor. #send前传激活值+recv反传梯度

        Args:
            output_dict: A dictionary mapping module names to tensors.
            tensor_shape: Expected gradient tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        grad_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items(): #遍历当前rank负责的各个模块
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1: #如果是该模块的尾stage rank
                # If last stage, and has outgoing modules, send forward activation and
                # receive backward gradient by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_src_module: #在这个模块中rank作为src rank，遍历作为src rank需要参与的BridgeCommunicator
                    grad = bridge_comm.send_forward_recv_backward(output_dict[module_name]) #调用BridgeCommunicator的send_forward_recv_backward方法发送前传激活值数据+接收后传梯度数据
                    grad_dict[bridge_comm.src_module_name] = grad #将接收的后传梯度数据存储到字典中，key是接收这个梯度数据的模块名
            else: #如果是该模块的非尾stage rank，那只需要进行模块内的通信
                # If not last stage, send forward activation and receive backward gradient
                # by using P2P communicator.
                tensor_to_send = _prepare_tensor_for_comm(output_dict[module_name]) #将数据适配到P2P通信可以接受的形状
                grad = rank_module_info.p2p_communicator.send_forward_recv_backward( #调用P2PCommunicator的send_forward_recv_backward方法发送前传激活值数据+接收后传梯度数据
                    tensor_to_send, tensor_shapes=tensor_shape, is_last_stage=False
                )
                grad_dict[module_name] = _restore_tensor_from_comm(grad) #将后传梯度数据还原并存储到字典中，key是当前模块名（因为是模块内部通信）
        return grad_dict

    def send_backward_recv_forward(
        self,
        grad_dict: Dict[str, torch.Tensor],
        tensor_shape: Optional[Shape] = None,
        is_first_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Send backward activation tensor and receive forward activation tensor. #send反传梯度+recv前传激活

        Args:
            grad_dict: A dictionary mapping module names to tensors.
            tensor_shape: Expected gradient tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        input_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items(): #遍历当前rank负责的各个模块
            if rank_module_info.pp_rank == 0: #如果是该模块的首stage rank
                for bridge_comm in rank_module_info.bridge_comms_as_dest_module: #在这个模块中rank作为dest rank，遍历作为dest rank需要参与的BridgeCommunicator
                    # If first stage, and has incoming modules, send backward gradient and
                    # receive forward activation by using bridge communicator.
                    received_tensor = bridge_comm.send_backward_recv_forward( #调用BridgeCommunicator的send_backward_recv_forward方法发送后传梯度数据+接收前传激活值数据
                        grad_dict[bridge_comm.src_module_name]
                    )
                    input_dict[bridge_comm.src_module_name] = received_tensor #将接收的前传激活值数据存储到字典中，key是发送这个激活值数据的模块名
            else: #如果是该模块的非首stage rank，那只需要进行模块内的通信
                # If not first stage, send backward gradient and receive forward activation
                # by using P2P communicator.
                grad_to_send = _prepare_tensor_for_comm(grad_dict[module_name]) #将数据适配到P2P通信可以接受的形状
                received_tensor = rank_module_info.p2p_communicator.send_backward_recv_forward( #调用P2PCommunicator的send_backward_recv_forward方法发送后传梯度数据+接收前传激活值数据
                    grad_to_send, tensor_shapes=tensor_shape, is_first_stage=False
                )
                input_dict[module_name] = _restore_tensor_from_comm(received_tensor) #将前传激活值数据还原并存储到字典中，key是当前模块名（因为是模块内部通信）
        return input_dict

    def recv_backward(
        self, tensor_shape: Optional[Shape] = None, is_last_stage: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Receive backward activation tensor. #接收后传梯度

        Args:
            tensor_shape: Expected gradient tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[recv_backward] tensor_shape: {tensor_shape}, is_last_stage: {is_last_stage}"
        )
        grad_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items(): #遍历当前rank负责的各个模块
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1: #如果是该模块的尾stage rank
                # If last stage, and has incoming modules, receive backward gradient
                # by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_src_module: #在这个模块中rank作为src rank，遍历作为src rank需要参与的BridgeCommunicator
                    grad = bridge_comm.recv_backward() #调用BridgeCommunicator的recv_backward方法接收后传梯度数据
                    grad_dict[bridge_comm.src_module_name] = grad #将接收的后传梯度数据存储到字典中，key是发送这个梯度数据的模块名
            else: #如果是该模块的非尾stage rank，那只需要进行模块内的通信
                # If not last stage, receive backward gradient by using P2P communicator.
                grad = rank_module_info.p2p_communicator.recv_backward( #调用P2PCommunicator的recv_backward方法接收后传梯度数据
                    tensor_shapes=tensor_shape, is_last_stage=False
                )
                grad_dict[module_name] = _restore_tensor_from_comm(grad) #将后传梯度数据还原并存储到字典中，key是当前模块名（因为是模块内部通信）
        return grad_dict

    def send_backward(self, grad_dict: Dict[str, torch.Tensor], is_first_stage: bool = False):
        """Send backward activation tensor. #发送后传梯度

        Args:
            grad_dict: A dictionary mapping module names to tensors.
        """
        for module_name, rank_module_info in self.rank_module_map.items(): #遍历当前rank负责的各个模块
            if rank_module_info.pp_rank == 0: #如果是该模块的首stage rank
                # If first stage, and has incoming modules, send backward activation
                # by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_dest_module: #在这个模块中rank作为dest rank，遍历作为dest rank需要参与的BridgeCommunicator
                    bridge_comm.send_backward(grad_dict[bridge_comm.src_module_name]) #调用BridgeCommunicator的send_backward方法发送后传梯度数据
            else: #如果是该模块的非首stage rank，那只需要进行模块内的通信
                # If not first stage, send backward activation by using P2P communicator.
                grad_to_send = _prepare_tensor_for_comm(grad_dict[module_name]) #将数据适配到P2P通信可以接受的形状
                rank_module_info.p2p_communicator.send_backward(grad_to_send, is_first_stage=False) #调用P2PCommunicator的send_backward方法发送后传梯度数据

    @staticmethod
    def compute_total_pipeline_stages(
        topology: Dict[str, List[str]],
        module_to_grid_map: Dict[str, HyperCommGrid],
        rank: Optional[int] = None,
        module_name: Optional[str] = None,
    ) -> int:
        """Compute the total number of pipeline stages across a multi-module chain. #计算多模块流水线「总共有多少个 pipeline stage」，以及某个 rank 处在整个多模块流水线的第几个 stage。

        Interprets ``topology`` as a directed acyclic graph (DAG) where nodes are modules
        and edges indicate forward data flow from source to destination modules. Each node
        is assigned a weight equal to its pipeline parallel size (number of PP stages).

        The total number of stages is defined as the length of the longest path in this DAG
        under node weights.

        If ``rank`` is None (default), returns the maximum over all terminal (sink) modules of
        the sum of PP sizes along a path ending at that terminal. For example, given:

            image_encoder ->\
                              -> llm -> generator
            audio_encoder  ->/

        the total is: max(pp(image_encoder), pp(audio_encoder)) + pp(llm) + pp(generator).

        If ``rank`` is provided, the result is the total number of pipeline stages up to (and
        including) the PP stage that ``rank`` occupies inside its module. In this case, the
        weight of the target module equals (pp_rank_index(rank) + 1) instead of the module's
        full PP size; other modules still contribute their full PP sizes. If the rank belongs to
        multiple modules (colocation), pass ``module_name`` to disambiguate; otherwise the
        maximum across all candidate modules containing the rank is returned.

        Args:
            topology: Mapping from a module to its list of outgoing modules.
            module_to_grid_map: Mapping from module name to its ``HyperCommGrid``.

        Returns:
            The total number of pipeline stages along the longest path given the constraints.

        Raises:
            ValueError: If the topology contains cycles; or has no terminal nodes when
                ``rank`` is None
        """
        nodes = set(module_to_grid_map.keys())
        # Build adjacency and reverse-adjacency (predecessors).
        adj: Dict[str, List[str]] = {node: list(topology.get(node, [])) for node in nodes}
        preds: Dict[str, List[str]] = {node: [] for node in nodes}
        for src, outs in adj.items():
            for dst in outs:
                preds[dst].append(src)

        # Identify terminal nodes (no outgoing edges) for the rank=None case.
        sinks = [node for node, outs in adj.items() if not outs]
        if rank is None and not sinks:
            raise ValueError(
                "Topology must be a DAG with at least one terminal (no outgoing) module."
            )

        def pp_size(name: str) -> int:
            grid = module_to_grid_map[name]
            pp_dim_index = grid.dim_names.index('pp')
            return grid.shape[pp_dim_index]

        def partial_weight_for_target(target: str) -> Optional[int]:
            if rank is None:
                return None
            grid = module_to_grid_map.get(target)
            rank_groups = grid._gen_rank_enum(['pp'])
            stage_index: Optional[int] = None
            for group in rank_groups:
                if rank in group:
                    stage_index = group.index(rank)
                    break
            return stage_index + 1

        def longest_path_to(target: str) -> int:
            visiting = set()
            partial = partial_weight_for_target(target) #获取rank所在模块内的stage序号

            def weight(name: str) -> int:
                if partial is not None and name == target:
                    return partial
                return pp_size(name)

            def dfs(node: str) -> int:
                if node in visiting:
                    raise ValueError("Topology contains cycles; expected a DAG.")
                visiting.add(node)
                best = 0
                for p in preds.get(node, []):
                    val = dfs(p)
                    if val > best:
                        best = val
                visiting.remove(node)
                return weight(node) + best

            return dfs(target) #dfs获得该模块到第一个模块的stage数量，中间会加上之前的partial

        if rank is None:
            return max(longest_path_to(sink) for sink in sinks)

        return longest_path_to(module_name)
