# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import nvtx_decorator

# Types
Shape = Union[List[int], torch.Size]


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []
    if tensor_send_prev is not None: #逐个生成P2POp对象
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_pipeline_rank, group
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group
        )
        ops.append(recv_next_op)
    if len(ops) > 0: #打包执行
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
): #逐个执行
    reqs = {}
    even_send_odd_recv_group = group
    if group.size() == 2 and torch.distributed.get_backend(group) != 'ucc':
        # Use the global process group for one of the two p2p communications
        # to allow the overlap of the independent communications.
        # Using the global process group is compatible because the pipeline-parallel
        # communications set the source and destination by global rank.
        # The only exception occurs when using the ‘ucc’ backend.
        # Because the global communicator always uses the ‘nccl’ backend,
        # we must ensure the else path is followed for the ‘ucc’ backend.
        even_recv_odd_send_group = torch.distributed.group.WORLD
    else:
        even_recv_odd_send_group = group

    if group.rank() % 2 == 0: #需要分奇偶rank来匹配NCCL的send和recv，否则会死锁。因为isend/irecv最终是ProcessGroupNCCL的点对点通信，对每对rank会创建一个ncclComm并使用一个cuda stream来完成NCCL kernel执行，而且不区分方向，进程不管是send还是recv都加入这个CUDA Stream，假如rank0，1进行p2p通信，都先启动send kernel再启动recv kernel，那rank0和rank1的CUDA Stream都在堵塞，相互死锁，不管是不是异步send/recv。这里通过调换顺序可以避免死锁
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["send_next"] = send_next_req

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["recv_prev"] = recv_prev_req

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["send_prev"] = send_prev_req

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["recv_next"] = recv_next_req

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["recv_prev"] = recv_prev_req

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["send_next"] = send_next_req

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["recv_next"] = recv_next_req

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["send_prev"] = send_prev_req
    return reqs


def is_single_shape(x) -> bool:
    """Check if the input is a single shape."""
    if isinstance(x, torch.Size):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False


class P2PCommunicator:
    """P2P (Point-to-Point) Communicator for pipeline parallelism.

    This class handles communication between pipeline stages by managing
    tensor exchanges between consecutive stages in the pipeline.
    """

    def __init__(self, pp_group: dist.ProcessGroup, config: ModelParallelConfig):
        # Basic attrs
        self.pp_group = pp_group
        self.config = config

        world_size = self.pp_group.size() #pp并行度
        curr_rank_in_pg = self.pp_group.rank() #当前进程在pp组中的rank

        next_rank_pg = (curr_rank_in_pg + 1) % world_size #在pp组中，当前进程的下一个进程的rank
        prev_rank_pg = (curr_rank_in_pg - 1) % world_size #在pp组中，当前进程的前一个进程的rank

        self.next_rank: int | None = dist.get_global_rank(self.pp_group, next_rank_pg) #获取下一个进程的全局rank
        self.prev_rank: int | None = dist.get_global_rank(self.pp_group, prev_rank_pg) #获取前一个进程的全局rank
        self.virtual_pipeline_model_parallel_size = (
            config.virtual_pipeline_model_parallel_size
            if config.virtual_pipeline_model_parallel_size is not None
            else None
        )

    @property
    def is_pp_first_stage(self) -> bool:
        """Return True if pp first stage."""
        return is_pp_first_stage(self.pp_group)

    @property
    def is_pp_last_stage(self) -> bool:
        """Return True if pp last stage."""
        return is_pp_last_stage(self.pp_group)

    @property
    def total_stages(self) -> int:
        """Return total number of pipeline stages."""
        return self.pp_group.size()

    @property
    def current_stage(self) -> int:
        """Return current pipeline stage index (0-indexed)."""
        return self.pp_group.rank()

    def _communicate_shapes(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
        """Communicate tensor shapes between stages. Used to communicate #用于在不同micro batch的序列长度变化场景下，在传输实际 tensor 数据之前先交换 shape 信息
        tensor shapes before the actual tensor communication happens.
        This is required when the sequence lengths across micro batches
        are not uniform.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if
                            set to None).
            tensor_send_prev: tensor to send to prev rank (no tensor sent if
                            set to None).
            recv_prev: boolean for whether tensor should be received from
                    previous rank.
            recv_next: boolean for whether tensor should be received from
                    next rank.
        Returns:
            (recv_prev_shape, recv_next_shape)
        """
        config = self.config
        recv_prev_shape_tensor = None
        recv_next_shape_tensor = None
        send_prev_shape_tensor = None
        send_next_shape_tensor = None
        if recv_prev: # 接收方的 buffer：空的 (3,) tensor，等着被填充。(seq, batch, hidden)
            recv_prev_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if recv_next: # 接收方的 buffer：空的 (3,) tensor，等着被填充。(seq, batch, hidden)
            recv_next_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_prev is not None: # 发送方：把实际 tensor 的 shape 转成 tensor
            send_prev_shape_tensor = torch.tensor(
                tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_next is not None: # 发送方：把实际 tensor 的 shape 转成 tensor
            send_next_shape_tensor = torch.tensor(
                tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )

        if config.use_ring_exchange_p2p: #默认False，PyTorch已弃用torch.distributed.ring_exchange
            torch.distributed.ring_exchange(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
            )
        else:
            ops = []
            if send_prev_shape_tensor is not None:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend, send_prev_shape_tensor, self.prev_rank, self.pp_group
                )
                ops.append(send_prev_op)
            if recv_prev_shape_tensor is not None:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_prev_shape_tensor, self.prev_rank, self.pp_group
                )
                ops.append(recv_prev_op)
            if send_next_shape_tensor is not None:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, send_next_shape_tensor, self.next_rank, self.pp_group
                )
                ops.append(send_next_op)
            if recv_next_shape_tensor is not None:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_next_shape_tensor, self.next_rank, self.pp_group
                )
                ops.append(recv_next_op)
            if len(ops) > 0:
                reqs = torch.distributed.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

            # To protect against race condition when using batch_isend_irecv().
            # should take this out once the bug with batch_isend_irecv is resolved.
            torch.cuda.synchronize()

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None: # 如果recv_prev_shape_tensor不是None，说明它被填充值了。(3,) tensor，等着被填充。(seq, batch, hidden)
            recv_prev_shape = recv_prev_shape_tensor.tolist()

        recv_next_shape = [0, 0, 0]
        if recv_next_shape_tensor is not None: # 如果recv_next_shape_tensor不是None，说明它被填充值了。(3,) tensor，等着被填充。(seq, batch, hidden)
            recv_next_shape = recv_next_shape_tensor.tolist()

        return recv_prev_shape, recv_next_shape

    def _communicate(
        self,
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        wait_on_reqs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Communicate tensors between stages. Used as helper method in other
        communication methods that are used in megatron/schedules.py.

        Args:
            tensor_send_next (torch.Tensor, optional):
                Tensor to send to next rank (no tensor sent if None)

            tensor_send_prev (torch.Tensor, optional):
                Tensor to send to prev rank (no tensor sent if None)

            recv_prev (boolean, required):
                whether tensor should be received from previous rank.

            recv_next (boolean, required):
                whether tensor should be received from next rank.

            tensor_shape (List[int] or torch.Size, required): #指定要接收的张量的形状
                shape of tensor to receive (this method assumes that all
                tensors sent and received in a single function call are
                the same shape).

            wait_on_reqs (boolean, optional, default=True): #控制 _communicate 内部是否会将通信完成Event插入到计算流中，以确保计算流数据依赖，不管是否batched p2p都可以指定，默认True
                For non-batched p2p communication, wait on each request #非批量时是一个一个 wait
                before returning.

        Returns:
            tuple containing

            - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
            - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

        """

        config = self.config
        tensor_recv_prev_func = None
        tensor_recv_next_func = None

        if config.variable_seq_lengths or config.mtp_standalone:
            recv_prev_shape, recv_next_shape = self._communicate_shapes( #变长序列（variable_seq_lengths）或 MTP 模式：先交换 shape，再分配 buffer
                tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )
        else: #固定序列长度：shape 由调用方通过参数传入，不需要额外的 shape 交换
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape

        def create_tensor_recv_prev(): #创建接收 buffer
            return torch.empty(
                recv_prev_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        def create_tensor_recv_next(): #创建接收 buffer
            return torch.empty(
                recv_next_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        if recv_prev:
            if config.pipeline_dtype is None:
                raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_prev is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_prev_func = create_tensor_recv_prev

        if recv_next:
            if config.pipeline_dtype is None:
                raise RuntimeError("dtype must be provided if recv_next is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_next is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_next_func = create_tensor_recv_next

        # Send tensors in both the forward and backward directions as appropriate. #确定使用哪种p2p通信方式
        if config.use_ring_exchange_p2p: #已弃用，理论上从不执行

            def _ring_exchange_wrapper(**kwargs):
                torch.distributed.ring_exchange(**kwargs)
                return []

            p2p_func = _ring_exchange_wrapper
        elif config.batch_p2p_comm: # 批量提交（batch_isend_irecv）
            assert wait_on_reqs
            p2p_func = _batched_p2p_ops
        else: # 逐个 isend/irecv
            p2p_func = _p2p_ops

        pp_group = self.pp_group
        next_rank = self.next_rank
        prev_rank = self.prev_rank

        if config.use_ring_exchange_p2p or config.batch_p2p_comm:
            reqs = []
        else:
            reqs = {}

        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func() #创建接收buffer

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func() #创建接收buffer

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=pp_group,
            prev_pipeline_rank=prev_rank,
            next_pipeline_rank=next_rank,
        ) #执行p2p通信，包括4个点对点通信
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
        else:
            reqs.update(p2p_reqs)

        if wait_on_reqs and len(reqs) > 0:
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait() #在返回给上层之前，把四个方向的通信完成事件都挂到当前 compute stream 上。这样上层拿到 tensor_recv_prev 直接送进下一层网络计算就是安全的，不需要关心通信是否真的结束——GPU 会自己在正确的位置等。
            reqs = None

        if config.batch_p2p_comm and config.batch_p2p_sync: #旧版 PyTorch 的 batch_isend_irecv 返回的 Work 对象 wait() 有竞态 bug（可能提前返回/事件没记录对），所以当时加了这个"全设备同步"兜底，注释明说：现代 torch 不需要这个
            # To protect against race condition when using batch_isend_irecv().
            # User should assert that we have a modern enough PyTorch to not need this
            torch.cuda.synchronize()

        return tensor_recv_prev, tensor_recv_next, reqs #返回接受的张量和通信handle

    @nvtx_decorator()
    def recv_forward(
        self, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        config = self.config
        for tensor_shape in tensor_shapes:
            if is_first_stage: #第一个stage没有前序rank，所以不需要从前序rank接收数据
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_tensor_shapes:
            return input_tensors[0]
        return input_tensors

    @nvtx_decorator()
    def recv_backward(
        self, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from next rank in pipeline (backward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        config = self.config
        output_tensor_grads = []
        for tensor_shape in tensor_shapes:
            if is_last_stage: #最后一个stage没有后序rank，所以不需要从后序rank接收数据
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
        if unwrap_tensor_shapes:
            return output_tensor_grads[0]
        return output_tensor_grads

    @nvtx_decorator()
    def send_forward(self, output_tensors, is_last_stage: bool) -> None:
        """Send tensor to next rank in pipeline (forward send).""" #前向把激活值送出去
        config = self.config
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]

        for output_tensor in output_tensors:
            if not is_last_stage:
                if config.timers is not None:
                    config.timers('forward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('forward-send').stop()

    @nvtx_decorator()
    def send_backward(self, input_tensor_grads, is_first_stage: bool) -> None:
        """Send tensor to previous rank in pipeline (backward send).""" #后向把梯度传出去
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        config = self.config
        for input_tensor_grad in input_tensor_grads:
            if not is_first_stage:
                if config.timers is not None:
                    config.timers('backward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('backward-send').stop()

    @nvtx_decorator()
    def send_forward_recv_backward(
        self, output_tensors, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Batched send and recv with next rank in pipeline.""" #一次性做好前传的send和反传的recv，都是和下一级stage的通信
        config = self.config
        unwrap_output_tensors = False
        if not isinstance(output_tensors, list):
            unwrap_output_tensors = True
            output_tensors = [output_tensors]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        output_tensor_grads = []
        for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('forward-send-backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-send-backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
        if unwrap_output_tensors:
            return output_tensor_grads[0]
        return output_tensor_grads

    @nvtx_decorator()
    def send_backward_recv_forward(
        self, input_tensor_grads, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Batched send and recv with previous rank in pipeline.""" #一次性做好后传的send和前传的recv，都是和上一级stage的通信
        config = self.config
        unwrap_input_tensor_grads = False
        if not isinstance(input_tensor_grads, list):
            unwrap_input_tensor_grads = True
            input_tensor_grads = [input_tensor_grads]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('backward-send-forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-send-forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_input_tensor_grads:
            return input_tensors[0]
        return input_tensors

    @nvtx_decorator()
    def send_forward_recv_forward(
        self,
        output_tensor: torch.Tensor,
        recv_prev: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False, #可选，控制这次 P2P 通信是同步等完还是异步返回 handle，从而让通信和计算重叠，PrcessGroupNCCL的P2P肯定是使用非compute stream，所以可以重叠
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline.""" #一次性做好前传的send和前传的recv，分别和下一级stage和上一级stage通信，recv可选
        config = self.config
        if config.timers is not None:
            config.timers('forward-send-forward-recv', log_level=2).start()
        input_tensor, _, wait_handles = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('forward-send-forward-recv').stop()
        if overlap_p2p_comm:
            return input_tensor, wait_handles
        return input_tensor

    @nvtx_decorator()
    def send_backward_recv_backward(
        self,
        input_tensor_grad: torch.Tensor,
        recv_next: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False, #可选，控制这次 P2P 通信是同步等完还是异步返回 handle，从而让通信和计算重叠，PrcessGroupNCCL的P2P肯定是使用非compute stream，所以可以重叠
    ) -> torch.Tensor:
        """Batched recv from next rank and send to previous rank in pipeline.""" #一次性做好后传的send和后传的recv，分别和上一级stage和下一级stage通信，recv可选
        config = self.config
        if config.timers is not None:
            config.timers('backward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, wait_handles = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('backward-send-backward-recv').stop()
        if overlap_p2p_comm:
            return output_tensor_grad, wait_handles
        return output_tensor_grad

    @nvtx_decorator()
    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
    ) -> torch.Tensor:
        """Batched send and recv with previous and next ranks in pipeline.""" #一次性做好前传反传的send和recv，recv可选
        config = self.config
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
        input_tensor, output_tensor_grad, _ = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
        )
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv').stop()
        return input_tensor, output_tensor_grad
