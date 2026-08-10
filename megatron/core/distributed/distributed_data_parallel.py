# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Optional

import torch

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..fp8_utils import is_float8tensor, post_all_gather_processing
from ..optimizer.param_layout import FullParamLayout
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from ..utils import log_single_rank
from .data_parallel_base import _BaseDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer, group_params_for_buffers, partition_buckets

logger = logging.getLogger(__name__)


class DistributedDataParallel(_BaseDataParallel):
    """ #DDP将grad统一放在连续的buffer中
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        pg_collection: Optional unified process group for distributed training.
        full_param_layout: Optional FullParamLayout providing pre-computed layouts for all
            dtype groups. When provided, each buffer uses the corresponding PerBufferParamLayout
            instead of computing a default one.

    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        full_param_layout: Optional[FullParamLayout] = None,
    ):
        super().__init__(config=config, module=module)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        # Setup process groups, handling both None and provided pg_collection values.
        process_group_dict = ProcessGroupCollection.setup_process_groups_for_ddp(
            pg_collection, config, ddp_config
        )#为 DDP 准备所有需要的进程组

        # If bucket_size is not provided as an input, use sane default based on dp_group size.
        dp_group = process_group_dict['dp_group']
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(40000000, 1000000 * dp_group.size())
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce: #没有启用overlap_grad_reduce就不进行bucket分桶
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )

        # Assign all required process groups #设置通信组，包含dp intra分层
        self.dp_group = process_group_dict['dp_group']
        self.dp_cp_group = process_group_dict['dp_cp_group']
        self.intra_dp_cp_group = process_group_dict['intra_dp_cp_group']
        self.expt_dp_group = process_group_dict['expt_dp_group']
        self.intra_expt_dp_group = process_group_dict['intra_expt_dp_group']
        self.tp_group = process_group_dict['tp_group']
        self.pp_group = process_group_dict['pp_group']
        self.ep_group = process_group_dict['ep_group']

        # Set inter_dist_opt_group if multiple optimizer instances #设置inter_dist_opt_group，应对多instance的DP
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            self.inter_dist_opt_group = process_group_dict['inter_dist_opt_group']

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        self.force_all_reduce = False
        if isinstance(self.pp_group, list):
            pp_rank = self.pp_group[0].rank()
        else:
            pp_rank = self.pp_group.rank()
        if disable_bucketing or pp_rank > 0:
            self.bucket_size = None

        self.param_to_bucket_group = {}

        # Collect all trainable parameters.
        param_to_name = {} #param到name的映射，方便后续通过param查找name
        self.params_with_grad = [] #记录了所有需要优化的参数
        all_params = [] #所有需要优化的参数的集合
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Track params with grad to enable direct setting
            # of param.grad_added_to_main_grad
            self.params_with_grad.append(param)

            param.grad_added_to_main_grad = False
            param_to_name[param] = name
            all_params.append(param)

        # Group parameters by (param_dtype, grad_dtype, is_expert_parallel). #将所有参数按照BufferKey分组
        buffer_groups = group_params_for_buffers(all_params, self.ddp_config.grad_reduce_in_fp32)

        # Auto-compute layouts when using distributed optimizer but no layout was provided.
        # This maintains backward compatibility for callers that create DDP directly
        # without pre-computing layouts (e.g., tests, external code).
        if full_param_layout is None and self.ddp_config.use_distributed_optimizer: #没有提供full_param_layout，就自动计算
            log_single_rank(
                logger,
                logging.WARNING,
                "DistributedDataParallel: full_param_layout not provided with "
                "use_distributed_optimizer=True. Auto-computing layout inside DDP. "
                "Callers should pre-compute layouts via "
                "DistributedOptimizer.compute_full_param_layout() and pass them in.",
            )
            from ..optimizer.distrib_optimizer import DistributedOptimizer

            full_param_layout = DistributedOptimizer.compute_full_param_layout(
                all_params,
                self.bucket_size,
                self.intra_dp_cp_group.size(), #world_size是intra实例内部的rank数量
                self.ddp_config,
                expert_data_parallel_world_size=self.intra_expt_dp_group.size(),
            )

        # When a full_param_layout is provided, verify that the grouping is consistent
        # with the layout (same buffer keys, same params per key, same param_indices).
        if full_param_layout is not None: #提供了full_param_layout，就进行验证
            assert set(buffer_groups.keys()) == set(full_param_layout.layouts.keys()), (
                f"Buffer keys from param grouping {set(buffer_groups.keys())} do not match "
                f"full_param_layout keys {set(full_param_layout.layouts.keys())}"
            )
            for buffer_key, (params, param_indices) in buffer_groups.items():
                layout = full_param_layout.layouts[buffer_key]
                assert set(params) == set(
                    layout.param_index_map.keys()
                ), f"Params for {buffer_key} do not match between grouping and layout"
                assert (
                    param_indices == layout.param_indices
                ), f"param_indices for {buffer_key} do not match between grouping and layout"

        # Compute gradient scaling factors. #计算gradient scaling factors
        if config.calculate_per_token_loss: #calculate_per_token_loss=True 时，DDP 不缩放（factor=1.0），通信做纯 SUM。真正的除法（/ 全局总 token 数）在 finalize_model_grads 中统一完成，那里已经拿到了跨所有 DP rank 汇总的 token 总数，所以不需要 DDP 层面再处理。
            assert (
                not self.ddp_config.average_in_collective
            ), "Cannot average in collective when calculating per-token loss!"
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            # The goal is to scale reduced gradients by 1/dp_size.
            # This can be achieved in two ways:
            #
            # Case 1: average_in_collective=True
            # - Non-expert parameters:
            #   1. No pre-scaling (gradient_scaling_factor=1.0)
            #   2. Do average reduction over dp group (equals to sum then divide by dp_size)
            #   3. Final result is scaled by 1/dp_size as desired
            #
            # - Expert parameters:
            #   1. Scale by edp_size/dp_size before reduction
            #   2. Do average reduction over edp group (equals to sum then divide by edp_size)
            #   3. Resulted scaling: (edp_size/dp_size) * (1/edp_size) = 1/dp_size as desired
            #   (edp_size = expert data parallel world size)
            #
            # Case 2: average_in_collective=False
            # - Both expert and non-expert parameters:
            #   1. Scale gradients by 1/dp_size before reduction
            #   2. Do sum reduction across data parallel ranks
            #   3. Final result is scaled by 1/dp_size as desired
            if self.ddp_config.average_in_collective: #通信做 average，本地只需补偿 expert 的 group size 差异（这样通信自动会除以expt_dp_group.size()，整体就是乘以 1/self.dp_cp_group.size()）。专家部分也是要除以self.dp_cp_group.size()的原因是本身就是dp_cp_group.size()个dp组，edp只是用部分rank来完成所有dp组的数据，所以梯度还是按照所有dp组来算，是dp_cp_group.size()
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = self.expt_dp_group.size() / self.dp_cp_group.size()
            else: #通信不做 average，dense和expert部分都是乘以 1/dp_size
                data_parallel_world_size = self.dp_cp_group.size()

                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Allocate buffers for each group.
        self.buffers = []
        self.expert_parallel_buffers = []
        pg_collection = ProcessGroupCollection(tp=self.tp_group, dp_cp=self.dp_cp_group)
        for buffer_key, (params, param_indices) in buffer_groups.items(): #对每一组param（buffer）
            if buffer_key.is_expert_parallel:
                data_parallel_group = self.intra_expt_dp_group ##intra_dp_cp_group 对应一个expert部分的分布式 optimizer 实例
                scaling_factor = expert_gradient_scaling_factor
            else:
                data_parallel_group = self.intra_dp_cp_group  #intra_dp_cp_group 对应一个dense部分的分布式 optimizer 实例
                scaling_factor = gradient_scaling_factor

            if not config.calculate_per_token_loss: #梯度缩放因子校验
                target_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()
                if self.ddp_config.average_in_collective:
                    if self.ddp_config.num_distributed_optimizer_instances == 1:
                        # Collective is averaging gradients in collective with data_parallel_group.
                        assert (
                            scaling_factor / data_parallel_group.size()
                            == target_gradient_scaling_factor
                        )
                    else:
                        # For non-expert parameters, gradient_scaling_factor is 1.
                        # For expert parameters, gradient_scaling_factor is edp_size/dp_size.
                        assert (scaling_factor == 1) or (
                            scaling_factor == (self.expt_dp_group.size() / self.dp_cp_group.size())
                        )
                else:
                    assert scaling_factor == target_gradient_scaling_factor

            param_layout = (
                full_param_layout.layouts.get(buffer_key) if full_param_layout is not None else None
            ) #获取这个 buffer_key 对应的布局信息
            params_with_names = [(p, param_to_name[p]) for p in params] #获取参数和参数名称的列表
            buffer = _ParamAndGradBuffer(
                self.ddp_config,
                buffer_key.param_dtype,
                buffer_key.grad_dtype,
                params_with_names,
                data_parallel_group,
                self.bucket_size,
                param_to_name,
                scaling_factor,
                param_indices,
                self.ddp_config.nccl_ub,
                pg_collection,
                param_layout=param_layout,
            )#按照param_layout构造_ParamAndGradBuffer对象，包含grad和param的buffer
            if buffer_key.is_expert_parallel: #将buffer添加到属性中
                self.expert_parallel_buffers.append(buffer)
            else:
                self.buffers.append(buffer)

        # In some scenarios, we want to put buckets from different buffers into a group so that
        # their communication can be aggregated. For example, when there are both fp8 buffers
        # and bf16 buffers in the model and vpp is enabled, each model chunk will have an fp8
        # bucket and a bf16 bucket, which doubles the number of communication kernels, and
        # because of the use of CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back
        # communications will prevent the overlap of the communication kernels with computation
        # kernels.
        # If bucketing is explicitly disabled, then put all buckets in a buffer into a single
        # bucket group.
        self.bucket_groups = partition_buckets(
            self.buffers,
            force_single_bucket_group=disable_bucketing,
            reduce_scatter_with_fp32_accumulation=(
                self.ddp_config.reduce_scatter_with_fp32_accumulation
            ),
        )#将bucket聚合为组_ParamAndGradBucketGroup，没有fp8就每个bucket自成一组
        self.expert_parallel_bucket_groups = partition_buckets(
            self.expert_parallel_buffers,
            force_single_bucket_group=disable_bucketing,
            reduce_scatter_with_fp32_accumulation=(
                self.ddp_config.reduce_scatter_with_fp32_accumulation
            ),
        )

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            assert (
                self.ddp_config.use_distributed_optimizer
            ), 'Partial DistOpt cannot be used without DistOpt'
            for bucket_groups in [self.bucket_groups, self.expert_parallel_bucket_groups]:
                communication_stream = torch.cuda.Stream(device=torch.cuda.current_device())
                for bucket_group in bucket_groups:
                    bucket_group.inter_distributed_optimizer_instance_group = (
                        self.inter_dist_opt_group
                    )
                    bucket_group.communication_stream = communication_stream #所有 bucket group 共享同一个 communication_stream，确保多个 DistOpt instance 之间的通信在同一个 stream 上串行化

        # Set `next_param_gather_bucket_group` for different bucket groups by iterating through
        # buckets in reverse order (since all-gathers happen in reverse order of buckets).
        # Note: overlap_param_gather covers both the distributed optimizer and the
        # layer-wise optimizer cases; the latter sets overlap_param_gather=True
        # without use_distributed_optimizer. #确定每个bucket group的next_param_gather_bucket_group
        if self.ddp_config.overlap_param_gather: #反向遍历：从最后一个 bucket group 开始，把它的 next_param_gather_bucket_group 指向前一个 bucket group（因为bucket是逆序构建的，指向前一个实际上代表后面的layer），指定每个bucket group的next_param_gather_bucket_group
            for bucket_groups in [self.bucket_groups, self.expert_parallel_bucket_groups]:
                num_bucket_groups = len(bucket_groups)
                for i in range(1, num_bucket_groups):
                    bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                        bucket_groups[num_bucket_groups - i - 1]
                    )

        # Create map from param to bucket group, used in pre_hook. #构建从param到bucket group的映射，用于pre_hook
        for bucket_groups in [self.bucket_groups, self.expert_parallel_bucket_groups]:
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bucket_group

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters(): #由于使用了grad buffer，需要拦截torch的自动梯度累计，注册backward钩子来手动进行梯度累计，累计到grad buffer中
            if param.requires_grad:
                # When delay_wgrad_compute is True and the param is marked with
                # skip_backward_post_hook, register the backward post hook for its module
                # instead of the param so that the wgrad accumulation and reduce will be performed
                # in backward_dw() method of the module instead of the hook of backward() method.
                # Otherwise, register the backward post hook for the param.
                if self.ddp_config.delay_wgrad_compute and getattr(
                    param, 'skip_backward_post_hook', False
                ):
                    for module in self.module.modules():
                        if hasattr(module, "register_wgrad_accumulation_and_reduce_hooks"):
                            for param_value in module.parameters():
                                if param is param_value:
                                    module.register_wgrad_accumulation_and_reduce_hooks(
                                        self._make_backward_post_hook(param)
                                    )
                                    break
                else:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param) #创建param_tmp就是为了访问到param叶子节点的grad_acc算子（这个是torch自动生成的）
                    # Get the gradient accumulator function.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0] #获取到param叶子节点的grad_acc算子
                    grad_acc.register_hook(self._make_backward_post_hook(param)) #对torch的grad_acc算子注册hook，在grad_acc累计完了grad之后，把grad累计到grad buffer中
                    self.grad_accs.append(grad_acc)

        # Note: overlap_param_gather covers both the distributed optimizer and the
        # layer-wise optimizer cases; the latter sets overlap_param_gather=True
        # without use_distributed_optimizer.
        self.use_forward_hook = self.ddp_config.overlap_param_gather #决定是否启用 forward pre-hook。overlap_param_gather=True 时，需要通过 pre-hook 在 forward 之前等待/触发 all-gather。注意注释：这个配置覆盖了 DO 和 layer-wise optimizer 两种场景
        self.remove_forward_pre_hook_handles = {}
        if self.use_forward_hook: #对每个module注册forward pre-hook，module在前传前进行allgather
            self.enable_forward_pre_hook()
        self.overlap_param_gather_with_optimizer_step = False #__init__ 默认 False，由后续代码（条件满足时）再打开

    def enable_forward_pre_hook(self):
        """
        Enable forward pre-hooks needed for param all-gather overlap with forward compute.
        """
        assert self.use_forward_hook
        assert len(self.remove_forward_pre_hook_handles) == 0
        # Register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )#添加hook，在forward之前进行allgather

    def disable_forward_pre_hook(self, param_sync: bool = True):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        Skip synchronous param all-gather if `param_sync` is False.
        """
        assert self.use_forward_hook
        # De-register forward pre-hook for all sub-modules.
        for module in self.module.modules(): #逐个反注册所有子模块的 pre-hook
            assert self.remove_forward_pre_hook_handles[module] is not None
            self.remove_forward_pre_hook_handles[module].remove()
            del self.remove_forward_pre_hook_handles[module]
        assert len(self.remove_forward_pre_hook_handles) == 0

        # Force synchronize parameters.
        if param_sync: #param_sync=True 是默认：禁用后做一次同步参数 all-gather；False 则跳过。
            self.start_param_sync(force_sync=True)

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather).
        """

        def hook(module, *unused):
            assert (
                self.use_forward_hook
            ), "Should use pre-hook only when overlap_param_gather is True"

            if is_graph_capturing():
                return

            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):#对module的每个param都进行finish_param_sync
                # Skip parameters without an associated buffer (such parameters have a
                # .requires_grad field equal to False).
                if param not in self.param_to_bucket_group:
                    continue
                assert param.requires_grad

                # If aligning param all-gather across pipeline stages, all-gather is dispatched
                # by start_param_sync calls in core/pipeline_parallelism/schedules.py.
                # If overlapping param all-gather with optimizer step, then all-gather has
                # already been dispatched in optimizer step.
                skip_next_bucket_dispatch = (#是否跳过下一个bucket的allgather
                    self.ddp_config.align_param_gather #参数 all-gather 需要跨 pipeline stage 对齐，由 pipeline schedule 统一调度
                    or self.overlap_param_gather_with_optimizer_step #all-gather 已在 optimizer step 中被提前发起了，不需要 pre-hook 再发下一个
                )
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )#完成当前param对应的bucket group的gather通信

        return hook

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
        ready (i.e., when all grads in a bucket have been computed in all microbatches
        in a batch).
        """

        def hook(*unused):
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data) #将param的梯度累加到main_grad（grad buffer）中
                param.grad = None #清空param的梯度

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(
                        param, self.force_all_reduce
                    )

        return hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                bucket_group.is_last_microbatch = True

    def start_param_sync(self, *unused, force_sync: bool = False, force_dispatch: bool = False):
        """
        Initiates param sync (all-gather) communication operations for all model parameters.

        By default, when overlap_param_gather is set to True, dispatches asynchronous communication
        calls; when overlap_param_gather is set to False, calls synchronous communication
        ops. Can override this default behavior using flags below.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings.
            force_dispatch (bool, optional): force dispatch regardless of other settings.
        """
        if not force_sync:
            # If overlapping param AG with optimizer step, AG should not be dispatched again
            # in forward_backward_step.
            if self.overlap_param_gather_with_optimizer_step and not force_dispatch:
                return

        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_param_sync(force_sync=force_sync)

            if not self.ddp_config.overlap_param_gather:
                # For MXFP8 params, we need to copy the all-gathered param data from the buffer to
                # the param.data, since param buffer is not mapped to model params for MXFP8 case.
                # The paramaters are cast from bf16 to MXFP8 during copy.
                # In the case of "overlap_param_gather=True", the param copy is done
                # in "finish_param_sync" stage after zeroing the shared gardient buffers.
                if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                    for bucket in bucket_group.buckets:
                        is_bf16_weight_bucket = False
                        for param in bucket.params:
                            # Skip copying since bf16 weights in the mxfp8 model
                            # are already mapped to param.data.
                            if not is_float8tensor(param):
                                is_bf16_weight_bucket = True
                                break
                            param_start, param_end = bucket.param_to_index[param]
                            param_slice = bucket.param_data.view(-1)[param_start:param_end]
                            param.data.copy_(param_slice.view(param.data.shape))
                        if is_bf16_weight_bucket:
                            continue
                        # All-gathered params are not needed after being copied to param.data.
                        # Zero out the param buffer (shared with grad buffer) for gradient
                        # accumulation. We cannot zero out the entire grad buffer because one grad
                        # buffer may correspond to multiple param buffers. If we zero out the entire
                        # grad buffer, it would clear the data of those param buffers that have not
                        # yet completed AG.
                        bucket.param_data.zero_()
                else:
                    fp8_params = []
                    for bucket in bucket_group.buckets:
                        for param in bucket.params:
                            if is_float8tensor(param):
                                fp8_params.append(param)
                    if len(fp8_params) > 0:
                        post_all_gather_processing(fp8_params)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_grad_sync()

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)

    def free_overlap_buffers(self):
        """Free overlap param-gather GPU buffers across all bucket groups."""
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.free_overlap_buffers()

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        if getattr(self.config, 'cuda_graph_impl', 'none') != 'transformer_engine':
            # Don't reset grad_added_to_main_grad when CUDA Graph is used.
            # Because in CUDA Graph it no longer has the opportunity to set it back
            # to True, and there will be a double-GA.
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset() #只会reset grad_buffer，不会reset param_buffer
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                data_parallel_group = self.expt_dp_group
            else:
                data_parallel_group = self.dp_cp_group
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0), #源 rank 在 group 中的 rank id（从 0 到 group.size()-1）
                group=data_parallel_group,
            )

    def offload_grad_buffers(self, synchronize: bool = True, empty_cache: bool = True) -> None:
        """
        Free all grad_data tensors to release GPU memory.

        Uses storage().resize_(0) to release memory while keeping tensor views intact.
        All bucket.grad_data and param.main_grad views remain valid tensor objects
        (though accessing them during offload is undefined behavior).

        Args:
            synchronize: Whether to call torch.cuda.synchronize() before freeing.
            empty_cache: Whether to call torch.cuda.empty_cache() after freeing.
        """
        if synchronize:
            torch.cuda.synchronize()

        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.offload_to_cpu(move_params=False, move_grads=True)

        if empty_cache:
            torch.cuda.empty_cache()

    def restore_grad_buffers(self, synchronize: bool = True) -> None:
        """
        Reallocate grad_data tensors on GPU.

        All existing views (bucket.grad_data, param.main_grad) automatically
        become valid again since they share the same storage. The grad_data
        is zeroed after reallocation.

        Args:
            synchronize: Whether to call torch.cuda.synchronize() after allocation.
        """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reload_from_cpu(move_params=False, move_grads=True)

        if synchronize:
            torch.cuda.synchronize()
