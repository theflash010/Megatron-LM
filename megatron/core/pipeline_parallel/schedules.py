# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.transformer.cuda_graphs import create_cudagraphs, set_current_microbatch
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    nvtx_range_pop,
    nvtx_range_push,
)

from .combined_1f1b import (
    combined_1f1b_schedule_for_interleaved_pipelining,
    combined_1f1b_schedule_for_no_pipelining,
)
from .hybrid_cp_schedule import hybrid_context_parallel_forward_backward

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func(pp_size: Optional[int] = None, vp_size: Optional[int] = None):
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state. #根据 PP 大小返回不同的PP调度函数

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step.

    collect_non_loss_data (optional, bool, default=False): TODO.

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    adjust_tensor_shapes_fn (Callable, optional): A function that adjusts the receive and send
        tensor shapes. Only applicable in forward_backward_pipelining_without_interleaving for now.
        Takes in a list of receive shapes and a list of send shapes and returns the adjusted
        respective list of shapes. Thus it is not used in the other forward-backward functions
        which have different shape handling.

    force_all_reduce (bool, optional): If true, force use of all-reduce for gradient reduction
        instead of reduce-scatter (if using distributed optimizer) in this iteration to ensure all
        data-parallel ranks have fully reduced gradients. This is useful for easier wgrad saving
        (can just inspect DP replica 0 to get full set of wgrads for entire model).

    Args:
        pp_size (Optional[int]): Pipeline model parallel size to use.
        vp_size (Optional[int]): Virtual pipeline model parallel size to use.
            If both pp_size and vp_size are None, both values fall back to parallel_state.
            Otherwise, provided values are used as-is and None is treated as an explicit input.

    """
    if pp_size is None and vp_size is None:
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    if pp_size > 1:
        if vp_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving #交错的 1F1B：多个虚拟 stage 轮流执行，进一步减小空泡
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving #1F1B：forward 和 backward 交错，减少空泡
    else:
        forward_backward_func = forward_backward_no_pipelining #顺序：N 个 micro batch 依次 forward，再依次 backward
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field. #deallocate_output_tensor 释放的是当前 pipeline stage 发送给下一个 stage 的那个输出张量，而不是 stage 内部各层计算中的中间激活。

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.

    Supports multiple formats:
    - torch.Tensor: Deallocates the tensor directly
    - List[Tensor]: Recursively deallocates each element
    - Dict[str, Tensor]: Recursively deallocates each value (for multi-module pipelines)
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return

    # Handle dict format (multi-module pipelines)
    if isinstance(out, dict):
        for value in out.values():
            deallocate_output_tensor(value, deallocate_pipeline_outputs)
        return

    # Handle list format
    if isinstance(out, list):
        for item in out:
            deallocate_output_tensor(item, deallocate_pipeline_outputs)
        return

    # Base case: deallocate tensor
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype) #把原张量的数据空间释放，只保留一个空壳


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None: #对于最后一个stage，由于grad_output不存在，所以这里构造一个grad_output，对loss的grad就是 1
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def get_tensor_device(tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]):
    """Get the device of a tensor or a dictionary of tensors."""
    if isinstance(tensor, dict):
        return next(iter(tensor.values())).device
    return tensor.device


def forward_step_calc_loss(
    model,
    output_tensor,
    loss_func,
    config,
    vp_stage,
    collect_non_loss_data,
    num_microbatches,
    forward_data_store,
    cp_group_size=None,
    is_last_stage=None,
):
    """Calculate the loss and number of tokens for forward_step()"""

    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert (
            vp_stage == model_vp_stage
        ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"

    if cp_group_size is None and is_last_stage is None:
        # fallback to parallel state
        cp_group_size = parallel_state.get_context_parallel_world_size()
        is_last_stage = parallel_state.is_pipeline_last_stage(
            ignore_virtual=False, vp_stage=vp_stage
        )
    else:
        assert is_last_stage is not None, "is_last_stage must be provided"
        if is_last_stage:
            assert cp_group_size is not None, "cp_group_size must be provided on last stage"

    num_tokens = torch.tensor(0, dtype=torch.int)
    if is_last_stage:
        if loss_func is None:
            forward_data_store.append(output_tensor)
        elif not collect_non_loss_data:
            outputs = loss_func(output_tensor) #loss_func 返回裸 sum（torch.sum(losses * loss_mask)）；可能返回 2 元组（legacy 样本粒度）或 3 元组（per-token 型：loss, num_tokens, loss_reduced）
            if len(outputs) == 3: #3 元组 = per-token 型 loss_func（用户自定义），框架按返回长度识别
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss: #默认（样本粒度）模式才本地归一化；真正的 per-token 模式（calculate_per_token_loss=True）跳过这里，loss 保持裸 sum，最后在 finalize_model_grads 统一除以全局 token 数
                    # Protect against division by zero when all tokens are masked
                    #   in a microbatch.
                    output_tensor /= torch.clamp(num_tokens, min=1) #默认模式：本地 per-token 平均（÷本 rank 自己的 token 数）——这是样本粒度归一化，不是真正的 per-token 模式（如果开了CP，只覆盖自己处理的那部分 token，后面有地方会聚合 loss）
                    output_tensor /= num_microbatches #再 ÷ microbatch 数 → 梯度累积平均，每个 microbatch 贡献 1/N 的梯度
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2 #老版本不用管
                output_tensor, loss_reduced = outputs
                output_tensor *= cp_group_size
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        device = get_tensor_device(output_tensor)
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            cp_size_for_scaling = cp_group_size if cp_group_size is not None else 1
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale * cp_size_for_scaling / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        device = get_tensor_device(output_tensor)
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    return output_tensor, num_tokens


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor, #其他stage前传过来的激活值，对于第一个stage是 None
    forward_data_store,
    config,
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    vp_stage=None,
    is_last_stage=True,
): #完成单个micro batch的前向计算
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.
        vp_stage (int, optional):
            The virtual pipeline stage. Defaults to None.
        is_last_stage (bool, optional):
            Whether it is the last stage. Defaults to True.
            Also considering virtual stages.
            In case of PP/VPP, is_last_stage/is_vp_last_stage.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list): #统一包装为list
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor) #set_input_tensor 负责让 PP 中间 stage的进程跳过 data_iterator，直接用上一 stage 传来的激活值。

    if config.enable_autocast: #根据 config 决定是否开启 autocast
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)  #torch.autocast 是一个上下文管理器，进入后 GPU 算子会自动选择 fp16/bf16 执行来加速，同时在需要精度的地方（如 loss、softmax、layernorm）保持 fp32。
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None: #不使用重计算
            output_tensor, loss_func = forward_step_func(data_iterator, model) #执行这个micro batch的前传（如果是pp stage0那就会通过data_iterator获取micro batch数据），#返回output_tensor（中间激活值/每个token的loss），还有计算最终loss的 loss_func
        else: #使用重计算
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
    output_tensor, num_tokens = forward_step_calc_loss( #用 loss_func 计算 loss 并处理 loss 的归一化/缩放。只在最后一个 stage 计算 loss，返回值：output_tensor, num_tokens——计算后的 per-token loss 和 token 计数，供后续 backward 使用。
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size,
        is_last_stage,
    )

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens #返回output_tensor和处理的token数


def backward_step(input_tensor, output_tensor, output_tensor_grad, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list): #统一将input_tensor打包成list，通过设置unwrap_input_tensor_grad来标识后续是否需要解包
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad() #设置输入张量的梯度需要保留（输入张量不是叶子张量，默认不保留梯度）

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None: #output_tensor_grad[0] is None代表是last stage 且 配置了 grad_scale_func
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # In multi-modal models like VLM, some batches may not have images.
    # When no image is present, the vision encoder (as a separate pipeline stage)
    # will not participate in the computation.
    # This results in a tensor that does not require gradients.
    # In such cases, we intentionally skip the backward pass while preserving zero gradients.
    if output_tensor[0].requires_grad: #如果某个 pipeline stage 在这一次 microbatch 中没有做任何前向计算，它的输出 tensor 就不会关联到计算图上，requires_grad 为 False，对它执行 backward 会出错。
        if config.deallocate_pipeline_outputs: #deallocate_pipeline_outputs 是一个显存优化选项，核心思想：output tensor 在发给下一个 stage 后，它的 .data 就没用了，唯一还有用的是 .grad_fn（用于反向传播），所以可以把 output tensor 的 .data 替换成一个标量空张量（前传的时候已经做了）。
            custom_backward(output_tensor[0], output_tensor_grad[0]) #这是配套的 backward 实现，直接调用 C++ autograd 引擎，绕过 PyTorch 的 shape 检查。因为 .data 已经被替换成了 (1,) 的标量，如果用标准的 torch.autograd.backward 会报 shape 不匹配的错。
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0]) #标准的 torch.autograd.backward

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None] #收集输入tensor（从上一个stage获得）的梯度
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad: #解包
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def backward_step_multimodule(
    input_tensor: Dict[str, torch.Tensor],
    output_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_tensor_grad: Optional[Dict[str, torch.Tensor]],
    config,
    language_model_module_name: str,
) -> Dict[str, torch.Tensor]:
    """Backward step for multi-module pipelines.

    In multi-module pipelines, tensors are organized as dictionaries with
    module names as keys. Each module's backward pass is performed independently.
    """
    # Retain gradients on all input tensors.
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is not None:
            tensor.retain_grad()

    # Last stage: output_tensor is a scalar loss from the language model.
    # Associate it with the language_model_module_name.
    if not isinstance(output_tensor, dict):
        output_tensor = {language_model_module_name: output_tensor}

    # Handle output_tensor_grad: None (last stage) or dict (intermediate stages).
    if not output_tensor_grad:
        output_tensor_grad = {key: None for key in output_tensor.keys()}

    # Apply grad scaling if needed (for last stage only).
    for module_name in output_tensor.keys():
        if output_tensor_grad[module_name] is None and config.grad_scale_func is not None:
            output_tensor[module_name] = config.grad_scale_func(output_tensor[module_name])

    # Perform backward pass for each module.
    for module_name in output_tensor.keys():
        output_tensor_module = output_tensor[module_name]
        output_tensor_grad_module = output_tensor_grad[module_name]

        # In multi-modal models like VLM, some batches may not have images.
        # In such cases, skip backward while preserving zero gradients.
        if output_tensor_module is not None and output_tensor_module.requires_grad:
            if config.deallocate_pipeline_outputs:
                custom_backward(output_tensor_module, output_tensor_grad_module)
            else:
                torch.autograd.backward(
                    output_tensor_module, grad_tensors=output_tensor_grad_module
                )

    # Collect gradients for input tensors.
    input_tensor_grad = {}
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is None:
            input_tensor_grad[module_name] = None
        else:
            input_tensor_grad[module_name] = tensor.grad

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    """Check if it is the first validation step."""
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]], #数据迭代器。非 PP 模式下只取第一个
    model: Union[torch.nn.Module, List[torch.nn.Module]], #模型。非 PP 模式不允许分块，取第一个
    num_microbatches: int, #microbatch 数量，每个 step 跑几个 microbatch 再合并梯度
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,  # unused
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run forward and backward passes with no pipeline parallelism""" #处理一个step的n个micro batch

    if pg_collection is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif pg_collection is not None:
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for non-pipeline-parallel schedule"

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func #sync 指的就是数据并行（Data Parallel）的梯度同步。no_sync是一个上下文管理器（context manager），它可以在其管理的代码块执行期间禁用梯度同步，而在代码块执行完毕后恢复梯度同步。使用DDP的话默认是DDP的 no_sync 方法
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = [] #收集训练信息用的（如(output_tensor, num_tokens, loss_reduced)）
    input_tensor, output_tensor_grad = None, None #input_tensor是上一个pp stage前传输入给这个stage的激活值，output_tensor_grad是下一个stage反传过来的grad
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda") #total_num_tokens 是一个0 维标量 tensor，放在 GPU 上，初始值为 0，用于跨所有 microbatch 累加 token 总数。当开启 calculate_per_token_loss 时，loss 是按 per-token 平均的，梯度缩放需要知道整个 global batch 里一共多少 token，而不是简单按 microbatch 数量平均。所以需要把所有 microbatch 的 token 数累加起来。

    if config.overlap_moe_expert_parallel_comm and not forward_only:
        forward_data_store, total_num_tokens = combined_1f1b_schedule_for_no_pipelining(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            partial(check_first_val_step, first_val_step, forward_only),
        )
    elif config.hybrid_context_parallel:
        forward_data_store, total_num_tokens = hybrid_context_parallel_forward_backward(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            check_first_val_step,
            model_type,
        )
    else:
        with no_sync_func(): #前n-1个microbatch不进行梯度同步（因为还没有反传，在梯度累积），只在最后1个microbatch进行梯度同步
            for i in range(num_microbatches - 1):
                output_tensor, num_tokens = forward_step( #执行单个micro batch的前传
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor, #其他stage前传过来的激活值，对于第一个stage是 None
                    forward_data_store,
                    config,
                    pg_collection.cp.size(),
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0), #判断当前是否是第一个 microbatch
                    current_microbatch=i, #当前是第几个 microbatch
                )
                total_num_tokens += num_tokens
                if not forward_only: #进行反传
                    backward_step(input_tensor, output_tensor, output_tensor_grad, config)
        # Run computation for last microbatch out of context handler (want to
        # synchronize gradients). #最后一个micro batch同步梯度（DDP逻辑）
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            pg_collection.cp.size(),
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, num_microbatches == 1
            ),
            current_microbatch=num_microbatches - 1,
        ) #返回output_tensor（输出中间激活值/加权平均后的loss）, num_tokens（处理的token数量）

        total_num_tokens += num_tokens #累加处理的token数量

        if not forward_only: #反传
            backward_step(input_tensor, output_tensor, output_tensor_grad, config)

    if config.finalize_model_grads_func is not None and not forward_only: #如果有需要，确保所有梯度同步完成了
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func( #梯度收尾，完成模型特定组件的额外的all-reduce操作
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()
    # Reset all_gather_pipeline bucket status before next validation iteration
    if forward_only:
        for model_chunk in [model]:
            if (
                hasattr(model_chunk, 'ddp_config')
                and model_chunk.ddp_config.use_megatron_fsdp
                and model_chunk.ddp_config.overlap_param_gather
            ):
                model_chunk.synchronize_param_gather()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ): #当使用 CUDA Graph 加速（cuda_graph_impl == "local"）且首次迭代时，创建一组 CUDA Graph。CUDA Graph 可以在后续迭代中直接回放，省去 kernel launch 的开销。这里的条件意味着：只在首个step创建 Graph，后续回放不复创建。
        create_cudagraphs()

    return forward_data_store


def clear_embedding_activation_buffer(config, model, is_last_stage):
    """Clear embedding activation buffer."""

    if is_last_stage and config.defer_embedding_wgrad_compute:
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)

        # Need to ensure no stray activations exists in this buffer
        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
    """Finish embedding wgrad compute.""" #补齐LM Head的梯度计算
    if is_last_stage and config.defer_embedding_wgrad_compute:
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight, tp_group
        )


def get_pp_rank_microbatches(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    forward_only=False,
    overlap_moe_expert_parallel_comm=False,
    p2p_communicator: Optional[P2PCommunicator] = None,
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    if p2p_communicator is not None:
        pipeline_parallel_size = p2p_communicator.pp_group.size()
        pipeline_parallel_rank = p2p_communicator.pp_group.rank()
        virtual_pipeline_parallel_size = p2p_communicator.virtual_pipeline_model_parallel_size
    else:
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        virtual_pipeline_parallel_size = (
            parallel_state.get_virtual_pipeline_model_parallel_world_size()
        )

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:
            # forward_backward_pipelining_without_interleaving
            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            # forward_backward_pipelining_with_interleaving
            # Run (num_model_chunks-1)*microbatch_group_size_per_vp_stage on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
            # When enabling overlap_moe_expert_parallel_comm, we schedule one extra micro-batch
            # forward step before the 1f1b stages. This is needed to ensure the forward
            # and backward computations are independent in all 1f1b steps.
            if overlap_moe_expert_parallel_comm:
                num_warmup_microbatches = num_warmup_microbatches + 1
    else:
        # forward_backward_no_pipelining
        # This path is only used for cuda graph capturing compatibility for the PP=1 case.
        num_warmup_microbatches = 0

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )


def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Get the schedule table for PP scheduling."""
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    config = get_model_config(model[0])
    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        # vp is ignored for clear_embedding_activation_buffer
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # Compute number of warmup and remaining microbatches.
    # seems only used for vpp
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            is_pp_first_stage(p2p_communicator.pp_group)
            if forward
            else is_pp_last_stage(p2p_communicator.pp_group)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        """Preprocess for forward_step_helper"""
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if _is_vp_first_stage(vp_stage=model_chunk_id) and is_pp_first_stage(pp_group):
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        return input_tensor

    def forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return

    def forward_step_helper(virtual_microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
            is_last_stage=_is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group),
        )

        forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens)

        return output_tensor

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id):
        """Preprocess for backward_step_helper"""
        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # pylint: disable=E0606
        if _is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group):
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        return input_tensor, output_tensor, output_tensor_grad

    def backward_step_helper_postprocess(virtual_microbatch_id):
        """Postprocess for backward_step_helper"""
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks"""
        nonlocal output_tensor_grads
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)

        input_tensor, output_tensor, output_tensor_grad = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id
        )

        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad

    def forward_backward_helper_wrapper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
    ):
        """
        wrap forward_helper, backward_helper, and combined_forward_backward_helper in a unified way
        """
        if config.overlap_moe_expert_parallel_comm and not forward_only:  # Combined 1F1B path
            return combined_1f1b_schedule_for_interleaved_pipelining(
                config,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                forward_data_store,
                forward_step_helper_preprocess,
                forward_step_helper_postprocess,
                backward_step_helper_preprocess,
                backward_step_helper_postprocess,
                get_microbatch_id_in_model_chunk,
                get_model_chunk_id,
                partial(check_first_val_step, first_val_step, forward_only),
                is_first_microbatch_for_model_chunk,
                collect_non_loss_data,
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:  # Conventional interleaved 1F1B path
            forward_output_tensor = None
            backward_input_tensor_grad = None
            # forward pass
            if f_virtual_microbatch_id is not None:
                forward_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
                if pre_forward is not None:
                    pre_forward()
                forward_output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    forward_output_tensor = post_forward(forward_output_tensor)

            # Backward pass.
            if b_virtual_microbatch_id is not None:
                backward_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
                if pre_backward is not None:
                    pre_backward()
                backward_input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    backward_input_tensor_grad = post_backward(backward_input_tensor_grad)
            return forward_output_tensor, backward_input_tensor_grad

    # ==============================main logic=========================================
    _is_vp_first_stage = partial(
        is_vp_first_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    _is_vp_last_stage = partial(
        is_vp_last_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    pp_group = p2p_communicator.pp_group

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    input_tensors[0].append(
        p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if is_pp_first_stage(p2p_communicator.pp_group):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if is_pp_last_stage(p2p_communicator.pp_group):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)

        if config.overlap_p2p_comm_warmup_flush:
            if (
                not (
                    _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)
                )
                and k != 0
            ):
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(
            p2p_communicator.pp_group
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communicator.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Don't send tensor downstream if on last stage.
        if _is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communicator.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not is_pp_first_stage(p2p_communicator.pp_group):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=False, tensor_shape=tensor_shape, overlap_p2p_comm=True
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
            # isend() copies asynchronously; wait until the copy is done before
            # freeing the source buffer, otherwise the next PP stage gets corrupted data.
            if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
                send_next_wait_handle.wait()
                send_next_wait_handle = None

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])
    nvtx_range_pop(suffix="warmup")

    # Run 1F1B in steady state.
    nvtx_range_push(suffix="steady")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        if config.overlap_p2p_comm:

            backward_k = k

            # Sync forward recv
            def pp_pre_forward(vp_stage=None):
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_prev_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                            'should have registered recv handle'
                        )
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()
                    else:
                        if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                            recv_prev_wait_handle.wait()

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Async forward send / receive
            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                # Last virtual stage no activation tensor to send.
                if _is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group):
                    output_tensor = None

                recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                    forward_k, forward=True
                )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_next_wait_handle is not None:
                    send_next_wait_handle.wait()
                if fwd_wait_handles is not None:
                    send_next_wait_handle = (
                        fwd_wait_handles.pop("send_next")
                        if "send_next" in fwd_wait_handles
                        else None
                    )
                    if "recv_prev" in fwd_wait_handles:
                        recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
                # isend() copies asynchronously; wait until the copy is done before
                # freeing the source buffer, otherwise the next PP stage gets corrupted data.
                if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
                    send_next_wait_handle.wait()
                    send_next_wait_handle = None
                # assert fwd_wait_handles is not None

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(
                        fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                    )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Sync backward recv
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not (_is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_next_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                            'should have registered recv next handle'
                        )
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()
                    else:
                        if recv_next_wait_handles is not None and recv_next_wait_handles:
                            recv_next_wait_handle = recv_next_wait_handles.pop(0)
                            recv_next_wait_handle.wait()

            # Async backward send / receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                # First virtual stage no activation gradient tensor to send.
                if _is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
                return input_tensor_grad

            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                pre_forward=pp_pre_forward,
                pre_backward=pp_pre_backward,
                post_forward=pp_post_forward,
                post_backward=pp_post_backward,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        else:  # No p2p overlap.
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            if _is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if _is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communicator.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communicator.recv_backward(
                    tensor_shape,
                    is_last_stage=(
                        _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                    ),
                )
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            if (
                not (_is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group))
                and k != 0
            ):
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not is_pp_last_stage(
                p2p_communicator.pp_group
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not is_pp_last_stage(p2p_communicator.pp_group):
                    _, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communicator.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    nvtx_range_pop(suffix="cooldown")

    nvtx_range_push(suffix="misc")
    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).

        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()
    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    return forward_data_store


def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Determine tensor shapes for pipeline communication. #确定P2P通信张量shape（单个micro batch通信）

    Returns [()] for variable_seq_lengths mode (shapes exchanged dynamically),
    or computed shapes for fixed sequence length mode.
    """
    tensor_shapes = []

    if config.variable_seq_lengths: #如果变长序列，返回空list，意思是张量shape通过P2P通信获取
        # Shapes exchanged dynamically during P2P communication
        tensor_shapes.append(())
        return tensor_shapes

    # Fixed sequence lengths - compute shape #如果固定序列长度，返回计算好的shape
    effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
    effective_seq_length = effective_seq_length // cp_group.size() #首先CP切分

    if config.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_group.size() #然后SP切分

    tensor_shapes.append((effective_seq_length, micro_batch_size, config.hidden_size)) #添加张量shape
    return tensor_shapes


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None, #调整 p2p 通信的 tensor shape
    p2p_communicator: Optional[P2PCommunicator] = None, #负责 stage 间 P2P 通信
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None, #还可以是 MultiModuleProcessGroupCollection（多模块 VLM）
    force_all_reduce: Optional[bool] = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list): #非交错1F1B不会有多个模型切分
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm: #P2P 通信重叠，就是让P2P Communicator的通信结果返回通信handle，后面必要同步时再用handle进行wait，只在 interleaved 1F1B 中用。非交错调度中，P2P 通信是同步的。
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    tp_group, cp_group, cp_size = None, None, None

    # Determine if this is a multi-module pipeline
    # (used for validation and backward function selection)
    is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection) or isinstance(
        p2p_communicator, MultiModulePipelineCommunicator
    ) #判断是不是多模态模型

    if p2p_communicator is None and pg_collection is None: #如果两个都没有提供，那就创建默认的
        # Default: single-module with parallel_state groups
        p2p_communicator = P2PCommunicator( #用pp组构建p2p_communicator
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection() #创建默认的pg_collection
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"

        if is_multimodule:
            # Multi-module: use language model's CP size for loss scaling
            if not config.variable_seq_lengths:
                raise ValueError(
                    "config.variable_seq_lengths=True required for multi-module pipelines"
                )
            if pg_collection.has_language_model():
                cp_size = pg_collection.get_language_model_cp_size()
            else:
                # Encoder-only ranks should not use CP loss scaling.
                cp_size = None

        elif isinstance(pg_collection, ProcessGroupCollection):
            # Single-module: extract tp/cp groups and cp_size
            assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
            assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()

        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection or "
                f"MultiModuleProcessGroupCollection, got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only: #对于延迟LM Head的wgrad计算，这里把用于保存激活值的buffer清空
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func #指定no_sync_func，用于控制DDP梯度通信的触发时机（overlap场景，overlap_grad_reduce=True）
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions""" #禁止异步DDP梯度通信
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions""" #启用异步DDP梯度通信
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync() #先禁止异步DDP梯度通信

    # Compute number of warmup microbatches.
    num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1  #warmup阶段的microbatch数量，p2p_communicator.current_stage从0开始编号
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches) #确保warmup阶段的microbatch数量不超过总microbatch数量
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches #剩余的microbatch数量（非warmup阶段）

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf #出自论文 Reducing Activation Recomputation 附录 C
    max_outstanding_backprops = None #每个stage为指定一些micro batch进行部分重计算所使用的窗口，窗口大小等于warmup阶段的microbatch数量+1（就是in-flight micro batch数量+1），越靠后的stage窗口越小
    if config.num_microbatches_with_partial_activation_checkpoints is not None: #num_microbatches_with_partial_activation_checkpoints确定使用哪些micro batch进行部分重计算，i%W < T的micro batch进行部分重计算（T就是num_microbatches_with_partial_activation_checkpoints，W就是max_outstanding_backprops），其余的micro batch全部重计算。越靠前的stage有更多的micro batch进行完全重计算来压显存占用
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Select backward function based on whether multi-module or single-module
    if is_multimodule: #确定使用哪种后向传播流程
        backward_func = partial(
            backward_step_multimodule,
            language_model_module_name=pg_collection.language_model_module_name,
        )
    else:
        backward_func = backward_step

    recv_tensor_shapes = get_tensor_shapes( #确定recv张量shape
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes( #确定send张量shape
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only: #input_tensors用于保存输入该stage的激活值，output_tensors用于保存该stage输出的激活值
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches): #开始执行warmup阶段的forward passes
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = ( #判断是否进行完全重计算
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward( #接受前序stage发送过来的张量
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        output_tensor, num_tokens = forward_step( #执行单个micro batch的前传
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage) #发送前向结果张量到后序stage
        total_num_tokens += num_tokens #累加tokens数量

        if not forward_only:
            input_tensors.append(input_tensor) #保存该stage接受到的输入张量
            output_tensors.append(output_tensor) #保存该stage输出的激活值
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs) #把output tensor的.data换为(1,)，相当于清空output tensor激活值的显存空间（反正反传也不需要这个激活值，只要保留空壳在后面反传能挂上grad_fn就行）
    
    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0: #准备进入steady阶段，这里先接收steady阶段第一个micro batch的输入张量
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining): #开始执行steady阶段，遵循1F1B
        last_iteration = i == (num_microbatches_remaining - 1) #判断是否是steady阶段的最后一个micro batch

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = ( #判断是否进行完全重计算
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step( #完成单个micro batch的前传，并得到该stage的输出张量和该micro batch处理的tokens数量
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor, #提前接收好了上游stage传递的输入激活值
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens #累加tokens数量

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else: #训练走这里，反传走这里
            output_tensor_grad = p2p_communicator.send_forward_recv_backward( #和下一个stage的通信，把计算出的激活值发给下一个stage，并从下一个stage获取反传梯度
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor) #保存该stage接受到的输入张量
            output_tensors.append(output_tensor) #保存该stage输出的激活值
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs) #把output tensor的.data换为(1,)，相当于清空output tensor激活值的显存空间（反正反传也不需要这个激活值，只要保留空壳在后面反传能挂上grad_fn就行）

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0) #弹出该stage接受到的输入张量，因为处理micro batch的顺序是固定的，最早进行前传的micro batch，也是最早进行反传的micro batch，所以直接出队列
            output_tensor = output_tensors.pop(0) #弹出该stage输出的激活值

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration: #如果是最后一个stage且已经处理完除最后一个micro batch外的所有micro batch的反传，那就开启DDP的梯度通信。其他stage不在这里开启，在cool down阶段开启。
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage: #grad_sync_func在align_grad_reduce才会被设置（align_grad_reduce控制的是多个 PP stage 的梯度归约通信要不要"对齐时刻"一起发起），一般不用。
                    enable_grad_sync() #开启DDP梯度通信就绪跟踪

            input_tensor_grad = backward_func( #完成单个micro batch的反传，并得到输入激活值梯度
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if last_iteration: #如果是steady阶段的最后一个micro batch，那就把input_tensor_grad发给前序stage，完成反传
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:#如果不是最后一个micro batch，那就和前序stage两个通信操作，把input_tensor_grad发给前序stage，并接收前序stage发过来的输入激活值，为后续前传做准备
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

    # Run cooldown backward passes.
    if not forward_only: #训练走这里，就执行cool down阶段的反传
        for i in range(num_warmup_microbatches): #cool down阶段需要反传的micro batch数量是和warmup阶段对应的（因为warmup阶段没做的反传在这里做）

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1: #如果是cool down阶段的最后一个micro batch，就开启DDP的梯度通信就绪跟踪
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage: #grad_sync_func在align_grad_reduce才会被设置（align_grad_reduce控制的是多个 PP stage 的梯度归约通信要不要"对齐时刻"一起发起），一般不用。如果使用align，这里对first stage也会提前开启DDP梯度通信就绪跟踪，因为first stage的DDP梯度通信是关键路径。
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0) #弹出该stage接受到的输入张量
            output_tensor = output_tensors.pop(0) #弹出该stage输出的激活值

            output_tensor_grad = p2p_communicator.recv_backward( #从后序stage接收输出激活值梯度
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad = backward_func( #完成单个micro batch的反传，并得到输入激活值梯度
                input_tensor, output_tensor, output_tensor_grad, config
            )

            p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage) #把input_tensor_grad发给前序stage，为前序stage的反传做准备

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync() #开启DDP梯度通信就绪跟踪（不过这里不会再进行反传了）
            if config.grad_sync_func is not None: #如果align_grad_reduce被设置为True，此时每个stage会一起发起DDP梯度归约通信
                config.grad_sync_func(model.parameters()) #手动调用梯度归约通信

    if config.finalize_model_grads_func is not None and not forward_only: #如果需要梯度收尾

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute( #先补齐LM Head梯度计算（如果defer_embedding_wgrad_compute被设置为True的话，否则不需要）
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func( #最后进行梯度收尾
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()

    return forward_data_store
