# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
from typing import Callable, List, Optional, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

try:
    from torch.distributed._tensor import DTensor, distribute_tensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core.pipeline_parallel.utils import (
    get_pp_last_rank,
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from .. import parallel_state
from ..transformer.moe.moe_utils import get_updated_expert_bias
from ..transformer.transformer_config import TransformerConfig
from ..utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
)


def _get_main_grad_attr(param: torch.nn.Parameter):
    if hasattr(param, "main_grad"):
        return "main_grad"
    return "grad"


def _unshard_if_dtensor(tensor: Union[torch.Tensor, "DTensor"]) -> torch.Tensor:
    """
    Unshards the input tensor if it is a DTensor and otherwise returns the
    tensor unmodified.

    Args:
        tensor (Union[torch.Tensor, DTensor]): The tensor to potentially unshard.

    Returns:
        An unsharded version of the input tensor if it is a DTensor, or the
        input tensor unmodified if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        unsharded_tensor = tensor.full_tensor()
        for k, v in vars(tensor).items():
            setattr(unsharded_tensor, k, v)
        return unsharded_tensor
    return tensor


def _reshard_if_dtensor(
    tensor_to_shard: torch.Tensor, reference_tensor: Union[torch.Tensor, "DTensor"]
) -> Union[torch.Tensor, "DTensor"]:
    """
    Reshards the input tensor to match the sharding configuration of the
    reference tensor if the reference tensor is a DTensor. Otherwise, returns
    the reference tensor unmodified.

    Args:
        tensor_to_shard (torch.Tensor): The tensor to be potentially sharded.
        reference_tensor (Union[torch.Tensor, DTensor]): The reference tensor
            for the sharding configuration.

    Returns:
        Union[torch.Tensor, DTensor]: The sharded tensor matching the reference tensor's
        configuration, or the reference tensor itself if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(reference_tensor, DTensor):
        sharded_tensor = distribute_tensor(
            tensor_to_shard,
            device_mesh=reference_tensor.device_mesh,
            placements=reference_tensor.placements,
        )
        for k, v in vars(reference_tensor).items():
            setattr(sharded_tensor, k, v)
        return sharded_tensor
    return reference_tensor


def _allreduce_conditional_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    All-reduce conditional embedding grads.

    Reduce grads across all the pp stages to ensure that parameters of the conditional embedders
    (e.g., timestep embedder, FPS embedder, label embedder) stay in sync.
    This is for the models with replicated embedders on each PP / VPP rank, like diffusion models.
    """
    if pp_group is None:
        pp_group = parallel_state.get_pipeline_model_parallel_group()

    if pp_group.size() > 1 and getattr(config, "has_cond_embedder", False):
        grads_dict = {}
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if param.requires_grad and getattr(param, 'pipeline_parallel', False):
                    grad = param.main_grad
                    if name in grads_dict:
                        # Add all the virtual PP rank's gradients to
                        # the first local virtual PP rank.
                        grads_dict[name][0].add_(grad)
                        # Append to the end for later update after cross-rank reduce.
                        grads_dict[name].append(grad)
                    else:
                        grads_dict[name] = [grad]
        if grads_dict:
            # All-reduce the gradient on the first VPP rank.
            grads = [param_grad[0] for _, param_grad in grads_dict.items()]
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=pp_group)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

            # Update the gradients on other VPP ranks.
            for grads in grads_dict.values():
                for grad in grads[1:]:
                    grad.copy_(grads[0])


def _get_shared_word_embedding_weight(
    model_module: torch.nn.Module, config: TransformerConfig
) -> Optional[torch.nn.Parameter]:
    """Return the shared word-embedding weight if it is duplicated across stages.

    Args:
        model_module: The model module from which to extract the
            word-embedding weight.
        config: Transformer config.

    Returns:
        The shared embedding or output weight if available; otherwise ``None``.
    """
    # Only reduce if weights are duplicated across stages. #判断：权重是否真的跨 stage 共享？
    if model_module.share_embeddings_and_output_weights or getattr(config, 'mtp_num_layers', 0): #是 → 返回那个共享的参数矩阵
        return model_module.shared_embedding_or_output_weight()
    return None #否 → 返回 None


def _get_position_embedding_weight(model_module: torch.nn.Module) -> torch.nn.Parameter:
    """Return the position-embedding weight tensor from the given model module.

    Args:
        model_module: The model module that owns the
            position-embedding parameter.

    Returns:
        The position-embedding weight tensor.
    """
    return getattr(model_module, 'position_embeddings').weight  # type: ignore[attr-defined]


def _allreduce_word_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    embd_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """All-reduce word-embedding gradients across the first and last PP stages.

    This ensures that the ``word_embeddings`` parameters stay in sync when they
    are shared between the input and output layers.

    Args:
        model: A list containing the pipeline chunks
            that constitute the model on the current rank (including any
            virtual pipeline chunks).
        config: Transformer configuration. Used for edge
            cases like MTP where embeddings might be shared differently.
        embd_group: The process
            group over which to all-reduce the word-embedding gradients. If
            ``None``, it will be looked up based on the current pipeline model
            parallel group.
        pp_group: The pipeline
            parallel process group used to identify first/last stages. If
            ``None``, it will be looked up.
    """
    if embd_group is None: #确定embedding组，即所有有共享嵌入参数的pp stage
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        if get_pg_size(embd_group) > 1:
            assert pp_group is None
            pp_group = parallel_state.get_pipeline_model_parallel_group()

    _allreduce_embedding_grad( #进行通信
        model,
        embd_group,
        pp_group,
        partial(_get_shared_word_embedding_weight, config=config), #指定word embedding的参数权重的获取函数
        config=config,
    )


def _allreduce_embedding_grad(
    model: List[torch.nn.Module],
    embd_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
    weight_getter: Callable[[torch.nn.Module], Optional[torch.nn.Parameter]],
    skip_if_none: bool = True,
    config: TransformerConfig = None,
):
    """Unified helper to all-reduce embedding parameters across pipeline stages.

    Args:
        model (List[torch.nn.Module]): A list of model chunks (PP/VPP).
        embd_group (torch.distributed.ProcessGroup): The process group over which to reduce.
        pp_group (torch.distributed.ProcessGroup): The pipeline parallel process group for
            first/last stage detection.
        weight_getter (Callable[[torch.nn.Module], Optional[torch.nn.Parameter]]): A function
            that takes the *pre-process* model chunk and returns the parameter to be reduced
            (or ``None`` if not applicable).
        skip_if_none (bool, optional): If True, quietly returns when the parameter or its
            gradient is ``None``. Defaults to True.
    """

    if (
        # embd_group can be None in cases there is no embd_group
        # get_pg_size(embd_group) will return 1 and the all-reduce will be skipped.
        get_pg_size(embd_group) > 1
        and torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(embd_group)
    ):

        if is_pp_first_stage(pp_group):
            model_module = model[0]
        elif is_pp_last_stage(pp_group):
            model_module = model[-1]
        elif getattr(config, 'mtp_num_layers', None) is not None and config.mtp_num_layers > 0:
            # Embedding for MTP layers is in the last virtual pipeline model parallel stage.
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        ddp_config = model_module.ddp_config
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)

        weight = weight_getter(model_module)
        if weight is None and skip_if_none:
            return

        grad_attr = _get_main_grad_attr(weight)
        orig_grad = getattr(weight, grad_attr)
        if ddp_config.use_megatron_fsdp:
            orig_grad = orig_grad._local_tensor if orig_grad is not None else None
        grad = _unshard_if_dtensor(orig_grad)
        # When the embedding is frozen, the grad is None.
        if grad is None and skip_if_none:
            return
        torch.distributed.all_reduce(grad, group=embd_group)
        setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_position_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    pos_emb_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
):
    """
    All-reduce position_embeddings grad across encoder and decoder stages to ensure that position
    embeddings parameters stay in sync.
    """

    _allreduce_embedding_grad(
        model, pos_emb_group, pp_group, _get_position_embedding_weight, skip_if_none=False
    )


def reset_model_temporary_tensors(config: TransformerConfig, model: List[torch.nn.Module]):
    """
    Reset the temporary tensors of the model.
    """
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, 'modules')():
            if config.moe_router_enable_expert_bias and hasattr(module, 'expert_bias'):
                module.local_tokens_per_expert.zero_()
            if (
                config.moe_router_load_balancing_type == "global_aux_loss"
                or "global_aux_loss" in config.moe_router_load_balancing_type
            ) and hasattr(module, 'reset_global_aux_loss_tracker'):
                module.reset_global_aux_loss_tracker()


def _update_router_expert_bias(model: List[torch.nn.Module], config: TransformerConfig):
    """
    Update the expert bias of the router for a global batch.
    This requires all-reduce of local_tokens_per_expert across TPxCPxDP ranks
    """
    tokens_per_expert_list = []
    expert_bias_list = []
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, 'modules')():
            # Only update expert_bias if this module is in the training mode. There are special
            # cases where only the student is in training mode but the teacher is in eval mode
            # when using online knoweldge-distillation with Model-Optimizer. In this case, we want
            # to avoid updating teacher's expert_bias.
            if hasattr(module, 'expert_bias') and module.training:
                tokens_per_expert_list.append(module.local_tokens_per_expert)
                expert_bias_list.append(module.expert_bias)
    # For hybrid models with both MoE and Dense layers, this list can be empty.
    if len(expert_bias_list) == 0:
        return
    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)
    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert, stacked_expert_bias, config.moe_router_bias_update_rate
    )

    for expert_bias, updated_expert_bias in zip(expert_bias_list, stacked_updated_expert_bias):
        expert_bias.copy_(updated_expert_bias)


def _allreduce_non_tensor_model_parallel_grads(  # 跨 tp_group 归约「非 TP 模块」的梯度
    model: List[torch.nn.Module],  # 模型 chunk 列表（VPP 下多个）
    config: TransformerConfig,  # Transformer 配置
    tp_group: Optional[torch.distributed.ProcessGroup] = None,  # TP 进程组（不传则用默认）
):
    """
    All-reduce both layernorm grads (for sequence parallelism) and
    gradients from modules with average_gradients_across_tp_domain=True
    across tensor-model-parallel ranks.
    """
    tp_group = get_tensor_model_parallel_group_if_none(tp_group)  # 未传 tp_group 时取默认 TP 组
    if tp_group.size() <= 1:  # TP 规模为 1（无 TP 并行）时无需归约
        return

    params_sum = []  # SUM 类参数列表（SP / qk_layernorm，梯度是部分和）
    grads_sum = []  # SUM 类梯度列表
    params_avg = []  # AVG 类参数列表（average_gradients_across_tp_domain，冗余梯度取平均）
    grads_avg = []  # AVG 类梯度列表

    for model_chunk in model:  # 遍历每个模型 chunk
        ddp_config = model_chunk.ddp_config  # 该 chunk 的 DDP 配置
        for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():  # 穿透模型包装遍历所有参数
            if param.requires_grad:  # 只处理需要梯度的参数
                # Check if this param needs average reduction (average_gradients_across_tp_domain)
                if getattr(param, "average_gradients_across_tp_domain", False):  # AVG 类：HF 包装模型打的标记
                    grad_attr = _get_main_grad_attr(param)  # 获取该参数存梯度的属性名（通常是 main_grad）
                    grad = getattr(param, grad_attr)  # 取出该参数的梯度
                    if grad is None:  # 本 step 无梯度（如冻结/未参与计算）则跳过
                        continue
                    params_avg.append(param)  # 记录参数到 AVG 列表
                    if ddp_config.use_megatron_fsdp:  # FSDP 模式下
                        grads_avg.append(grad._local_tensor.data)  # 取本地分片数据参与归约
                    else:  # 非 FSDP 模式
                        grad = _unshard_if_dtensor(grad)  # DTensor 先展开成完整张量（否则归约只覆盖本地分片）
                        grads_avg.append(grad.data)  # 收集梯度数据
                # Check if this param needs sum reduction (sequence parallel or qk_layernorm)
                elif (config.sequence_parallel and getattr(param, "sequence_parallel", False)) or (  # SUM 类：SP 下复制层（序列被切分）
                    config.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name)  # 或 qk_layernorm（head 被 TP 切分）
                ):
                    grad_attr = _get_main_grad_attr(param)  # 获取该参数存梯度的属性名
                    grad = getattr(param, grad_attr)  # 取出该参数的梯度
                    if grad is None:  # 本 step 无梯度则跳过
                        continue
                    params_sum.append(param)  # 记录参数到 SUM 列表
                    if ddp_config.use_megatron_fsdp:  # FSDP 模式下
                        grads_sum.append(grad._local_tensor.data)  # 取本地分片数据参与归约
                    else:  # 非 FSDP 模式
                        grad = _unshard_if_dtensor(grad)  # DTensor 先展开成完整张量
                        grads_sum.append(grad.data)  # 收集梯度数据

    # Loop grads and perform correct all-reduce
    for params, grads, all_reduce_op in zip(  # 依次处理 SUM 组与 AVG 组
        [params_sum, params_avg],  # 两组参数列表
        [grads_sum, grads_avg],  # 两组梯度列表
        [torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.AVG],  # SUM（部分和求和）/ AVG（冗余取平均）
    ):
        if grads:  # 该组收集到梯度才通信
            coalesced = _flatten_dense_tensors(grads)  # 多个小梯度拼成一个连续大张量（一次通信替代 N 次小通信）
            torch.distributed.all_reduce(coalesced, op=all_reduce_op, group=tp_group)  # 跨 TP rank 做一次 SUM/AVG all-reduce
            for param, buf, synced in zip(  # 逐个参数写回归约结果
                params, grads, _unflatten_dense_tensors(coalesced, grads)  # 按原梯度形状把大张量拆回
            ):
                buf.copy_(synced)  # 归约结果写回该参数的梯度 buffer（buf 即 main_grad 的引用）
                grad_attr = _get_main_grad_attr(param)  # 获取 main_grad 属性名
                orig_grad = getattr(param, grad_attr)  # 取回原梯度对象
                if ddp_config.use_megatron_fsdp:  # FSDP 模式下
                    setattr(param, grad_attr, orig_grad)  # 原样放回（copy_ 已生效，无需换对象）
                else:  # 非 FSDP 模式
                    setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))  # 完整张量按原 DTensor 布局重新 shard 放回


"""
This is an alias to _allreduce_non_tensor_model_parallel_grads that we must
maintain for legacy tests. We can remove this proxy in mcore 0.14.
"""
_allreduce_layernorm_grads = _allreduce_non_tensor_model_parallel_grads


def finalize_model_grads(
    model: List[torch.nn.Module],
    num_tokens: Optional[torch.Tensor] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])
    if pg_collection is not None:
        assert hasattr(pg_collection, 'tp')
        assert hasattr(pg_collection, 'pp')
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group."
            " If you are using the default process group, please use"
            " `parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group."
            " If you are using the default process group, please use "
            " `parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'dp_cp')
        tp_group = pg_collection.tp
        pp_group = pg_collection.pp
        embd_group = pg_collection.embd
        pos_emb_group = pg_collection.pos_embd
        dp_cp_group = pg_collection.dp_cp
    else:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model: #DDP 通信，overlap 模式下这是「等待最后一次 backward 异步发起的 reduce」；非 overlap 模式下这是「同步发起并完成唯一一次 reduce」。
        model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce t_embedder grads (for pp & vpp of DiT). #DiT相关
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_conditional_embedding_grads(model, config, pp_group)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # All-reduce layer-norm grads (for sequence parallelism) and non-tensor parallel modules.
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_non_tensor_model_parallel_grads(model, config, tp_group) #进行非TP部分的all-reduce（使用SP的layernrom，qk_layrernorm这些）
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_word_embedding_grads(model, config, embd_group, pp_group) #完成word embedding部分的all-reduce，如果有pp，首尾stage都只有一半的grad，需要allreduce
    _allreduce_position_embedding_grads(model, config, pos_emb_group, pp_group) #完成position embedding部分的all-reduce，只有T5老模型会这样做，现在基本都不用绝对位置编码

    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    if config.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)

    reset_model_temporary_tensors(config, model) #重置模型临时张量

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None: #per-token loss模式的梯度归一化操作

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        assert not isinstance(pp_group, list)
        last_rank = get_pp_last_rank(pp_group)
        torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group) #num_tokens 只在末 stage 算出来（loss 在末 stage），但每个 pp stage 的 rank 都要对自己的本地梯度做除法，所以先从末 rank 广播给整个 pp 组。

        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=dp_cp_group) #dp_cp 求和：per-token 的分母必须是全局所有 replica（含 CP 分片）的 token 总和，跨 dp_cp_group 求和得到
        for model_chunk in model: #先纯 SUM 累加、最后统一除法
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)


def finalize_colocated_encoder_grads(
    model: List[torch.nn.Module],
    encoder_inner_dp_group: torch.distributed.ProcessGroup,
    encoder_inner_dp_last_rank: int,
    dp_cp_group: torch.distributed.ProcessGroup,
    num_tokens: Optional[torch.Tensor] = None,
):
    """Finalize the colocated encoder's grads: outer-DP reduce, inner-DP SUM, token scaling.

    共置 encoder 的梯度收尾（Task 4.6/5，与 ``finalize_model_grads`` 并列的**独立函数**，
    不是它的分支）。为什么不复用 ``finalize_model_grads``：那个函数的其余四段都建立在
    "模型被 PP 切开"的前提上——word/position embedding 归约要 ``embd_group``/``pp_group``
    且只对首尾 stage 有意义、非 TP 参数归约要 backbone 的 TP 组——而共置 encoder 在**每个
    rank 上都是完整副本、没有 PP**，这些段落对它全部不适用。加一个 colocated 标志位会让
    这四段各自再长一个分支，反而更难读。

    共置 encoder 的梯度有**两个数据并行维度**：
    - **outer（跨副本，复用 ``dp_group``）**：不同副本处理不同的 global batch 分片 →
      标准 DDP 负责（``finish_grad_sync``，与 backbone 用完全相同的 DDP 配置）；
    - **inner（副本内 P 个 rank，``encoder_inner_dp_group``）**：轮盘让这 P 个 rank 处理
      **同一副本的不同 microbatch**（producer p 拿 p, p+P, ...）→ 必须 **SUM**，不是 AVG。

    为什么 inner 是 SUM：这 P 个 rank 的本地梯度是**同一份 loss 在不同 microbatch 上的
    分量**，合起来才是完整的一份，与"单 rank 顺序跑完 n 个 microbatch 做梯度累积"完全
    等价（inner 是**跨 rank 的梯度累积**，不是重复计算的平均）。若取 AVG 会小 P 倍。
    与 MoE 的 expert-DP 分层同理：expert 参数在 edp 组内也处理不同 token，Megatron 用
    ``expert_gradient_scaling_factor = edp_size/dp_size`` + 组内 AVERAGE，净效果同样是
    "组内 SUM + 全局归一化"（distributed_data_parallel.py:196-203）。

    **per-token 归一化（``calculate_per_token_loss=True``）**：encoder 自己不算 loss，
    "per-token" 对它只意味着**分母**——它收到的边界梯度是"未归一化的 loss 之和"对
    image_embeddings 的导数（per-token 模式下 ``forward_step_calc_loss`` 跳过本地除法，
    schedules.py:270），DDP 也不做缩放（``gradient_scaling_factor=1.0``，纯 SUM，
    distributed_data_parallel.py:169-174），所以 encoder 参数梯度同样是未归一化的和，
    必须除以**与 backbone 完全相同的全局 token 数**。
    token 数的两级规约与梯度对称：``num_tokens`` 只在 backbone 的 pp 末 stage 统计出来
    （它是 ``loss_func`` 返回的三元组之一，中间 stage 不算 loss，schedules.py:262-269），
    而 inner DP 组的成员恰好就是 pp 组成员（doc §2.1）——所以先在 inner 组内从末 stage
    ``broadcast`` 得到本副本的 token 数，再对 ``dp_cp_group`` 做一次 SUM 得到全局 token 数
    （与 ``finalize_model_grads`` 的 broadcast + all_reduce 逐句对应，:494/:497）。
    这里**不用 inner 组的 all_reduce(SUM) 代替 broadcast**：那样做依赖"非末 stage 的
    ``num_tokens`` 恒为 0"这条**上游实现行为**（rebase 后若改成每个 stage 都上报，会静默
    变成 P 倍）；broadcast 只依赖"inner 组的 rank 列表按 pp stage 有序"这条**本项目自己
    建组时的约定**，更可控，语义也直接说出了"值只存在于末 stage"。
    传入的 ``num_tokens`` 会被 ``clone()`` 后再规约，**不修改调用方的张量**——backbone 的
    ``finalize_model_grads`` 会就地把同一个张量改成全局值，两边互不干扰。


    顺序：outer → inner → token 归一化。前两者都是线性运算、数学上可交换，但
    ``overlap_grad_reduce`` 下 DDP 的桶归约在最后一个 microbatch 的反传中就已异步发起，
    ``finish_grad_sync`` 是"等它完成"，所以先 outer 才不会与在飞的桶通信抢同一片
    ``grad_data``；归一化必须在两级求和都完成之后。

    TODO（Task 5）：encoder 若要支持 TP/SP，非 TP 参数（layernorm 等）的组内归约要加在
    这里——对应 ``finalize_model_grads`` 的 ``_allreduce_non_tensor_model_parallel_grads``，
    但通信组要用 encoder 自己的 TP 组。当前最小实现假定 encoder TP=1。
    """
    config = get_model_config(model[0])
    # num_tokens 与 loss 归一化模式必须一致（与 finalize_model_grads 的调用约定相同：
    # per-token 模式才传 num_tokens）。
    # num_tokens must match the loss normalization mode (same convention as
    # finalize_model_grads: only per-token loss passes num_tokens).
    assert (num_tokens is not None) == config.calculate_per_token_loss, (
        f"num_tokens (given: {num_tokens is not None}) must be provided exactly when "
        f"calculate_per_token_loss is set (currently {config.calculate_per_token_loss})"
    )

    # ① outer DP：标准 DDP 归约（等最后一个 microbatch 反传时发起的桶通信完成）。
    # ① outer DP: the standard DDP reduce over dp_group.
    for model_chunk in model:
        model_chunk.finish_grad_sync()

    # ② inner DP：副本内 P 个 rank 的梯度**求和**（跨 rank 的梯度累积，见 docstring）。
    # 直接对 DDP 的扁平梯度 buffer 做一次 all_reduce（每个 buffer 一次通信，不逐参数）。
    # ② inner DP: SUM the grads of the P ranks inside one replica (cross-rank gradient
    # accumulation) — one all_reduce per flat DDP grad buffer.
    for model_chunk in model:
        assert not model_chunk.ddp_config.use_distributed_optimizer, (
            "colocated encoder grad finalize assumes use_distributed_optimizer=False "
            "(doc §2.9): with reduce-scatter the flat grad buffer only holds a shard"
        )
        for buffer in model_chunk.buffers:
            torch.distributed.all_reduce(
                buffer.grad_data,
                op=torch.distributed.ReduceOp.SUM,
                group=encoder_inner_dp_group,
            )

    # ③ per-token 归一化：inner 广播（token 数只在 pp 末 stage 有）+ dp_cp 求和 → 全局
    # token 数，再统一除（与 backbone 同一个分母）。
    # ③ per-token normalization: broadcast inside the replica (the count only exists on the
    # pp last stage) + SUM across dp_cp gives the global token count — the same divisor as the
    # backbone — then scale once.
    if num_tokens is not None:
        num_tokens = num_tokens.clone()  # 不改调用方的张量 / do not mutate the caller's tensor
        torch.distributed.broadcast(
            num_tokens, src=encoder_inner_dp_last_rank, group=encoder_inner_dp_group
        )
        torch.distributed.all_reduce(
            num_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group
        )
        if num_tokens > 0:
            for model_chunk in model:
                model_chunk.scale_gradients(1.0 / num_tokens)
