# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import copy
import logging
import warnings
from collections import defaultdict
from dataclasses import astuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import SGD as CPUSGD
from torch.optim import AdamW as CPUAdam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD

    USING_PYTORCH_OPTIMIZER = False
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD

        USING_PYTORCH_OPTIMIZER = False
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import SGD
        from torch.optim import AdamW as Adam

        USING_PYTORCH_OPTIMIZER = True

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    _eo_ver = tuple(int(x) for x in _pkg_version('emerging-optimizers').split('.')[:2])
except (ImportError, PackageNotFoundError):
    _eo_ver = (0, 0)

HAVE_EMERGING_OPTIMIZERS = _eo_ver >= (0, 2)

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.scalar_optimizers import Lion

from megatron.core import parallel_state
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer_param_scheduler import (
    ParamGroupOverride,
    combine_param_group_overrides,
    param_group_override_to_tuple,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name

from ..distributed.param_and_grad_buffer import _ParamAndGradBuffer
from ..transformer.module import MegatronModule
from ..utils import get_model_config, get_pg_rank, get_pg_size, is_te_min_version, log_single_rank
from .distrib_optimizer import DistributedOptimizer
from .emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    HAVE_EMERGING_OPTIMIZERS,
    _create_emerging_optimizer,
)
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
    param_group_identifier_keys,
)

# Subclass aliases kept for backward compatibility; all are OptimizerConfig.
from .optimizer_config import (
    AdamOptimizerConfig,
    OptimizerConfig,
    ParamKey,
    ParamPredicate,
    ParamWithNamePredicate,
    SGDOptimizerConfig,
)

logger = logging.getLogger(__name__)


def get_standard_config_overrides(config: OptimizerConfig) -> Dict[ParamKey, ParamGroupOverride]:
    """Get standard config overrides for the optimizer, handling decoupled LR and common wd skips.

    Args:
        config (OptimizerConfig): optimizer configuration object.

    Returns:
        Dict[ParamKey, ParamGroupOverride]: standard config overrides.
    """
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = {} #一个映射表，Key = ParamKey：参数匹配规则，Value = ParamGroupOverride：对命中的参数应用什么覆盖
    # First, figure out how we are going to do wd skipping. The two main approaches are:
    #  1. The classic megatron approach of skipping all len 1 and bias parameters.
    #  2. The Qwen3-Next approach of doing 1, other than qk layernorm parameters.
    if config.apply_wd_to_qk_layernorm:
        shape_1_not_qkln_param = ParamWithNamePredicate(
            name="s1_not_qkln",
            fn=lambda param, name: (len(param.shape) == 1 or name.endswith(".bias"))
            and not ("q_layernorm." in name or "k_layernorm." in name),
        ) #shape=1 或 bias 的参数 → 跳过 wd，但排除 QK LayerNorm（因为这些参数即使 shape=1 也要施加 wd）。
        param_wd_mult_key = ParamKey(with_name_predicate=shape_1_not_qkln_param)
    else:
        param_length_1_match = ParamPredicate(#ParamPredicate是对匹配函数fn的包装
            name="param_len_1", fn=lambda param: len(param.shape) == 1
        ) #shape=1 的参数 + 所有 bias → 跳过 wd。QK LayerNorm 的 1 维参数也跳过。
        param_wd_mult_key = ParamKey(name="*.bias", predicate=param_length_1_match) #ParamKey 是参数分组 key，包含三种匹配方式，任意一种匹配即命中（OR 关系）：name — 按参数名做 fnmatch 模式匹配（"*.bias" 匹配任何以 .bias 结尾的参数名）；attr — 按参数属性匹配（此处未使用）；predicate — 按自定义 predicate 匹配（上面定义的 1 维参数）

    config_overrides[param_wd_mult_key] = ParamGroupOverride(wd_mult=0.0) #把所有ParamKey命中参数的 weight decay 乘数设为 0，即对这些参数不做 weight decay。

    if config.decoupled_lr is not None:
        decoupled_lr_config: ParamGroupOverride = {"max_lr": config.decoupled_lr} #对选定的参数设置解耦的lr
        decoupled_param_key = ParamKey(attr="is_embedding_or_output_parameter") #筛选 is_embedding_or_output_parameter=True 的参数
        if config.decoupled_min_lr is not None: #如果 decoupled_min_lr 也设置了，一并覆盖 min_lr
            decoupled_lr_config["min_lr"] = config.decoupled_min_lr
        config_overrides[decoupled_param_key] = decoupled_lr_config

    return config_overrides


def get_mup_config_overrides(
    config: OptimizerConfig, mup_width_mult: float, optimizer_type: str = 'adam'
) -> Dict[ParamKey, ParamGroupOverride]:
    """Get MuP config overrides for per-layer LR and Adam epsilon scaling.

    In MuP, optimizer learning rates are adjusted by parameter class to ensure
    stable update scales across model widths and enable hyperparameter transfer.

    MuP optimizer scaling rules (as implemented here):
    - Adam/AdamW:
      - hidden (matrix-like) lr = base_lr / width_mult
      - hidden (matrix-like) eps = base_eps / width_mult
      - vector-like params keep base lr and eps
    - SGD:
      - vector-like lr = base_lr * width_mult
      - hidden (matrix-like) lr keeps base_lr in the current uniform-width setup
      - no eps override is applied
    - Non-Adam optimizers:
      - hidden (matrix-like) lr = base_lr / width_mult
      - no eps override is applied.
      - for Muon optimizers, matrix-like params managed by Muon itself are
        excluded from these Adam-style MuP overrides.

    With decoupled_lr enabled, embedding/output params continue using decoupled LR
    and MuP will not override those explicit decoupled values.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        mup_width_mult (float): Width multiplier (hidden_size / base_hidden_size).
        optimizer_type (str): Optimizer type string from config.optimizer.

    Returns:
        Dict[ParamKey, ParamGroupOverride]: MuP optimizer overrides.
    """
    optimizer_type_lower = optimizer_type.lower()
    is_sgd_optimizer = optimizer_type_lower == 'sgd'
    is_adam_optimizer = 'adam' in optimizer_type_lower
    is_muon_optimizer = 'muon' in optimizer_type_lower

    decoupled_lr_enabled = config.decoupled_lr is not None
    if decoupled_lr_enabled:
        message = (
            "Both decoupled_lr and MuP LR scaling are enabled. decoupled_lr sets an "
            "absolute LR for embedding+output params, and MuP LR scaling will not "
            "override those parameters."
        )
        if is_adam_optimizer:
            message += " MuP Adam epsilon scaling remains applied to hidden matrix-like parameters."
        log_single_rank(logger, logging.WARNING, message)

    if is_muon_optimizer:
        muon_scale_mode = getattr(config, 'muon_scale_mode', 'spectral')
        if muon_scale_mode == 'spectral':
            log_single_rank(
                logger,
                logging.WARNING,
                "Both MuP and muon_scale_mode=spectral are enabled. "
                "Muon-managed matrix parameters will continue using spectral Muon scaling. "
                "Set --muon-scale-mode unit_rms_norm to use unit_rms_norm scaling for "
                "Muon-managed matrices with MuP.",
            )

    if mup_width_mult == 1.0:
        # No scaling needed when width_mult is 1
        return {}

    hidden_lr_mult = 1.0 / mup_width_mult
    base_lr = config.lr
    base_min_lr = config.min_lr

    # Hidden matrix-like layers get scaled LR/eps; vector-like params keep base values.
    # Prefer the explicit parameter attribute set by LanguageModule. Fall back to
    # a conservative name check for older or non-language modules.
    def is_embedding_parameter(param: torch.nn.Parameter, param_name: str) -> bool:
        if getattr(param, 'shared_embedding', False):
            return True
        if hasattr(param, 'is_embedding_parameter'):
            return bool(param.is_embedding_parameter)
        return 'embedding' in param_name.lower()

    def is_vector_like_parameter(param: torch.nn.Parameter, param_name: str) -> bool:
        if is_embedding_parameter(param, param_name):
            return True
        if param.dim() <= 1:
            return True
        return False

    def is_muon_managed_matrix_parameter(param: torch.nn.Parameter, _: str) -> bool:
        if not is_muon_optimizer:
            return False
        return param.dim() == 2 and not getattr(param, 'is_embedding_or_output_parameter', False)

    def should_scale_lr_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if decoupled_lr_enabled and getattr(param, 'is_embedding_or_output_parameter', False):
            return False
        if is_muon_managed_matrix_parameter(param, param_name):
            return False
        return not is_vector_like_parameter(param, param_name)

    def should_scale_vector_like_lr_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if decoupled_lr_enabled and getattr(param, 'is_embedding_or_output_parameter', False):
            return False
        return is_vector_like_parameter(param, param_name)

    def should_scale_eps_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if is_vector_like_parameter(param, param_name):
            return False
        if is_muon_managed_matrix_parameter(param, param_name):
            return False
        # MuP Appendix B.3: eps scales with fan_in when non-negligible.
        # This implementation follows the common denominator form: sqrt(v) + eps.
        return True

    mup_overrides: Dict[ParamKey, ParamGroupOverride] = {}

    if is_sgd_optimizer:
        vector_like_lr_mult = mup_width_mult
        vector_like_lr_override: ParamGroupOverride = {}
        if base_lr is not None:
            vector_like_lr_override["max_lr"] = base_lr * vector_like_lr_mult
        if base_min_lr is not None:
            vector_like_lr_override["min_lr"] = base_min_lr * vector_like_lr_mult

        if vector_like_lr_override:
            vector_like_predicate = ParamWithNamePredicate(
                name="mup_sgd_vector_like_excluding_embedding_output",
                fn=should_scale_vector_like_lr_with_mup,
            )
            mup_overrides[ParamKey(with_name_predicate=vector_like_predicate)] = (
                vector_like_lr_override
            )

        return mup_overrides

    lr_override: ParamGroupOverride = {}
    if base_lr is not None:
        lr_override["max_lr"] = base_lr * hidden_lr_mult
    if base_min_lr is not None:
        lr_override["min_lr"] = base_min_lr * hidden_lr_mult

    eps_override: ParamGroupOverride = {}
    if is_adam_optimizer and config.adam_eps is not None:
        eps_override["eps"] = config.adam_eps * hidden_lr_mult

    if decoupled_lr_enabled:
        if lr_override:
            hidden_predicate = ParamWithNamePredicate(
                name="mup_hidden_only_excluding_embedding_output", fn=should_scale_lr_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_predicate)] = lr_override

        if eps_override:
            hidden_output_predicate = ParamWithNamePredicate(
                name="mup_hidden_only_for_adam_eps", fn=should_scale_eps_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_output_predicate)] = eps_override
    else:
        combined_override: ParamGroupOverride = {}
        combined_override.update(lr_override)
        combined_override.update(eps_override)
        if combined_override:
            hidden_output_predicate = ParamWithNamePredicate(
                name="mup_hidden_and_output", fn=should_scale_eps_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_output_predicate)] = combined_override

    return mup_overrides


def _get_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups from provided optimizer config object.

    NOTE There can be more than one match between a ParamKey and a parameter.
        What we do is merge all of the matching ParamKey overrides into a single ParamGroupOverride
        for that parameter and use that as the key for that parameter. Any parameters that get
        the same set of merged overrides will be mapped into the same parameter group.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer overrides,
            specified on a per-layer basis. NOTE: if you want to skip applying weight decay on bias
            and length 1 parameters, and also do not want to do any other overrides, set this to an
            empty dictionary rather than the default value of None.
    Returns:
        List of parameter groups.
    """

    # Map (pg_overrides, is_expert_parallel) to params.
    params_map = {} #将param按照(param_override, is_expert_parallel)进行分组并生成字典

    for model_chunk in model_chunks: #遍历model_chunk
        for name, param in model_chunk.named_parameters(): #遍历model_chunk中的参数
            if not param.requires_grad: #不需要更新的参数，优化器不考虑
                continue

            uses_default_config = False
            # Get optimizer config overrides for this parameter.
            param_overrides_list: list[ParamGroupOverride] = [] #将这个参数命中的ParamKey对应ParamGroupOverride记录下来，一个参数可能命中多个 ParamKey
            if config_overrides is not None:
                for param_key, param_override in config_overrides.items():
                    if param_key.matches(param, name): #如果参数命中某个ParamKey，则将该ParamKey的ParamGroupOverride记录下来
                        param_overrides_list.append(param_override)

            if param_overrides_list:
                param_override: ParamGroupOverride | None = combine_param_group_overrides(
                    param_overrides_list
                ) #param_overrides_list可能有多个元素，将针对该参数的重覆盖规则进行合并，检查是否有冲突的超参数设置，最终生成param_override
            else:
                param_override = None

            is_expert_parallel = not getattr(param, 'allreduce', True) #setattr(self.weight, "allreduce", not (self.is_expert and self.expert_parallel))，只有is_expert和expert_parallel都为True时，allreduce为False，其他情况为True，所以针对expert部分参数且使用ep的话，这些参数就会标记is_expert_parallel=True


            # Create config_tuple that is hash-able, and has a consistent ordering of the keys.
            param_override_tuple: tuple[tuple[str, Any], ...] | None = (
                param_group_override_to_tuple(param_override)
            )#把dict类型的param_group_override转换为tuple类型，可以作为key使用，因为dict是不可哈希的
            key = (param_override_tuple, is_expert_parallel) #将param_override_tuple和is_expert_parallel一起作为key
            if key not in params_map: #如果之前这种类型没有就创建
                params_map[key] = []
            params_map[key].append(param) #把参数添加到这个key的list中

    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    params_key = list(params_map.keys()) #获取params_map的所有key
    gathered_params_key = [None for _ in range(torch.distributed.get_world_size())] #构建一个列表，用于接收从其他rank收集到的key，其他rank范围是所有节点所有rank
    torch.distributed.all_gather_object(gathered_params_key, params_key) #allgather所有rank的key
    for keys in gathered_params_key: #将其他rank的key收集到的key整合到自己的key中
        for key in keys:
            if key not in params_key:
                params_key.append(key)
    # Need to pick one of the param_override_tuples to use for the param group.
    param_groups = []
    # Sort keys, None first.
    for key in sorted(params_key, key=lambda x: (x[0] is not None, x[0])): #按param_override是否为None排序，None排在前面；然后按param_override的tuple排序
        param_override_tuple, is_expert_parallel = key
        params = params_map[key] if key in params_map else [] #如果key不存在于params_map中，则params为空，说明这个key来源于其他的rank
        if param_override_tuple is None:
            param_override: ParamGroupOverride = {}
        else:
            param_override: ParamGroupOverride = {k: v for (k, v) in param_override_tuple} #把tuple类型的param_override还原为dict类型

        # False if param_group_override is None or empty tuple or if we do not modify the
        #  LR schedule.
        #  NOTE: "default_config" is used for logging the learning rate in training.py.
        #   so set to True if we do not modify the learning rate.
        #  if param_group['default_config']:
        #    learning_rate = param_group['lr']
        uses_default_lr_schedule: bool = (not bool(param_override_tuple)) or not any(
            ["lr" in k for k in param_override]
        )

        # TODO: Remove "backwards compatible" fields below eventually.
        default_config: ParamGroupOverride = { #基础default值
            'wd_mult': 1.0,
            'lr_mult': 1.0,
            'is_decoupled_lr': False,
            # The following two fields may be important to keep even when we remove the
            #   above "backwards compatible" fields.
            "max_lr": config.lr,  # user may override this in param_override
            "min_lr": config.min_lr,  # user may override this in param_override
        }
        assert (
            "params" not in param_override
        ), "'params' should not be in param_override, this is a protected key"
        param_group = {
            'params': params,
            'is_expert_parallel': is_expert_parallel,
            'default_config': uses_default_lr_schedule,
            **default_config,
            **param_override,  # keep **param_override last so that users can override other fields. #Python 的 ** 解包顺序决定了同名 key 时后者胜出,所以如果用户通过 config_overrides 给某组参数指定了 max_lr，它就会覆盖 default_config 中的 max_lr。如果没指定，就用 config.lr 作为默认 max_lr。
        }#构建param_group字典
        param_groups.append(param_group)#添加param_group到param_groups列表中

    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer/scheduler
            overrides, specified on the basis of ParamKey matches with each parameter.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    param_groups = _get_param_groups(model_chunks, config, config_overrides) #获取参数分组
    param_groups = list(filter(filter_fn, param_groups)) #读取param_groups的is_expert_parallel属性来过滤不符合条件的参数组param group
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name): #buffer_name="buffers"
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name) #获取model_chunk对应的buffer，_ParamAndGradBuffer包括grad和param的buffer

    return param_groups, buffers #返回参数组和buffer


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None, #非分布式场景的优化器包装（不使用 DistributedOptimizer 时才走这条分支），用model_parallel_group来梯度统计
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    distributed_optimizer_instance_id: Optional[int] = 0, #tp_pp通信组的idx，每个 dp_group_idx key 对应一个 TP×PP 分片
    pg_collection: Optional[ProcessGroupCollection] = None,
    skip_megatron_wrapping: bool = False, #是否跳过Megatron优化器的包装
) -> Union[MegatronOptimizer, Tuple[Optional[torch.optim.Optimizer], Optional[Callable]]]:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.
        distributed_optimizer_instance_id (int, optional): Distributed optimizer instance. Defaults
            0.
        skip_megatron_wrapping (bool): if True, return a
            ``(optimizer, init_state_fn)`` tuple of the raw PyTorch optimizer
            without any Megatron wrapping. Useful when the caller
            (e.g. LayerWiseDistributedOptimizer) performs its own wrapping.

    Returns:
        Instance of MegatronOptimizer, or ``(optimizer, init_state_fn)`` when
        *skip_megatron_wrapping=True*.
    """
    # All param_groups passed here must belong to the same optimizer type (adam / sgd).
    # Callers are responsible for splitting by optimizer type before calling this function.

    if skip_megatron_wrapping and config.use_precision_aware_optimizer:
        raise ValueError(
            "skip_megatron_wrapping=True is incompatible with use_precision_aware_optimizer."
        )
    if skip_megatron_wrapping and config.optimizer_cpu_offload:
        raise ValueError("skip_megatron_wrapping=True is incompatible with optimizer_cpu_offload.")

    # When freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions.
    if param_groups:
        if config.optimizer_cpu_offload: #如果需要offload优化器状态
            if torch.__version__ < '2.3.0':
                warnings.warn(
                    "CPU offload is recommended for PyTorch >= 2.3.0, "
                    "untested versions below this may have convergence issues."
                )
            assert (
                config.decoupled_weight_decay
            ), "CPU offloading only supported with decoupled_weight_decay enabled (AdamW mode)."
            gpu_optimizer_cls = Adam if config.optimizer == 'adam' else SGD #默认optimizer是Adam
            cpu_optimizer_cls = CPUAdam if config.optimizer == 'adam' else CPUSGD #默认optimizer是Adam
            if config.use_torch_optimizer_for_cpu_offload:
                gpu_optimizer_cls = cpu_optimizer_cls
            if config.optimizer == 'adam':
                gpu_optimizer_cls = Adam
                cpu_optimizer_cls = CPUAdam
                optimizer_defaults = dict(
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_eps,
                    bias_correction=True,
                    fused=True,  # this flag is used to improve the performance of the cpu optimizer
                )
            else:
                gpu_optimizer_cls = SGD
                cpu_optimizer_cls = CPUSGD
                optimizer_defaults = dict(
                    lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum
                )
            optimizer = HybridDeviceOptimizer(
                param_groups,
                offload_fraction=config.optimizer_offload_fraction,
                cpu_optimizer_cls=cpu_optimizer_cls,
                gpu_optimizer_cls=gpu_optimizer_cls,
                overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
                pin_cpu_grads=config.pin_cpu_grads,
                pin_cpu_params=config.pin_cpu_params,
                param_update_in_fp32=True,
                **optimizer_defaults,
            )
            init_state_fn = None
        elif config.optimizer == 'adam': #默认优化器是Adam
            kwargs = {
                "params": param_groups, #param_groups是param_group的列表，每个param_group是字典，包括params，is_expert_parallel，max_lr这些超参
                "lr": config.lr, #传给优化器作为兜底的默认值
                "weight_decay": config.weight_decay, #传给优化器作为兜底的默认值
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
                "capturable": config.optimizer_cuda_graph,
            }

            # set Adam class and weight decay mode depending
            # on source of optimizer (Torch or TE/Apex)
            if USING_PYTORCH_OPTIMIZER: #对于PyTorch的优化器，区分使用Adam还是AdamW
                adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
            else:
                kwargs["adam_w_mode"] = config.decoupled_weight_decay
                adam_cls = Adam

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )
                # Master weight is managed by MCore when main_params_dtype is fp32. This is
                # because we want to use fp8 primary weight with precision aware optimizer.
                # Otherwise, master weight will be managed by TransformerEngine.
                # Delayed scaling is an exception because casting as well as the computation
                # of the scaling factor can be conducted in the adam kernel.
                if config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                    kwargs.update(
                        {
                            "master_weights": True,
                            "use_decoupled_grad": True,
                            "master_weight_dtype": config.main_params_dtype,
                        }
                    )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": config.store_param_remainders})

            optimizer = adam_cls(**kwargs) #根据优化器类型进行创建

            def init_state_fn(opt, config=None): #初始化函数，可以调用阻止lazy初始化，提前完成初始化（初始化指创建优化器状态）
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

        elif config.optimizer == 'lion':
            if not HAVE_EMERGING_OPTIMIZERS:
                raise ImportError(
                    "Lion optimizer requires emerging_optimizers >= 0.2. "
                    "Please install or upgrade it to use --optimizer lion."
                )
            optimizer = Lion(  # pylint: disable=possibly-used-before-assignment
                param_groups,
                lr=config.lr,
                betas=(config.lion_beta1, config.lion_beta2),
                weight_decay=config.weight_decay,
            )

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            opt.state[p]['exp_avg'] = torch.zeros_like(p.data)

        elif config.optimizer == 'sgd':
            optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
            init_state_fn = None
        else:
            raise Exception('{} optimizer is not supported.'.format(config.optimizer))
    else:
        optimizer = None
        init_state_fn = None

    if skip_megatron_wrapping: #如果配置项skip_megatron_wrapping为True，则直接返回优化器和初始化状态函数
        return optimizer, init_state_fn

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer: #如果使用混合精度或者分布式优化器，进行Megatron封装

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale: #构建grad_scaler对象
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [optimizer, config, grad_scaler, init_state_fn] #聚合优化器参数
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                model_chunks=model_chunks,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )#DistributedOptimizer封装
            # This is needed for case where num_distributed_optimizer_instances > 1. In this case,
            # weight gradients are all-reduced across optimizer instances, so each instance has
            # the duplicated weight gradients, need to reduce gradient stats inside each instance.
            setattr(optimizer, 'grad_stats_parallel_group', intra_dist_opt_group) #设置优化器的grad_stats_parallel_group属性为intra_dist_opt_group，后续用于梯度统计
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args) #fp16/bf16 非分布式场景的优化器包装（不使用 DistributedOptimizer 时才走这条分支），用model_parallel_group来梯度统计
            setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        optimizer = FP32Optimizer(optimizer, config, init_state_fn) #最轻量的包装器，纯fp32训练，参数、梯度、优化器状态都是fp32
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    if pg_collection is None or not hasattr(pg_collection, 'tp'):
        tp_group = parallel_state.get_tensor_model_parallel_group()
    else:
        tp_group = pg_collection.tp
    # TODO(M4): plumb tp_group through optimizer constructors so this setattr disappears.
    setattr(optimizer, 'tp_group', tp_group) #设置优化器的tp通信组

    return optimizer


def check_config_overrides_consistency(
    config: OptimizerConfig, config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]]
):
    """Check if the config overrides are consistent with the config."""

    # TODO: Remove `optimizer` from this eventually (e.g., if we use Muon for some layers and
    # Adam for other layers). This would need some more refactoring to work though (param_groups
    # filtered by optimizer passed into _get_megatron_optimizer_based_on_param_groups).
    if config_overrides is not None:
        fields_to_check_for_consistency = [
            'overlap_param_gather_with_optimizer_step',
            'optimizer',
            'optimizer_cpu_offload',
        ] #对 config_overrides 做一致性校验，防止用户在 override 中修改了不该动的全局配置。
        for field_name in fields_to_check_for_consistency:
            base_field = getattr(config, field_name, None)
            all_config_overrides = list(config_overrides.values())
            for config_override in all_config_overrides:
                if field_name in config_override:
                    field = config_override[field_name]
                    if field != base_field:
                        raise ValueError(
                            f"Field {field_name} should not be overriden in a config override."
                        )
    return True


def _get_megatron_emerging_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, Any]] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Build an emerging optimizer (e.g. Muon) for the given model chunks.

    Parameter separation (e.g., linear weights -> Muon, rest -> Adam) is expressed as a
    config_override, the same mechanism used for weight-decay and learning-rate overrides.
    Adam/SGD groups are delegated to _get_megatron_optimizer_based_on_param_groups so they
    go through the exact same code path as the standard optimizer factory.

    When ``config.use_layer_wise_distributed_optimizer`` is True, the underlying optimizers
    are wrapped with :class:`LayerWiseDistributedOptimizer`.
    """
    eopt_name = config.optimizer
    use_layer_wise = config.use_layer_wise_distributed_optimizer

    # Handle legacy "dist_*" optimizer names (e.g. "dist_muon" → "muon" + layer-wise).
    if eopt_name.startswith('dist_'):
        bare_name = eopt_name[len('dist_') :]
        warnings.warn(
            f"optimizer='{eopt_name}' is deprecated. "
            f"Use optimizer='{bare_name}' with use_layer_wise_distributed_optimizer=True.",
            DeprecationWarning,
            stacklevel=3,
        )
        eopt_name = bare_name
        use_layer_wise = True

    if not HAVE_EMERGING_OPTIMIZERS:
        raise ImportError(
            f"emerging-optimizers package is required for optimizer='{eopt_name}'. "
            "Install it with: pip install emerging-optimizers"
        )
    if eopt_name not in _EMERGING_OPTIMIZERS:
        raise ValueError(f"Unsupported emerging optimizer: {eopt_name}")
    if config.fp16:
        raise ValueError('emerging optimizer with fp16 is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Tag parameters with optimizer-specific attributes (expert_tp, is_qkv).
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True

    # Apply optimizer-specific default param overrides (e.g. muon: non-linear -> adam).
    config_overrides.update(_EMERGING_OPTIMIZERS[eopt_name].default_param_overrides)

    # Build param groups and bucket by (optimizer_name, is_expert_parallel).
    # Layer-wise distributed optimizer handles expert params internally so we skip that split.
    all_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    grouped_param_groups = defaultdict(list)
    for group in all_param_groups:
        opt_name = group.get('optimizer', eopt_name)
        is_expert = group['is_expert_parallel'] and not use_layer_wise
        grouped_param_groups[(opt_name, is_expert)].append(group)

    # Build an optimizer for each (optimizer_name, is_expert) bucket and combine.
    results = []
    for (opt_name, is_expert), groups in grouped_param_groups.items():
        if not groups:
            continue

        model_parallel_group = pg_collection.tp_ep_pp if is_expert else pg_collection.mp

        if opt_name in _EMERGING_OPTIMIZERS:
            optimizer, init_state_fn = _create_emerging_optimizer(
                config, groups, eopt_name, model_chunks, pg_collection
            )
            if use_layer_wise:
                result = (optimizer, init_state_fn)
            else:
                if config.bf16:
                    optimizer = Float16OptimizerWithFloat16Params(
                        optimizer, config, None, init_state_fn
                    )
                else:
                    optimizer = FP32Optimizer(optimizer, config, init_state_fn)
                setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
                if pg_collection is None or not hasattr(pg_collection, 'tp'):
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                else:
                    tp_group = pg_collection.tp
                setattr(optimizer, 'tp_group', tp_group)
                result = optimizer
        else:
            fallback_config = copy.copy(config)
            fallback_config.optimizer = opt_name
            fallback_config.use_distributed_optimizer = False
            result = _get_megatron_optimizer_based_on_param_groups(
                config=fallback_config,
                model_chunks=model_chunks,
                param_groups=groups,
                model_parallel_group=model_parallel_group,
                pg_collection=pg_collection,
                skip_megatron_wrapping=use_layer_wise,
            )
            # TODO(deyuf): ChainedOptimizer currently asserts all sub-optimizers
            # share the same config. Revisit this design now that emerging
            # optimizers mix different optimizer types (e.g. Muon + Adam).
            # For now, reset to the top-level config so the assertion holds.
            if not use_layer_wise and hasattr(result, 'config'):
                result.config = config
        results.append(result)

    if use_layer_wise:
        base_optimizers, init_fns = (), ()
        if results:
            base_optimizers, init_fns = zip(*results)
        log_single_rank(
            logger, logging.INFO, f'Using LayerWiseDistributedOptimizer for {eopt_name}'
        )
        return LayerWiseDistributedOptimizer(
            list(base_optimizers),
            config,
            pg_collection,
            init_state_fn_list=list(init_fns),
            model_chunks=model_chunks if config.overlap_param_gather else None,
        )

    return ChainedOptimizer(results)


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
    dump_param_to_param_group_map: Optional[str] = None,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    Handles both standard optimizers (Adam, SGD) and emerging optimizers (e.g. Muon).
    We use separate optimizers for expert parameters and non-expert parameters.
    For emerging optimizers with ``config.use_layer_wise_distributed_optimizer=True``,
    the optimizer is automatically wrapped with :class:`LayerWiseDistributedOptimizer`.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        config_overrides (Optional[Dict[ParamKey, OptimizerConfig]]): optional dictionary of
            optimizer configuration objects to override default optimizer behavior for different
            subsets of parameters (identified by ParamKey).
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        pg_collection: Optional unified process group for distributed training.
        dump_param_to_param_group_map (Optional[str]): path to dump parameter to param group map.

    Returns:
        Instance of MegatronOptimizer.
    """

    # None → apply standard defaults. To extend defaults with custom overrides,
    # start from get_standard_config_overrides(config) and merge yours in.
    if config_overrides is None:
        config_overrides = get_standard_config_overrides(config)

    check_config_overrides_consistency(config, config_overrides) #对 config_overrides 做一致性校验，防止用户在 override 中修改了不该动的全局配置。

    # TODO: the standard and emerging optimizer paths handle pg_collection differently;
    # unify them so both use a single pg_collection-based flow.
    if config.optimizer not in ('adam', 'sgd'): #为新兴优化器（Muon、Lion、Soap 等）构建 Megatron 优化器。
        return _get_megatron_emerging_optimizer(
            config=config,
            model_chunks=model_chunks,
            config_overrides=config_overrides,
            pg_collection=pg_collection,
        )

    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    # Separate out first model chunk if overlapping param AG with optimizer step.
    if config.overlap_param_gather_with_optimizer_step: #如果指定优化器更新与参数gather通信重叠执行的话，就将model_chunk划分为第一个chunk和其他chunk，这里只做第一个chunk的gather和后续chunk优化器更新重叠，因为前传的时候第一个chunk是关键节点，其他的chunk的gather不会和优化器的更新重叠，会在前传的时候和前传重叠（overlap_param_gather_with_optimizer_step开启一定需要--overlap-param-gather开启）
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]] #将模型chunk进行分组，第一个chunk和其他chunk分开
        overlap_param_gather_with_optimizer_step_flags = [True, False] #第一个chunk重叠，其他chunk不重叠
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]

    # Setup process groups using helper method
    process_groups_dict = ProcessGroupCollection.setup_process_groups_for_optimizer(
        pg_collection, model_chunks, use_gloo_process_groups
    )#设置优化器相关通信组

    dp_cp_group = process_groups_dict['dp_cp_group']
    intra_dp_cp_group = process_groups_dict['intra_dp_cp_group']
    intra_expt_dp_group = process_groups_dict['intra_expt_dp_group']
    mp_group = process_groups_dict['mp_group']
    expt_tp_pp_group = process_groups_dict['expt_tp_pp_group']
    intra_dp_cp_group_gloo = process_groups_dict['intra_dp_cp_group_gloo']
    intra_expt_dp_group_gloo = process_groups_dict['intra_expt_dp_group_gloo']
    intra_dist_opt_group = process_groups_dict['intra_dist_opt_group']

    model_parallel_rank = get_pg_rank(mp_group) #确定在tp_pp通信组内的rank

    if get_pg_size(dp_cp_group) > get_pg_size(intra_dp_cp_group): #如果有多个优化器实例
        inter_dist_opt_group = process_groups_dict['inter_dist_opt_group']
        distributed_optimizer_instance_id = get_pg_rank(inter_dist_opt_group)
    else:
        distributed_optimizer_instance_id = 0

    optimizers = []
    model_chunk_offset = 0 #offset偏移
    ddp_config = model_chunks[0].ddp_config  # Use the first model chunk's DDP config
    if ddp_config.use_megatron_fsdp:#如果使用megatron fsdp
        for model_chunk, overlap_param_gather_with_optimizer_step in zip(
            all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
        ):
            param_groups, buffers = _get_param_groups_and_buffers(
                model_chunk,
                model_chunk_offset=model_chunk_offset,
                config=config,
                config_overrides=config_overrides,
                filter_fn=lambda g: True,
                buffer_name='buffers',
            )

            optimizer_part = _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunk,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
            if (
                not USING_PYTORCH_OPTIMIZER
                and config.use_precision_aware_optimizer
                and getattr(optimizer_part.optimizer, "master_weights", None) is not None
            ):
                # NOTE(@cspades): FusedAdam is provided Megatron-FSDP's main weights as
                # non-quantized DTensor(s). Megatron-FSDP should NEVER use FusedAdam's
                # main weights, complete waste of memory as the optimizer step is still
                # applied to the Megatron-FSDP main weight and extended to FusedAdam
                # main weights. Override this here.
                setattr(optimizer_part.optimizer, "master_weights", False)
                # Megatron-FSDP always uses a decoupled gradient when using FusedAdam.
                setattr(optimizer_part.optimizer, "use_decoupled_grad", True)

            optimizers.append(optimizer_part)
            model_chunk_offset += 1

        if len(optimizers) == 1:
            return optimizers[0]

        return ChainedOptimizer(optimizers)

    if dump_param_to_param_group_map is not None:
        param_to_param_group = {}
        param_group_id = 0
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ): #处理dense部分，按overlap_param_gather_with_optimizer_step对dense部分的model_chunk进行了分组
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset, # #model_chunk_offset 确保在多组 model chunks 场景下 key 全局唯一
            config=config,
            config_overrides=config_overrides,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
        )#获取参数组和buffer
        for model_chunk in dense_model_chunks:#对每个model_chunk赋值overlap_param_gather_with_optimizer_step
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )
        if dump_param_to_param_group_map is not None:
            for param_group in param_groups:
                for param in param_group["params"]:
                    param_name = get_global_unique_param_name(model_chunks, param)
                    param_to_param_group[param_name] = param_group_id
                param_group_id += 1

        # Pass Gloo process groups into optimizer only if needed.
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups( #构造优化器
                config=config,
                model_chunks=dense_model_chunks, #模型分片列表
                param_groups=param_groups, #模型分片列表中所有参数分组
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=intra_dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )
        model_chunk_offset += 1 #按道理应该+= len(dense_model_chunks)，这里只加1，是因为model_chunk分组，只分为第一个chunk和其余chunk，所以第一个dense_model_chunks里面的model_chunk数量肯定为1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )#获取expert且ep的参数组和buffer
    if dump_param_to_param_group_map is not None:
        for param_group in moe_param_groups:
            for param in param_group["params"]:
                param_name = get_global_unique_param_name(model_chunks, param)
                param_to_param_group[param_name] = param_group_id
            param_group_id += 1
    if len(moe_param_groups) > 0: #如果有expert且ep的参数
        expt_model_parallel_rank = get_pg_rank(expt_tp_pp_group)
        # Pass Gloo process groups into optimizer only if needed.
        if use_gloo_process_groups:
            expt_data_parallel_group_gloo = intra_expt_dp_group_gloo
        else:
            expt_data_parallel_group_gloo = None
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=expt_tp_pp_group, #非分布式优化器（不使用 DistributedOptimizer 时才走这条分支），用model_parallel_group来梯度统计
                data_parallel_group=intra_expt_dp_group, #dp通信组
                data_parallel_group_gloo=expt_data_parallel_group_gloo,
                data_parallel_group_idx=expt_model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group, #分布式优化器（使用 DistributedOptimizer 时才走这条分支），用model_parallel_group来梯度统计
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )#添加expert且ep的优化器
        )

    if dump_param_to_param_group_map is not None:
        torch.distributed.checkpoint.save(
            state_dict=param_to_param_group, checkpoint_id=dump_param_to_param_group_map
        )

    return ChainedOptimizer(optimizers) #把创建的多个优化器包装为ChainedOptimizer对象
