# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""
import logging
import os
import random
import time
import warnings
from datetime import timedelta

import numpy as np
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train
from megatron.core.fusions.fused_bias_gelu import bias_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
from megatron.core.parallel_state import create_group
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    initialize_rerun_state_machine,
)
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    enable_batch_invariant_mode,
)
from megatron.core.utils import get_pg_rank, get_te_version, is_te_min_version, is_torch_min_version
from megatron.training import (
    get_adlr_autoresume,
    get_args,
    get_tensorboard_writer,
    inprocess_restart,
)
from megatron.training.async_utils import init_persistent_async_worker
from megatron.training.utils import is_rank0, print_rank_0, warn_rank_0

logger = logging.getLogger(__name__)


def initialize_megatron(
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    store=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    args = get_args()

    # set logging level
    setup_logging()

    if args.async_save and args.use_persistent_ckpt_worker: #设置异步检查点机制（创建后台进程，专门写检查点）
        init_persistent_async_worker(args.rank, 'forkserver')

    # init rerun state
    def state_save_func():
        return {'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict['rng_tracker_states']:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])

    args = get_args()
    initialize_rerun_state_machine( # Rerun 状态机 的初始化，用于训练任务失败后自动重跑修复
        state_save_func=state_save_func, # 如何保存当前 RNG 状态
        state_restore_func=state_restore_func, # 如何恢复 RNG 状态
        mode=RerunMode(args.rerun_mode), # 重跑模式（由 --rerun-mode 控制）
        error_injector=RerunErrorInjector(
            error_injection_rate=args.error_injection_rate,
            error_injection_type=RerunDiagnostic(args.error_injection_type),
        ),
        result_rejected_tracker_filename=args.result_rejected_tracker_filename,
    )

    if args.batch_invariant_mode:
        print_rank_0("Enabling batch invariant mode globally")
        enable_batch_invariant_mode()

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store)

        # Random seeds for reproducibility.
        print_rank_0("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(
            args.seed,
            args.data_parallel_random_init,
            args.te_rng_tracker,
            args.inference_rng_tracker,
            use_cudagraphable_rng=args.cuda_graph_impl != "none",
        )

        # Setup MoE aux loss scale value.
        if args.num_experts is not None:
            from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler

            MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init: #是否延迟分布式环境的初始化
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else: #默认，立即初始化
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init() #完成分布式环境的初始化

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies() #把数据集的 C++ 索引构建器编译出来。

        if args.tp_comm_overlap:
            # TODO: Should this be activated with just decoder-tp-comm-overlap too?
            _initialize_tp_communicators() #初始化TP通算并行，预分配好通信缓冲区

        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    torch.distributed.barrier()

def _initialize_tp_communicators():
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    try:
        import transformer_engine
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
            "'transformer_engine' packages"
        )

    args = get_args()

    if args.tp_comm_overlap_cfg is not None:
        with open(args.tp_comm_overlap_cfg, "r") as stream:
            ub_cfgs = yaml.safe_load(stream)
    else:
        ub_cfgs = {}

    if getattr(args, 'decoder_tp_comm_overlap', False):
        input_shape = [ #确定输入激活值shape
            (args.decoder_seq_length * args.micro_batch_size) // args.context_parallel_size,
            args.hidden_size,
        ]
    else:
        input_shape = [ #确定输入激活值shape
            (args.seq_length * args.micro_batch_size) // args.context_parallel_size,
            args.hidden_size,
        ]

    if is_te_min_version("2.7.0"):
        UserBufferQuantizationMode = te_module.base.UserBufferQuantizationMode
        quantization_modes = [
            UserBufferQuantizationMode.FP8 if args.fp8 else UserBufferQuantizationMode.NONE
        ]
        if (
            args.fp8 is not None
            and args.first_last_layers_bf16
            and (args.num_layers_at_start_in_bf16 > 0 or args.num_layers_at_end_in_bf16 > 0)
        ):
            quantization_modes.append(UserBufferQuantizationMode.NONE)
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub( #初始化用户缓冲区（UB）用于张量模型并行通信，缓冲区大小为输入激活值shape
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            quantization_modes=quantization_modes,
            ub_cfgs=ub_cfgs,
            bootstrap_backend=args.tp_comm_bootstrap_backend,
        )
    elif is_te_min_version("1.9.0"):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=args.tp_comm_bootstrap_backend,
        )
    else:
        if args.tp_comm_bootstrap_backend != 'mpi':
            warnings.warn(
                f"Transformer Engine v{get_te_version()} supports only MPI bootstrap backend."
            )
        # Create a MPI process group to help with TP communication overlap bootstrap.
        create_group(backend='mpi', group_desc='TP_BOOTSTRAP_GROUP_MPI')

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
            ub_cfgs=ub_cfgs,
        )


def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store):
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized(): #判断torch的分布式环境是否已经初始化

        print_rank_0("torch distributed is already initialized, skipping initialization ...")
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        print_rank_0("> initializing torch distributed ...") #全局RANK = 0的节点print信息
        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(args.local_rank) #绑定每个进程使用的GPU
            device_id = torch.device(f'cuda:{args.local_rank}')
        else:
            device_id = None

        # Set to non-default stream for cudagraph capturing.
        if args.cuda_graph_impl == "transformer_engine":
            torch.cuda.set_stream(torch.cuda.Stream())

        # Set flight recorder env vars if specified.
        # Priority: pre-existing environment variable > MLM argument.
        # All vars follow the same setdefault semantics: if already set in the
        # environment we warn and keep the user's value; otherwise we apply the
        # value derived from the MLM argument / flag.
        # The block is also triggered when either path env var is already set
        # so that the remaining defaults are applied consistently.
        _fr_path = (
            args.flight_recorder_dump_path
            or os.environ.get('TORCH_FR_DUMP_TEMP_FILE')
            or os.environ.get('TORCH_NCCL_DEBUG_INFO_TEMP_FILE')
        )
        if _fr_path is not None:
            _fr_dump_prefix = _fr_path
            if os.path.isdir(_fr_path):
                _fr_dump_prefix = os.path.join(_fr_path, '_dump_')
                warn_rank_0(
                    "Flight recorder: using directory "
                    f"'{_fr_path}' for dump path, appending per-rank prefix "
                    f"'{_fr_dump_prefix}'."
                )
            _fr_env_defaults = {
                'TORCH_FR_DUMP_TEMP_FILE': _fr_dump_prefix,
                'TORCH_NCCL_DEBUG_INFO_TEMP_FILE': _fr_dump_prefix,
                'TORCH_NCCL_TRACE_BUFFER_SIZE': str(args.flight_recorder_trace_buffer_size),
                'TORCH_NCCL_DUMP_ON_TIMEOUT': str(int(args.flight_recorder_dump_on_timeout)),
                'TORCH_INCLUDE_STACK_TRACE': str(int(args.flight_recorder_include_stack_trace)),
                'TORCH_INCLUDE_ONLY_ACTIVE': str(int(args.flight_recorder_include_only_active)),
                'TORCH_NCCL_EXTRA_DUMP_ON_EXEC': str(int(args.flight_recorder_extra_dump_on_exec)),
            }
            for _var, _default in _fr_env_defaults.items():
                if _var in os.environ:
                    warn_rank_0(
                        f"Flight recorder: environment variable {_var} is already set to "
                        f"'{os.environ[_var]}'; ignoring config value '{_default}'."
                    )
                else:
                    os.environ[_var] = _default
            print_rank_0(
                "Flight recorder env vars:\n"
                + "\n".join(f"  {k}={os.environ[k]}" for k in _fr_env_defaults)
            )

        # Call the init process
        init_process_group_kwargs = {
            'backend': args.distributed_backend,
            'store': store,
            'world_size': args.world_size,
            'rank': args.rank,
            'timeout': timedelta(minutes=args.distributed_timeout_minutes),
        }#分布式进程组的参数，包括分布式通信后端、world_size、rank、超时时间等
        if args.fake_process_group:
            assert is_torch_min_version(
                "2.3.0"
            ), "Fake process group is only supported with PyTorch 2.3.0 and above."
            from torch.testing._internal.distributed.fake_pg import FakeStore

            store = FakeStore()
            init_process_group_kwargs['backend'] = 'fake'
            init_process_group_kwargs['store'] = store

        torch.distributed.init_process_group(**init_process_group_kwargs) #初始化PyTorch分布式环境，创建default precess group
        inprocess_restart.maybe_force_nccl_backend_init(device_id)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators. #TP/DP/PP等分布式训练的进程组初始化
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(#构建分布式训练的通信组
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
                use_sharp=args.use_sharp,
                context_parallel_size=args.context_parallel_size,
                hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
                hybrid_context_parallel=args.hybrid_context_parallel,
                expert_model_parallel_size=args.expert_model_parallel_size,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp',
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=args.use_gloo_process_groups,
                high_priority_stream_groups=args.high_priority_stream_groups,
                sharp_enabled_group=args.sharp_enabled_group,
                # Colocated encoder training: creates the encoder inner dp group.
                # The encoder's own tensor parallel size comes from the colocated
                # arguments (colocated_args.py) and is absent for non-colocated
                # entry points, hence getattr.
                # 共置 encoder 训练：创建 encoder inner dp 组（仅在开启该开关时）。
                # encoder 自己的 tp 并行度来自共置参数（colocated_args.py），非共置入口
                # 没有这个字段，故用 getattr 取。
                use_colocated_encoder=getattr(args, "use_colocated_encoder", False),
                colocated_encoder_tensor_model_parallel_size=getattr(
                    args, "colocated_encoder_tensor_model_parallel_size", None
                ),
                colocated_encoder_num_distributed_optimizer_instances=getattr(
                    args, "colocated_encoder_num_distributed_optimizer_instances", 1
                ),
            )
            print_rank_0(
                f"> initialized tensor model parallel with size "
                f"{mpu.get_tensor_model_parallel_world_size()}"
            )
            print_rank_0(
                f"> initialized pipeline model parallel with size "
                f"{mpu.get_pipeline_model_parallel_world_size()}"
            )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    is_colocated_encoder: bool = False,
):
    """Set random seed for reproducability.

    The random streams are derived from the job-wide (mpu) coordinates by default.
    For colocated encoder training the two components have DIFFERENT topologies and
    therefore need different seeds: the seed offsets below decorrelate distinct model
    SHARDS, so they must be computed from the coordinates of the component being built.
    The encoder is replicated across the pipeline dimension (its own pipeline rank is
    always 0), while the backbone is sharded over it - see get_colocated_model.
    随机流默认按作业级（mpu）坐标推导。共置 encoder 训练时两个组件拓扑不同、种子也必须
    不同：下面的偏移目的是让**不同的模型分片**互不相关，因此必须按"正在构建的那个组件"
    的坐标来算——encoder 在 pipeline 维上是副本（它自己的 pipeline rank 恒为 0），backbone
    才是按该维切分的，详见 get_colocated_model。

    ``is_colocated_encoder``：True 时本次调用为共置 encoder 组件设种子。此时**所有**坐标
    （pipeline / tensor / dp / 专家并行）都在函数内部从 parallel_state 的
    ``get_colocated_encoder_*`` 系列 getter 推导——encoder 的 pipeline / 专家并行组都是
    单成员组（处处 rank 0），dp 组是共置数据并行组，tp 组是它自己的通信子——无需调用方
    传入 encoder_pg。False 时走正常流程、与 encoder 坐标无关（共置模式下的 backbone 组件
    就落在这个分支）。

    注意本函数每次调用都会 reset 整个 RNG tracker（``model_parallel_cuda_manual_seed``），
    因此 encoder 播下的命名状态会被随后 backbone 那次调用冲掉。encoder 在训练前传时要用
    的那套状态由 ``snapshot_colocated_encoder_rng_tracker`` 在 encoder 构建完立刻整体冻结
    （见 get_colocated_model），与本函数职责分离。
    """
    if seed_ is not None and seed_ > 0:
        pipeline_rank = None
        tensor_rank = None
        data_parallel_rank = None
        expert_model_parallel_rank = None
        expert_tensor_parallel_rank = None
        if mpu.is_colocated_encoder_enabled() and is_colocated_encoder:
            # The encoder's OWN coordinates, derived from its own process groups (no
            # caller-supplied encoder_pg needed): the pipeline group is single-member
            # (rank 0 on every rank), the tensor group is its own communicator, the
            # dp group is the colocated data-parallel group, and the expert-parallel
            # groups are single-member too (the encoder has no MoE experts).
            # 共置 encoder 自身坐标取自它自己的通信组（无需调用方传入 encoder_pg）：
            # pipeline 组单成员（处处 rank 0）、tp 组是它自己的通信子、dp 组是共置
            # 数据并行组、专家并行组也是单成员（encoder 无 MoE 专家）。
            encoder_pipeline_group = mpu.get_colocated_encoder_pipeline_model_parallel_group()
            encoder_tensor_group = mpu.get_colocated_encoder_tensor_model_parallel_group()
            encoder_data_parallel_group = mpu.get_colocated_data_parallel_group()
            pipeline_rank = get_pg_rank(encoder_pipeline_group)
            tensor_rank = get_pg_rank(encoder_tensor_group)
            data_parallel_rank = get_pg_rank(encoder_data_parallel_group)
            expert_model_parallel_rank = get_pg_rank(encoder_pipeline_group)
            expert_tensor_parallel_rank = get_pg_rank(encoder_pipeline_group)
        if pipeline_rank is None:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        if data_parallel_rank is None:
            data_parallel_rank = mpu.get_data_parallel_rank()
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * pipeline_rank)
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * data_parallel_rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(
                seed,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
                tp_rank=tensor_rank,
                ep_rank=expert_model_parallel_rank,
                etp_rank=expert_tensor_parallel_rank,
            )
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    if is_torch_min_version("2.2.0a0"):
        pass  # we're using torch.compile for jit fusion
    elif is_torch_min_version("1.10.0a0"):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function() #预热消除了第一个 training step 中 JIT 编译带来的额外延迟，让第一个 step 就能达到稳定性能。只编译代码中写死的几个 Megatron 自定义 fused 算子（bias_gelu / bias_swiglu）


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(
        args.ffn_hidden_size // args.tensor_model_parallel_size, dtype=dtype, device="cuda"
    )
    input = torch.rand(
        (
            args.seq_length // args.context_parallel_size,
            args.micro_batch_size,
            args.ffn_hidden_size // args.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            if args.swiglu:
                output = bias_swiglu(input, bias)
            else:
                output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input = torch.rand(
        (seq_length // args.context_parallel_size, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    residual = torch.rand(
        (seq_length // args.context_parallel_size, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(residual)
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train([input, bias], residual, dropout_rate)
    del bias, input, residual, output
    torch.cuda.empty_cache()


def setup_logging() -> None:
    """Sets the default logging level based on cmdline args and env vars.

    Precedence:
    1. Command line argument `--logging-level`
    2. Env var `MEGATRON_LOGGING_LEVEL`
    3. Default logging level (INFO)

    Returns: None
    """
    args = get_args()
    logging_level = None
    env_logging_level = os.getenv('MEGATRON_LOGGING_LEVEL', None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)
    if args.logging_level is not None:
        logging_level = args.logging_level

    if logging_level is not None:
        if is_rank0():
            logger.info(f'Setting logging level to {logging_level}')
        logging.getLogger().setLevel(logging_level)
