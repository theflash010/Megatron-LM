# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command line arguments of colocated training (Task 6.5).

共置训练的专属参数入口。**不往 ``multimodal_args.py`` 里加字段**，而是独立成文件：那里的
参数描述的是"LLaVA 这个模型"，这里的参数描述的是"共置这套并行拓扑"，两者的生命周期不同。
本文件只**组合调用** ``add_multimodal_extra_args``，因为 ``model_provider`` 仍然读 llava
的那批参数。

为什么 encoder 需要自己的并行度参数：共置实现里 encoder 的 dp 组是"全部 W 个 rank"、pp 组
是单成员组，这两者是从 backbone 的 inner dp × pp 推导出来的；而 tp 在此之前一直**隐式**沿用
作业的 mpu tp 组。隐式沿用在当前配置（TP=1）下结果正确，但没有显式契约——换一套卡数/并行
配置时，"encoder 用哪个 tp 组"这件事没有任何地方声明，也就无从校验。因此把 encoder 的
tp/pp/cp 并行度提成显式参数，默认值与当前实现完全一致，由 ``validate_colocated_args``
守住实现已经验证过的边界。

The encoder's parallel sizes are explicit arguments rather than implicitly inherited from the
job's mpu groups, so that a different device count / parallel layout is a validated
configuration error instead of a silent mismatch.
"""

from multimodal_args import add_multimodal_extra_args


def add_colocated_extra_args(parser):
    """Add the colocated encoder arguments on top of the multimodal ones."""
    parser = add_multimodal_extra_args(parser)

    group = parser.add_argument_group(title='colocated encoder arguments')
    group.add_argument(
        "--colocated-encoder-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Tensor model parallel size of the colocated vision encoder. Defaults to the "
        "job's --tensor-model-parallel-size, which is currently the only supported value "
        "(see validate_colocated_args).",
    )
    group.add_argument(
        "--colocated-encoder-pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Pipeline model parallel size of the colocated vision encoder. Every rank holds "
        "a full encoder replica, so the only supported value is 1.",
    )
    group.add_argument(
        "--colocated-encoder-context-parallel-size",
        type=int,
        default=1,
        help="Context parallel size of the colocated vision encoder. The encoder consumes one "
        "micro batch of images on a single rank, so the only supported value is 1.",
    )
    group.add_argument(
        "--colocated-encoder-num-distributed-optimizer-instances",
        type=int,
        default=1,
        help="Number of distributed optimizer instances across the colocated encoder's own "
        "data-parallel domain. --use-distributed-optimizer enables the distributed optimizer "
        "for both the encoder and the language model.",
    )

    return parser


def validate_colocated_args(args):
    """Fill the defaults of the colocated encoder parallel sizes and check them.

    在 ``parse_and_validate_args`` 之后、``pretrain`` 之前调用：此时作业的
    ``tensor_model_parallel_size`` 已定，可以据它补 encoder tp 的默认值。
    并行度断言对应三处实现现状，DistOpt 参数再独立校验正数与 encoder DP 域整除关系：
      * tp 必须与 backbone 相等——边界包里的 image_embeddings 是 tp 全量副本，producer 与
        consumer 按 tp 槽位一一对应发送（Task 3），tp 数不等时槽位对不上，需要额外的跨 tp
        重分发逻辑，尚未实现；
      * pp 必须为 1——每个 rank 持有完整 encoder 副本，共置 encoder 的 pipeline 组就是单成员组；
      * cp 必须为 1——encoder 在单 rank 上吃完一个 micro batch 的图像，没有序列切分；
      * encoder DistOpt instance 数必须大于 0，且能整除 encoder 自己的完整 DP 域。
    """
    if not args.use_colocated_encoder:
        return

    if args.colocated_encoder_tensor_model_parallel_size is None:
        args.colocated_encoder_tensor_model_parallel_size = args.tensor_model_parallel_size

    assert (
        args.colocated_encoder_tensor_model_parallel_size == args.tensor_model_parallel_size
    ), (
        "colocated encoder tensor model parallel size "
        f"({args.colocated_encoder_tensor_model_parallel_size}) must equal the job's tensor "
        f"model parallel size ({args.tensor_model_parallel_size}): the boundary packet carries "
        "image embeddings between the tensor-parallel slot of the same index on the producer "
        "and the consumer, so a different encoder tensor parallel size would need a cross "
        "tensor-parallel redistribution that is not implemented"
    )
    assert args.colocated_encoder_pipeline_model_parallel_size == 1, (
        "colocated encoder pipeline model parallel size must be 1, got "
        f"{args.colocated_encoder_pipeline_model_parallel_size}: every rank holds a full "
        "encoder replica, the encoder is not pipeline parallel"
    )
    assert args.colocated_encoder_context_parallel_size == 1, (
        "colocated encoder context parallel size must be 1, got "
        f"{args.colocated_encoder_context_parallel_size}: the encoder consumes one micro batch "
        "of images on a single rank, there is no sequence to split"
    )
    if args.use_distributed_optimizer:
        assert args.colocated_encoder_tensor_model_parallel_size == 1, (
            "colocated encoder distributed optimizer currently requires encoder tensor model "
            "parallel size 1; encoder TP > 1 needs an intra optimizer group spanning both the "
            "tensor-parallel and data-parallel dimensions"
        )
    assert args.colocated_encoder_num_distributed_optimizer_instances > 0, (
        "colocated encoder distributed optimizer instances must be greater than 0, got "
        f"{args.colocated_encoder_num_distributed_optimizer_instances}"
    )
    colocated_encoder_data_parallel_size = (
        args.world_size // args.colocated_encoder_tensor_model_parallel_size
    )
    assert (
        colocated_encoder_data_parallel_size
        % args.colocated_encoder_num_distributed_optimizer_instances
        == 0
    ), (
        "colocated encoder data parallel size "
        f"({colocated_encoder_data_parallel_size}) must be divisible by its number of "
        "distributed optimizer instances "
        f"({args.colocated_encoder_num_distributed_optimizer_instances})"
    )
