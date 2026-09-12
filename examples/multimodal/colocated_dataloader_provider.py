# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Colocated dataloader provider (Task 6.1).

共置训练的 dataloader provider。与 ``dataloader_provider.py`` 只有两处实质差异，都源于
"每个 rank 都持有完整 encoder 副本、都要自己取数"这一点：

① **取数 rank 的范围**：非共置下 dataloader 只建在流水线首/末 stage（
   ``is_first_or_last_stage``，dataloader_provider.py:71-81），因为只有那两个 stage 需要
   数据。共置下**每个 rank 都跑 encoder 前传**（唯一的 ``next(data_iterator)`` 在
   ``colocated_encoder_get_batch``，colocated_train.py），所以这道门要改按 **encoder 自己的
   pipeline 组**判断——那是单成员组，first stage 恒真 ⇒ 每个 rank 都取数；tp rank 0 取数再
   broadcast 的那层保留，但同样改用 encoder 自己的 tp 组。

② **分片域**：非共置传的是常规数据并行组（$$D_{outer}$$），共置要传**共置 dp 组（全 W
   个 rank = $$D_{outer} \\times P$$）**。这样每 rank 每 step 取 $$n/P$$ 个互不相同的
   micro batch，$$W \\cdot n/P = D_{outer} \\cdot n = \\text{GBS}/mbs$$ 恰好一遍全局 batch，
   因此 ``num_microbatches`` 计算器不用改（它仍按 ``get_data_parallel_world_size()`` =
   $$D_{outer}$$ 算）。

已核实的三件事（2026-08-30）：
  * Energon 只在 dataloader **状态存取**里用 ``data_parallel_group``（``gather_object`` /
    ``scatter_object_list``，savable_loader.py），分片本身只看 ``rank`` / ``world_size``
    ⇒ 它不假设组成员是"同一份分片的副本"，传共置组语义上正确；
  * 分片按**样本**而非整个 tar 切（``split_samples_to_workers``，sharder.py），一个 tar 可
    被切给多个 worker ⇒ 不存在"tar shard 数 >= world_size"的限制，只需样本数远大于
    ``num_workers * world_size``；
  * dataloader checkpoint 分支整段在 ``if args.load is not None`` 内（
    dataloader_provider.py），本项目 ``--load`` 恒空 ⇒ 不会进入。
"""

from dataloader_provider import EnergonDataloader, datasets_provider
from dataset_helpers import TaskEncoder

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.utils import get_pg_rank, get_pg_size
from megatron.energon import WorkerConfig, get_savable_loader
from megatron.training import get_args


def colocated_encoder_merged_batch_size():
    """Dataloader batch size for the merged encoder forward (optimization spec Task 1).

    合并前传（优化 spec doc §1.2 设计 A）要求本 rank 在一个 iteration 内的
    ``num_microbatches / num_producers`` 个 micro batch **一次取回**，因此 Energon 的
    batch_size 取 ``micro_batch_size * num_microbatches / num_producers``。两个前提在
    这里运行时钉住：

    * **不支持 rampup batch size**——dataloader 的 batch_size 在建表时固定，而 rampup
      会在训练中改变 ``num_microbatches``，两者矛盾；当前共置脚本也未使用 rampup。
    * **``num_microbatches`` 必须能被 producer 数整除**——轮盘分配
      （``get_microbatches_for_producer``）与合并粒度共用这条不变量。

    Returns:
        合并后的 batch size（样本/批）。GBS=64、MBS=1、4 producers 时为 16。
    """
    args = get_args()
    assert args.rampup_batch_size is None, (
        "the merged encoder forward fixes the dataloader batch size at build time, which "
        f"conflicts with --rampup-batch-size {args.rampup_batch_size} (num_microbatches "
        "changes per iteration); colocated training does not support rampup batch size"
    )
    num_microbatches = get_num_microbatches()
    num_producers = get_pg_size(parallel_state.get_colocated_data_parallel_group())
    assert num_microbatches % num_producers == 0, (
        f"num_microbatches ({num_microbatches}) must be divisible by the number of "
        f"producers ({num_producers}); see get_microbatches_for_producer"
    )
    return args.micro_batch_size * (num_microbatches // num_producers)


def is_colocated_dataloader_rank():
    """Check if we should have the dataloader on this encoder tensor and pipeline rank.

    两层门都按 **encoder 自己的**进程组书写，而不是作业（mpu）的组：
      * ``encoder pp 组的 first stage``——非共置那道门是
        ``is_first_or_last_stage``（dataloader_provider.py:82-92），首 stage 要 tokens/images、
        末 stage 要 labels 算 loss，中间 stage 根本不建 dataloader（``data_iterator`` 恒
        ``None``）。共置下 encoder 的 pipeline 组是单成员组（每 rank 一份完整副本），因此
        first stage 恒真、每个 rank 都取数；末 stage 也不再需要取数，backbone 的
        tokens/labels/num_image_tiles 随 ForwardPacket 到达。判断写成"encoder pp 组内 rank
        为 0"而不是删掉，才与 encoder 的拓扑一致、也才在 encoder pp 并行度改变时仍然正确。
      * ``encoder tp 组的 rank 0``——取数后在 encoder tp 组内 broadcast（
        ``colocated_encoder_get_batch``），取数 rank 必须与 broadcast 的源 rank 是同一个组
        内的同一个成员，否则建了 dataloader 的 rank 与广播源不一致会挂死。
    """
    encoder_pipeline_group = parallel_state.get_colocated_encoder_pipeline_model_parallel_group()
    encoder_tensor_group = parallel_state.get_colocated_encoder_tensor_model_parallel_group()
    return get_pg_rank(encoder_pipeline_group) == 0 and get_pg_rank(encoder_tensor_group) == 0


def colocated_train_valid_test_dataloaders_provider(
    train_val_test_num_samples, task_encoder=None
):
    """Build the colocated train dataloader (no validation / test).

    返回值形状与非共置 provider 一致（三元组），但验证与测试恒为 None：共置不支持评估，
    ``arguments.py`` 的共置校验块已要求 ``--eval-iters 0`` 且禁止
    ``--online-evaluation-config``。
    """
    args = get_args()
    assert parallel_state.is_colocated_encoder_enabled(), (
        "colocated_train_valid_test_dataloaders_provider requires colocated encoder "
        "training; pass --use-colocated-encoder or use the non-colocated provider in "
        "dataloader_provider.py"
    )

    if task_encoder is None:
        task_encoder = TaskEncoder()

    if not is_colocated_dataloader_rank():
        return None, None, None

    # 共置 dp 组 = 全部 W 个 rank；rank / world_size 必须与该组一致，Energon 的分片
    # （sharder.py 第 4 步取 ``offsets[rank*nw : (rank+1)*nw+1]``）就是按这两个值划的。
    # The colocated data-parallel group spans all W ranks; Energon shards purely by
    # rank / world_size, so both must come from that same group.
    colocated_data_parallel_group = parallel_state.get_colocated_data_parallel_group()
    worker_config = WorkerConfig(
        rank=get_pg_rank(colocated_data_parallel_group),
        world_size=get_pg_size(colocated_data_parallel_group),
        num_workers=args.num_workers,
        data_parallel_group=colocated_data_parallel_group,
        worker_debug_path=None,
        worker_log_level=0,
    )

    train_dataset, _, _ = datasets_provider(
        task_encoder,
        worker_config,
        build_validation=False,
        batch_size=colocated_encoder_merged_batch_size(),
    )
    train_dataloader = get_savable_loader(train_dataset, worker_config=worker_config)

    return EnergonDataloader(train_dataloader), None, None
