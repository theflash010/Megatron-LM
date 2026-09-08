# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Dataloader-only driver of the colocated path (Task 6.6).

只建共置 dataloader、取若干个 micro batch，然后校验**全部 W 个 rank 取到的样本互不重复**——
这是"分片域从常规 dp 组放大到共置 dp 组"这条设计的唯一硬指标（doc §2.2）：encoder 在
pipeline 维上是副本，如果分片域仍是常规 dp 组，同一个外层副本内的 P 个 rank 会取到**同一批**
样本，一步就把同一批数据算了 P 次。

不建模型、不训练：这样验证一次只花建数据集的时间，失败时的信息也不会被训练日志淹没。
Builds only the colocated dataloader and checks that no sample key repeats across the W ranks.
"""
import os
import sys
from collections import Counter

import torch

# ``examples/multimodal`` 下的模块按"脚本同目录导入"的方式书写，必须先把该目录与仓库根目录
# 加入 ``sys.path``。本文件位于 ``<repo>/tests/functional_tests/colocated/``，仓库根目录上溯三级。
MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from colocated_args import add_colocated_extra_args, validate_colocated_args
from colocated_dataloader_provider import colocated_train_valid_test_dataloaders_provider
from train import llava_embedding_ranks, llava_position_embedding_ranks

from megatron.core import parallel_state as mpu
from megatron.core.utils import get_pg_rank, get_pg_size
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import parse_and_validate_args
from megatron.training.initialize import initialize_megatron

MICRO_BATCHES_PER_RANK = 8


def batch_sample_keys(batch):
    """Return the sample keys of one micro batch as a list of strings.

    Energon 的 batch 是 dataclass（``ImageTaskBatchPacked``，dataset_helpers.py:71-81），
    ``__key__`` 打包时是 ``",".join(...)``（:932）或每个样本一项的列表（:852），两种都处理。
    """
    keys = getattr(batch, "__key__", None)
    if keys is None:
        keys = batch["__key__"]
    if isinstance(keys, str):
        return keys.split(",")
    return [str(key) for key in keys]


def check_no_duplicate_samples(data_iterator):
    """Pull micro batches on every rank and assert global uniqueness over the colocated dp group."""
    local_keys = []
    for _ in range(MICRO_BATCHES_PER_RANK):
        local_keys.extend(batch_sample_keys(next(data_iterator)))

    colocated_data_parallel_group = mpu.get_colocated_data_parallel_group()
    group_size = get_pg_size(colocated_data_parallel_group)
    gathered_keys = [None] * group_size
    torch.distributed.all_gather_object(
        gathered_keys, local_keys, group=colocated_data_parallel_group
    )

    print(
        f"[rank {torch.distributed.get_rank()}] colocated dp rank "
        f"{get_pg_rank(colocated_data_parallel_group)}/{group_size}: "
        f"{len(local_keys)} samples, first={local_keys[0]}, last={local_keys[-1]}",
        flush=True,
    )

    all_keys = [key for rank_keys in gathered_keys for key in rank_keys]
    duplicates = {key: count for key, count in Counter(all_keys).items() if count > 1}
    assert not duplicates, (
        f"the same sample was loaded by more than one rank: {sorted(duplicates)[:8]} "
        f"({len(duplicates)} duplicated keys out of {len(all_keys)} samples). The colocated "
        "dataloader must shard over the colocated data parallel group (all W ranks), not over "
        "the regular data parallel group"
    )
    print_rank_0(
        f">>> no duplicate samples: {len(all_keys)} distinct samples over {group_size} ranks "
        f"({MICRO_BATCHES_PER_RANK} micro batches each)"
    )


if __name__ == "__main__":
    arguments = parse_and_validate_args(
        extra_args_provider=add_colocated_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    validate_colocated_args(arguments)
    initialize_megatron(
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
    arguments = get_args()
    assert mpu.is_colocated_encoder_enabled(), (
        "This driver only covers the colocated dataloader; pass --use-colocated-encoder."
    )

    train_dataloader, _, _ = colocated_train_valid_test_dataloaders_provider(None)
    assert train_dataloader is not None, (
        "every rank must own a dataloader under colocated training (TP=1 here), got None"
    )
    check_no_duplicate_samples(train_dataloader)

    torch.distributed.barrier()
    print_rank_0(">>> colocated dataloader verification passed")
    torch.distributed.destroy_process_group()
