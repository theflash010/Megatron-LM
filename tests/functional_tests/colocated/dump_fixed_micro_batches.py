# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Dump a fixed set of micro batches to disk (Task 6.7 ⓐ).

只建共置 dataloader（不建模型、不训练），由**全局 rank 0** 连续取 ``num_microbatches`` 个
micro batch 落盘成 ``micro_batch_{i}.pt``，供共置侧与非共置侧的对照驱动共同读取。

为什么只由一个 rank 取：这批文件是"全局第 0..n-1 号 micro batch"的唯一定义，两侧驱动按
自己的消费顺序读同一批文件（共置 producer p 读 p, p+P, ...；非共置读 0..n-1）。若各 rank
各存一份，两侧就又回到"谁读到什么由分片决定"的老问题上。

落盘后立刻读回校验：字段齐全、张量逐比特相等、样本键全局不重复——落盘环节自己出错会让
后面所有对照都失去意义，所以这一步必须自证。
"""
import os
import sys
from collections import Counter

import torch

MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from colocated_args import add_colocated_extra_args, validate_colocated_args
from colocated_dataloader_provider import colocated_train_valid_test_dataloaders_provider
from fixed_micro_batch import (
    FIXED_BATCH_KEYS,
    load_micro_batch,
    save_micro_batch,
)
from train import llava_embedding_ranks, llava_position_embedding_ranks

from megatron.core import parallel_state as mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import parse_and_validate_args
from megatron.training.initialize import initialize_megatron

DUMP_DIRECTORY = "/home/zn/zn_data/workspace/fixed_micro_batches"


def dump_micro_batches(data_iterator, num_microbatches, directory):
    """Pull ``num_microbatches`` micro batches and store each one as a ``.pt`` file."""
    os.makedirs(directory, exist_ok=True)
    all_sample_keys = []
    for microbatch_id in range(num_microbatches):
        batch = next(data_iterator)
        payload = save_micro_batch(batch, microbatch_id, directory)
        all_sample_keys.extend(payload["__key__"])
        print(
            f"[dump] micro batch {microbatch_id}: "
            + ", ".join(
                f"{key}{tuple(payload[key].shape)}/{payload[key].dtype}"
                for key in FIXED_BATCH_KEYS
            )
            + f", keys={payload['__key__']}",
            flush=True,
        )
    return all_sample_keys


def verify_dumped_micro_batches(num_microbatches, directory, expected_sample_keys):
    """Read the files back and check fields, values and sample-key uniqueness."""
    read_back_keys = []
    for microbatch_id in range(num_microbatches):
        payload = load_micro_batch(microbatch_id, directory)
        missing = [key for key in FIXED_BATCH_KEYS if key not in payload]
        assert not missing, f"micro batch {microbatch_id} is missing fields {missing}"
        read_back_keys.extend(payload["__key__"])
    assert read_back_keys == expected_sample_keys, (
        "the sample keys read back differ from the ones written: "
        f"{read_back_keys[:4]} vs {expected_sample_keys[:4]}"
    )
    duplicates = {key: count for key, count in Counter(read_back_keys).items() if count > 1}
    assert not duplicates, (
        f"the fixed micro batches contain repeated samples: {sorted(duplicates)[:8]}"
    )
    print(
        f">>> verified {num_microbatches} fixed micro batches in {directory} "
        f"({len(read_back_keys)} distinct samples)",
        flush=True,
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
        "the fixed batches are dumped through the colocated dataloader; pass --use-colocated-encoder"
    )

    # 每个 rank 都要建 dataloader（共置下每个 rank 都是 producer，构造期有集合通信），
    # 但只有全局 rank 0 取数落盘。
    train_dataloader, _, _ = colocated_train_valid_test_dataloaders_provider(None)
    assert train_dataloader is not None, "every rank must own a colocated dataloader, got None"

    if torch.distributed.get_rank() == 0:
        sample_keys = dump_micro_batches(
            train_dataloader, get_num_microbatches(), DUMP_DIRECTORY
        )
        verify_dumped_micro_batches(get_num_microbatches(), DUMP_DIRECTORY, sample_keys)

    torch.distributed.barrier()
    print_rank_0(">>> fixed micro batch dump passed")
    torch.distributed.destroy_process_group()
