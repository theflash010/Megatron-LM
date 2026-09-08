# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Fixed micro batch storage shared by the colocated and non-colocated comparison drivers (Task 6.7 ⓐ).

逐元素对照的第一个前置条件是**两侧吃到完全相同的数据**。不能靠 dataloader 复现：两侧的
分片域不同（共置是全 W 的共置 dp 组，非共置是常规 dp 组），Energon 的取样顺序也不由我们
控制。做法是把若干个 micro batch 的原始 dataloader 输出落盘成 ``.pt``，两侧都从文件读。

落盘的是 **dataloader 的原始字段**（``get_batch`` 之前），不是变换后的张量：这样两侧各自
跑自己的 ``get_batch``（broadcast_data、labels 左移、loss_mask 重建），那段代码也一并进入
对照范围；若只存变换后的结果，就等于假定变换一致，而它恰恰是要被验证的东西之一。

六个字段取自两侧 ``get_batch`` 实际读的键：共置侧读 4 个
（colocated_train.py:97-109），非共置侧多读 ``cu_lengths`` / ``max_lengths``
（train.py:71-72，用于构造 PackedSeqParams），故按并集存。
"""
import os

import torch

# 两侧 get_batch 读到的字段并集。The union of the fields both get_batch paths read.
FIXED_BATCH_KEYS = ("tokens", "labels", "imgs", "num_tiles", "cu_lengths", "max_lengths")


def batch_field(batch, key):
    """Read one field of an Energon batch (dict-like or dataclass)."""
    try:
        return batch[key]
    except TypeError:
        return getattr(batch, key)


def batch_sample_keys(batch):
    """Return the sample keys of one micro batch as a list of strings.

    ``__key__`` 打包时是 ``",".join(...)``（dataset_helpers.py:932）或每样本一项的列表（:852）。
    """
    keys = getattr(batch, "__key__", None)
    if keys is None:
        keys = batch["__key__"]
    if isinstance(keys, str):
        return keys.split(",")
    return [str(key) for key in keys]


def micro_batch_path(directory, microbatch_id):
    return os.path.join(directory, f"micro_batch_{microbatch_id}.pt")


def save_micro_batch(batch, microbatch_id, directory):
    """Save the six raw fields (+ sample keys) of one micro batch to a ``.pt`` file."""
    payload = {key: batch_field(batch, key).detach().cpu() for key in FIXED_BATCH_KEYS}
    payload["__key__"] = batch_sample_keys(batch)
    torch.save(payload, micro_batch_path(directory, microbatch_id))
    return payload


def load_micro_batch(microbatch_id, directory):
    """Load one micro batch; the returned dict is what ``get_batch`` indexes by key."""
    return torch.load(micro_batch_path(directory, microbatch_id), weights_only=False)


class FixedMicroBatchIterator:
    """Iterator over a fixed list of stored micro batches, cycling forever.

    ``microbatch_ids`` 是**本 rank 在本 step 内按顺序消费的 microbatch 号**——共置侧是
    ``get_microbatches_for_producer(p, n, P)``，非共置侧是 ``range(n)``。循环（而不是耗尽）
    使每个 iteration 都吃到同一批数据：对照要的是可复现，不是遍历数据集。
    """

    def __init__(self, microbatch_ids, directory):
        assert microbatch_ids, "the fixed micro batch iterator needs at least one microbatch id"
        self.microbatch_ids = list(microbatch_ids)
        self.directory = directory
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        microbatch_id = self.microbatch_ids[self.position % len(self.microbatch_ids)]
        self.position += 1
        return load_micro_batch(microbatch_id, self.directory)
