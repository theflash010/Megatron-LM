# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Prescan driver for the encoder batch-merge optimization (optimization spec Task 1.1).

不建模型、不训练，只建共置 dataloader 取数，回答合并计算方案的三个前置问题：

  ① ``num_tiles`` 是否恒定 —— 决定合并后的 image_embeddings 按 batch 维**等分**，
     还是必须按 ``num_tiles.cumsum()`` 做**变长切分**；
  ② 文本 token 长度分布，以及 batch 化（padding 宽度从"单样本长度"变成"组内最大
     长度"）之后是否仍满足下面这条约束——超了就会在
     ``colocated_llava_model._preprocess_data`` 触发截断：

         L_text + sum(num_tiles) * img_seq_len - num_images <= decoder_seq_length

  ③ 把 dataloader 的 batch_size 从 ``micro_batch_size`` 提到
     ``micro_batch_size * (num_microbatches / num_producers)`` 之后，各字段的真实形状。

Prescan only: builds the colocated dataloader twice (current batch size, then the merged
batch size) and reports the distributions that decide the merge design.
"""
import os
import statistics
import sys
from collections import Counter

import torch

# ``examples/multimodal`` 下的模块按"脚本同目录导入"书写，先把该目录与仓库根目录加入 sys.path。
MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from colocated_args import add_colocated_extra_args, validate_colocated_args
from colocated_dataloader_provider import colocated_train_valid_test_dataloaders_provider
from train import llava_embedding_ranks, llava_position_embedding_ranks

from megatron.core import parallel_state as mpu
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.utils import get_pg_rank, get_pg_size
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.arguments import parse_and_validate_args
from megatron.training.initialize import initialize_megatron

# 取多少个当前粒度（micro_batch_size）的 batch 做逐样本统计。
MICRO_BATCHES_TO_SCAN = int(os.environ.get("PRESCAN_MICRO_BATCHES", "160"))
# 第二段用合并粒度取多少个大 batch。
MERGED_BATCHES_TO_SCAN = int(os.environ.get("PRESCAN_MERGED_BATCHES", "4"))
def field_of(batch, name):
    """Read one field of an Energon batch (dataclass or dict)."""
    value = getattr(batch, name, None)
    if value is None and isinstance(batch, dict):
        value = batch[name]
    return value


def text_token_lengths(tokens, pad_token_id):
    """Return the real (non-pad) text token count of every sample in the batch.

    batcher 左对齐填充、尾部补 pad（``dataset_helpers.py::batch``），所以真实长度 =
    总宽度 - 尾部连续 pad 数。中间也可能出现 pad（罕见），因此按"最后一个非 pad 的下标"算，
    这与 batcher 的 ``tokens[i, :text_len] = s.tokens[:text_len]`` 语义一致。
    """
    lengths = []
    for row in tokens:
        non_pad_positions = (row != pad_token_id).nonzero()
        if non_pad_positions.numel() == 0:
            lengths.append(0)
        else:
            lengths.append(int(non_pad_positions[-1].item()) + 1)
    return lengths


def describe(name, values):
    """Print min/mean/p50/p90/p99/max of an integer sample."""
    if not values:
        print_rank_0(f"  {name}: (empty)")
        return
    ordered = sorted(values)
    count = len(ordered)

    def percentile(fraction):
        return ordered[min(count - 1, int(fraction * count))]

    print_rank_0(
        f"  {name}: n={count} min={ordered[0]} mean={statistics.mean(ordered):.1f} "
        f"p50={percentile(0.5)} p90={percentile(0.9)} p99={percentile(0.99)} max={ordered[-1]}"
    )
def scan_per_sample(data_iterator, batches_to_scan, image_tokens_per_tile, pad_token_id):
    """Pull batches at the CURRENT granularity and collect per-sample statistics."""
    args = get_args()
    per_sample_text_lengths = []
    per_sample_tiles = []
    per_batch_padded_width = []
    per_sample_total_length = []
    tiles_counter = Counter()
    text_only_batches = 0

    for _ in range(batches_to_scan):
        batch = next(data_iterator)
        tokens = field_of(batch, "tokens")
        num_tiles = field_of(batch, "num_tiles")
        images = field_of(batch, "imgs")

        # 文本-only 样本的哨兵形状（``colocated_encoder_get_batch`` 也按它判断）。
        if tuple(images.shape) == (1, 1):
            text_only_batches += 1

        lengths = text_token_lengths(tokens, pad_token_id)
        per_sample_text_lengths.extend(lengths)
        per_batch_padded_width.append(int(tokens.shape[1]))

        tiles_list = [int(x) for x in num_tiles.flatten().tolist()]
        per_sample_tiles.extend(tiles_list)
        tiles_counter.update(tiles_list)

        # 组合序列长度：图像 embedding 展开后与 decoder_seq_length 比较的那个量。
        # 每个 batch 内 tiles 与 images 的对应关系由 num_tiles 的展平顺序给出，MBS=1 时
        # 一个样本一项；这里按"整个 batch 的 tiles 总数 / 样本数"保守地按样本聚合。
        sample_count = int(tokens.shape[0])
        tiles_per_sample = max(1, len(tiles_list) // max(1, sample_count))
        for index, length in enumerate(lengths):
            begin = index * tiles_per_sample
            end = begin + tiles_per_sample
            sample_tiles = sum(tiles_list[begin:end]) or 1
            images_in_sample = max(1, tiles_per_sample)
            per_sample_total_length.append(
                length + sample_tiles * image_tokens_per_tile - images_in_sample
            )

    budget = args.decoder_seq_length - image_tokens_per_tile + 1
    over_budget = [x for x in per_sample_text_lengths if x > budget]
    over_total = [x for x in per_sample_total_length if x > args.decoder_seq_length]

    print_rank_0("")
    print_rank_0(f"=== per-sample scan ({batches_to_scan} batches of size {args.micro_batch_size}) ===")
    describe("text token length (real, non-pad)", per_sample_text_lengths)
    describe("batch padded width (tokens.shape[1])", per_batch_padded_width)
    describe("combined length (text + tiles*img_tok - imgs)", per_sample_total_length)
    print_rank_0(f"  num_tiles histogram: {dict(sorted(tiles_counter.items()))}")
    print_rank_0(f"  text-only batches (imgs shape [1,1]): {text_only_batches}")
    print_rank_0(
        f"  text budget = decoder_seq_length({args.decoder_seq_length}) - "
        f"img_tokens_per_tile({image_tokens_per_tile}) + 1 = {budget}"
    )
    print_rank_0(
        f"  samples with text length > budget: {len(over_budget)}/{len(per_sample_text_lengths)}"
    )
    print_rank_0(
        f"  samples with combined length > decoder_seq_length: "
        f"{len(over_total)}/{len(per_sample_total_length)}"
    )
    return per_sample_text_lengths, per_sample_tiles
def simulate_merged_padding(text_lengths, merge_factor):
    """Simulate the padded width after merging: max over each group of ``merge_factor`` samples.

    合并后 batcher 的 ``max_seq_len`` 取的是**组内最大文本长度**（``dataset_helpers.py::batch``
    在 ``--dataloader-seq-length`` 未设时走这条），因此把逐样本长度按 merge_factor 分组取 max，
    即为合并后每个 micro batch 的 padding 宽度。
    """
    groups = [
        text_lengths[begin : begin + merge_factor]
        for begin in range(0, len(text_lengths) - merge_factor + 1, merge_factor)
    ]
    return [max(group) for group in groups if group]
def scan_merged(image_tokens_per_tile, pad_token_id, merged_batch_size):
    """Rebuild the dataloader at the MERGED batch size and report the real field shapes."""
    args = get_args()
    original_batch_size = args.micro_batch_size
    # provider 只把 ``args.micro_batch_size`` 透给 Energon 的 batcher（dataloader_provider.py:42），
    # 改这一个字段即可模拟合并后的取数粒度；模型/并行状态在本 driver 里根本没建。
    args.micro_batch_size = merged_batch_size
    try:
        merged_dataloader, _, _ = colocated_train_valid_test_dataloaders_provider(None)
        assert merged_dataloader is not None
        print_rank_0("")
        print_rank_0(f"=== merged scan (dataloader batch_size = {merged_batch_size}) ===")
        padded_widths = []
        tiles_counter = Counter()
        for index in range(MERGED_BATCHES_TO_SCAN):
            batch = next(merged_dataloader)
            tokens = field_of(batch, "tokens")
            labels = field_of(batch, "labels")
            images = field_of(batch, "imgs")
            num_tiles = field_of(batch, "num_tiles")
            padded_widths.append(int(tokens.shape[1]))
            tiles_counter.update(int(x) for x in num_tiles.flatten().tolist())
            lengths = text_token_lengths(tokens, pad_token_id)
            print_rank_0(
                f"  batch {index}: tokens={tuple(tokens.shape)} labels={tuple(labels.shape)} "
                f"imgs={tuple(images.shape)} num_tiles={tuple(num_tiles.shape)} "
                f"real text len min={min(lengths)} max={max(lengths)}"
            )
        # 合并后每个样本都被 pad 到组内最大宽度，组合长度按该宽度算（单图单 tile）。
        worst_case_combined = [width + image_tokens_per_tile - 1 for width in padded_widths]
        print_rank_0(f"  padded widths: {padded_widths}")
        print_rank_0(
            f"  worst-case combined length (padded width + {image_tokens_per_tile} - 1): "
            f"{worst_case_combined}  vs decoder_seq_length={args.decoder_seq_length}"
        )
        print_rank_0(f"  num_tiles histogram: {dict(sorted(tiles_counter.items()))}")
        return padded_widths
    finally:
        args.micro_batch_size = original_batch_size
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
        "This prescan only covers the colocated data path; pass --use-colocated-encoder."
    )

    # 每个 tile 的图像 token 数：与 model.py:53 完全同一个调用，避免用脚本里写的
    # ``--seq-length``（那个值会被 model.py:66 覆写）。
    image_tokens_per_tile = get_num_image_embeddings(
        arguments.img_h,
        arguments.img_w,
        arguments.patch_dim,
        arguments.vision_model_type,
        arguments.disable_vision_class_token,
        1,
        arguments.pixel_shuffle,
        arguments.use_tile_tags,
        arguments.max_num_tiles,
        arguments.tokenizer_prompt_format,
    )
    pad_token_id = get_tokenizer().pad
    colocated_data_parallel_group = mpu.get_colocated_data_parallel_group()
    num_producers = get_pg_size(colocated_data_parallel_group)
    num_microbatches = get_num_microbatches()
    merge_factor = num_microbatches // num_producers
    merged_batch_size = arguments.micro_batch_size * merge_factor

    print_rank_0("")
    print_rank_0("=== configuration ===")
    print_rank_0(
        f"  world={torch.distributed.get_world_size()} colocated_dp(producers)={num_producers} "
        f"num_microbatches={num_microbatches} merge_factor={merge_factor}"
    )
    print_rank_0(
        f"  micro_batch_size={arguments.micro_batch_size} -> merged batch_size={merged_batch_size} "
        f"global_batch_size={arguments.global_batch_size}"
    )
    print_rank_0(
        f"  img_h={arguments.img_h} img_w={arguments.img_w} patch_dim={arguments.patch_dim} "
        f"use_tiling={arguments.use_tiling} max_num_tiles={arguments.max_num_tiles} "
        f"use_thumbnail={arguments.use_thumbnail}"
    )
    print_rank_0(
        f"  image tokens per tile (get_num_image_embeddings) = {image_tokens_per_tile}; "
        f"hidden_size(h_lang)={arguments.hidden_size}; "
        f"dataloader_seq_length={arguments.dataloader_seq_length}; "
        f"packing_buffer_size={arguments.packing_buffer_size}; "
        f"decoder_seq_length={arguments.decoder_seq_length}"
    )
    packet_bytes = image_tokens_per_tile * arguments.micro_batch_size * arguments.hidden_size * 2
    print_rank_0(
        f"  one boundary packet image_embeddings (bf16) = {image_tokens_per_tile} x "
        f"{arguments.micro_batch_size} x {arguments.hidden_size} x 2B = "
        f"{packet_bytes / 1e6:.2f} MB; all {num_microbatches} = "
        f"{packet_bytes * num_microbatches / 1e6:.1f} MB"
    )

    train_dataloader, _, _ = colocated_train_valid_test_dataloaders_provider(None)
    assert train_dataloader is not None, "every rank must own a dataloader under colocated training"

    text_lengths, _ = scan_per_sample(
        train_dataloader, MICRO_BATCHES_TO_SCAN, image_tokens_per_tile, pad_token_id
    )

    simulated_widths = simulate_merged_padding(text_lengths, merge_factor)
    print_rank_0("")
    print_rank_0(f"=== simulated merged padding (groups of {merge_factor} samples) ===")
    describe("simulated padded width (group max)", simulated_widths)
    if simulated_widths:
        worst = max(simulated_widths) + image_tokens_per_tile - 1
        print_rank_0(
            f"  worst-case combined length = {worst} vs decoder_seq_length="
            f"{arguments.decoder_seq_length} -> "
            f"{'OK' if worst <= arguments.decoder_seq_length else 'TRUNCATION'}"
        )

    scan_merged(image_tokens_per_tile, pad_token_id, merged_batch_size)

    torch.distributed.barrier()
    print_rank_0("")
    print_rank_0(">>> prescan finished")
    torch.distributed.destroy_process_group()
