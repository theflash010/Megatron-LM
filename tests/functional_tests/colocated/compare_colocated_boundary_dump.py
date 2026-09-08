# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Offline checker for the dumps of ``check_colocated_boundary_and_tokens.py`` (Task 6.7 前置).

读取每个 rank 落盘的 json，检查三组不变量，全部以数字形式打印（不只打印通过/失败）：

① **覆盖性**：所有 producer 记录的 microbatch id 合起来恰好是 0..n-1，每个 id 只被一个
   producer 负责（轮盘分配的定义），consumer 记录的 id 集合与之相同。
② **一致性**：同一个 id 的 producer 摘要与 consumer 摘要逐字段相等——边界通信没有错位、
   没有把别的 microbatch 的包交给消费者。
③ **per-token 分母**：末 stage 记录的每个 microbatch 的 ``loss_mask.sum()`` 之和（本 step
   的真实 token 数）必须等于 backbone finalize 与 encoder finalize 规约后的 num_tokens。

用法：python compare_colocated_boundary_dump.py <dump_dir>
"""
import glob
import json
import os
import sys


def load_dumps(dump_directory):
    dumps = []
    for path in sorted(glob.glob(os.path.join(dump_directory, "boundary_dump_rank*.json"))):
        with open(path) as dump_file:
            dumps.append(json.load(dump_file))
    assert dumps, f"no dumps found in {dump_directory}"
    return dumps


def check_coverage(dumps):
    num_microbatches = dumps[0]["num_microbatches"]
    owner_of_id = {}
    failures = []
    for dump in dumps:
        for microbatch_id in dump["producer_packets"]:
            if microbatch_id in owner_of_id:
                failures.append(
                    f"microbatch {microbatch_id} produced by rank {owner_of_id[microbatch_id]} "
                    f"and rank {dump['rank']}"
                )
            owner_of_id[microbatch_id] = dump["rank"]
    produced = sorted(int(key) for key in owner_of_id)
    expected = list(range(num_microbatches))
    print(f"[coverage] num_microbatches={num_microbatches} produced={len(produced)}")
    for dump in dumps:
        print(
            f"  rank {dump['rank']} (producer {dump['producer_id']}/{dump['num_producers']}, "
            f"pp {dump['pipeline_rank']}): produced "
            f"{sorted(int(k) for k in dump['producer_packets'])}, consumed "
            f"{sorted(int(k) for k in dump['consumer_packets'])}"
        )
    if produced != expected:
        failures.append(f"produced ids {produced} != expected {expected}")

    consumed = sorted(
        int(key) for dump in dumps for key in dump["consumer_packets"]
    )
    if consumed != expected:
        failures.append(f"consumed ids {consumed} != expected {expected}")
    return failures


def check_packet_equality(dumps):
    producer_digests = {}
    for dump in dumps:
        for microbatch_id, digest in dump["producer_packets"].items():
            producer_digests[microbatch_id] = (dump["rank"], digest)

    failures = []
    compared = 0
    for dump in dumps:
        for microbatch_id, consumer_digest in dump["consumer_packets"].items():
            if microbatch_id not in producer_digests:
                failures.append(f"microbatch {microbatch_id} consumed but never produced")
                continue
            producer_rank, producer_digest = producer_digests[microbatch_id]
            for field in producer_digest:
                if producer_digest[field] != consumer_digest[field]:
                    failures.append(
                        f"microbatch {microbatch_id} field {field} mismatch: producer rank "
                        f"{producer_rank} {producer_digest[field]} vs consumer rank "
                        f"{dump['rank']} {consumer_digest[field]}"
                    )
            compared += 1
    print(f"[packets] compared {compared} producer/consumer packet pairs")
    return failures


def check_token_accounting(dumps):
    """末 stage 的 per-microbatch token 数之和，必须等于两侧 finalize 规约后的 num_tokens.

    只有末 pipeline stage 的 rank 会真正调用 loss_func，所以只有它们有 ``loss_num_tokens``；
    若 TP > 1，末 stage 的若干 TP rank 是彼此的副本，记录应完全相同（这里断言），本 step 的
    真实 token 数取其中一份，而不是求和。
    """
    num_microbatches = dumps[0]["num_microbatches"]
    failures = []

    dumps_with_loss = [dump for dump in dumps if dump["loss_num_tokens"]]
    print("[tokens] per-rank loss_mask.sum() records:")
    for dump in dumps:
        print(
            f"  rank {dump['rank']} (pp {dump['pipeline_rank']}): "
            f"{dump['loss_num_tokens']} (sum={sum(dump['loss_num_tokens'])})"
        )
    if not dumps_with_loss:
        failures.append("no rank recorded any loss_num_tokens")
        return failures

    pipeline_ranks = {dump["pipeline_rank"] for dump in dumps_with_loss}
    if len(pipeline_ranks) != 1:
        failures.append(
            f"loss_num_tokens recorded on multiple pipeline stages {sorted(pipeline_ranks)}"
        )

    reference = dumps_with_loss[0]
    for dump in dumps_with_loss[1:]:
        if dump["loss_num_tokens"] != reference["loss_num_tokens"]:
            failures.append(
                f"tensor-parallel replicas disagree: rank {reference['rank']} "
                f"{reference['loss_num_tokens']} vs rank {dump['rank']} {dump['loss_num_tokens']}"
            )
    if len(reference["loss_num_tokens"]) != num_microbatches:
        failures.append(
            f"rank {reference['rank']} recorded {len(reference['loss_num_tokens'])} microbatches, "
            f"expected {num_microbatches}"
        )
    expected_total = sum(reference["loss_num_tokens"])
    print(f"[tokens] step token count from loss_mask = {expected_total}")

    print("[tokens] finalize num_tokens (before / after the in-place reduce):")
    modules_seen = set()
    for dump in dumps:
        for record in dump["finalize_num_tokens"]:
            modules_seen.add(record["module"])
            print(
                f"  rank {dump['rank']} {record['module']}: "
                f"before={record['before']} after={record['after']}"
            )
            if record["after"] != expected_total:
                failures.append(
                    f"rank {dump['rank']} {record['module']} finalize num_tokens "
                    f"{record['after']} != loss_mask total {expected_total}"
                )
    for module_name in ("encoder", "language_model"):
        if module_name not in modules_seen:
            failures.append(f"no finalize record for the {module_name} chunk")
    return failures


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <dump_dir>")
        return 2
    dumps = load_dumps(sys.argv[1])
    print(f"loaded {len(dumps)} rank dumps from {sys.argv[1]}\n")

    failures = []
    failures += check_coverage(dumps)
    print()
    failures += check_packet_equality(dumps)
    print()
    failures += check_token_accounting(dumps)

    print()
    if failures:
        print(f"FAILED ({len(failures)} problems):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASSED: boundary packets match and per-token denominators agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
