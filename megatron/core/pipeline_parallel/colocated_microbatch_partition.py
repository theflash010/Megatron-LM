# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Colocated microbatch→producer partition strategies (dual-channel-p2p).

共置 encoder 的 microbatch→producer 划分策略集合。每个策略是一个独立函数，签名统一为
``(num_microbatches: int, num_producers: int) -> list[list[int]]``：第 i 项是 producer i
（producer 0 即 consumer）的 mb id 升序表，所有 mb 恰好出现一次。

切换方式：在 ``examples/multimodal/colocated_train.py`` 入口由环境变量
``COLOCATED_MICROBATCH_PARTITION`` 选择（2026-09-25 起，替代早先"手动改函数名"）：
未设或 0 = 默认轮盘；1 = reverse_block_partition；2 = consumer_head_tail_reverse_partition；
3 = consumer_head_block_partition。默认的轮盘划分（``default_round_robin_partition``）
保留在 ``examples/multimodal/colocated_args.py``；本文件放非默认的调度策略。
head-tail-reverse 系（mode 2/4）的 owned 总数档位可经 ``COLOCATED_PARTITION_PROFILE``
（逗号分隔 4 整数 consumer/p1/p2/p3）覆盖（2026-09-26 显存均衡 sweep）。

策略与调度正确性的关系（2026-09-23 定案）：供给/预取/grad 界（steady 紧界 ``i+x``、
warmup ``+1``）只依赖 wave offset x，对任意划分通用；但"phase ④ 无需分批等 grad"依赖
"最后 ≥P 个 mb 归 consumer"——此时各 producer 的 boundary grad 在其 backbone 结束前必然
全部到齐（consumer steady 覆盖 mb 0..N-P-1，尾部 P-1 个在 consumer 自己的 cooldown）。这是
策略层的构造性质，选策略时需满足（reverse 系天然满足）。

Colocated microbatch→producer partition strategies. Each strategy is a standalone function
with the unified signature ``(num_microbatches, num_producers) -> list[list[int]]``: entry i
is producer i's (producer 0 = consumer) ascending mb-id list, covering every mb exactly once.
Switch by env ``COLOCATED_MICROBATCH_PARTITION`` in ``examples/multimodal/colocated_train.py``
(2026-09-25, replacing manual function-name edits): unset/0 = default round-robin;
1 = reverse_block_partition; 2 = consumer_head_tail_reverse_partition;
3 = consumer_head_block_partition. The default round-robin stays in
``examples/multimodal/colocated_args.py``; this file hosts the non-default strategies only.
The owned-total profile of the head-tail-reverse family (modes 2/4) is overridable via env
``COLOCATED_PARTITION_PROFILE`` (memory-balance sweep, 2026-09-26).
Supply/prefetch/grad bounds are partition-agnostic; the
"no grad-arrival batching needed" property requires the last >=P microbatches to be owned by
the consumer (a construction-time property of the chosen strategy).
"""

import os


def reverse_block_partition(num_microbatches: int, num_producers: int) -> list[list[int]]:
    """Contiguous microbatch blocks in REVERSE producer order (stress: owned>D + far-stage head).

    连续 microbatch 分块、按 producer 逆序分配（16 mb / 4 producer：mb0-3→producer3、
    mb4-7→producer2、mb8-11→producer1、mb12-15→producer0）。专压两点：(1) 最前一整块 mb 集中在
    单个远端 producer；(2) 每 producer owned 数可 > D（如 16/4=4），检验"每 producer 多个 mb"的
    正确性（轮盘下 owned 分散会掩盖这条）。不整除时靠前的块多分一个，每个内层列表升序。
    满足"最后 ≥P 个 mb 归 consumer"（producer0 恒持尾部块），故 phase ④ 无需分批等 grad。
    Contiguous blocks assigned to producers in reverse order: the headmost contiguous block
    lands on the farthest producer and each producer's owned count can exceed D. Uneven
    remainder goes to the headmost blocks; each inner list is ascending. The consumer keeps
    the tail block, so all producers' boundary grads arrive before their backbone ends.
    """
    base_length, remainder = divmod(num_microbatches, num_producers)
    owned_by_producer: list[list[int]] = [[] for _ in range(num_producers)]
    microbatch_start = 0
    for chunk_index in range(num_producers):
        chunk_length = base_length + (1 if chunk_index < remainder else 0)
        producer_id = num_producers - 1 - chunk_index
        owned_by_producer[producer_id] = list(
            range(microbatch_start, microbatch_start + chunk_length)
        )
        microbatch_start += chunk_length
    return owned_by_producer


def consumer_head_block_partition(num_microbatches: int, num_producers: int) -> list[list[int]]:
    """Uniform CONTIGUOUS blocks in FORWARD order: the consumer takes the headmost block
    (experiment 2026-09-25: uniform but non-round-robin; P=4 with 64 mbs ⇒ 16/16/16/16).

    均匀连续分块、按 producer 正序分配（64 mb / 4 producer：mb0-15→consumer(producer0)、
    mb16-31→producer1、mb32-47→producer2、mb48-63→producer3；不整除时靠前的块多分一个，
    每个内层列表升序）。与轮盘的差异只在"归属连续"：consumer 的 encoder 前传集中在前 16 个
    mb（头 P 在内，warmup 全本地），三个 producer 各拿一整块。
    **不满足**"最后 ≥P 个 mb 归 consumer"的构造性质：producer3 持尾部块（含 mb63），其
    boundary grad 要到 consumer cooldown 的最末尾才到齐，phase ④ 前的 _finish_owned_grads
    等待相应变长——这正是本实验要测的时间效应（正确性不受影响：merged backward 本就排在
    全部 grad drain 之后）。
    Uniform contiguous blocks in forward order: the consumer takes the headmost block and
    each producer takes one full block (16/16/16/16 at 64 mbs, P=4; uneven remainder goes
    to the headmost blocks; each inner list ascending). Unlike round-robin, only the
    ownership is contiguous. This strategy does NOT keep the "last >=P mbs on the consumer"
    property: producer3 owns the tail block (incl. mb63), so its boundary grads arrive only
    at the very end of the consumer's cooldown, lengthening the _finish_owned_grads wait
    before phase ④ - exactly the timing effect this experiment measures (correctness is
    unaffected: the merged backward already runs after all grads are drained).
    """
    base_length, remainder = divmod(num_microbatches, num_producers)
    owned_by_producer: list[list[int]] = [[] for _ in range(num_producers)]
    microbatch_start = 0
    for producer_id in range(num_producers):
        block_length = base_length + (1 if producer_id < remainder else 0)
        owned_by_producer[producer_id] = list(
            range(microbatch_start, microbatch_start + block_length)
        )
        microbatch_start += block_length
    return owned_by_producer


# consumer_head_tail_reverse_partition 的 owned 总数档位（P=4）：consumer / producer1 /
# producer2 / producer3。中段 = 总数 - 头 P - 尾 P（consumer 的中段份额 = 12 - 8 = 4）。
# Owned-total profile for consumer_head_tail_reverse_partition (P=4): consumer / producer1 /
# producer2 / producer3. A producer's middle share = its total (the consumer's middle share
# is its total minus the head and tail P blocks).
_CONSUMER_HEAD_TAIL_REVERSE_PROFILE = [12, 14, 18, 20]


def _consumer_head_tail_profile() -> list[int]:
    """Resolve the owned-total profile, honoring the ``COLOCATED_PARTITION_PROFILE`` override.

    从环境变量 ``COLOCATED_PARTITION_PROFILE`` 解析 owned 总数档位（逗号分隔 4 个整数
    consumer/producer1/producer2/producer3，如 ``12,14,18,20``）；未设置时回退默认档
    ``_CONSUMER_HEAD_TAIL_REVERSE_PROFILE``。供 head-tail-reverse 系（mode 2/4）共用：
    一次改动即可扫任意档位（2026-09-26 显存均衡 sweep），无需逐档加函数/分支。
    Resolves the owned-total profile from env ``COLOCATED_PARTITION_PROFILE`` (four
    comma-separated ints consumer/p1/p2/p3, e.g. ``12,14,18,20``); falls back to the
    default ``_CONSUMER_HEAD_TAIL_REVERSE_PROFILE`` when unset. Shared by the
    head-tail-reverse family (modes 2/4) so any profile can be swept without new
    functions or branches (memory-balance sweep, 2026-09-26).
    """
    raw_profile = os.environ.get("COLOCATED_PARTITION_PROFILE")
    if not raw_profile:
        return list(_CONSUMER_HEAD_TAIL_REVERSE_PROFILE)
    profile = [int(part.strip()) for part in raw_profile.split(",")]
    assert len(profile) == len(_CONSUMER_HEAD_TAIL_REVERSE_PROFILE), (
        f"COLOCATED_PARTITION_PROFILE expects {len(_CONSUMER_HEAD_TAIL_REVERSE_PROFILE)} "
        f"comma-separated ints consumer/p1/p2/p3, got '{raw_profile}'"
    )
    assert all(count >= 0 for count in profile), (
        f"COLOCATED_PARTITION_PROFILE must be non-negative, got '{raw_profile}'"
    )
    return profile


def consumer_head_tail_reverse_partition(
    num_microbatches: int, num_producers: int
) -> list[list[int]]:
    """Non-uniform partition: consumer owns the head-P, tail-P and a middle remainder; the
    middle blocks tile the producers in REVERSE stage order (farthest stage gets the earliest
    and largest block).

    非均匀划分（用户 2026-09-25 策略，P=4、num_microbatches=64 档：consumer 12 /
    producer1 14 / producer2 18 / producer3 20）：
        mb0-3     → consumer（头 P，warmup 全本地）
        mb4-23    → producer3（20，最远 stage 拿最早最大的块）
        mb24-41   → producer2（18）
        mb42-55   → producer1（14）
        mb56-59   → consumer（中段尾份 = 12 - 头P - 尾P）
        mb60-63   → consumer（尾 P）
    结构性质（选此策略的原因）：① 头 P 归 consumer ⇒ warmup 阶段零跨机依赖；② 尾 P 归
    consumer ⇒ 所有 producer 的 `max(owned) ≤ N-P-1`，其 boundary grad 在自己 backbone 结束
    前必然全部到齐（consumer steady 覆盖 mb 0..N-P-1），phase ④ 每图一次 backward 即可、
    无需分批等 grad；③ 中段按逆序铺且块尺寸递增向最远 stage 倾斜，兼顾计算时间与显存对冲。
    每个内层列表升序；consumer 的列表跨头/中/尾但保持升序（合并 forward 只按样本批次
    组织、mb id 经 owner 表回填，不要求连续）。
    Non-uniform partition (user's 2026-09-25 strategy): the consumer owns the head-P block,
    the tail-P block and a middle remainder; the middle is tiled to producers in reverse
    stage order with increasing block sizes toward the farthest stage. Structural properties:
    (1) head-P on the consumer keeps warmup fully local; (2) tail-P on the consumer guarantees
    every producer's boundary grads arrive before its backbone ends (max owned <= N-P-1), so
    phase ④ runs one backward per graph with no grad-arrival batching; (3) the reverse tiling
    with size ramp balances encoder compute time and hedges activation memory. Inner lists
    are ascending; the consumer's list spans head/middle/tail but stays ascending (the merged
    forward is organized by sample batch, mb ids are recovered via the owner table).
    """
    profile = _consumer_head_tail_profile()
    assert num_producers == len(profile), (
        f"consumer_head_tail_reverse_partition: profile {profile} is defined for "
        f"{len(profile)} producers, got num_producers={num_producers}"
    )
    assert sum(profile) == num_microbatches, (
        f"consumer_head_tail_reverse_partition: profile sums to {sum(profile)} but "
        f"num_microbatches={num_microbatches}"
    )

    head_and_tail = num_producers  # 头块与尾块长度均为 P / both head and tail blocks are P long
    consumer_middle = profile[0] - 2 * head_and_tail
    owned_by_producer: list[list[int]] = [[] for _ in range(num_producers)]

    # 头 P 归 consumer / head-P block to the consumer.
    owned_by_producer[0].extend(range(0, head_and_tail))

    # 中段按逆序铺：producer3（profile 末位）拿最前的中段块，随后 producer2、producer1，
    # 最后一段（consumer 的中段份额）紧挨尾块 / middle tiling in reverse stage order: the
    # farthest producer takes the earliest middle block, the consumer's middle remainder is
    # the last one, adjacent to its tail block.
    middle_start = head_and_tail
    for producer_index in range(num_producers - 1, 0, -1):
        block_length = profile[producer_index]
        producer_id = producer_index
        owned_by_producer[producer_id] = list(
            range(middle_start, middle_start + block_length)
        )
        middle_start += block_length

    # consumer 的中段份额（紧挨尾块）+ 尾 P 归 consumer / the consumer's middle remainder
    # (adjacent to the tail block) plus the tail-P block.
    owned_by_producer[0].extend(
        range(middle_start, middle_start + consumer_middle)
    )
    owned_by_producer[0].extend(
        range(num_microbatches - head_and_tail, num_microbatches)
    )
    return owned_by_producer


def consumer_head_tail_reverse_scattered_partition(
    num_microbatches: int, num_producers: int
) -> list[list[int]]:
    """Non-uniform SCATTERED partition: head-P/tail-P stay on the consumer and the producers'
    quotas are dealt round-robin in REVERSE stage order (experiment 2026-09-25, env mode 4).

    数量档位默认非均匀 [12,14,18,20]（consumer/p1/p2/p3），可经 ``COLOCATED_PARTITION_PROFILE``
    覆盖（2026-09-26 显存均衡 sweep：consumer = 总数 − P − 三 producer 配额，自动得出），
    但中段不再连续成块，而是
    反轮盘逐个发放：mb 升序、循环 producer3→producer2→producer1，谁的配额发满就退出循环
    （p1 发满 14 后只剩 p3↔p2 交替），producer 全部发满后剩余中段一股脑归 consumer（紧挨
    尾块），最后尾 P 也归 consumer。64 mb / P=4 档的实发：
        mb0-3          → consumer（头 P）
        mb4,7,...,43   → producer3（14）
        mb46,48,50,52  → producer3（+4）
        mb54,55        → producer3（+2 → 20，发满）
        mb5,8,...,44   → producer2（14）
        mb47,49,51,53  → producer2（+4 → 18，发满）
        mb6,9,...,45   → producer1（14，最先发满）
        mb56-59        → consumer（中段剩余，一股脑）
        mb60-63        → consumer（尾 P）
    动机（2026-09-25）：头块均分（mode 3，~8.42s）≈ 非均匀连续（mode 2，~8.48s），都比轮盘
    （mode 0，~8.04s）慢 ~0.4s——代价主要来自"连续块"而非数量分布。本策略保留非均匀数量、
    把中段散开，检验散开能否追回轮盘的时间。
    结构性质：头 P 与尾 P 仍归 consumer，"phase ④ 无需分批等 grad"保持成立；散开的中段与
    默认轮盘同构（owner 表对任意划分通用，供给/grad 界不依赖划分）。
    Non-uniform scattered partition (experiment 2026-09-25, env mode 4): the owned profile
    defaults to [12,14,18,20] and is overridable via ``COLOCATED_PARTITION_PROFILE``
    (memory-balance sweep 2026-09-26: the consumer's share = total - P - the three producer
    quotas, derived automatically), and instead of contiguous tiling the middle mbs are dealt
    round-robin in ascending order over the cycle producer3→producer2→producer1, skipping
    producers whose quota is filled; once all producer quotas are filled the remaining
    middle mbs lump onto the consumer (adjacent to its tail block), and the tail-P block
    stays on the consumer. Motivation: head-block uniform (mode 3, ~8.42s) ≈ non-uniform
    contiguous (mode 2, ~8.48s), both ~0.4s slower than round-robin (mode 0, ~8.04s) - the
    cost comes from contiguity, not the count distribution. Scattering the middle tests
    whether round-robin-style spreading recovers the time. Structural properties: head-P and
    tail-P remain on the consumer (the no-grad-batching property holds); scattered middle is
    isomorphic to the default round-robin (the owner table works for any partition).
    """
    profile = _consumer_head_tail_profile()
    assert num_producers == len(profile), (
        f"consumer_head_tail_reverse_scattered_partition: profile {profile} is defined for "
        f"{len(profile)} producers, got num_producers={num_producers}"
    )
    assert sum(profile) == num_microbatches, (
        f"consumer_head_tail_reverse_scattered_partition: profile sums to {sum(profile)} "
        f"but num_microbatches={num_microbatches}"
    )

    head_and_tail = num_producers  # 头块与尾块长度均为 P / both head and tail blocks are P long
    # 中段总跨度校验：producer 配额不得超过中段长度，否则发放循环退出时仍有配额未发出。
    # Middle-span validation: producer quotas must fit the middle span, otherwise the
    # dealing loop exits with unfilled quotas.
    assert sum(profile[1:]) <= num_microbatches - 2 * head_and_tail, (
        f"consumer_head_tail_reverse_scattered_partition: producer quotas {profile[1:]} "
        f"exceed the middle span {num_microbatches - 2 * head_and_tail}"
    )
    owned_by_producer: list[list[int]] = [[] for _ in range(num_producers)]

    # 头 P 归 consumer / head-P block to the consumer.
    owned_by_producer[0].extend(range(0, head_and_tail))

    # 中段反轮盘发放：mb 升序，循环 (P-1 → ... → 1)，配额发满的 producer 跳过；
    # producer 全满后剩余中段一股脑归 consumer（紧挨尾块）。
    # Dealing loop: ascending mbs over the cycle (P-1 → ... → 1), skipping filled quotas;
    # leftover middle mbs lump onto the consumer once every quota is filled.
    producer_remaining = {
        producer_index: profile[producer_index] for producer_index in range(1, num_producers)
    }
    middle_stop = num_microbatches - head_and_tail
    microbatch_id = head_and_tail
    while microbatch_id < middle_stop and any(q > 0 for q in producer_remaining.values()):
        for producer_index in range(num_producers - 1, 0, -1):
            if producer_remaining[producer_index] <= 0:
                continue
            owned_by_producer[producer_index].append(microbatch_id)
            microbatch_id += 1
            producer_remaining[producer_index] -= 1
            if microbatch_id >= middle_stop:
                break

    # 剩余中段一股脑归 consumer + 尾 P 归 consumer。
    # Leftover middle mbs lump onto the consumer, plus the tail-P block.
    owned_by_producer[0].extend(range(microbatch_id, middle_stop))
    owned_by_producer[0].extend(range(middle_stop, num_microbatches))
    return owned_by_producer
