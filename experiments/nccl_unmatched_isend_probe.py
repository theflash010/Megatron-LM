# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Minimal probe: does a SECOND unmatched isend head-of-line-block the sender's own compute
under CUDA_DEVICE_MAX_CONNECTIONS=1? (2026-09-25, follow-up to the steady-slack experiment.)

背景：steady slack=1 实验中 producer 一拍内发出两个 isend（第二个无匹配 irecv），=1 下整个
rank 挂死、且与分配器冷热无关（iter1 冷挂 / iter21 暖挂）。本脚本把该机制抽成 2-rank 最小
复现，单独回答"两个背靠背 isend 会不会堵塞"：

- Scenario A（对照）：receiver 开局就把两个 irecv 都 post 上 → 两个 isend 立即匹配，
  sender 的后续计算应立即执行（时长 ≈ 纯计算量）。
- Scenario B（探测）：receiver 只 post irecv #1，等数据、算一段、host sleep 数秒，最后才
  post irecv #2。sender 侧固定顺序：isend #1 → isend #2（均 wait=False）→ 一段 matmul
  计算。若"未匹配 isend 在 =1 单队列队头自旋、挡住其后所有 kernel"成立，sender 的计算要
  拖到 receiver post irecv #2 之后才完成（时长 ≈ sleep 时长）；若不受影响 → 队头阻塞
  假设被否定。

判定标准（sender 侧 compute 墙钟；kernel 单队列按发射序执行）：
  B ≈ 纯计算量（且 << sleep 时长）→ 无队头阻塞，机制否定；
  B ≈ receiver 的延迟（sleep 时长）→ 未匹配 isend 队头阻塞坐实。

运行（zju-a800，=1 与 =2 各跑一遍对比；注意先 export CUDA_DEVICE_MAX_CONNECTIONS）：
  CUDA_DEVICE_MAX_CONNECTIONS=1 uv run python -m torch.distributed.run \
      --nproc_per_node=2 experiments/nccl_unmatched_isend_probe.py
"""

import os
import time

import torch
import torch.distributed as dist

PAYLOAD_BYTES = 10 * 1024 * 1024      # ≈ 真实边界包 image_embeddings 的大小
PAYLOAD_NUMEL = PAYLOAD_BYTES // 2    # bf16
COMPUTE_MILLISECONDS = 40             # sender 每场景的计算量（循环 4096³ bf16 matmul）
SLEEP_SECONDS = 3                     # scenario B 中 receiver 延迟 post 第二个 irecv 的时长


def _log(rank: int, message: str) -> None:
    print(f"[probe rank{rank} {time.time():.3f}] {message}", flush=True)


def _sender_compute() -> float:
    """跑约 COMPUTE_MILLISECONDS 的大 matmul，返回墙钟秒；末尾同步，测的是 GPU 真实完成点。

    单队列按发射序执行：若队头被未匹配 isend 占住，这里的墙钟会被拉长到阻塞解除为止。
    Run ~COMPUTE_MILLISECONDS of large matmuls and return the wall-clock seconds; the trailing
    synchronize makes this the GPU completion point, which stretches under head-of-line
    blocking.
    """
    matrix_a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    matrix_b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    start = time.time()
    while (time.time() - start) < COMPUTE_MILLISECONDS / 1000:
        matrix_a @ matrix_b
    torch.cuda.synchronize()
    return time.time() - start


def _warmup(rank: int) -> None:
    """一发配对的 isend/irecv，把 NCCL 首次连接的握手开销排除在计时之外。"""
    buffer = torch.zeros(PAYLOAD_NUMEL, device="cuda", dtype=torch.bfloat16)
    if rank == 1:
        dist.isend(buffer, dst=0).wait()
    else:
        dist.irecv(buffer, src=1).wait()
    dist.barrier()


def run_scenario(scenario: str, rank: int) -> None:
    # 2026-09-25 更正：数据 op（isend/irecv）必须在 barrier 之后 post——同一 PG 的 NCCL op
    # 按 CPU 发射序执行，本脚本初版在 receiver 上"先 post irecv 再 barrier"、sender 上
    # "先 barrier 再 isend"，barrier 被排在 irecv 之后等数据、isend 又排在 barrier 之后，
    # 交叉成环直接死锁（与要测的机制无关，纯脚本排序 bug）。
    # 2026-09-25 fix: data ops (isend/irecv) MUST be posted after the barrier - NCCL ops on
    # one PG execute in CPU issue order, and the initial ordering (receiver posted irecvs
    # before its barrier, sender issued isends after its barrier) closed a barrier<->recv
    # cycle unrelated to the mechanism under test.
    dist.barrier()
    if rank == 1:
        send_buffer_1 = torch.full((PAYLOAD_NUMEL,), 1, device="cuda", dtype=torch.bfloat16)
        send_buffer_2 = torch.full((PAYLOAD_NUMEL,), 2, device="cuda", dtype=torch.bfloat16)
        dist.barrier()  # 双方就位后才开始发 / both sides ready before any data op
        _log(rank, f"{scenario}: isend #1 -> isend #2 -> compute BEGIN")
        handle_1 = dist.isend(send_buffer_1, dst=0)
        handle_2 = dist.isend(send_buffer_2, dst=0)
        compute_seconds = _sender_compute()
        _log(
            rank,
            f"{scenario}: compute DONE in {compute_seconds * 1000:.1f} ms "
            f"(pure compute ≈ {COMPUTE_MILLISECONDS} ms)",
        )
        handle_1.wait()
        handle_2.wait()
        _log(rank, f"{scenario}: both isend handles completed")
    else:
        recv_buffer_1 = torch.empty(PAYLOAD_NUMEL, device="cuda", dtype=torch.bfloat16)
        recv_buffer_2 = torch.empty(PAYLOAD_NUMEL, device="cuda", dtype=torch.bfloat16)
        dist.barrier()  # 双方就位后再 post 数据 op / data ops posted only after this point
        if scenario == "A":
            handle_1 = dist.irecv(recv_buffer_1, src=1)
            handle_2 = dist.irecv(recv_buffer_2, src=1)
            handle_1.wait()
            handle_2.wait()
            _log(rank, "A: both irecvs posted after barrier, both completed")
        else:
            handle_1 = dist.irecv(recv_buffer_1, src=1)
            handle_1.wait()
            _log(rank, "B: irecv #1 completed; compute + sleep before posting #2")
            _sender_compute()
            time.sleep(SLEEP_SECONDS)
            _log(rank, "B: posting irecv #2 NOW")
            handle_2 = dist.irecv(recv_buffer_2, src=1)
            handle_2.wait()
            _log(rank, "B: irecv #2 completed")
    dist.barrier()


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    _log(
        rank,
        "CUDA_DEVICE_MAX_CONNECTIONS="
        f"{os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset')}",
    )
    _warmup(rank)
    for scenario in ("A", "B"):
        if rank == 0:
            _log(rank, f"=== scenario {scenario} ===")
        run_scenario(scenario, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
