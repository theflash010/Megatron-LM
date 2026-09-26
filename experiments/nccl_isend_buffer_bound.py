# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sweep probe: is the unmatched-isend capacity BYTE-based (buffer occupancy) or COUNT-based
(number of in-flight sends)? (2026-09-25, follow-up to the K*=1 finding at 10MiB packets.)

背景：已实测 10MiB 包的未匹配 isend 缓冲上界 K*=1（1 个被通道缓冲吸收、第 2 个起 send
kernel 自旋等对端收包）。本脚本用三种包长各扫一遍 K，判定上界的量纲：
- **字节论**（缓冲占满）：K* 随包长缩小而放大——5MiB 的 K*≈2、2MiB 的 K*≈5-8
  （总未匹配字节大致恒定 ~10-20MiB）；
- **数量论**（isend 个数）：K* 与包长无关，恒为 1。

每个 (包长, K) 档：sender 背靠背发 K 个未匹配 isend → 跑 ~40ms matmul → device 同步；
receiver 整窗（3s）不 post 任何 irecv，然后一口气 post K 个收尾放行。compute 墙钟
≈ 纯计算 → 该 K 被缓冲吸收；墙钟 ≈ sleep 时长 → 该 K 已进自旋区。

运行（zju-a800）：
  CUDA_DEVICE_MAX_CONNECTIONS=1 uv run python -m torch.distributed.run \
      --nproc_per_node=2 experiments/nccl_isend_buffer_bound.py
"""

import os
import time

import torch
import torch.distributed as dist

COMPUTE_MILLISECONDS = 40         # sender 每档的计算量
SLEEP_SECONDS = 3                 # receiver 延迟 post 的窗口（长于任何一轮计算）
# (包长 MiB, K 序列)：10MiB 复现 K*=1 作参照；5MiB/2MiB 判定量纲。
SWEEP_PLAN = (
    (10, (1, 2, 3, 4)),
    (5, (1, 2, 3, 4, 6, 8)),
    (2, (1, 2, 4, 8, 16, 32)),
)


def _log(rank: int, message: str) -> None:
    print(f"[sweep rank{rank} {time.time():.3f}] {message}", flush=True)


def _sender_compute() -> float:
    """约 COMPUTE_MILLISECONDS 的大 matmul，末尾 device 级同步——若队列头有自旋 isend，
    墙钟被拉长到自旋解除为止。"""
    matrix_a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    matrix_b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    start = time.time()
    while (time.time() - start) < COMPUTE_MILLISECONDS / 1000:
        matrix_a @ matrix_b
    torch.cuda.synchronize()
    return time.time() - start


def _warmup(rank: int) -> None:
    buffer = torch.zeros(10 * 1024 * 1024 // 2, device="cuda", dtype=torch.bfloat16)
    if rank == 1:
        dist.isend(buffer, dst=0).wait()
    else:
        dist.irecv(buffer, src=1).wait()
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

    for size_megabytes, k_values in SWEEP_PLAN:
        payload_numel = size_megabytes * 1024 * 1024 // 2  # bf16
        for k_unmatched in k_values:
            # 每档恰好一个 barrier（2026-09-25 更正：初版 rank0 多了一个 barrier，配对数
            # 不一致直接卡死——与被测机制无关，纯脚本 bug）。
            # Exactly one barrier per K (2026-09-25 fix: the initial version had an extra
            # barrier on the receiver, mismatching the pair count).
            dist.barrier()
            if rank == 1:
                buffers = [
                    torch.full(
                        (payload_numel,), k_unmatched, device="cuda", dtype=torch.bfloat16
                    )
                    for _ in range(k_unmatched)
                ]
                handles = []
                for buffer in buffers:  # 背靠背 K 发，中间零计算
                    handles.append(dist.isend(buffer, dst=0))
                compute_seconds = _sender_compute()
                _log(
                    rank,
                    f"size={size_megabytes}MiB K={k_unmatched}: unmatched isends + compute "
                    f"= {compute_seconds * 1000:.1f} ms wall "
                    f"(pure compute ≈ {COMPUTE_MILLISECONDS} ms, "
                    f"unmatched bytes ≈ {size_megabytes * k_unmatched} MiB)",
                )
                for handle in handles:
                    handle.wait()
                _log(rank, f"size={size_megabytes}MiB K={k_unmatched}: handles waited")
            else:
                buffers = [
                    torch.empty(payload_numel, device="cuda", dtype=torch.bfloat16)
                    for _ in range(k_unmatched)
                ]
                # sender 开始发；本端整段窗口不 post 任何 irecv。
                # Sender starts issuing; this side posts NO irecv during the whole window.
                time.sleep(SLEEP_SECONDS)
                handles = [dist.irecv(buffer, src=1) for buffer in buffers]
                for handle in handles:
                    handle.wait()
                _log(
                    rank,
                    f"size={size_megabytes}MiB K={k_unmatched}: receiver done",
                )
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
