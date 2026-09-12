# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Minimal demo: a big NCCL P2P transfer overlapping a long GEMM chain, for nsys viewing.

最小演示：rank0 ``isend`` 2 GiB、rank1 ``irecv``，两侧同时入队一段长 GEMM 链。
P2P 收发先 post、计算后入队：两侧立即匹配上，``ncclDevKernel_SendRecv`` 在 NCCL
内部流上立刻开跑，与 compute stream 上的 sgemm 链在墙钟上重叠——nsys 时间线里
两条 lane 同时有 bar，即通算并行的实物例子。

运行（zju-a800，无 torchrun 入口）：
  /home/zn/.local/bin/uv run --no-sync --project /home/zn/code/Megatron_LM_Code \
      python -m torch.distributed.run --nproc_per_node=2 probe_p2p_overlap_demo.py
"""
import os
import time

import torch
import torch.distributed as dist

# 2 GiB float32：传输要足够大，ncclDevKernel_SendRecv 才有几十毫秒的重叠窗口。
MESSAGE_ELEMENTS = 512 * 1024 * 1024
COMPUTE_MATRIX_SIZE = 4096
COMPUTE_ITERATIONS = 64
WARMUP_ITERATIONS = 8


def run_compute(left_matrix, accumulator, iterations):
    """Enqueue a chained GEMM loop that keeps the compute stream busy."""
    for _ in range(iterations):
        accumulator = torch.matmul(left_matrix, accumulator)
    return accumulator


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "this demo needs exactly two ranks"

    device = torch.cuda.current_device()
    message = torch.ones(MESSAGE_ELEMENTS, dtype=torch.float32, device=device)
    left_matrix = torch.randn(
        COMPUTE_MATRIX_SIZE, COMPUTE_MATRIX_SIZE, dtype=torch.float32, device=device
    )
    accumulator = torch.randn(
        COMPUTE_MATRIX_SIZE, COMPUTE_MATRIX_SIZE, dtype=torch.float32, device=device
    )

    # 预热 NCCL 建链与 cuBLAS kernel 选择，避免一次性开销混进测量段。
    torch.cuda.nvtx.range_push("warmup")
    warmup_message = torch.ones(1024, dtype=torch.float32, device=device)
    if rank == 0:
        dist.send(warmup_message, dst=1)
    else:
        dist.recv(warmup_message, src=0)
    run_compute(left_matrix, accumulator, WARMUP_ITERATIONS)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # 测量段：先 post P2P（两侧立即匹配，传输立刻开始），再入队 GEMM 链。
    # 传输与计算无数据依赖，ncclDevKernel_SendRecv 与 sgemm 链墙钟重叠。
    torch.cuda.nvtx.range_push("p2p-overlap")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    if rank == 0:
        send_handle = dist.isend(message, dst=1)
    else:
        receive_handle = dist.irecv(message, src=0)
    compute_start = time.perf_counter()
    accumulator = run_compute(left_matrix, accumulator, COMPUTE_ITERATIONS)
    end_event.record()
    if rank == 0:
        send_handle.wait()
    else:
        receive_handle.wait()
    torch.cuda.synchronize()
    compute_milliseconds = (time.perf_counter() - compute_start) * 1000.0
    torch.cuda.nvtx.range_pop()
    print(
        f"[rank {rank}] GEMM chain wall : {compute_milliseconds:9.1f} ms "
        f"(transfer of {MESSAGE_ELEMENTS * 4 / 1024**3:.0f} GiB hidden inside)",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
