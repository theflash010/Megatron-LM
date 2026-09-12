# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Minimal demo: a guaranteed comm/compute overlap, plus its dependent control group.

最小演示：一个**必然发生**的通算并行（NCCL all-reduce 与长段 GEMM 墙钟重叠），
外加一个**必然串行**的对照组（通信的输入依赖计算输出）。

原理。通算重叠的充要条件是：NCCL kernel 与计算 kernel 之间**没有 stream/数据依赖**。
  * A 相（overlap）：先在 compute stream 入队一长串 GEMM 链，再对一块**从不参与计算**
    的专用 buffer 做 ``dist.all_reduce``。该 buffer 与 GEMM 无任何依赖，ProcessGroupNCCL
    不会插入对 compute stream 的等待，``ncclDevKernel_AllReduce`` 在 NCCL 内部流上立刻
    开跑 —— 必然与 GEMM 重叠。
  * B 相（control，串行）：把 GEMM 链的**输出**拿去 all-reduce。通信输入依赖计算结果，
    NCCL 流必须等 compute stream 的 event —— 必然"先算后通"。

判据（不依赖 nsys，程序直接给数）：串行时合跑墙钟 ≈ compute_alone + comm_alone；
重叠时 ≈ max(两者) + SM 争抢损耗，显著小于两者之和。nsys 报告则用来**亲眼看**：
compute stream 与 NCCL stream 两条 lane 上的 kernel bar 在同一时间窗内重叠。

一个必须绕开的机制（2026-09-10 首跑实测，A 相被误判 SERIAL 后定位）：阻塞式 c10d 集合通信
会在**调用时刻把输入 tensor 记账到当前流**，NCCL 内部流必须等当前流的 event——于是
"在同一当前流上先算后通信"必然串行，与数据是否依赖无关。所以 A 相把 ``all_reduce`` 从一条
**空侧流**上发起（``async_op=True``）：侧流上没有 event 可等 ⇒ NCCL 流不再等 GEMM 链 ⇒
必然重叠。B 相刻意留在当前流上，"依赖串行"照旧成立。

运行（zju-a800，无 torchrun 入口）：
  /home/zn/.local/bin/uv run --no-sync --project /home/zn/code/Megatron_LM_Code \
      python -m torch.distributed.run --nproc_per_node=2 probe_comm_compute_overlap_demo.py
nsys 版本把整条命令包进 nsys profile --trace=cuda,nvtx,osrt 即可。
"""
import os
import time

import torch
import torch.distributed as dist

# 通信 buffer：2 GiB float32（每 rank）。专用、独立，全程不与计算 buffer 相碰。
# 2 GiB 的 all-reduce 在 NVLink 上是几十毫秒量级，重叠窗口在 nsys 里一眼可见。
# Dedicated comm buffer: 2 GiB float32 per rank, never touched by the compute chain.
MESSAGE_ELEMENTS = 512 * 1024 * 1024
# 计算段：中等规模 GEMM 链，链式依赖让 compute stream 连续忙一段（几百毫秒）。
# Compute segment: a chained GEMM loop keeps the compute stream busy continuously.
COMPUTE_MATRIX_SIZE = 4096
COMPUTE_ITERATIONS = 64
WARMUP_ITERATIONS = 8
# 判决用"被藏住的通信时长"：hidden = (compute_alone + comm_alone) - 合跑墙钟。
# hidden 超过 comm_alone 的一半即判为重叠（串行时 hidden ≈ 0，完全重叠时 ≈ comm_alone）。
# Verdict by hidden comm time: serial gives hidden ~= 0, full overlap ~= comm_alone.
HIDDEN_FRACTION_THRESHOLD = 0.5


def run_compute(left_matrix, accumulator, iterations):
    """Enqueue a chained GEMM loop; internally dependent, externally independent."""
    for _ in range(iterations):
        accumulator = torch.matmul(left_matrix, accumulator)
    return accumulator


def measure_compute_milliseconds(left_matrix, accumulator, iterations):
    """Time the compute chain with device events, so the number is GPU time."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = run_compute(left_matrix, accumulator, iterations)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event), result


def measure_comm_milliseconds(message):
    """Time a lone all-reduce with wall clock around a full synchronize.

    NCCL 的内部流拿不到 event 句柄，墙钟 + synchronize 是它的干净量法：
    enqueue 是微秒级，墙钟几乎就是 kernel 实跑时间。
    The NCCL stream is not directly event-able; wall clock around a full
    synchronize is clean because enqueue cost is microsecond-scale.
    """
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    dist.all_reduce(message)
    torch.cuda.synchronize()
    return (time.perf_counter() - wall_start) * 1000.0


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "this demo needs exactly two ranks"

    device = torch.cuda.current_device()
    comm_message = torch.ones(MESSAGE_ELEMENTS, dtype=torch.float32, device=device)
    left_matrix = torch.randn(
        COMPUTE_MATRIX_SIZE, COMPUTE_MATRIX_SIZE, dtype=torch.float32, device=device
    )
    accumulator = torch.randn(
        COMPUTE_MATRIX_SIZE, COMPUTE_MATRIX_SIZE, dtype=torch.float32, device=device
    )

    # 预热：NCCL bootstrap/建链与 cuBLAS kernel 选择都是一次性开销，不进测量段。
    # Warm up NCCL bootstrap and cuBLAS kernel selection before measuring anything.
    torch.cuda.nvtx.range_push("warmup")
    measure_comm_milliseconds(comm_message)
    _, accumulator = measure_compute_milliseconds(left_matrix, accumulator, WARMUP_ITERATIONS)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # ---- 基线段：算、通各自单独跑 ----
    torch.cuda.nvtx.range_push("baseline-compute-alone")
    compute_alone_milliseconds, accumulator = measure_compute_milliseconds(
        left_matrix, accumulator, COMPUTE_ITERATIONS
    )
    torch.cuda.nvtx.range_pop()
    dist.barrier()

    torch.cuda.nvtx.range_push("baseline-comm-alone")
    comm_alone_milliseconds = measure_comm_milliseconds(comm_message)
    torch.cuda.nvtx.range_pop()
    dist.barrier()
    torch.cuda.synchronize()

    # ---- A 相：独立通算，必然重叠 ----
    # 关键：all_reduce 必须从一条**空侧流**上发起。阻塞式 c10d 会在调用时刻把输入
    # tensor 记账到当前流并让 NCCL 流等它的 event；当前流是空的侧流 ⇒ NCCL 流无等待
    # ⇒ 与 compute stream 上的 GEMM 链必然重叠。async_op=True 避免在侧流上插回等。
    # The all-reduce must be issued from an empty side stream: c10d books the input
    # tensor on the current stream and makes the NCCL stream wait for its event.
    torch.cuda.nvtx.range_push("phaseA-independent-overlap")
    torch.cuda.synchronize()
    comm_stream = torch.cuda.Stream()
    comm_stream.wait_stream(torch.cuda.current_stream())
    wall_start = time.perf_counter()
    compute_start_event = torch.cuda.Event(enable_timing=True)
    compute_end_event = torch.cuda.Event(enable_timing=True)
    compute_start_event.record()
    accumulator = run_compute(left_matrix, accumulator, COMPUTE_ITERATIONS)
    compute_end_event.record()
    with torch.cuda.stream(comm_stream):
        dist.all_reduce(comm_message, async_op=True)
    torch.cuda.synchronize()
    phase_a_wall_milliseconds = (time.perf_counter() - wall_start) * 1000.0
    torch.cuda.nvtx.range_pop()
    dist.barrier()

    # ---- B 相：通信输入依赖计算输出，必然串行（对照组）----
    # 链自身的 GPU 时间必须单独量出来，否则"墙钟 ≈ 基线计算"无法区分
    # "串行（= 链 + 通信）"与"重叠（≈ 链）"。
    # The chain's own GPU time must be measured separately to tell the two apart.
    torch.cuda.nvtx.range_push("phaseB-dependent-serial")
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    b_start_event = torch.cuda.Event(enable_timing=True)
    b_end_event = torch.cuda.Event(enable_timing=True)
    b_start_event.record()
    dependent_result = run_compute(left_matrix, accumulator, COMPUTE_ITERATIONS)
    b_end_event.record()
    dist.all_reduce(dependent_result)
    torch.cuda.synchronize()
    phase_b_wall_milliseconds = (time.perf_counter() - wall_start) * 1000.0
    phase_b_chain_milliseconds = b_start_event.elapsed_time(b_end_event)
    torch.cuda.nvtx.range_pop()

    sum_alone = compute_alone_milliseconds + comm_alone_milliseconds
    phase_a_hidden_milliseconds = sum_alone - phase_a_wall_milliseconds
    a_overlapped = (
        phase_a_hidden_milliseconds > comm_alone_milliseconds * HIDDEN_FRACTION_THRESHOLD
    )
    b_serial = (
        phase_b_wall_milliseconds
        >= phase_b_chain_milliseconds + comm_alone_milliseconds * HIDDEN_FRACTION_THRESHOLD
    )
    print(
        f"[rank {rank}] compute alone            : {compute_alone_milliseconds:9.1f} ms\n"
        f"[rank {rank}] comm alone (all-reduce)  : {comm_alone_milliseconds:9.1f} ms\n"
        f"[rank {rank}] sum of the two           : {sum_alone:9.1f} ms\n"
        f"[rank {rank}] phase A wall (independent): {phase_a_wall_milliseconds:9.1f} ms"
        f"  (compute chain GPU time {compute_start_event.elapsed_time(compute_end_event):9.1f} ms,"
        f" hidden comm {phase_a_hidden_milliseconds:6.1f} ms"
        f" = {phase_a_hidden_milliseconds / comm_alone_milliseconds * 100:5.1f}% of comm)\n"
        f"[rank {rank}] phase B wall (dependent) : {phase_b_wall_milliseconds:9.1f} ms"
        f"  (compute chain GPU time {phase_b_chain_milliseconds:9.1f} ms,"
        f" expected serial {phase_b_chain_milliseconds + comm_alone_milliseconds:9.1f} ms)\n"
        f"[rank {rank}] verdict A: "
        + (
            "OVERLAP - the all-reduce ran concurrently with the GEMM chain"
            if a_overlapped
            else "SERIAL - unexpected, check stream dependencies"
        )
        + f"\n"
        f"[rank {rank}] verdict B: "
        + (
            "SERIAL - as expected, the collective waited for its input"
            if b_serial
            else "OVERLAP - unexpected for a dependent collective"
        ),
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
