# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Minimal probe: does a pending NCCL P2P transfer block the dispatch of later compute?

最小探针：一个**尚未能完成**的 NCCL P2P 传输，会不会挡住排在它后面入队的计算 kernel？

背景。``CUDA_DEVICE_MAX_CONNECTIONS`` 决定 GPU 前端能从几条硬件队列独立取指。设为 1 时所有
stream 的命令挤进一条 FIFO，于是"后入队的 kernel 不可能比先入队的先开始"。但**派发按序不等于
执行串行**——一个 kernel 发射出去之后，只要没有依赖且资源允许，后面的 kernel 就能并发执行。
真正待确认的是：当队首那个 NCCL kernel 因为**对端迟到**而长时间不能结束时，排在它后面、与它
毫无数据依赖的计算，能不能照常开始。这一点跟架构与 NCCL 实现相关，不能靠推理，只能测。

判据（不依赖 nsys 时间线，直接给数）。发送侧在 ``isend`` **之前**打一个参考 event，在计算**起始**
再打一个 event，量这两个 event 之间的**派发空档**；对端故意晚 ``RECEIVER_DELAY_SECONDS`` 秒才
post ``irecv``，所以那个 ``isend`` 在这段时间里必然还在等。
**为什么量空档而不量计算时长**：如果只在计算前后各打一个 event，一旦真的发生队首阻塞，那个起始
event 本身也会被一起推迟，两个 event 的差反而完全正常，什么也测不到。
判读：
  * 空档 ≈ 0                            ⇒ 计算照常派发，**没有队首阻塞**；在途通信只是自己在等；
  * 空档 ≈ RECEIVER_DELAY_SECONDS        ⇒ 计算被压在通信后面，**存在队首阻塞**；
  * 空档 ≈ 0 但计算比基线慢一些          ⇒ 不是阻塞，是 SM 资源争抢（NCCL kernel 占了 SM）。

两侧的排列刻意不同，一次跑同时回答两个方向的问题：
  * rank 0（发送侧）：**先** isend、**后** 计算 —— 测"队首的在途通信是否挡住后续计算"；
  * rank 1（接收侧）：**先** 计算、**后** irecv —— 测"长段计算是否推迟通信的开始"。

一个必须知道的实现细节：``dist.isend`` / ``dist.irecv`` 用的是 PyTorch ``ProcessGroupNCCL``
**自己的内部流**，靠 event 与当前流同步；用户显式 ``torch.cuda.stream(...)`` 并不能决定 NCCL
在哪条流上跑。所以这里不自建通信流——那只会造成"我控制了通信流"的错觉，而不改变实际行为。

运行：torchrun --nproc_per_node 2 probe_p2p_dispatch_overlap.py
      CUDA_DEVICE_MAX_CONNECTIONS 分别取 1 与 8 各跑一遍，差值才是答案。
"""
import os
import time

import torch
import torch.distributed as dist

# 消息要大到超过 NCCL 的 staging buffer，否则小消息拷进暂存区就返回，send 根本不会等对端，
# 这个实验就什么也没测到。512 MiB 的 float32。
# The message must exceed NCCL's staging buffer, otherwise the send never waits for the peer.
MESSAGE_ELEMENTS = 128 * 1024 * 1024
# 计算段用中等规模 GEMM 循环：太小量不出来，太大就把 SM 占满、把"资源争抢"混进"派发阻塞"里。
COMPUTE_MATRIX_SIZE = 4096
COMPUTE_ITERATIONS = 200
# 对端故意迟到的秒数：必须显著大于 compute_alone，这样"阻塞"与"不阻塞"两种结果相差一个数量级。
RECEIVER_DELAY_SECONDS = 5.0
WARMUP_ITERATIONS = 20
# 两种模式，回答 CUDA_DEVICE_MAX_CONNECTIONS 机制的两半：
#   pending_send（默认）——先 isend、后入队计算、最后才 wait。测"**正在跑**的通信 kernel 会不会
#     挡住后续计算"。已实测：不会（派发空档 0，=1 与 =8 相同）。
#   wait_first——先 isend、**立刻 handle.wait()**、然后才入队计算，且计算放在**另一条 stream** 上。
#     测"**未满足的 semaphore wait** 会不会挡住无关 stream 的计算"。计算必须换流，否则它与 wait
#     同流、被推迟是 stream 语义的必然结果，跟队列数无关，测了也说明不了任何事。
# The second mode isolates the real head-of-line candidate: an unsatisfied semaphore wait.
PROBE_MODE = os.environ.get("PROBE_MODE", "pending_send")
def run_compute(left_matrix, accumulator, iterations):
    """Enqueue a fixed amount of GEMM work; no dependency on any communication buffer."""
    for _ in range(iterations):
        accumulator = torch.matmul(left_matrix, accumulator)
    return accumulator


def measure_compute_milliseconds(left_matrix, accumulator, iterations):
    """Time the compute段 with device events, so the number is GPU time, not CPU wall time."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = run_compute(left_matrix, accumulator, iterations)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event), result


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "this probe needs exactly two ranks"

    device = torch.cuda.current_device()
    message = torch.ones(MESSAGE_ELEMENTS, dtype=torch.float32, device=device)
    left_matrix = torch.randn(
        COMPUTE_MATRIX_SIZE, COMPUTE_MATRIX_SIZE, dtype=torch.float32, device=device
    )
    accumulator = torch.randn(
        COMPUTE_MATRIX_SIZE, COMPUTE_MATRIX_SIZE, dtype=torch.float32, device=device
    )

    # 预热：首次 NCCL 调用要做 bootstrap 与建链（CPU 侧、几百毫秒量级），首次 GEMM 要选 kernel。
    # 不预热的话这些一次性开销会被算进测量段。
    # Warm up NCCL bootstrap and cuBLAS kernel selection before measuring anything.
    warmup_message = torch.ones(1024, dtype=torch.float32, device=device)
    if rank == 0:
        dist.send(warmup_message, dst=1)
    else:
        dist.recv(warmup_message, src=0)
    _, accumulator = measure_compute_milliseconds(left_matrix, accumulator, WARMUP_ITERATIONS)
    dist.barrier()
    torch.cuda.synchronize()

    # ---- 基线段：没有任何在途通信 ----
    compute_alone_milliseconds, accumulator = measure_compute_milliseconds(
        left_matrix, accumulator, COMPUTE_ITERATIONS
    )
    dist.barrier()
    torch.cuda.synchronize()

    # ---- 测试段 ----
    connections = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "<unset>")
    if rank == 0:
        # 发送侧：先入队 isend（对端要 5 秒后才 post irecv，所以它必然在等），再入队计算。
        # Sender: enqueue the isend first, then the independent compute behind it.
        #
        # **量的是 dispatch 延迟，不是计算时长**。这一点很关键：如果只用"计算前后两个 event"
        # 去测，一旦真的发生队首阻塞，那个**起始 event 本身也会被一起推迟**，两个 event 的差
        # 反而看不出任何异常。所以在 isend **之前**先打一个参考 event，看它到计算起始 event
        # 之间的空档——没有阻塞时应当接近 0，被阻塞时应当接近对端的迟到秒数。
        # Measure the dispatch gap, not the compute duration: under real blocking the start
        # event would be delayed too, so an event pair around the compute would show nothing.
        reference_event = torch.cuda.Event(enable_timing=True)
        compute_start_event = torch.cuda.Event(enable_timing=True)
        compute_end_event = torch.cuda.Event(enable_timing=True)

        enqueue_start = time.perf_counter()
        reference_event.record()
        send_handle = dist.isend(message, dst=1)
        if PROBE_MODE == "wait_first":
            # 立刻 wait：当前流上被插入一个"等通信 event"的 semaphore wait，此刻条件不满足。
            # 计算换到另一条 stream 上——它与那个 wait 没有任何 stream 依赖，唯一可能把它拖住的
            # 就是"两条 stream 共用一条硬件队列"。同时打 CPU 时间戳，用来区分另一种解释：
            # 如果 wait() 阻塞了 CPU 线程，那计算根本没被入队，与队列数无关。
            # Immediately wait: an unsatisfied semaphore wait now sits on the current stream.
            # The compute moves to another stream, so only channel sharing can delay it.
            cpu_before_wait = time.perf_counter()
            send_handle.wait()
            cpu_wait_seconds = time.perf_counter() - cpu_before_wait
            compute_stream = torch.cuda.Stream()
            with torch.cuda.stream(compute_stream):
                compute_start_event.record()
                accumulator = run_compute(left_matrix, accumulator, COMPUTE_ITERATIONS)
                compute_end_event.record()
            torch.cuda.synchronize()
        else:
            cpu_wait_seconds = 0.0
            compute_start_event.record()
            accumulator = run_compute(left_matrix, accumulator, COMPUTE_ITERATIONS)
            compute_end_event.record()
            send_handle.wait()
            torch.cuda.synchronize()
        total_seconds = time.perf_counter() - enqueue_start

        dispatch_gap_seconds = reference_event.elapsed_time(compute_start_event) / 1000.0
        compute_milliseconds = compute_start_event.elapsed_time(compute_end_event)
        print(
            f"[rank 0] PROBE_MODE={PROBE_MODE} "
            f"CUDA_DEVICE_MAX_CONNECTIONS={connections}\n"
            f"[rank 0] compute alone                 : {compute_alone_milliseconds:9.1f} ms\n"
            f"[rank 0] compute behind a pending send : {compute_milliseconds:9.1f} ms\n"
            f"[rank 0] cpu time inside handle.wait() : {cpu_wait_seconds:9.3f} s\n"
            f"[rank 0] dispatch gap (send -> compute): {dispatch_gap_seconds:9.3f} s"
            f"  (peer posts the recv {RECEIVER_DELAY_SECONDS} s late)\n"
            f"[rank 0] wall time of the test phase   : {total_seconds:9.3f} s\n"
            f"[rank 0] verdict: "
            + (
                "HEAD-OF-LINE BLOCKING - the compute could not start until the transfer ran"
                if dispatch_gap_seconds > RECEIVER_DELAY_SECONDS * 0.5
                else "no head-of-line blocking - the compute started while the send was pending"
            ),
            flush=True,
        )
    else:
        # 接收侧：先入队长计算，再**故意延迟** post irecv，制造对端迟到。
        # 顺带回答另一个方向：长段计算会不会推迟通信的开始（看 total 与 compute 的差）。
        # Receiver: long compute first, then deliberately post the irecv late.
        enqueue_start = time.perf_counter()
        compute_before_recv_milliseconds, accumulator = measure_compute_milliseconds(
            left_matrix, accumulator, COMPUTE_ITERATIONS
        )
        time.sleep(RECEIVER_DELAY_SECONDS)
        receive_handle = dist.irecv(message, src=0)
        receive_handle.wait()
        torch.cuda.synchronize()
        total_seconds = time.perf_counter() - enqueue_start
        print(
            f"[rank 1] compute before recv      : {compute_before_recv_milliseconds:9.1f} ms "
            f"(alone {compute_alone_milliseconds:9.1f} ms)\n"
            f"[rank 1] wall time of the test段   : {total_seconds:9.3f} s",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

