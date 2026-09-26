# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Feasibility experiment: partial (chunked) backward on a merged per-sample graph.

可行性实验：合并 forward 的图能否做**输出子集的局部反传**——dual-channel-p2p 非均匀切分
方案的"分批 encoder 反传"就依赖这个性质。

与真实结构的对应（关键：种子在**输出侧**）：
- encoder 主体：输入 x_in [B, H] → per-sample 算子链（Linear→GELU→LayerNorm→残差×2，
  LayerNorm 只在 hidden 维归一化，不跨样本）→ 输出 out [B, H]。ViT encoder 没有任何
  跨样本计算，batch 维内独立；
- out 就是 **boundary**（真实代码里 = encoder 输出 image_embeddings，consumer 按 mb 切走
  的 view）；
- g [B, H] = consumer 反传后发回的"边界梯度"（= dL/d(image_embeddings) 按 mb 到达）；
- 真实 phase ④ 的做法 = `torch.autograd.backward(out 的 per-mb view, 收到的 grad)`——种子
  在图的**输出侧**，往 encoder 图里回传。本实验逐字复刻：全量一次 / 只 seed 前 CHUNK 行
  （retain_graph）/ 再 seed 余下行，三者对比。

验证三件事：
1. 只 seed 前 CHUNK 行的局部反传，参数梯度与"只用这 CHUNK 行单独 forward+backward"的
   参考值一致 → 子图按 mb 不相交、切片反传正确；
2. 分批（CHUNK + 余下）累积的参数梯度，与一次全量 backward 一致 → 分批反传数值上等价于
   统一反传（fp64 下容差 ~1e-12）；
3. 反例场景 B：图中一旦存在**跨样本 op**（batch-mean 分支），局部反传不再等价——证明
   "encoder 无跨样本计算"是该方案的必要前提（ViT 满足）。

fp64 + 固定种子，纯 CPU。
"""

import torch

torch.manual_seed(7)
torch.set_default_dtype(torch.float64)

B, H, CHUNK = 8, 16, 5


def make_params():
    """Two per-sample blocks worth of weights (no cross-batch parameters)."""
    return (
        torch.randn(H, H, requires_grad=True), torch.randn(H, requires_grad=True),  # block1 linear
        torch.randn(H, H, requires_grad=True), torch.randn(H, requires_grad=True),  # block2 linear
        torch.randn(H, H, requires_grad=True), torch.randn(H, requires_grad=True),  # block3 linear
    )


def encoder_body(x, p, cross_batch_term=None, mix=False):
    """Mini encoder: per-sample ops only unless cross_batch_term/mix is injected (scenario B).

    迷你 encoder：默认纯 per-sample（Linear→GELU→LayerNorm→残差×2）；场景 B 注入跨样本耦合
    （batch-mean 线性层 / 输出行间混合）用于反例。
    """
    w1, b1, w2, b2, w3, b3 = p
    y = torch.nn.functional.gelu(x @ w1 + b1)
    y = torch.nn.functional.layer_norm(y, (H,))
    y = torch.nn.functional.gelu(y @ w2 + b2) + y  # 残差 / residual
    y = torch.nn.functional.layer_norm(y, (H,))
    y = y @ w3 + b3
    if cross_batch_term is not None:
        # 跨样本分支：batch-mean 会把所有样本混合进每行输出（真实 encoder 中不存在，
        # ViT attention/MLP/projection 都在 batch 维内独立）。
        y = y + cross_batch_term(x)
    if mix:
        # 行间混合：每行输出依赖所有行（切片前就耦合）——局部反传必然不等价的构造。
        # Row mixing: every output row depends on all rows BEFORE any slicing, so a
        # subset-seeded backward is structurally different from a subset-only graph.
        y = y + 0.01 * y.sum(dim=0, keepdim=True)
    return y


def param_grads(params):
    return [p.grad.clone() if p.grad is not None else None for p in params]


def zero_grads(params, x_in):
    for p in params:
        p.grad = None
    x_in.grad = None


def max_diff(a, b):
    return (a - b).abs().max().item()


def run_scenario(title, cross_batch_term=None, mix=False):
    """Full vs chunk1-only-reference vs accumulated-chunked, seeded at the OUTPUT side."""
    print(f"\n========== {title} ==========")
    params = make_params()
    x_in = (torch.randn(B, H) * 0.5).requires_grad_(True)  # encoder 输入 / encoder input
    g = torch.randn(B, H)  # consumer 回传的边界梯度 / incoming boundary grads

    # --- reference 0: 全量一次 backward（现行 phase ④ 的做法）---
    zero_grads(params, x_in)
    out = encoder_body(x_in, params, cross_batch_term, mix)
    torch.autograd.backward(out, g)  # 种子在输出侧 / seeded at the boundary (output side)
    full_pg = param_grads(params)
    full_x_grad = x_in.grad.clone()

    # --- reference 1: 只用前 CHUNK 行单独 forward+backward（局部反传的理论等价物）---
    small_params = [t.clone().detach().requires_grad_(True) for t in params]
    x5 = x_in[:CHUNK].detach().requires_grad_(True)
    out5 = encoder_body(x5, small_params, cross_batch_term, mix)
    torch.autograd.backward(out5, g[:CHUNK])
    ref5_pg = [p.grad.clone() for p in small_params]
    ref5_x_grad = x5.grad.clone()

    # --- chunked: 先只 seed 前 CHUNK 行（retain_graph），再 seed 余下行（释放整图）---
    zero_grads(params, x_in)
    out = encoder_body(x_in, params, cross_batch_term, mix)
    torch.autograd.backward(out[:CHUNK], g[:CHUNK], retain_graph=True)
    chunk1_pg = param_grads(params)
    x_grad_after_chunk1 = x_in.grad.clone()
    torch.autograd.backward(out[CHUNK:], g[CHUNK:])  # 最后一块：释放整图
    chunked_pg = param_grads(params)

    # --- 比较一：chunk1 局部反传 vs CHUNK 行独立参考 ---
    diffs1 = [max_diff(a, b) for a, b in zip(chunk1_pg, ref5_pg)]
    print(f"[chunk1 vs {CHUNK}-row reference]  param grad max|diff| = {max(diffs1):.3e}")

    # --- 比较二：分批累积 vs 一次全量 ---
    diffs2 = [max_diff(a, b) for a, b in zip(chunked_pg, full_pg)]
    print(f"[chunked accumulated vs full]     param grad max|diff| = {max(diffs2):.3e}")

    # --- 检查三：chunk1 后输入的梯度只在被反传的行上非零 ---
    untouched = x_grad_after_chunk1[CHUNK:]
    untouched_max = untouched.abs().max().item() if untouched.numel() else 0.0
    print(f"[input grad rows >={CHUNK} after chunk1]  max|grad| = {untouched_max:.3e} (expect 0)")
    input_row_diff = max_diff(x_grad_after_chunk1[:CHUNK], ref5_x_grad)
    print(f"[input grad rows <{CHUNK} vs reference]    max|diff| = {input_row_diff:.3e}")

    verdict1 = max(diffs1) < 1e-10
    verdict2 = max(diffs2) < 1e-10
    verdict3 = untouched_max == 0.0
    print(f"==> chunk1 == {CHUNK}-row reference: {'PASS' if verdict1 else 'FAIL'}; "
          f"accumulated == full: {'PASS' if verdict2 else 'FAIL'}; "
          f"untouched rows zero: {'PASS' if verdict3 else 'FAIL'}")
    return verdict1 and verdict2 and verdict3


def main():
    ok_a = run_scenario("Scenario A: per-sample graph only (encoder-like, ViT-faithful)")
    # 场景 B 反例：在切片前注入行间混合 → 局部反传必然不再等价（证明前提的必要性）。
    # Scenario B counterexample: inject row mixing BEFORE slicing — the subset-seeded backward
    # is then structurally different from a subset-only graph, showing "no cross-sample ops in
    # the encoder" is a necessary premise of the chunked design.
    ok_b = run_scenario("Scenario B: with row mixing before slicing (counterexample, expect FAIL)",
                        mix=True)

    print("\n========== CONCLUSION ==========")
    print(f"Scenario A (per-sample only):        {'ALL PASS — chunked backward is valid' if ok_a else 'FAIL'}")
    print(f"Scenario B (cross-sample injected):  {'equivalence BROKEN as expected' if not ok_b else 'UNEXPECTED PASS'}")
    if ok_a and not ok_b:
        print("结论：合并 forward 的 per-sample 图支持输出侧（boundary view）子集的局部反传；分批")
        print("累积数值上等价于一次全量 backward。前提是图中不存在跨样本计算——ViT encoder")
        print("（attention/MLP/projection 均在 batch 维内独立）满足。非均匀方案的'分批 encoder")
        print("反传'可行，且前传无需为反传而拆分。")


if __name__ == "__main__":
    main()
