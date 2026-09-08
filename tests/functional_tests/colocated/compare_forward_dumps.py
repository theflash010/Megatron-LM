# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Offline comparison of the forward dumps of every side (Task 6.7 ①②).

读 ``forward_dumps/`` 下所有 ``{side}_rank{R}.json``（摘要 + 标量）与 ``{side}_rank{R}.pt``
（选定 microbatch 的原张量），以 ``colocated_*`` 为基准，逐个把其余侧与它对照。

位置名与拓扑无关（``layer{NN}/mb{j}``、``vision_layer{NN}/mb{j}``、``encoder_output/mb{j}``、
``final_token_loss/mb{j}``、``loss``/``num_tokens``），所以 TP1/PP4 与 TP4/PP1 能放进同一张表：
层号是全局层号，PP 只决定它落在哪个 rank，TP 只决定它是否被切分（层输出经 all-reduce 后是
副本，可直接比）。

判据：
  * 结构性问题（某位置只有一侧有、形状不同）一律失败——那是接线错误；
  * 数值用相对误差 ``max_abs_diff / max(|reference|)`` 加阈值。同 TP 的两侧应当 **bitwise
    相等（0）**；TP 不同的两侧不可能为 0——row-parallel 把归约维切开后先分片累加再 all-reduce，
    浮点加法不满足结合律，且切开后的 GEMM 形状会让 cuBLAS 选不同 kernel ⇒ 每层差 1~2 个 ULP、
    经 32 层放大到 1e-3~1e-2 量级。因此阈值按侧给：``--tolerance`` 只对 TP 不同的侧生效。

用法：python compare_forward_dumps.py <dump_dir> [tolerance]
"""
import glob
import json
import os
import re
import sys

import torch

REFERENCE_SIDE_PREFIX = "colocated"

# encoder（ViT + vision_projection）的参数梯度**不可能 bitwise 相等**，即使两侧 TP 相同：
#   * 共置侧每个 rank 只算自己负责的 num_microbatches/P 个 microbatch，再在共置 dp 组（全 W）上
#     做一次 NCCL SUM ⇒ 求和顺序是"(2 个的局部和) 四份相加"；
#   * 非共置侧把 8 个 microbatch 顺序累加进同一个 main_grad buffer。
# 同样 8 项、不同结合顺序，fp32 加法不满足结合律 ⇒ 差在 fp32 舍入量级（实测 rel_l2 ≤ 5.3e-8）。
# 语言侧没有这个问题：两侧都是"同一个 stage 顺序累加全部 8 个 microbatch"，实测精确为 0。
FLOAT32_ACCUMULATION_TOLERANCE = 1e-6

# Task 9.4 ④ 的键分三类，判据严格程度不同：
#   * ``param_grad/{name}``——**裁剪之前**的归约后梯度。语言侧仍要求 bitwise 相等（两侧都是同一个
#     stage 顺序累加全部 microbatch），只有 encoder 侧放宽到 fp32 结合序量级；
#   * ``{param,exp_avg,exp_avg_sq}/step{n}/{name}``——**裁剪之后**的产物，语言侧也**不可能**
#     bitwise 相等：裁剪系数是全模型的一个标量 $$\min(1, c/\|g\|)$$，而 $$\|g\|$$ 里含 encoder
#     那一项的结合序差异 ⇒ 这点差异被乘进**每一个**参数的更新。这是 ChainedOptimizer 有意的耦合
#     （optimizer.py:1361-1383），不是错误，故这三类键一律放宽到 fp32 结合序量级；
#   * ``grad_norm`` / ``clip_coefficient``——同上，含 encoder 那一项。
# 放宽到 1e-6 仍然是很紧的判据：归约域若设错，偏差是 $$\sqrt{P}$$ 或 $$\sqrt{D}$$ 量级（几十个
# 百分点），不是 1e-6。
GRADIENT_TAG = "param_grad"
# ``reduced_grad`` 是把 DistOpt 的 reduce-scatter 分片 all-gather 拼回来的完整梯度（键里带 rank），
# 与非 DistOpt 的整块归约梯度同口径可比；encoder 侧仍允许 fp32 结合序量级的差异。
MERGED_GRADIENT_TAG = "reduced_grad"
CLIPPED_PARAMETER_SPACE_TAGS = ("param", "exp_avg", "exp_avg_sq", "model_bf16", "main_fp32")
ENCODER_PARAMETER_NAME_MARKERS = ("vision_model.", "vision_projection.")
FULL_MODEL_SCALAR_TAGS = ("grad_norm", "clip_coefficient")
# ``encoder_replica_sum`` 是 encoder 全部权重的 float64 摘要，里面含 encoder 梯度的 fp32 结合序
# ⇒ 与 ``param`` 同类，放宽到结合序量级。**``*_spread`` 不在此列**：它是同一步内跨 rank 的极差，
# 是"参数 all-gather 正确"的判据本身，必须严格为 0（Task 8.9.4 / 8.7）。
# The replica digest carries the encoder accumulation order; the spread is the criterion itself.
ENCODER_REPLICA_DIGEST_TAGS = ("encoder_replica_sum",)


def tolerance_for_key(key, base_tolerance):
    """Per-key tolerance: quantities carrying the encoder's fp32 accumulation order are relaxed."""
    tag = key.split("/", 1)[0]
    if tag in FULL_MODEL_SCALAR_TAGS or tag in CLIPPED_PARAMETER_SPACE_TAGS:
        return max(base_tolerance, FLOAT32_ACCUMULATION_TOLERANCE)
    if tag in ENCODER_REPLICA_DIGEST_TAGS:
        return max(base_tolerance, FLOAT32_ACCUMULATION_TOLERANCE)
    if tag in (GRADIENT_TAG, MERGED_GRADIENT_TAG) and any(
        marker in key for marker in ENCODER_PARAMETER_NAME_MARKERS
    ):
        return max(base_tolerance, FLOAT32_ACCUMULATION_TOLERANCE)
    return base_tolerance


def discover_sides(dump_directory):
    """Return the side names present in the dump directory, reference side first."""
    sides = set()
    for path in glob.glob(os.path.join(dump_directory, "*_rank*.json")):
        match = re.fullmatch(r"(.+)_rank\d+\.json", os.path.basename(path))
        if match:
            sides.add(match.group(1))
    assert sides, f"no dumps found in {dump_directory}"
    reference = sorted(side for side in sides if side.startswith(REFERENCE_SIDE_PREFIX))
    assert reference, f"no {REFERENCE_SIDE_PREFIX}* side found among {sorted(sides)}"
    others = sorted(sides - set(reference))
    return reference[0], reference[1:] + others


def load_side(dump_directory, side):
    """Merge every rank's digests / scalars / raw tensors of one side."""
    digests = {}
    scalars = {}
    tensors = {}
    for json_path in sorted(glob.glob(os.path.join(dump_directory, f"{side}_rank*.json"))):
        with open(json_path) as json_file:
            payload = json.load(json_file)
        digests.update(payload["digests"])
        scalars.update(payload["scalars"])
        tensor_path = json_path[: -len(".json")] + ".pt"
        if os.path.exists(tensor_path):
            tensors.update(torch.load(tensor_path, weights_only=False))
    print(
        f"[{side}] {len(digests)} digests, {len(scalars)} scalars, {len(tensors)} raw tensors"
    )
    return {"digests": digests, "scalars": scalars, "tensors": tensors}


def tensor_parallel_size_of(side):
    """Read the TP size out of the side name (``..._tp4pp1`` -> 4)."""
    match = re.search(r"_tp(\d+)pp(\d+)$", side)
    assert match, f"side name {side} does not carry its topology (expected ..._tp{{N}}pp{{M}})"
    return int(match.group(1))


def pipeline_parallel_size_of(side):
    """Read the PP size out of the side name (``..._tp4pp1`` -> 1)."""
    match = re.search(r"_tp(\d+)pp(\d+)$", side)
    assert match, f"side name {side} does not carry its topology (expected ..._tp{{N}}pp{{M}})"
    return int(match.group(2))


def encoder_distributed_optimizer_instances_of(side):
    """Read the encoder DistOpt instance count out of the side name (``..._enc2_...`` -> 2).

    没有 ``enc{N}`` 段的是 Task 8.7 之前的 dump，按 1 个 instance 处理。
    Dumps written before Task 8.7 carry no instance segment and count as one instance.
    """
    match = re.search(r"_enc(\d+)_", side)
    return int(match.group(1)) if match else 1


def align_shapes(reference_tensor, other_tensor):
    """Cut both tensors down to the common prefix when exactly one dimension differs.

    PP>1 时 LLaVA 把组装后的序列 **padding 到 ``--decoder-seq-length``**（流水线 P2P 要求固定
    形状），PP=1 则用真实长度（576 图像 token + 文本长度 - 1）。于是同一个位置在 TP1/PP4 上是
    1024、在 TP4/PP1 上是 788——多出来的是 padding，取公共前缀比较才有意义。
    只允许**恰好一个维度**不同，其余维度不同一律当作接线错误（返回 None）。
    """
    reference_shape = list(reference_tensor.shape)
    other_shape = list(other_tensor.shape)
    if reference_shape == other_shape:
        return reference_tensor, other_tensor, None
    if reference_tensor.numel() == other_tensor.numel() and (
        reference_tensor.ndim == 1 or other_tensor.ndim == 1
    ):
        # DistOpt stores main parameters and Adam states as flat shard views, while the
        # non-DistOpt path keeps the original parameter shape. The values are comparable
        # when the element counts match; restore the reference shape without changing data.
        return reference_tensor, other_tensor.reshape_as(reference_tensor), ("reshape",)
    if len(reference_shape) != len(other_shape):
        return None, None, None
    differing = [
        dim for dim in range(len(reference_shape)) if reference_shape[dim] != other_shape[dim]
    ]
    if len(differing) != 1:
        return None, None, None
    dim = differing[0]
    common = min(reference_shape[dim], other_shape[dim])
    index = [slice(None)] * len(reference_shape)
    index[dim] = slice(0, common)
    return reference_tensor[tuple(index)], other_tensor[tuple(index)], (dim, common)


def relative_difference(reference_tensor, other_tensor):
    """Return (max_abs_diff, max-based relative, L2-based relative).

    两个相对误差都要看：``rel_max`` 对**单个离群元素**极敏感（CLIP ViT 的 outlier 维度量级可达
    几百，bf16 在那里的 ulp 就是 1），``rel_l2`` 才反映整体偏差水平。判定用 ``rel_l2``，
    ``rel_max`` 作为信息打印。
    """
    difference = (reference_tensor.double() - other_tensor.double()).abs()
    max_abs_difference = float(difference.max().item())
    scale = float(reference_tensor.double().abs().max().item())
    norm = float(reference_tensor.double().norm().item())
    relative_max = max_abs_difference / scale if scale > 0 else 0.0
    relative_l2 = float(difference.norm().item()) / norm if norm > 0 else 0.0
    return max_abs_difference, relative_max, relative_l2


def compare_pair(
    reference_side,
    reference,
    other_side,
    other,
    tolerance,
    same_pipeline_parallel_size=True,
    same_encoder_distributed_optimizer_instances=True,
):
    """Compare one side against the reference; return the list of failures."""
    failures = []
    print(f"\n=== {other_side} vs {reference_side} (tolerance {tolerance:.1e}) ===")

    only_reference = sorted(set(reference["digests"]) - set(other["digests"]))
    only_other = sorted(set(other["digests"]) - set(reference["digests"]))
    # PP 不同的两侧，记录发生在各自 rank 0 上，而 rank 0 持有的语言层区间本来就不同（PP=4 的
    # 第一个 stage 只有 1/4 的层，PP=1 则是全部层）⇒ 只在一侧出现的键是拓扑差异，不是错误；
    # PP 相同的两侧则必须逐键一一对应，缺键说明接线错了。
    for side_name, keys in ((reference_side, only_reference), (other_side, only_other)):
        for key in keys:
            if same_pipeline_parallel_size:
                failures.append(f"{key}: recorded on {side_name} only")
            else:
                print(
                    f"{key:30s} {'':20s} {'pp coverage':>13s} {'':>10s} {'':>10s}  "
                    f"skipped (recorded on {side_name} only)"
                )

    print(
        f"{'position':30s} {'shape':20s} {'max_abs_diff':>13s} {'rel_max':>10s} {'rel_l2':>10s}  "
        "note"
    )
    for key in sorted(set(reference["digests"]) & set(other["digests"])):
        reference_digest = reference["digests"][key]
        other_digest = other["digests"][key]
        shape_text = str(tuple(reference_digest["shape"]))
        digest_delta = (
            f"sum {other_digest['sum'] - reference_digest['sum']:+.3e} "
            f"abs_sum {other_digest['abs_sum'] - reference_digest['abs_sum']:+.3e}"
        )
        # 参数梯度在 TP 不同的两侧是**分片 vs 整块**（只在 TP rank 0 记录 ⇒ 那一片是沿
        # partition_dim 的第一段），元素数不同就没法整块比，跳过而不是判失败；活化值的长度不同
        # 是 PP 的 padding，走下面的公共前缀比较。
        if key.startswith("param_grad/") and reference_digest["numel"] != other_digest["numel"]:
            print(
                f"{key:30s} {shape_text:20s} {'tp shard':>13s} {'':>10s} {'':>10s}  "
                f"skipped (other side {other_digest['shape']})"
            )
            continue
        # ``param_grad`` 记的是本 rank 的 ``main_grad``——DistOpt 下**只有本 rank 那段分片**是
        # reduce-scatter 归约完成的（8.9.4）。encoder instance 数不同的两侧，分片大小本来就不同
        # （intra=4 时是 1/4 buffer，intra=2 时是 1/2），未归约的那几段自然天差地别 ⇒ 这个 tag
        # 在这种对照里没有意义，只打印。同口径可比的是 ``reduced_grad``（分片 all-gather 拼回的
        # 整块），它仍然被严格判定。
        # The local main_grad is only partially reduced under DistOpt, and the valid region
        # differs when the two sides use different instance counts. Judge reduced_grad instead.
        if key.startswith(f"{GRADIENT_TAG}/") and not same_encoder_distributed_optimizer_instances:
            print(
                f"{key:30s} {shape_text:20s} {'local shard':>13s} {'':>10s} {'':>10s}  "
                f"not judged (encoder instance counts differ)  {digest_delta}"
            )
            continue
        if key not in reference["tensors"] or key not in other["tensors"]:
            # 只有摘要的位置（未选中的 microbatch、或大到不存原值的梯度）：改判摘要的相对偏差
            # （用 abs_sum，它对符号抵消不敏感）。元素数不同（PP 的 padding）时无法公平比较，
            # 只打印不判定，判定交给有原张量的那个 microbatch。
            reference_abs_sum = reference_digest["abs_sum"]
            relative = abs(other_digest["abs_sum"] - reference_abs_sum) / max(
                abs(reference_abs_sum), 1e-12
            )
            if reference_digest["numel"] != other_digest["numel"]:
                print(
                    f"{key:30s} {shape_text:20s} {'digest only':>13s} {'':>10s} {'':>10s}  "
                    f"length differs {other_digest['shape']}, not judged  {digest_delta}"
                )
                continue
            over = relative > tolerance_for_key(key, tolerance)
            if over:
                failures.append(
                    f"{key}: abs_sum relative difference {relative:.3e} > tolerance "
                    f"{tolerance_for_key(key, tolerance):.1e} (digest only)"
                )
            print(
                f"{key:30s} {shape_text:20s} {'digest only':>13s} {'':>10s} {relative:10.3e}  "
                f"{digest_delta}{'  <-- OVER TOLERANCE' if over else ''}"
            )
            continue

        aligned_reference, aligned_other, alignment = align_shapes(
            reference["tensors"][key], other["tensors"][key]
        )
        if aligned_reference is None:
            failures.append(
                f"{key}: incompatible shapes {reference_digest['shape']} vs "
                f"{other_digest['shape']}"
            )
            print(
                f"{key:30s} {shape_text:20s} {'SHAPE MISMATCH':>13s} {'':>10s} {'':>10s}  "
                f"{other_digest['shape']}"
            )
            continue
        max_abs_difference, relative_max, relative_l2 = relative_difference(
            aligned_reference, aligned_other
        )
        note = digest_delta
        if alignment is not None:
            if alignment[0] == "reshape":
                note = "reshaped flat tensor  " + note
            else:
                note = f"prefix dim{alignment[0]}[:{alignment[1]}]  " + note
        over = relative_l2 > tolerance_for_key(key, tolerance)
        if over:
            failures.append(
                f"{key}: rel_l2 {relative_l2:.3e} > tolerance {tolerance:.1e} "
                f"(max_abs_diff {max_abs_difference:.3e}, rel_max {relative_max:.3e})"
            )
        print(
            f"{key:30s} {shape_text:20s} {max_abs_difference:13.6e} {relative_max:10.3e} "
            f"{relative_l2:10.3e}  {note}{'  <-- OVER TOLERANCE' if over else ''}"
        )

    print(f"\n{'scalar':30s} {reference_side:>22s} {other_side:>22s} {'rel':>10s}")
    for key in sorted(set(reference["scalars"]) | set(other["scalars"])):
        if key not in reference["scalars"] or key not in other["scalars"]:
            failures.append(f"{key}: scalar recorded on one side only")
            continue
        reference_value = reference["scalars"][key]
        other_value = other["scalars"][key]
        relative = abs(other_value - reference_value) / max(abs(reference_value), 1e-12)
        # 标量也要过 ``tolerance_for_key``：``grad norm`` 与裁剪系数在共置侧是 python float
        # （``ChainedOptimizer.get_grad_norm`` 走 ``math.sqrt``），在非共置侧是 fp32 tensor，
        # **表示精度**本身就带来 ~1.2e-7 的差，这是该标量能达到的下限，不是数值差异。
        key_tolerance = tolerance_for_key(key, tolerance)
        # ``num_tokens`` 是整数、由数据决定，任何拓扑下都必须**完全相等**——它不等就说明两侧
        # 吃的数据或 loss_mask 不同，是硬错误。``loss`` 是标量求和的派生量，对隐状态的扰动高度
        # 敏感（本实验的 ``vision_projection`` 是随机初始化，语言模型的输入本就在分布外，CE 对
        # 输入扰动的放大倍数很大）⇒ TP 不同的两侧只打印、不作判定，判定交给逐层的 rel_l2。
        # 判"是否硬失败"要看**两侧拓扑是否相同**（基准阈值为 0 才是同 TP 对），不能看逐键放宽后
        # 的阈值——否则被 ``tolerance_for_key`` 放宽过的键（grad norm 等）永远只是 informational。
        is_hard_check = tolerance == 0.0 or key.startswith("num_tokens/")
        over = relative > key_tolerance
        if over and is_hard_check:
            failures.append(
                f"{key}: {other_value} vs {reference_value} (relative {relative:.3e})"
            )
        marker = ""
        if over:
            marker = "  <-- OVER TOLERANCE" if is_hard_check else "  (informational)"
        print(
            f"{key:30s} {reference_value:22.8f} {other_value:22.8f} {relative:10.3e}{marker}"
        )
    return failures


def main():
    if len(sys.argv) not in (2, 3):
        print(f"usage: {sys.argv[0]} <dump_dir> [tolerance]")
        return 2
    dump_directory = sys.argv[1]
    cross_topology_tolerance = float(sys.argv[2]) if len(sys.argv) == 3 else 2e-2

    reference_side, other_sides = discover_sides(dump_directory)
    print(f"reference side: {reference_side}; other sides: {other_sides}\n")
    reference = load_side(dump_directory, reference_side)

    failures = []
    for other_side in other_sides:
        other = load_side(dump_directory, other_side)
        # 同 TP 的两侧必须 bitwise 相等；TP 不同的用给定阈值（原因见模块 docstring）。
        same_tensor_parallel_size = tensor_parallel_size_of(other_side) == tensor_parallel_size_of(
            reference_side
        )
        tolerance = 0.0 if same_tensor_parallel_size else cross_topology_tolerance
        same_pipeline_parallel_size = pipeline_parallel_size_of(
            other_side
        ) == pipeline_parallel_size_of(reference_side)
        same_encoder_distributed_optimizer_instances = (
            encoder_distributed_optimizer_instances_of(other_side)
            == encoder_distributed_optimizer_instances_of(reference_side)
        )
        failures += compare_pair(
            reference_side,
            reference,
            other_side,
            other,
            tolerance,
            same_pipeline_parallel_size=same_pipeline_parallel_size,
            same_encoder_distributed_optimizer_instances=(
                same_encoder_distributed_optimizer_instances
            ),
        )

    print()
    if failures:
        print(f"FAILED ({len(failures)} problems):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASSED: every recorded position agrees within its tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
