# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Per-rank, per-step drift table for the recorded gradients / weights (Task 9.4 ④ / 9.5 ⑤).

与 ``compare_forward_dumps.py`` 的分工：那个脚本回答"**某个位置**是否超阈值"，本脚本回答
"**误差的形状**是什么"——逐步、逐组件给出 (张量数, 位级相等数, rel_l2 中位数, 最大值及其键)，
用来区分两种形态：

  * 差异**集中**在少数张量上 ⇒ 接线/归约域错误的形状；
  * 差异**弥散**在全部张量上且随步数单调增长 ⇒ "两侧只是走到了轨迹上的不同点"的形状。

**必须逐 rank 比对**：``named_parameters()`` 里的 ``decoder.layers.N`` 是**流水段内的局部下标**，
把各 rank 的 digest 合并成一个字典会拿 rank0 的 layer0 去比 rank1 的 layer0（还会因后写覆盖丢掉
大部分张量）。2026-08-31 我第一版就是这么错的，得出"语言侧 step1 中位数 0.53"的假结论。

用法：python compare_gradient_drift.py <dump_dir> [reference_side] [other_side]
两侧都省略时，目录里必须恰好有两个侧。
"""
import glob
import json
import os
import re
import statistics
import sys

import torch

ENCODER_PARAMETER_NAME_MARKERS = ("vision_model.", "vision_projection.")
# 比对哪些标签：``param_grad`` 是裁剪前的梯度（判据的主体），``param``/``exp_avg``/``exp_avg_sq``
# 是"梯度 → 权重"之后的产物，``model_bf16``/``main_fp32`` 是每个 iteration 换轮点的快照——后两个
# 放进来是为了直接检验一条机制：**bf16 舍入边界翻转**。fp32 主副本只差 1e-13 时 bf16 权重本该
# 完全相同（bf16 的 ulp 相对量级 4e-3），但正好落在舍入中点附近的元素会翻到相邻的 bf16 值、
# 一下差整个 ulp；7B 个元素里只要有几个翻转，下一步的前向/反向就不再是同一个函数。若
# ``model_bf16`` 的偏差显著大于同一时刻 ``main_fp32`` 的偏差 ⇒ 这条机制成立。
COMPARED_TAGS = ("param_grad", "param", "exp_avg", "exp_avg_sq", "model_bf16", "main_fp32")
STEP_SCALAR_TAGS = ("grad_norm", "clip_coefficient", "num_zeros")


def discover_sides(dump_directory):
    """Return every side name present in the dump directory."""
    sides = set()
    for path in glob.glob(os.path.join(dump_directory, "*_rank*.json")):
        match = re.fullmatch(r"(.+)_rank\d+\.json", os.path.basename(path))
        if match:
            sides.add(match.group(1))
    return sorted(sides)


def ranks_of_side(dump_directory, side):
    """Return the ranks this side dumped, in ascending order."""
    ranks = []
    for path in glob.glob(os.path.join(dump_directory, f"{side}_rank*.json")):
        match = re.fullmatch(rf"{re.escape(side)}_rank(\d+)\.json", os.path.basename(path))
        if match:
            ranks.append(int(match.group(1)))
    return sorted(ranks)


def load_rank(dump_directory, side, rank):
    """Load one rank's payload; raw tensors stay separate from digests."""
    json_path = os.path.join(dump_directory, f"{side}_rank{rank}.json")
    with open(json_path) as json_file:
        payload = json.load(json_file)
    tensor_path = json_path[: -len(".json")] + ".pt"
    tensors = torch.load(tensor_path, weights_only=False) if os.path.exists(tensor_path) else {}
    return payload["digests"], payload["scalars"], tensors


def normalized_key(key):
    """Drop the wrapper prefix so the two sides' parameter names line up.

    共置侧的 backbone chunk 是 ``GPTModel``（``decoder.layers.*``），非共置侧是 ``LLaVAModel``
    （``language_model.decoder.layers.*``）；encoder 两侧都是 ``vision_model.*`` /
    ``vision_projection.*``。故只需去掉 ``language_model.`` 这一层。
    """
    return key.replace("language_model.", "")


def component_of(key):
    """Classify one key as ``encoder`` or ``language`` by its parameter name."""
    if any(marker in key for marker in ENCODER_PARAMETER_NAME_MARKERS):
        return "encoder"
    return "language"


def parse_key(key):
    """Split ``{tag}/step{n}/{name}`` or ``{tag}/it{n}/{name}`` into (tag, index, name).

    优化器状态用 ``step{n}``、每个 iteration 的参数快照用 ``it{n}``，两者的编号语义不同
    （``it{n}`` 是第 n 个 iteration **开始前**的状态 = 第 n-1 步之后），但同一个 tag 内部编号是
    自洽的，放进同一张表即可；返回 None 表示这个键不参与本脚本的比对。
    """
    match = re.fullmatch(r"([^/]+)/(?:step|it)(\d+)/(.+)", key)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def relative_l2(reference_tensor, other_tensor):
    """``||other - reference||_2 / ||reference||_2`` in float64."""
    reference = reference_tensor.double()
    difference = (other_tensor.double() - reference).norm().item()
    norm = reference.norm().item()
    if norm == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / norm


def collect_measurements(dump_directory, reference_side, other_side):
    """Compare what the two sides recorded, rank by rank.

    Returns ``(raw_measurements, digest_measurements)``，两者的元素都是
    ``(tag, step, component, relative, label)``：
      * raw——两侧都存了原张量（≤1M 元素）的键，判据是逐元素 rel_l2；
      * digest——只有摘要的大张量（参数量级上不可能逐步存原值），判据退化为 ``abs_sum`` 的相对差。
        它对符号抵消不敏感、比 rel_l2 弱，但**覆盖全部参数**，"差异是否弥散"这个形状问题必须看
        全部参数才能回答。
    """
    raw_measurements = []
    digest_measurements = []
    reference_ranks = ranks_of_side(dump_directory, reference_side)
    other_ranks = ranks_of_side(dump_directory, other_side)
    common_ranks = [rank for rank in reference_ranks if rank in other_ranks]
    assert common_ranks, (
        f"no common rank between {reference_side} {reference_ranks} and "
        f"{other_side} {other_ranks}"
    )
    for rank in common_ranks:
        reference_digests, _, reference_tensors = load_rank(
            dump_directory, reference_side, rank
        )
        other_digests, _, other_tensors = load_rank(dump_directory, other_side, rank)
        other_tensor_keys = {normalized_key(key): key for key in other_tensors}
        other_digest_keys = {normalized_key(key): key for key in other_digests}
        for key, reference_digest in reference_digests.items():
            parsed = parse_key(key)
            if parsed is None or parsed[0] not in COMPARED_TAGS:
                continue
            tag, step, name = parsed
            normalized = normalized_key(key)
            other_key = other_digest_keys.get(normalized)
            if other_key is None:
                continue
            other_digest = other_digests[other_key]
            if other_digest["numel"] != reference_digest["numel"]:
                continue
            label = f"rank{rank}/{name}"
            if key in reference_tensors and normalized in other_tensor_keys:
                relative = relative_l2(
                    reference_tensors[key], other_tensors[other_tensor_keys[normalized]]
                )
                raw_measurements.append((tag, step, component_of(name), relative, label))
                continue
            reference_absolute_sum = reference_digest["abs_sum"]
            relative = abs(other_digest["abs_sum"] - reference_absolute_sum) / max(
                abs(reference_absolute_sum), 1e-12
            )
            digest_measurements.append((tag, step, component_of(name), relative, label))
    return raw_measurements, digest_measurements


def print_table(measurements, tag, title):
    """Print the per-step, per-component summary of one tag."""
    steps = sorted({step for measured_tag, step, _, _, _ in measurements if measured_tag == tag})
    if not steps:
        return
    print(f"\n=== {tag} — {title} ===")
    print(
        f"{'step':>4s} {'component':>9s} {'n':>5s} {'exact':>6s} {'median':>11s} "
        f"{'max':>11s}  argmax"
    )
    for step in steps:
        for component in ("language", "encoder"):
            values = [
                (value, label)
                for measured_tag, measured_step, measured_component, value, label in measurements
                if measured_tag == tag
                and measured_step == step
                and measured_component == component
            ]
            if not values:
                continue
            numbers = [value for value, _ in values]
            exact = sum(1 for value in numbers if value == 0.0)
            worst_value, worst_label = max(values, key=lambda item: item[0])
            print(
                f"{step:4d} {component:>9s} {len(numbers):5d} {exact:6d} "
                f"{statistics.median(numbers):11.3e} {worst_value:11.3e}  {worst_label}"
            )


def print_step_scalars(dump_directory, reference_side, other_side):
    """Print the per-step scalars (grad norm 等) of the lowest common rank."""
    reference_ranks = ranks_of_side(dump_directory, reference_side)
    other_ranks = ranks_of_side(dump_directory, other_side)
    common_ranks = [rank for rank in reference_ranks if rank in other_ranks]
    rank = common_ranks[0]
    _, reference_scalars, _ = load_rank(dump_directory, reference_side, rank)
    _, other_scalars, _ = load_rank(dump_directory, other_side, rank)
    print(f"\n=== per-step scalars (rank{rank}) ===")
    print(f"{'scalar':>24s} {reference_side:>24s} {other_side:>24s} {'rel':>11s}")
    for key in sorted(set(reference_scalars) & set(other_scalars)):
        if key.split("/", 1)[0] not in STEP_SCALAR_TAGS:
            continue
        reference_value = reference_scalars[key]
        other_value = other_scalars[key]
        relative = abs(other_value - reference_value) / max(abs(reference_value), 1e-12)
        print(f"{key:>24s} {reference_value:24.10f} {other_value:24.10f} {relative:11.3e}")


def main():
    if len(sys.argv) not in (2, 4):
        print(f"usage: {sys.argv[0]} <dump_dir> [reference_side] [other_side]")
        return 2
    dump_directory = sys.argv[1]
    if len(sys.argv) == 4:
        reference_side, other_side = sys.argv[2], sys.argv[3]
    else:
        sides = discover_sides(dump_directory)
        assert len(sides) == 2, (
            f"{dump_directory} holds {sides}; pass the two side names explicitly"
        )
        reference_side, other_side = sides
    print(f"reference side: {reference_side}\nother side:     {other_side}")

    measurements, digest_measurements = collect_measurements(
        dump_directory, reference_side, other_side
    )
    for tag in COMPARED_TAGS:
        print_table(measurements, tag, "raw tensors, rel_l2, per rank per step")
    for tag in COMPARED_TAGS:
        print_table(digest_measurements, tag, "digest only, abs_sum relative, per rank per step")
    print_step_scalars(dump_directory, reference_side, other_side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
