# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Recording of forward activations, shared by every side of the comparison (Task 6.7 ①②).

三侧（共置 TP1/PP4、非共置 TP1/PP4、非共置 TP4/PP1）各自记录"每个 microbatch 在每个位置上的
张量"，再由 ``compare_forward_dumps.py`` 两两比 ``max_abs_diff`` 与相对误差。

**位置用与拓扑无关的名字标识**（``{tag}/mb{id}``，不带 pp 号）——这是能把三种拓扑放在一张表
里比的前提：
  * ``layer{NN}``——语言侧第 NN 个 ``TransformerLayer`` 的输出（``layer_number`` 是**全局**
    层号，PP 切分只决定它落在哪个 rank 上）；stage 边界因此被层号自动覆盖（stage s 的输出
    就是 layer 8(s+1) 的输出）；
  * ``vision_layer{NN}``——ViT 的逐层输出；
  * ``encoder_output``——``vision_projection`` 的输出（共置侧是 encoder chunk 的返回值）；
  * ``final_token_loss``——末 stage 的逐 token loss；``loss``/``num_tokens``——末 stage 标量。

两条量级约束：
  * **只在 TP rank 0 记**——层输出经 row-parallel 的 all-reduce 后在 TP 组内是副本，四份一样
    的东西没必要落盘四遍（TP=4 侧会直接翻四倍）；
  * **只有 ``RAW_TENSOR_MICROBATCH_IDS`` 里的 microbatch 存原张量**，其余只留摘要。32 层 ×
    8 microbatch × [1024,1,4096] fp32 ≈ 4 GB，全存是不可行的；逐元素 ``max_abs_diff`` 只需要
    一个 microbatch 就能定位"误差在哪一层突然跳变"，其余 microbatch 用摘要兜底。
"""
import json
import os

import torch

from megatron.core import parallel_state as mpu
from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.core.optimizer.optimizer import ChainedOptimizer, MixedPrecisionOptimizer
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import get_model_config, get_pg_rank, unwrap_model

# 单个张量存原值的上限：hidden state（1024×4096 ≈ 4.2e6）存得下，logits（≈3.4e7）不存。
RAW_TENSOR_ELEMENT_LIMIT = 8_388_608
# 只有这些 microbatch 存原张量（其余只留摘要），见模块 docstring 的量级说明。
RAW_TENSOR_MICROBATCH_IDS = (0,)
# 优化器状态（主权重 + 两个 Adam 动量）的原值上限比上面小得多：这三类张量的**数量是参数量级**，
# 而且要按步数翻倍。按 8M 存的话，仅 ViT 一侧（24 层 × 若干 3~4M 的矩阵）单步单类就有 1.5GB
# 左右，三类两步就到 9GB。故只对小张量（layernorm / bias 这类，≤1M 元素 = 4MB fp32）存原值做
# 逐元素定位，大矩阵一律只留 float64 摘要——摘要里的 sum/abs_sum/max_abs 已足以暴露"语言侧不再
# bitwise 相等"这类问题。
OPTIMIZER_STATE_RAW_TENSOR_ELEMENT_LIMIT = 1_048_576

# ---------------------------------------------------------------------------
# 对照实验用的梯度扰动（Task 9.5）
# ---------------------------------------------------------------------------
# 目的：共置与非共置之间 encoder 梯度那点差异是**不可避免**的（两侧 all-reduce 的结合顺序不同，
# fp32 加法不满足结合律），实测第 1 步的张量级 rel_l2 ≤ 1.267e-10。有了非零种子，后续步数出现
# 差异是必然的——问题是"第 3 步就长到百分之几"这个**速率**是否只由放大解释。共置侧的单边数据
# 无法回答：任何系统性接线错误也会呈现"弥散 + 单调增长"。故需要一条标尺：两侧拓扑完全相同
# （都非共置、归约树一致、本该 bitwise 相等），只往其中一侧注入同量级的种子，测出"纯放大"的
# 漂移曲线。
#
# 扰动模型必须与真实种子同形，这一点有个硬约束：**fp32 里做不出"每个元素相对差 1e-10"**——
# $$1 + 10^{-10}$$ 在 fp32 下就等于 1（ulp 相对量级 $$2^{-23} \approx 1.2\times10^{-7}$$）。
# 结合序差异的真实形态是"**少数**元素差 1~2 个末位"，张量级 rel_l2 因而远小于单元素的 1 ulp。
# 于是这里按同一形态构造：随机挑 k 个元素、各改 1 个 ulp，k 由目标 rel_l2 反解——
# 均匀抽样下 $$\text{rel\_l2} \approx 2^{-23}\sqrt{k/n}$$ ⇒ $$k = n\,(\text{target}/2^{-23})^2$$。
# 小张量（bias / layernorm）按此式会得到 k=0，那样种子就不存在了，故下限取 1 个元素——代价是
# 小张量上的实际 rel_l2 会高于目标（4096 元素时 1.9e-9，约 15 倍），所以函数**返回实测的
# rel_l2**，判读时用实测值而不是目标值。
GRADIENT_PERTURBATION_ULP = 2.0**-23
# lr=1e-6 那轮实测的 encoder 梯度 step1 **单张量最大 rel_l2**（`compare_gradient_drift.py` 的
# 定义：$$\|g^{other}-g^{ref}\|_2/\|g^{ref}\|_2$$），即真实种子的量级。注意真实种子还是**稀疏**的
# （221 个被比对的 encoder 张量里 181 个位级相等），而本函数会给**每个**张量都注入至少 1 个元素
# ⇒ 注入的种子比真实的**更密**，方向上偏保守（对照组会漂得不少于共置侧）。判读时以对照组自己
# 的 step1 表（实测种子）为准，而不是以这个常数为准。
DEFAULT_GRADIENT_PERTURBATION_RELATIVE_L2 = 3.238e-9
ENCODER_PARAMETER_NAME_MARKERS = ("vision_model.", "vision_projection.")

_perturbation_generators = {}


def _perturbation_generator(device):
    """One dedicated generator per device, so the perturbation never touches the global RNG stream.

    绝不能用全局 RNG：那会把两条本该逐位相同的轨迹在**别的地方**也拉开（dropout 已关，但初始化
    之外的随机源仍共用全局流），对照实验就不再是"只差一个种子"。
    """
    generator = _perturbation_generators.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(20260831)
        _perturbation_generators[device] = generator
    return generator


def perturb_gradients_in_place(
    module, target_relative_l2, name_markers=ENCODER_PARAMETER_NAME_MARKERS
):
    """Flip a few last bits of the selected parameters' reduced gradients (control experiment).

    调用点与 ``record_parameter_gradients`` 相同、且必须在它**之前**：``finalize_model_grads``
    之后、``optimizer.step()`` 之前——那时 ``main_grad`` 已是优化器真正会用的值，扰动才会进入
    更新，同时也会被记录下来（落盘里能看到注入的种子有多大）。

    形态与量级的推导见模块顶部的注释块。Returns ``(张量数, 实测的整体 rel_l2)``。
    """
    perturbed_fraction = min(1.0, (target_relative_l2 / GRADIENT_PERTURBATION_ULP) ** 2)
    squared_difference = 0.0
    squared_reference = 0.0
    count = 0
    for name, parameter in module.named_parameters():
        if not any(marker in name for marker in name_markers):
            continue
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None:
            continue
        # ``view`` 而不是 ``reshape``：必须原地写进梯度本身的存储，非连续时宁可当场报错也不要
        # 静默拿到一份副本、扰动被丢掉。
        flat = gradient.view(-1)
        element_count = flat.numel()
        selected_count = max(1, int(round(element_count * perturbed_fraction)))
        generator = _perturbation_generator(flat.device)
        indices = torch.randint(
            0, element_count, (selected_count,), device=flat.device, generator=generator
        )
        signs = torch.randint(
            0,
            2,
            (selected_count,),
            device=flat.device,
            generator=generator,
            dtype=flat.dtype,
        )
        signs.mul_(2.0).sub_(1.0)
        before = flat[indices].clone()
        flat[indices] = before + signs * before.abs() * GRADIENT_PERTURBATION_ULP
        squared_difference += float((flat[indices] - before).double().pow(2).sum().item())
        squared_reference += float(flat.double().pow(2).sum().item())
        count += 1
    achieved_relative_l2 = (
        (squared_difference**0.5) / (squared_reference**0.5) if squared_reference > 0.0 else 0.0
    )
    return count, achieved_relative_l2


def scale_gradients_in_place(module, relative_scale):
    """Multiply **every** parameter's reduced gradient by ``1 + relative_scale`` (control experiment).

    这是第二个对照种子，形态与第一个（稀疏末位翻转、只在 encoder 上）完全不同，因为实测指出
    共置与非共置之间真正占主导的种子是**全模型、稠密、一致**的：``grad norm`` 在共置侧是
    ``math.sqrt`` 给的 python float、在非共置侧是 fp32 tensor，两者差 1.442e-7（fp32 表示下限），
    于是**裁剪系数**差同样的相对量，而裁剪系数是乘到**每一个**参数梯度上的一个标量
    （optimizer.py:1361-1383）⇒ 等价于"把所有梯度整体乘上 $$1+1.442\\times10^{-7}$$"。
    这个量级恰好在 fp32 ulp（$$2^{-23}\\approx1.19\\times10^{-7}$$）之上，所以真的能被表示出来，
    每个元素都会动 1 个末位——与第一个种子"只有极少数元素动末位"是两回事。
    """
    factor = 1.0 + relative_scale
    count = 0
    for _, parameter in module.named_parameters():
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None:
            continue
        gradient.mul_(factor)
        count += 1
    return count


def tensor_digest(tensor):
    """Deterministic summary of one tensor, computed in float64."""
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1).double()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(flat.numel()),
        "sum": float(flat.sum().item()),
        "abs_sum": float(flat.abs().sum().item()),
        "max_abs": float(flat.abs().max().item()) if flat.numel() else None,
        "first": float(flat[0].item()) if flat.numel() else None,
        "last": float(flat[-1].item()) if flat.numel() else None,
    }


class ForwardRecorder:
    """Collect digests (+ raw tensors for selected microbatches) keyed by ``{tag}/mb{id}``."""

    def __init__(self, side, is_recording_rank):
        self.side = side
        # TP rank 0 之外不记：层输出在 TP 组内是副本。
        self.is_recording_rank = is_recording_rank
        self.digests = {}
        self.scalars = {}
        self.raw_tensors = {}
        # 前向键（``{tag}/mb{id}{suffix}``）的后缀与开关：落盘点在第 N 步优化器步之后，而逐层
        # hook 每个 iteration 都会触发，若键不带 iteration 标识就会被后续 iteration 覆盖，落盘的
        # 便不再是第 1 个 iteration 的前向。第 1 个 iteration 用空后缀（与 9.1/9.2 的键名一致），
        # 第 2 个用 ``/it2``（供"更新是否晚一个 iteration 生效"的诊断），之后直接停记。
        self.forward_key_suffix = ""
        self.forward_recording_stopped = False

    def set_forward_key_suffix(self, suffix):
        """Suffix appended to every forward key, so different iterations do not overwrite."""
        self.forward_key_suffix = suffix

    def stop_forward_recording(self):
        """Stop recording forward tensors and per-microbatch scalars (parameters keep going)."""
        self.forward_recording_stopped = True

    def record_tensor(self, tag, microbatch_id, tensor):
        if tensor is None or not self.is_recording_rank or self.forward_recording_stopped:
            return
        if isinstance(tensor, (tuple, list)):  # TransformerLayer 返回 (hidden_states, context)
            tensor = tensor[0]
        if not torch.is_tensor(tensor):
            return
        key = f"{tag}/mb{microbatch_id}{self.forward_key_suffix}"
        self.digests[key] = tensor_digest(tensor)
        if (
            microbatch_id in RAW_TENSOR_MICROBATCH_IDS
            and tensor.numel() <= RAW_TENSOR_ELEMENT_LIMIT
        ):
            self.raw_tensors[key] = tensor.detach().to(device="cpu", dtype=torch.float32)

    def record_scalar(self, tag, microbatch_id, value):
        if not self.is_recording_rank or self.forward_recording_stopped:
            return
        self.scalars[f"{tag}/mb{microbatch_id}{self.forward_key_suffix}"] = float(value)


    def record_step_scalar(self, tag, step_index, value):
        """Record a per-step scalar (``{tag}/step{n}``), e.g. the grad norm of one optimizer step."""
        if not self.is_recording_rank:
            return
        self.scalars[f"{tag}/step{step_index}"] = float(value)


    def record_parameter_gradients(self, module, step_index, tag_prefix="param_grad"):
        """Record every parameter's reduced gradient of ``module`` (Task 6.7 ③ / 9.4 ④).

        调用点必须在 ``finalize_model_grads`` **之后**、下一步 ``zero_grad_buffer`` 之前：那时
        ``main_grad`` 才是"DDP 归约 + 按全局 token 数归一化"之后的最终值，也就是优化器真正会用的
        东西。参数名两侧一致（``vision_model.*`` / ``vision_projection.*`` / ``language_model.*``），
        所以直接用名字作键，与拓扑无关；键里带步号 ``{tag}/step{n}/{name}``，因为逐步都要记——
        只看 ``grad norm`` 这一个标量无法区分"差异弥散在所有参数上"（两侧落在不同点的必然结果）
        与"差异集中在某个组件上"（bug 的形状）。

        梯度张量太大（7B 参数）不存原值，只存 float64 摘要；小张量（≤1M 元素，layernorm / bias
        这类）逐步都存原值——数量不多，逐元素定位要用。
        """
        if not self.is_recording_rank:
            return 0
        count = 0
        for name, parameter in module.named_parameters():
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is None:
                continue
            key = f"{tag_prefix}/step{step_index}/{name}"
            self.digests[key] = tensor_digest(gradient)
            if gradient.numel() <= OPTIMIZER_STATE_RAW_TENSOR_ELEMENT_LIMIT:
                self.raw_tensors[key] = gradient.detach().to(device="cpu", dtype=torch.float32)
            count += 1
        return count

    def record_reduced_parameter_gradients(
        self, model_chunk, step_index, full_bucket_gradients, tag_prefix="reduced_grad"
    ):
        """Record每个参数**跨 rank 合并之后**的完整归约梯度（Task 8.9.4）。

        ``full_bucket_gradients`` 由 ``gather_reduced_bucket_gradients`` 给出：非 DistOpt 是
        all-reduce 之后的整块 buffer，DistOpt 是把各 rank 的 reduce-scatter 分片 all-gather 回来
        再拼接的结果。参数在 bucket 内的字节范围取自 ``bucket.param_to_index``，即生产代码自己
        用来给 ``main_grad`` 建视图的那份映射，所以切出来的就是该参数的完整梯度。

        键里带 rank（``{tag}/step{n}/rank{R}/{name}``）：encoder 参数在所有 rank 上同名，语言侧的
        ``decoder.layers.N`` 也是**流水段内的局部下标**，不带 rank 会在离线合并时互相覆盖，跨 rank
        的分歧就此消失——那正是"只有 DistOpt 出问题"这类现象最需要看到的信号。
        """
        rank = torch.distributed.get_rank()
        module = unwrap_model(model_chunk)
        names_by_parameter = {
            parameter: name for name, parameter in module.named_parameters()
        }
        count = 0
        for buffer in list(model_chunk.buffers) + list(model_chunk.expert_parallel_buffers):
            for bucket in buffer.buckets:
                full_gradient = full_bucket_gradients[id(bucket)]
                for parameter in bucket.params_list:
                    name = names_by_parameter.get(parameter)
                    if name is None:
                        continue
                    start_index, end_index = bucket.param_to_index[parameter]
                    gradient = full_gradient[start_index:end_index].view(parameter.shape)
                    key = f"{tag_prefix}/step{step_index}/rank{rank}/{name}"
                    self.digests[key] = tensor_digest(gradient)
                    if gradient.numel() <= OPTIMIZER_STATE_RAW_TENSOR_ELEMENT_LIMIT:
                        self.raw_tensors[key] = gradient.detach().to(
                            device="cpu", dtype=torch.float32
                        )
                    count += 1
        return count

    def record_model_parameters(self, model_chunks, tag):
        """Record both the bf16 model parameters and their fp32 master copies (diagnostic).

        用于区分"更新没回拷到 bf16 模型参数"与"前向输入是旧的"：调用点放在**每个 iteration 的
        第一次前向之前**，键 ``model_bf16/{tag}/{name}`` 与 ``main_fp32/{tag}/{name}``。
        若某个 iteration 的 ``main_fp32`` 变了而 ``model_bf16`` 没变 ⇒ 回拷没生效；两者都变而前向
        输出不变 ⇒ 前向吃到的是旧输入。参数量级大，只存摘要。
        """
        if not self.is_recording_rank:
            return 0
        count = 0
        for model_chunk in model_chunks:
            for name, parameter in unwrap_model(model_chunk).named_parameters():
                self.digests[f"model_bf16/{tag}/{name}"] = tensor_digest(parameter.data)
                main_parameter = getattr(parameter, "main_param", None)
                if main_parameter is not None:
                    self.digests[f"main_fp32/{tag}/{name}"] = tensor_digest(main_parameter)
                count += 1
        return count

    def record_optimizer_state(
        self, model_chunks, optimizer, step_index, store_raw_tensors=False
    ):
        """Record the fp32 master weights and the Adam moments after one optimizer step (Task 9.4 ④).

        逐步记录"梯度 → 权重"这一段的产物，键里带步号（``param/step{n}/{name}``）：
          * ``param``——**fp32 主副本**（``main_param``），也就是 Adam 真正更新的那份；模型上的
            bf16 参数只是它的舍入结果，比它多一层量化噪声，不适合当判据；
          * ``exp_avg`` / ``exp_avg_sq``——Adam 的一阶/二阶动量，误差若在累积就先在二阶动量上显形。

        参数名两侧一致，所以直接用名字作键、与拓扑无关。张量数量级：7B 参数逐步存原值不可行，
        故默认只存 float64 摘要，``store_raw_tensors`` 为真时才对小张量额外存原值（用于最后一步
        做逐元素定位）。
        """
        if not self.is_recording_rank:
            return 0
        state_by_parameter = {}
        for inner_optimizer in inner_torch_optimizers(optimizer):
            state_by_parameter.update(inner_optimizer.state)
        count = 0
        for model_chunk in model_chunks:
            for name, parameter in unwrap_model(model_chunk).named_parameters():
                main_parameter = getattr(parameter, "main_param", None)
                if main_parameter is None:
                    main_parameter = parameter
                tensors = {"param": main_parameter}
                parameter_state = state_by_parameter.get(main_parameter)
                if parameter_state is not None:
                    for moment_name in ("exp_avg", "exp_avg_sq"):
                        moment = parameter_state.get(moment_name)
                        if moment is not None:
                            tensors[moment_name] = moment
                for tag, tensor in tensors.items():
                    key = f"{tag}/step{step_index}/{name}"
                    self.digests[key] = tensor_digest(tensor)
                    if store_raw_tensors and (
                        tensor.numel() <= OPTIMIZER_STATE_RAW_TENSOR_ELEMENT_LIMIT
                    ):
                        self.raw_tensors[key] = tensor.detach().to(
                            device="cpu", dtype=torch.float32
                        )
                count += 1
        return count

    def dump(self, directory):
        os.makedirs(directory, exist_ok=True)
        rank = torch.distributed.get_rank()
        payload = {
            "side": self.side,
            "rank": rank,
            "digests": self.digests,
            "scalars": self.scalars,
        }
        json_path = os.path.join(directory, f"{self.side}_rank{rank}.json")
        with open(json_path, "w") as json_file:
            json.dump(payload, json_file, indent=1)
        tensor_path = os.path.join(directory, f"{self.side}_rank{rank}.pt")
        torch.save(self.raw_tensors, tensor_path)
        print(
            f"[rank {rank}] wrote {json_path} ({len(self.digests)} digests, "
            f"{len(self.scalars)} scalars) and {tensor_path} ({len(self.raw_tensors)} tensors)",
            flush=True,
        )


def inner_torch_optimizers(optimizer):
    """Return the underlying ``torch.optim`` instances that actually hold the Adam state.

    共置侧是一层 ``ChainedOptimizer``（每个组件一个子优化器，training.py:1706
    ``get_colocated_optimizer``），非共置侧是单个 ``MegatronOptimizer``；两者的 Adam 状态都挂在
    ``.optimizer`` 上（``MegatronOptimizer.optimizer``）。这里只取状态持有者，不关心裁剪逻辑。
    """
    if isinstance(optimizer, ChainedOptimizer):
        return [
            sub_optimizer.optimizer
            for sub_optimizer in optimizer.chained_optimizers
            if getattr(sub_optimizer, "optimizer", None) is not None
        ]
    inner = getattr(optimizer, "optimizer", None)
    return [inner] if inner is not None else []


def register_transformer_layer_hooks(root_module, recorder, microbatch_id_getter):
    """Hook every ``TransformerLayer`` output of the language decoder and the ViT.

    语言侧与视觉侧分别遍历，是因为两者的 ``layer_number`` 各自从 1 开始、直接混在一起会撞名。
    Returns the number of hooks registered on each side, for logging.
    """

    def hook_layers(parent, tag_prefix):
        decoder = getattr(parent, "decoder", None) if parent is not None else None
        layers = getattr(decoder, "layers", None) if decoder is not None else None
        if layers is None:
            return 0
        count = 0
        for layer in layers:
            if not isinstance(layer, TransformerLayer):
                continue
            tag = f"{tag_prefix}{layer.layer_number:02d}"

            def make_hook(tag):
                def hook(module, inputs, output):
                    recorder.record_tensor(tag, microbatch_id_getter(), output)

                return hook

            layer.register_forward_hook(make_hook(tag))
            count += 1
        return count

    language_hooks = hook_layers(getattr(root_module, "language_model", None), "layer")
    vision_hooks = hook_layers(getattr(root_module, "vision_model", None), "vision_layer")
    return language_hooks, vision_hooks


def wrap_finalize_to_record_gradients(
    model_chunk,
    recorder,
    wrapped_configs,
    step_index_getter,
    gradient_perturbation=0.0,
    gradient_relative_scale=0.0,
):
    """Wrap this chunk's finalize callback so parameter gradients are recorded right after it.

    ``config.finalize_model_grads_func`` 是 ``train()`` 在运行期挂上的（training.py），所以只能在
    第一次前向时包。组件名从**本次调用传入的 chunk** 上读——共置的 encoder 与 backbone 可能共用
    同一个 config 对象（那时只包一次），但两次调用要各记各的参数。

    ``step_index_getter`` 返回**本次即将执行的优化器步号**（已完成的步数 + 1）：finalize 发生在
    ``optimizer.step()`` 之前，所以不能直接用已完成步数。返回值 > 记录步数时不再记录。

    ``gradient_perturbation`` > 0 时，在记录**之前**往 encoder 的梯度里注入该量级（目标 rel_l2）
    的末位扰动；``gradient_relative_scale`` > 0 时改成把**全部**梯度整体乘 $$1+\\text{scale}$$
    （两个种子的形态差别见各自的函数注释）。都用于 Task 9.5 的对照实验，扰动**每一步都注入**
    （不受记录步数限制），否则第 6 步起两侧又回到同一条轨迹上。
    """
    config = get_model_config(model_chunk)
    if id(config) in wrapped_configs:
        return False
    original = config.finalize_model_grads_func
    if original is None:
        return False

    def recording_finalize(*args, **kwargs):
        result = original(*args, **kwargs)
        chunks = kwargs.get("model") if "model" in kwargs else args[0]
        if not isinstance(chunks, (list, tuple)):
            chunks = [chunks]
        if gradient_perturbation > 0.0:
            for chunk in chunks:
                count, achieved = perturb_gradients_in_place(
                    unwrap_model(chunk), gradient_perturbation
                )
                if count:
                    print(
                        f"[rank {torch.distributed.get_rank()}] perturbed {count} encoder "
                        f"gradient tensors, target rel_l2 {gradient_perturbation:.3e}, "
                        f"achieved {achieved:.3e}",
                        flush=True,
                    )
        if gradient_relative_scale > 0.0:
            for chunk in chunks:
                count = scale_gradients_in_place(unwrap_model(chunk), gradient_relative_scale)
                print(
                    f"[rank {torch.distributed.get_rank()}] scaled {count} gradient tensors by "
                    f"1 + {gradient_relative_scale:.3e}",
                    flush=True,
                )
        step_index = step_index_getter()
        if step_index is None:
            return result
        for chunk in chunks:
            count = recorder.record_parameter_gradients(unwrap_model(chunk), step_index)
            # 分片合并：DistOpt 下上面记到的 ``main_grad`` 只有本 rank 那一段是归约完成的，
            # 这里再记一份 all-gather 拼回来的完整梯度（两侧同名、可直接比较）。
            # 集合通信在**所有 rank 上同构执行**，不放在任何 rank 分支里。
            full_bucket_gradients = gather_reduced_bucket_gradients(chunk)
            reduced_count = recorder.record_reduced_parameter_gradients(
                chunk, step_index, full_bucket_gradients
            )
            print(
                f"[rank {torch.distributed.get_rank()}] step {step_index}: recorded {count} "
                f"parameter gradients and {reduced_count} merged full gradients of "
                f"{type(unwrap_model(chunk)).__name__}",
                flush=True,
            )
        return result

    config.finalize_model_grads_func = recording_finalize
    wrapped_configs.add(id(config))
    return True


def patch_optimizer_step_to_record(
    recorder_getter, model_chunks_getter, recorded_steps, raw_tensor_steps=(), on_recorded=None
):
    """Record ``grad norm`` / clip coefficient / weights / Adam moments after each of N steps.

    为什么打的是**类方法**而不是包某个实例：优化器由 ``pretrain`` 内部建出，驱动拿不到它。
    被打的两个类恰好覆盖两侧实际走的路径——共置侧是一层 ``ChainedOptimizer``（其 ``step`` 直接
    调子优化器的 ``step_with_ready_grads``、不会再进 ``step``，所以不会重复计数），非共置侧的
    ``Float16OptimizerWithFloat16Params`` 继承 ``MixedPrecisionOptimizer.step``。仍然加一道
    重入保护，万一将来嵌套只记最外层那次。

    ``step`` 的返回值是 ``(update_successful, grad_norm, num_zeros_in_grad)``（optimizer.py:1390），
    裁剪系数不在返回值里，按上游同一公式由 ``grad_norm`` 与 ``clip_grad`` 推出
    （``clip_grad_by_total_norm_fp32``：``clip_coeff = max_norm / (total_norm + 1e-6)``，仅当 < 1
    才生效）——两侧只要 ``grad norm`` 相同，系数必然相同，记下来是为了让判据能直接读。
    """
    counter = {"steps": 0, "inside": False}

    def make_recording_step(original_step):
        def recording_step(self):
            if counter["inside"]:
                return original_step(self)
            counter["inside"] = True
            try:
                update_successful, grad_norm, num_zeros_in_grad = original_step(self)
            finally:
                counter["inside"] = False
            if counter["steps"] >= recorded_steps:
                return update_successful, grad_norm, num_zeros_in_grad
            counter["steps"] += 1
            step_index = counter["steps"]
            recorder = recorder_getter()
            if grad_norm is not None:
                recorder.record_step_scalar("grad_norm", step_index, grad_norm)
                clip_grad = self.config.clip_grad
                clip_coefficient = 1.0
                if clip_grad > 0.0:
                    clip_coefficient = min(1.0, clip_grad / (grad_norm + 1.0e-6))
                recorder.record_step_scalar("clip_coefficient", step_index, clip_coefficient)
            if num_zeros_in_grad is not None:
                recorder.record_step_scalar("num_zeros", step_index, num_zeros_in_grad)
            count = recorder.record_optimizer_state(
                model_chunks_getter(),
                self,
                step_index,
                store_raw_tensors=step_index in raw_tensor_steps,
            )
            print(
                f"[rank {torch.distributed.get_rank()}] step {step_index}: grad norm {grad_norm}, "
                f"recorded {count} parameters",
                flush=True,
            )
            # 参数 all-gather 已在 ``step`` 内完成（overlap_param_gather 关闭时是同步的），此刻
            # 模型上的权重就是下一轮前向真正会用的那份 ⇒ 是检查副本是否分叉的正确时刻。
            if mpu.is_colocated_encoder_enabled():
                for chunk in model_chunks_getter():
                    module = unwrap_model(chunk)
                    if getattr(module, "colocated_module_name", None) != "encoder":
                        continue
                    spread = record_colocated_encoder_replica_spread(
                        chunk, recorder, step_index
                    )
                    print(
                        f"[rank {torch.distributed.get_rank()}] step {step_index}: encoder "
                        f"replica digest spread {spread}",
                        flush=True,
                    )
            if on_recorded is not None:
                on_recorded(step_index)
            return update_successful, grad_norm, num_zeros_in_grad

        return recording_step

    for optimizer_class in (ChainedOptimizer, MixedPrecisionOptimizer):
        optimizer_class.step = make_recording_step(optimizer_class.step)


def gather_reduced_bucket_gradients(model_chunk):
    """Return ``{id(bucket): full reduced gradient}`` for one DDP-wrapped chunk (Task 8.9.4).

    非 DistOpt 下 ``bucket.grad_data`` 在 all-reduce 之后**本身就是整块归约梯度**；DistOpt 下
    reduce-scatter 只让每个 rank 持有自己那一段是归约完成的，其余段仍是本 rank 的局部值，因此
    必须把各 rank 的分片 all-gather 回来再拼接，才能得到与非 DistOpt 可比的完整梯度。分片边界
    取自 ``shard_buffer``（与生产 reduce-scatter 用的是同一个函数），拼接顺序即 rank 顺序。

    The reduce-scatter shard of each rank is gathered back so the full gradient becomes
    comparable with the non-distributed-optimizer path.
    """
    full_gradients = {}
    for buffer in list(model_chunk.buffers) + list(model_chunk.expert_parallel_buffers):
        data_parallel_group = buffer.data_parallel_group
        data_parallel_world_size = buffer.data_parallel_world_size
        uses_shards = (
            model_chunk.ddp_config.use_distributed_optimizer and data_parallel_world_size > 1
        )
        for bucket in buffer.buckets:
            if not uses_shards:
                full_gradients[id(bucket)] = bucket.grad_data
                continue
            local_shard = shard_buffer(bucket.grad_data, data_parallel_world_size)[
                get_pg_rank(data_parallel_group)
            ].contiguous()
            gathered_shards = [
                torch.empty_like(local_shard) for _ in range(data_parallel_world_size)
            ]
            torch.distributed.all_gather(gathered_shards, local_shard, group=data_parallel_group)
            full_gradients[id(bucket)] = torch.cat(gathered_shards)
    return full_gradients


def record_colocated_encoder_replica_spread(model_chunk, recorder, step_index):
    """Record how far the encoder replicas drifted apart after one optimizer step (Task 8.9.4).

    encoder 在共置 dp 组的每个 rank 上都是**完整副本**。DistOpt 把它的优化器状态沿该组切成
    W 份，每个 rank 只更新自己那一段，再靠参数 all-gather 把整份权重还原回来 ⇒ **只要 shard
    偏移、归约域或参数 gather 有一处不对，各 rank 的副本就会分叉**；而梯度归约只同步梯度、
    从不同步参数，训练不会报错，loss 也仍然像模像样。这条探针就是把那个静默分叉变成数字：
    按参数名排序在 float64 上累加 (和, 平方和)，再在组内取 MIN 与 MAX，记录两者的差。

    非 DistOpt 路径每个 rank 都自己更新整份权重、输入又完全相同，差值应当恒为 0；DistOpt 若
    实现正确也应为 0（bf16 权重逐位相同）。非 0 即为参数侧的实现问题，与浮点结合序无关——
    结合序只影响梯度的值，不会让同一份被 gather 回来的权重在 rank 之间不一致。
    """
    digest = torch.zeros(2, dtype=torch.float64, device=torch.cuda.current_device())
    module = unwrap_model(model_chunk)
    parameters = dict(module.named_parameters())
    for name in sorted(parameters):
        values = parameters[name].detach().double()
        digest[0] += values.sum()
        digest[1] += (values * values).sum()
    colocated_data_parallel_group = mpu.get_colocated_data_parallel_group()
    minimum, maximum = digest.clone(), digest.clone()
    torch.distributed.all_reduce(
        minimum, op=torch.distributed.ReduceOp.MIN, group=colocated_data_parallel_group
    )
    torch.distributed.all_reduce(
        maximum, op=torch.distributed.ReduceOp.MAX, group=colocated_data_parallel_group
    )
    spread = (maximum - minimum).tolist()
    recorder.record_step_scalar("encoder_replica_sum_spread", step_index, spread[0])
    recorder.record_step_scalar("encoder_replica_sum_squares_spread", step_index, spread[1])
    recorder.record_step_scalar("encoder_replica_sum", step_index, digest[0].item())
    return spread


