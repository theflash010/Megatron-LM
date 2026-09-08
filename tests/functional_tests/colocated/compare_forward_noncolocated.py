# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Non-colocated reference side of the module-level forward comparison (Task 6.7 ①).

与 ``compare_forward_colocated.py`` 成对使用：同一批固定 micro batch、同一份
``vision_projection`` 初始化、dropout 0，跑 TP=1/PP=4 的**非共置** LLaVA（层切分与共置
backbone 完全相同），把相同位置的张量记下来落盘。

记录的位置与共置侧一一对应：
  * ``encoder_output``——在 ``vision_projection`` 上挂 forward hook 取其输出。共置侧这个位置
    是 encoder chunk 的返回值（``image_embeddings``），非共置侧它是 LLaVAModel forward 内部的
    中间量，只能用 hook 取；
  * ``backbone_stage_output``——``forward_step`` 的返回值（末 stage 是逐 token loss）；
  * ``loss`` / ``num_tokens``——末 stage 每个 microbatch 的标量。

microbatch 号按**每个 stage 的前向调用顺序**取：1F1B 下每个 stage 的前向就是 0,1,...,n-1。
"""
import os
import sys

import torch

MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from dataloader_provider import is_first_or_last_stage
from fixed_micro_batch import FixedMicroBatchIterator
from fixed_vision_projection import sync_vision_projection
from forward_record import (
    ForwardRecorder,
    patch_optimizer_step_to_record,
    register_transformer_layer_hooks,
    wrap_finalize_to_record_gradients,
)
from model import model_provider
from multimodal_args import add_multimodal_extra_args
from train import forward_step, llava_embedding_ranks, llava_position_embedding_ranks

from megatron.core import parallel_state as mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.utils import get_attr_wrapped_model, unwrap_model
from megatron.training import pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args

FIXED_BATCH_DIRECTORY = "/home/zn/zn_data/workspace/fixed_micro_batches"
VISION_PROJECTION_PATH = "/home/zn/zn_data/workspace/fixed_vision_projection.pt"
# 对照实验要往别的目录落盘，否则会盖掉主对照（共置 vs 非共置）那一轮的产物。
DUMP_DIRECTORY = os.environ.get(
    "FORWARD_DUMP_DIRECTORY", "/home/zn/zn_data/workspace/forward_dumps"
)
# ④ 优化器步等价性（Task 9.4）：与共置侧同一组常量，两侧必须记同样多的步数。
RECORDED_OPTIMIZER_STEPS = 5
RAW_TENSOR_OPTIMIZER_STEPS = (1, RECORDED_OPTIMIZER_STEPS)
# ⑤ 多步漂移的对照实验（Task 9.5）：>0 时本侧在 encoder 梯度里注入该目标 rel_l2 的末位扰动，
# 侧名也随之加 ``perturbed`` 标记（否则两次非共置运行会写同名文件互相覆盖）。两次运行除这个
# 种子之外完全相同 ⇒ 漂移曲线就是"纯放大"的标尺。
ENCODER_GRADIENT_PERTURBATION = float(os.environ.get("ENCODER_GRADIENT_PERTURBATION", "0.0"))
# 第二个对照种子：把**全部**梯度整体乘 1+scale，用来复现"共置与非共置的裁剪系数差 1.442e-7"
# 这件事——它是全模型稠密的，与上面那个稀疏的 encoder 末位扰动形态完全不同。
GRADIENT_RELATIVE_SCALE = float(os.environ.get("GRADIENT_RELATIVE_SCALE", "0.0"))

state = {"forward_calls": 0, "dumped": False, "prepared": False, "recorder": None}
state["current_microbatch_id"] = 0
state["wrapped_configs"] = set()
state["iteration"] = 0
state["optimizer_steps"] = 0
state["model_chunks"] = []


def side_name():
    """Topology-derived side name, so TP1/PP4 and TP4/PP1 dumps never collide."""
    marker = ""
    if ENCODER_GRADIENT_PERTURBATION > 0.0:
        marker = "_perturbed"
    elif GRADIENT_RELATIVE_SCALE > 0.0:
        marker = "_scaled"
    # 标记插在拓扑后缀**之前**：``compare_forward_dumps.py`` 靠 ``_tp{N}pp{M}$`` 读拓扑。
    return (
        f"noncolocated{marker}"
        f"_tp{mpu.get_tensor_model_parallel_world_size()}"
        f"pp{mpu.get_pipeline_model_parallel_world_size()}"
    )


def get_recorder():
    if state["recorder"] is None:
        state["recorder"] = ForwardRecorder(
            side=side_name(),
            is_recording_rank=mpu.get_tensor_model_parallel_rank() == 0,
        )
    return state["recorder"]


def fixed_micro_batch_dataloaders_provider(train_val_test_num_samples):
    """Mirror the upstream gate: only the first and last pipeline stages own a dataloader."""
    pipeline_size = mpu.get_pipeline_model_parallel_world_size()
    if not is_first_or_last_stage(pipeline_size):
        return None, None, None
    microbatch_ids = list(range(get_num_microbatches()))
    print(
        f"[rank {torch.distributed.get_rank()}] fixed micro batches {microbatch_ids}",
        flush=True,
    )
    return FixedMicroBatchIterator(microbatch_ids, FIXED_BATCH_DIRECTORY), None, None


fixed_micro_batch_dataloaders_provider.is_distributed = True


def next_optimizer_step_index():
    """The step about to run (finalize happens before ``step()``); None once N steps are done."""
    step_index = state["optimizer_steps"] + 1
    return step_index if step_index <= RECORDED_OPTIMIZER_STEPS else None


def prepare_once(model_chunk):
    """Register the per-layer hooks, and on the stage that owns it load the vision_projection."""
    if state["prepared"]:
        return
    state["prepared"] = True
    module = unwrap_model(model_chunk)
    # 优化器步的记录需要按名字遍历参数，而驱动拿不到 pretrain 内部的 model 列表。
    state["model_chunks"].append(model_chunk)

    language_hooks, vision_hooks = register_transformer_layer_hooks(
        module, get_recorder(), lambda: state["current_microbatch_id"]
    )
    # ③ 反向：finalize 之后记每个参数的 main_grad（归约 + 归一化之后的最终值）。逐步都记，
    # 键带步号，与共置侧同构。
    wrap_finalize_to_record_gradients(
        model_chunk,
        get_recorder(),
        state["wrapped_configs"],
        next_optimizer_step_index,
        gradient_perturbation=ENCODER_GRADIENT_PERTURBATION,
        gradient_relative_scale=GRADIENT_RELATIVE_SCALE,
    )
    print(
        f"[rank {torch.distributed.get_rank()}] hooked {language_hooks} language layers and "
        f"{vision_hooks} vision layers",
        flush=True,
    )

    vision_projection = getattr(module, "vision_projection", None)
    if vision_projection is None:
        return

    action, count = sync_vision_projection([model_chunk], VISION_PROJECTION_PATH)
    print(
        f"[rank {torch.distributed.get_rank()}] vision_projection {action} ({count} parameters)",
        flush=True,
    )

    def record_projection_output(module_, inputs, output):
        # 停记与后缀都由 recorder 自己管，这里无条件调用即可。
        get_recorder().record_tensor("encoder_output", state["current_microbatch_id"], output)

    vision_projection.register_forward_hook(record_projection_output)


def reference_model_provider(pre_process=True, post_process=True, **keyword_arguments):
    """Build the vision tower **only on the first pipeline stage**.

    ``ModelType.encoder_or_decoder`` 下 ``get_model`` 不传 ``add_encoder``，于是
    ``model_provider`` 的默认值 True 会让**每个 stage 都建 ViT**；而中间/末 stage 的
    ``images`` 是 None（``get_batch`` 对非首末 stage 直接返回 None、末 stage 在 pp>1 时把
    imgs 置 None），``LLaVAModel.forward`` 的 ``elif self.add_encoder and not has_images``
    分支会去读 ``images.dtype`` 直接 AttributeError。这不是共置引入的问题——它是
    "epp=0 + PP>1 的非共置 LLaVA" 本身的状态，只是基线一直跑 PP=1 所以没暴露。
    这里把 ``add_encoder`` 绑到 ``pre_process``：vision 只在 stage 0，正好与 7.5 建出的
    ``llava_noncolocated_pp4``（stage 0 才有 vision_model / vision_projection）以及共置
    backbone 的层切分一致。
    """
    keyword_arguments.pop("add_encoder", None)
    return model_provider(
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=pre_process,
        **keyword_arguments,
    )


def recording_loss_func(original_loss_func, microbatch_id):
    def wrapper(*call_args, **call_kwargs):
        result = original_loss_func(*call_args, **call_kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            get_recorder().record_scalar("loss", microbatch_id, result[0].item())
            get_recorder().record_scalar("num_tokens", microbatch_id, result[1].item())
        return result

    return wrapper


def instrumented_forward_step(data_iterator, model):
    """``train.forward_step`` 的记录版。"""
    prepare_once(model)

    num_microbatches = get_num_microbatches()
    # 换轮点与共置侧同构：第 2 个 iteration 的前向用 ``/it2`` 后缀继续记（诊断更新是否晚一轮
    # 生效，本侧是对照组），第 3 个起停记；每个换轮点记一次 bf16 参数与 fp32 主副本。
    iteration = state["forward_calls"] // num_microbatches + 1
    if iteration != state["iteration"]:
        state["iteration"] = iteration
        # 与共置侧同构：第 1 个 iteration 不拍快照（那侧此刻只登记了 encoder chunk，拍了会变成
        # 一堆"只在一侧出现"的键），其内容已由 ``param/step1`` 覆盖（第 1 步 lr=0）。
        if iteration > 1:
            get_recorder().record_model_parameters(state["model_chunks"], f"it{iteration}")
        if iteration == 2:
            get_recorder().set_forward_key_suffix("/it2")
        elif iteration > 2:
            get_recorder().stop_forward_recording()

    microbatch_id = state["forward_calls"] % num_microbatches
    # 逐层 hook 在 forward 内部触发，microbatch 号必须在调用之前设好。
    state["current_microbatch_id"] = microbatch_id
    output_tensor, loss_function = forward_step(data_iterator, model)
    state["forward_calls"] += 1

    # 末 stage 的返回值是逐 token loss；非末 stage 的返回值等于本 stage 最后一层的输出，已由
    # 逐层 hook 覆盖。
    if get_attr_wrapped_model(model, "post_process"):
        get_recorder().record_tensor("final_token_loss", microbatch_id, output_tensor)
    return output_tensor, recording_loss_func(loss_function, microbatch_id)


def record_optimizer_steps_and_dump(step_index):
    """Called after each recorded optimizer step; the last one triggers the dump."""
    state["optimizer_steps"] = step_index
    if step_index >= RECORDED_OPTIMIZER_STEPS and not state["dumped"]:
        get_recorder().dump(DUMP_DIRECTORY)
        state["dumped"] = True


if __name__ == "__main__":
    arguments = parse_and_validate_args(
        extra_args_provider=add_multimodal_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    assert not getattr(arguments, "use_colocated_encoder", False), (
        "this is the non-colocated reference side; do not pass --use-colocated-encoder"
    )
    assert arguments.train_iters >= RECORDED_OPTIMIZER_STEPS, (
        "this driver records the first "
        f"{RECORDED_OPTIMIZER_STEPS} optimizer steps, so --train-iters must be at least that many"
    )
    # ⓑ dropout=0：与共置侧同因——``--use-checkpoint-args`` 会用 ckpt 里的 0.1 覆盖命令行的 0，
    # 而它带来的 ``padded_vocab_size`` 又是必须的，故解析后显式清零再断言。
    arguments.hidden_dropout = 0.0
    arguments.attention_dropout = 0.0
    assert arguments.hidden_dropout == 0.0 and arguments.attention_dropout == 0.0, (
        "element-wise comparison requires dropout 0 on both sides"
    )
    full_config = pretrain_cfg_container_from_args(arguments)
    print(
        f"dump directory {DUMP_DIRECTORY}, encoder gradient perturbation "
        f"{ENCODER_GRADIENT_PERTURBATION:.3e}, gradient relative scale "
        f"{GRADIENT_RELATIVE_SCALE:.3e}",
        flush=True,
    )

    # ④ 优化器步：类方法打桩必须在 pretrain 建出优化器**之前**完成。
    patch_optimizer_step_to_record(
        get_recorder,
        lambda: state["model_chunks"],
        RECORDED_OPTIMIZER_STEPS,
        raw_tensor_steps=RAW_TENSOR_OPTIMIZER_STEPS,
        on_recorded=record_optimizer_steps_and_dump,
    )

    pretrain(
        full_config,
        fixed_micro_batch_dataloaders_provider,
        reference_model_provider,
        ModelType.encoder_or_decoder,
        instrumented_forward_step,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
