# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Load-only driver for the colocated LLaVA checkpoint (Task 7.8).

This driver runs the real ``examples/multimodal`` model provider against the real
converted checkpoint, but stops right after ``setup_model_and_optimizer``, so it
exercises the whole checkpoint path (``load_checkpoint`` -> per chunk key checks
-> load hooks -> cross replica check) without needing the dataset work of Task 6.
该驱动使用真实的 ``examples/multimodal`` 模型构建函数与真实转换出来的检查点，但在
``setup_model_and_optimizer`` 之后立即停止，因此可以在不依赖 Task 6 的数据集工作的
前提下，完整走一遍检查点通路（``load_checkpoint`` -> 分 chunk 键校验 -> 加载钩子
-> 跨副本一致性校验）。
"""
import os
import sys

import torch

# ``examples/multimodal`` 下的 ``model.py`` / ``multimodal_args.py`` 是按“脚本同目录
# 导入”的方式写的（例如 ``from config import ...``），所以必须先把该目录以及仓库根目录
# 加入 ``sys.path``，再导入其中的符号。本文件位于
# ``<repo>/tests/functional_tests/colocated/``，故仓库根目录是上溯三级。
MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from model import model_provider
from multimodal_args import add_multimodal_extra_args

from megatron.core import parallel_state as mpu
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import parse_and_validate_args
from megatron.training.checkpointing import get_checkpoint_name
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer


def llava_embedding_ranks(pipeline_parallel_ranks):
    """Copy of ``examples/multimodal/train.py``'s ``llava_embedding_ranks`` (:303-313).

    It is duplicated instead of imported because importing ``train.py`` would drag in
    the Energon dataloader provider, which is exactly the part Task 6 has not adapted yet.
    这里复制而不是导入，是因为导入 ``train.py`` 会连带引入 Energon 数据加载模块，
    而那正是 Task 6 尚未适配的部分。
    """
    last_rank = pipeline_parallel_ranks[-1]
    if len(pipeline_parallel_ranks) == 1:
        return [last_rank]
    return [pipeline_parallel_ranks[0], last_rank]


def llava_position_embedding_ranks(pipeline_parallel_ranks):
    """Copy of ``examples/multimodal/train.py``'s ``llava_position_embedding_ranks`` (:316-326)."""
    last_rank = pipeline_parallel_ranks[-1]
    if len(pipeline_parallel_ranks) == 1:
        return [last_rank]
    return [pipeline_parallel_ranks[0]]


def pretrained_checkpoint_file(checkpoint_directory):
    """Return this rank's checkpoint file inside ``--pretrained-checkpoint``.

    ``get_checkpoint_name`` derives the ``mp_rank_{tp:02d}_{pp:03d}`` sub directory from
    the current process groups, so every rank naturally opens its own shard.
    ``get_checkpoint_name`` 会根据当前进程组推导 ``mp_rank_{tp:02d}_{pp:03d}`` 子目录，
    因此每个 rank 自然打开属于自己的那一份分片。
    """
    tracker_path = os.path.join(checkpoint_directory, "latest_checkpointed_iteration.txt")
    with open(tracker_path, "r") as tracker_file:
        iteration = int(tracker_file.read().strip())
    return get_checkpoint_name(checkpoint_directory, iteration, release=False)


def verify_parameters_match_checkpoint(model):
    """Compare every checkpoint tensor against the live model parameter bit for bit.

    The guards added in Task 7.7 prove the *key sets* agree; they cannot prove the
    *values* were actually written, because ``load_model_state_dict`` falls back to
    ``strict=False`` on any exception (checkpointing.py:1893-1901) and a silently skipped
    tensor would leave the parameter at its random initialization. This function closes
    that hole by re-reading the checkpoint file and comparing tensors.
    Task 7.7 的守卫只能证明**键集合**一致，无法证明**数值**真的写进去了：
    ``load_model_state_dict``（checkpointing.py:1893-1901）在任何异常下都会回落到
    ``strict=False``，被静默跳过的张量会停留在随机初始化状态。本函数重新读取检查点
    文件并逐张量比对，堵住这个洞。
    """
    args = get_args()
    checkpoint_path = pretrained_checkpoint_file(args.pretrained_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for chunk_index, model_chunk in enumerate(model):
        module_name = get_attr_wrapped_model(model_chunk, "colocated_module_name")
        checkpoint_parameters = checkpoint["model%d" % chunk_index]
        model_parameters = get_attr_wrapped_model(model_chunk, "state_dict")()

        compared_tensor_count = 0
        compared_element_count = 0
        mismatched_names = []
        for name, checkpoint_value in checkpoint_parameters.items():
            # ``_extra_state`` 的值恒为 None（transformer_engine 兼容占位），没有可比内容。
            if not torch.is_tensor(checkpoint_value):
                continue
            model_value = model_parameters[name].detach().cpu()
            assert model_value.shape == checkpoint_value.shape, (
                f"[{module_name}] shape mismatch for {name}: "
                f"model {tuple(model_value.shape)} vs checkpoint {tuple(checkpoint_value.shape)}"
            )
            if not torch.equal(model_value, checkpoint_value.to(model_value.dtype)):
                mismatched_names.append(name)
            compared_tensor_count += 1
            compared_element_count += checkpoint_value.numel()

        assert not mismatched_names, (
            f"[{module_name}] {len(mismatched_names)} tensors differ from the checkpoint, "
            f"which means they were never loaded. First few: {sorted(mismatched_names)[:5]}"
        )
        print(
            f"[rank {torch.distributed.get_rank()}] chunk {chunk_index} ({module_name}): "
            f"{compared_tensor_count} tensors / {compared_element_count} elements "
            f"match {checkpoint_path}",
            flush=True,
        )


def _optional_attribute(model_chunk, name):
    """Return the chunk's attribute, or "n/a" when the component does not define it.

    ``ColocatedViTEncoder`` has no ``pre_process`` / ``post_process``: those describe a
    position inside a pipeline, and the encoder has none (its pipeline group has a single
    member). Only the backbone chunk carries them.
    ``ColocatedViTEncoder`` 没有 ``pre_process`` / ``post_process``——这两个字段描述的是
    在流水线里的位置，而 encoder 没有流水线（它的 pp 组只有一个成员）。只有 backbone
    chunk 才带这两个字段。
    """
    try:
        return get_attr_wrapped_model(model_chunk, name)
    except RuntimeError:
        return "n/a"


def report_model_shape(model):
    """Print the per chunk parameter inventory, so a wrong split is visible in the log."""
    for chunk_index, model_chunk in enumerate(model):
        module_name = get_attr_wrapped_model(model_chunk, "colocated_module_name")
        parameter_count = sum(
            parameter.numel() for parameter in model_chunk.parameters()
        )
        print(
            f"[rank {torch.distributed.get_rank()}] chunk {chunk_index} ({module_name}): "
            f"{parameter_count} parameters, "
            f"pre_process={_optional_attribute(model_chunk, 'pre_process')}, "
            f"post_process={_optional_attribute(model_chunk, 'post_process')}",
            flush=True,
        )


if __name__ == "__main__":
    parse_and_validate_args(
        extra_args_provider=add_multimodal_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    initialize_megatron(
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
    arguments = get_args()
    assert mpu.is_colocated_encoder_enabled(), (
        "This driver only covers the colocated encoder path; pass --colocated-encoder."
    )
    assert not arguments.load, "--load must stay empty so the pretrained checkpoint is used."

    # ``setup_model_and_optimizer`` itself runs ``load_checkpoint`` plus the Task 7.7
    # cross replica check, so no extra call is needed here.
    # ``setup_model_and_optimizer`` 内部已经串起了 ``load_checkpoint`` 与 Task 7.7 的
    # 跨副本校验，这里无需额外调用。
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, ModelType.encoder_or_decoder, checkpointing_context={}
    )
    report_model_shape(model)
    verify_parameters_match_checkpoint(model)

    torch.distributed.barrier()
    print_rank_0(">>> colocated checkpoint load verification passed")
    torch.distributed.destroy_process_group()
