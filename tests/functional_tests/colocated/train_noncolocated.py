# Copyright (c) 2026, Zhang Nan. All rights reserved.
"""Non-colocated LLaVA training entry point usable at **any** pipeline size.

与 ``examples/multimodal/train.py`` 的唯一区别是 ``add_encoder`` 的取值，其余（数据集、
``forward_step``、embedding rank 规则、在线评估回调）全部复用，以保证三侧的训练口径一致。

**为什么必须换一个入口**：LLaVA 用 ``ModelType.encoder_or_decoder``，而 ``get_model`` 对这个
model type 不传 ``add_encoder``，于是 ``model_provider`` 的默认值 ``True`` 会让**每个 stage 都
构建 ViT**；但只有 stage 0 拿得到 ``images``（``get_batch`` 对非首末 stage 直接返回 None，末
stage 在 pp>1 时也置 None），于是 ``LLaVAModel.forward`` 的 ``elif self.add_encoder and not
has_images`` 分支会去读 ``images.dtype``，在 ``llava_model.py:858`` 抛
``AttributeError: 'NoneType' object has no attribute 'dtype'``。

这是"epp=0（视觉与语言同栈）+ PP>1"这个组合本身跑不起来，与共置无关——上游基线一直是 PP=1
所以从未暴露。这里把 ``add_encoder`` 绑到 ``pre_process``（ViT 只在 stage 0），与共置侧
Task 7.5 建出的 PP4 产物形态一致。PP=1 时 ``pre_process`` 恒为真，行为与 train.py 完全相同，
因此 TP4/PP1 与 TP1/PP4 两侧可以共用这一个入口，避免两份几乎相同的 driver 漂移。
"""
import os
import sys

MEGATRON_SOURCE_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(MEGATRON_SOURCE_DIRECTORY, "examples", "multimodal"))
sys.path.insert(0, MEGATRON_SOURCE_DIRECTORY)

from dataloader_provider import train_valid_test_dataloaders_provider
from model import model_provider
from multimodal_args import add_multimodal_extra_args
from train import (
    forward_step,
    llava_embedding_ranks,
    llava_position_embedding_ranks,
    run_online_eval,
    write_online_eval_to_tensorboard,
)

from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args


def noncolocated_model_provider(pre_process=True, post_process=True, **keyword_arguments):
    """Build the LLaVA model with the vision encoder only on the first pipeline stage."""
    # get_model 可能已经传了 add_encoder（未来上游若补上），这里以 pre_process 为准并去重，
    # 避免 "got multiple values for keyword argument" 。
    keyword_arguments.pop("add_encoder", None)
    return model_provider(
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=pre_process,
        **keyword_arguments,
    )


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    arguments = parse_and_validate_args(
        extra_args_provider=add_multimodal_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    full_config = pretrain_cfg_container_from_args(arguments)

    pretrain(
        full_config,
        train_valid_test_dataloaders_provider,
        noncolocated_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
        non_loss_data_func=run_online_eval,
    )
