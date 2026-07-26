# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain or SFT multimodal."""
import math
import os
import sys
from functools import partial

import torch
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from dataloader_provider import train_valid_test_dataloaders_provider, is_first_or_last_stage
from model import model_provider
from multimodal_args import add_multimodal_extra_args

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import nvtx_range_pop, nvtx_range_push
from megatron.core.enums import ModelType
from megatron.core.models.multimodal import context_parallel
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, LLaVAModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    is_pipeline_last_stage,
)
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import is_last_rank, get_batch_on_this_cp_rank
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args
import torch.distributed as dist

def get_batch(data_iterator, image_token_index, img_seq_len):
    """Generate a batch

    Note: attn_mask_type in layer_specs.py sets the attention mask. Attention mask is None here.
    """
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None
    num_tiles = None
    packed_seq_params = None

    args = get_args()

    # Dataloader doesn't run on the middle stages in a pipeline parallel model.
    pp_size = get_pipeline_model_parallel_world_size() #获取PP并行度
    if not is_first_or_last_stage(pp_size): #如果不是第一个或最后一个stage，直接返回None，因为不需要数据
        # Note these are all set to None above.
        return tokens, labels, loss_mask, attention_mask, position_ids, imgs, num_tiles, packed_seq_params

    # Broadcast data.
    nvtx_range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0: #TP rank = 0的进程从data_iterator中取出新的micro batch数据
        data = next(data_iterator)
    else:
        data = None

    data_text = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"] #发送token数据，文本token经过tokenizer分词
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"] #发送label数据，还没有进行偏移（已经mask了系统提示+<image>+用户prompt）

    imgs = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"] #发送image数据，torch数据
    num_tiles = tensor_parallel.broadcast_data(["num_tiles"], data, torch.int32)["num_tiles"] #发送image分块数量

    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32)["cu_lengths"] #发送cu_lengths数据（pack 在一个序列里的所有子样本中，最长那个的长度。）
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32)["max_lengths"] #发送max_lengths数据（每个子样本的累加长度（cumulative sum）。）

    # No image input (text-only sample) if the dataloader returned a size 1 image.
    if imgs.shape == torch.Size([1, 1]):
        # FSDP can hang with text-only samples. A workaround is to run a valid dummy image through the vision
        # model and then add image embeddings with a zero multiplier.
        if args.use_torch_fsdp2:
            imgs = torch.zeros((1, 3, args.img_h, args.img_w), dtype=torch.float32, device=data_text.device)
            num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)
        else:
            # Similar workaround is not needed without FSDP and we can use an empty image.
            # FIXME: text-only data can cause still cause a hang in the special case where
            # the vision model is own its own pipeline rank and --freeze-ViT is enabled.
            imgs = torch.tensor([], dtype=torch.float32, device=data_text.device)
            num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)

    # Last pipeline parallel stage doesn't need images.
    if pp_size > 1 and is_pipeline_last_stage():
        imgs = None

    # If cu_lengths and max_lengths are non-dummy, construct PackedSeqParams. Otherwise, leave it at None.
    if cu_lengths.shape != torch.Size([1, 1]):
        assert (
            cu_lengths.shape[0] == max_lengths.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_lengths = cu_lengths[0]
        max_lengths = max_lengths[0]

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths,
            cu_seqlens_kv=cu_lengths,
            max_seqlen_q=max_lengths,
            max_seqlen_kv=max_lengths,
        )

    nvtx_range_pop("get_data")

    tokens_ = data_text.long() #把 data_text 强转成 PyTorch 的 torch.long 类型。

    nvtx_range_push("index tokens")
    tokenizer = get_tokenizer()
    text_length = tokens_.shape[1]
    tokens = tokens_[:, :text_length].contiguous() # 取 columns [0, 1, ..., text_length-1]
    labels = labels[:, 1 : text_length + 1].contiguous() # 取 columns [1, 2, ..., text_length]，这里对label进行了偏移（左移一位），真正的label

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    nvtx_range_pop("index tokens")

    nvtx_range_push("get_ltor_masks_and_position_ids")
    loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, labels, tokenizer.pad) #获取loss掩码和位置编码
    nvtx_range_pop("get_ltor_masks_and_position_ids")

    # If context parallel is enabled, must shard inputs to CP ranks.
    if args.context_parallel_size > 1 or args.sequence_parallel:
        assert tokens.shape[0], "micro-batch-size > 1 not supported yet with CP"

        num_image_tokens = torch.sum(tokens == image_token_index).item()
        num_image_embeddings = img_seq_len * imgs.shape[0] - num_image_tokens
        seq_len = text_length + num_image_embeddings

        # CP expects sequence length is divisible by CP size so apply padding.
        mp_padding_needed = context_parallel.get_padding(
            seq_len, args.context_parallel_size,
            args.tensor_model_parallel_size, args.sequence_parallel,
        )
        tokens, position_ids, labels, loss_mask = [torch.nn.functional.pad(item, (0, mp_padding_needed)) for item in (tokens, position_ids, labels, loss_mask)]

        # Get PackedSeqParams that indicate the amount of padding for TransformerEngine.
        packed_seq_params = context_parallel.get_packed_seq_params(tokens, num_image_embeddings, mp_padding_needed, args.context_parallel_size, True)

    return (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        imgs,
        num_tiles,
        packed_seq_params,
    )


def get_ltor_masks_and_position_ids(input_ids, target, pad_token):
    """Build masks and position id for left to right model."""
    seq_length = input_ids.shape[1]

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) #生成一个简单的 从 0 到 seq_len-1 的递增序列 [0, 1, 2, ..., seq_len-1]
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #将其复制到 batch 维度上，每个样本得到完全相同的位置编码 [0, 1, 2, ..., seq_len-1]

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device) # 初始全部为 1（计算 loss）
    loss_mask[target == pad_token] = 0.0  # mask paddings # padding 位置 → 不计算 loss
    loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts  # prompt 位置 → 不计算 loss ，IGNORE_INDEX就代表系统提示和用户prompt

    return loss_mask, position_ids


def get_mask_start_and_end_idx(arr):
    """
    Returns a list of tuples holding the start and end index in arr of the non-zeros contiguuous
    sub arrays.

    For instance, if arr = [0, 1, 0, 0, 1, 1]
    get_mask_start_and_end_idx(arr) = [(1, 1), (4, 5)]
    such that arr[1:1+1] = [1] and arr[4:5+1] = [1, 1]
    """
    mask = (arr != 0)

    mask_int = mask.int()

    diff = mask_int[1:] - mask_int[:-1]
    start_indices = (diff == 1).nonzero(as_tuple=False).flatten() + 1
    end_indices = (diff == -1).nonzero(as_tuple=False).flatten()
    if len(mask)==0: return []
    if mask[0]:
        start_indices = torch.cat((torch.tensor([0], device=arr.device), start_indices))
    if mask[-1]:
        end_indices = torch.cat((end_indices, torch.tensor([len(arr) - 1], device=arr.device)))
    sequences = list(zip(start_indices.tolist(), end_indices.tolist()))
    return sequences


def scaled_loss_func(loss_mask, output_tensor):
    """
    Scaled loss function

    Scale the loss for each conversation turn using the formula:

    1 / sum_j[ sqrt(length(loss_turn_j)) ] * sum_i[ sum(loss_turn_i) / sqrt(length(loss_turn_i)) ]

    Where we use the loss mask to infer the start / end of the conversation turns.
    """
    args = get_args()
    losses = output_tensor.float()

    loss_list = []
    num_valid_labels_list = []
    for idx in range(losses.shape[0]):
        loss_this_sample = losses[idx]
        turn_start_end_list = get_mask_start_and_end_idx(loss_mask[idx])
        for turn_start, turn_end in turn_start_end_list:
            # compute loss for each turn
            loss_this_turn = loss_this_sample[turn_start:turn_end+1].sum()
            assert (1 - loss_mask)[idx][turn_start:turn_end+1].sum() < 1.0
            num_valid_labels_this_turn = turn_end - turn_start + 1
            loss_this_turn = loss_this_turn / num_valid_labels_this_turn
            loss_list.append(loss_this_turn)
            # append num of valid labels for each turn
            num_valid_labels_list.append(num_valid_labels_this_turn)
    base_num = sum([math.sqrt(each) for each in num_valid_labels_list])
    for idx in range(len(loss_list)):
        # normalize loss for each turn
        loss_list[idx] = loss_list[idx] * math.sqrt(num_valid_labels_list[idx]) / base_num

    # Some ranks may not get loss tokens due to Context Parallel Sharding
    if len(loss_list) > 0:
        total_loss = torch.stack(loss_list).sum()
        total_tokens = torch.ones_like(total_loss)
    elif len(loss_list) == 0 and args.context_parallel_size > 1:
        total_tokens = loss_mask.sum()
        total_loss = torch.sum(losses.view(-1) * loss_mask)
    else:
        raise RuntimeError("loss_list for loss scaling per conversation unexpectedly got empty list")

    num_tokens = total_tokens.clone().detach().to(torch.int)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), num_tokens.view(1)])

    return (total_loss, num_tokens, {'lm loss': reporting_loss})


def loss_func(loss_mask, output_tensor):
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.contiguous().view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: LLaVAModel):
    """Forward training step.

    Args:
        data_iterator (torch.utils.data.dataloader): Input data iterator
        model: Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    ( #获取micro batch数据
        tokens, #系统提示 + <image> + 用户prompt + 回答
        labels, #回答labels，对于系统提示 + <image> + 用户prompt进行了mask，只🈶回答数据，而且左移了一个token
        loss_mask, #loss mask，系统提示 + <image> + 用户prompt这些token的loss_mask是0.0，只有回答部分是1.0可以计算loss
        attention_mask, #none，modulespec指定
        position_ids, #位置编码
        images, #图像tensor数据
        num_image_tiles, #每个样本的图像tile数量
        packed_seq_params,
    ) = get_batch(data_iterator, model.module.module.image_token_index, model.module.module.img_seq_len) #两层 .module 是 DDP 和 Float16Module 的包装。实际拿到的是 LLaVAModel
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(
        images,
        tokens,
        position_ids,
        attention_mask,
        labels,
        loss_mask,
        num_image_tiles=num_image_tiles,
        packed_seq_params=packed_seq_params,
    ) #执行model的forward。如果不是最后一个pp stage，那output_tensor就是中间activation，如果是最后一个stage，output_tensor就是最终的loss
    args = get_args()
    if args.use_loss_scaling:
        loss_function = partial(scaled_loss_func, loss_mask)
    else:
        loss_function = partial(loss_func, loss_mask) #拿 loss_mask 做加权平均算最终 loss 值——模型已经帮你把每个token的交叉熵loss算完了，loss_func 只需要做 sum(loss * mask) / sum(mask) 的加权平均（不计算被mask的那些token）。

    return output_tensor, loss_function #返回output_tensor（中间激活值/token loss），还有计算最终loss的func


def llava_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    # With no separate encoder pipeline stages (epp=0), the decoder starts at rank 0
    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[0], last_rank]


def llava_position_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    # With no separate encoder pipeline stages (epp=0), the decoder starts at rank 0
    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[0]]


def run_online_eval(model):
    """Run an evaluation benchmark during training."""
    args = get_args()

    # Online evaluation config is not defined. Do nothing.
    if not args.online_evaluation_config:
        return []

    from config import EvaluationConfig
    # Import the common evaluation functions
    from run_text_generation import get_evaluation_configs, run_evaluation_loop

    # Use the common config loading function
    configs = get_evaluation_configs(config_path=args.online_evaluation_config)

    # The inference code assumes the first rank is the leader.
    # Tensorboard writer is on the last rank.
    # We must write to a storage space that all ranks see.
    output_dir = os.path.join(args.save, "online_eval")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the common evaluation loop
    scores = run_evaluation_loop(model[0].module, configs, output_dir_override=output_dir, print_output=False)

    return [scores]


def write_eval_to_tensorboard(data, iteration, writer, walltime=None):
    """Write evaluation data to Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration, walltime=walltime)


def write_online_eval_to_tensorboard(data, iteration, writer, walltime=None):
    """Write online evaluation data to Tensorboard."""
    import shutil
    args = get_args()

    # Define source and destination directories
    source_dir = os.path.join(args.save, "online_eval")
    destination_dir = os.path.join(args.save, f"online_eval_{iteration}")
    if os.path.exists(source_dir):
        print("Moving online eval data from", source_dir, "to", destination_dir)

        # Move the directory (back up the generation)
        shutil.move(source_dir, destination_dir)

    write_eval_to_tensorboard(data, iteration, writer, walltime)


if __name__ == "__main__":

    train_valid_test_dataloaders_provider.is_distributed = True

    # import debugpy
    # local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # try:#使用异常处理适配多进程代码，这样只有一个进程会监听5670端口
    #     # 仅主进程开启调试监听
    #     if local_rank == 0:
    #         debugpy.listen(("localhost", 5670))
    #         print("Waiting for debugger attach")
    #         debugpy.wait_for_client()#强制等待vscode调试点击
    #     else:
    #         # 非0号进程：永久阻塞，卡死在这里，不执行任何后续代码
    #         while True: pass
    # except Exception as e:
    #     pass

    args = parse_and_validate_args(
        extra_args_provider=add_multimodal_extra_args, #添加用户自定义参数的函数
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}, #针对那些没有default的参数进行兜底
    )
    full_config = pretrain_cfg_container_from_args(args) #获取完整配置

    pretrain(
        full_config,
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
        non_loss_data_func=run_online_eval,
    )
