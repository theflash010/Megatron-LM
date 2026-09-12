# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Colocated training entry helpers: data, forward step, loss (implementation layer).

共置训练入口辅助（对应 ``examples/multimodal/train.py`` 的职责）：数据获取、模型
forward 的分支函数、loss。schedule（``colocated_schedule.py``）只做 orchestration：
- phase ① 循环调 ``colocated_forward_step`` 的 encoder 分支拿回包存 buffer；
- phase ② 调其 backbone 分支（输入已由 schedule 设置好：consumer 的包 / 非首 stage
  的 ``set_input_tensor`` 激活）。

``colocated_forward_step`` 是按注入给 train_step 的唯一 forward_step_func，按
``model[0]`` 的 chunk 类型分支：encoder（phase ①，schedule 传 ``model=[encoder_chunk]``）
/ backbone（phase ②，1F1B 传 ``model=[backbone_chunk]``）。

职责边界（2026-08-12 用户确认）：
- forward step **只做纯前传**：encoder 分支取数 + ``encoder_chunk(images)`` 返回包；
  backbone 分支读模型已设输入调 ``backbone_chunk(...)`` 返回 ``(output, loss_func)``；
- 数据/模型细节（``image_token_index`` / ``img_seq_len`` 等）全在本文件内部读取，
  **不经过 schedule**；schedule 不接收 ``get_batch_fn`` 之类参数。

``colocated_get_batch`` / ``get_ltor_masks_and_position_ids`` / ``loss_func`` 复制自
``train.py``（L36-152 / L155-168 / L244-254），差异：去掉"中间 stage 不取数"守卫
（colocated 里所有 rank 都做 encoder 前传、都需要数据）。
"""

from functools import partial

import os
import sys

import torch

# 与 ``train.py`` 同款：把仓库根目录加入 sys.path，使以脚本方式启动时也能 import megatron。
# Same as train.py: make the repository root importable when launched as a script.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.multimodal import context_parallel
from megatron.core.models.multimodal.colocated_llava_model import (
    ColocatedGPTBackbone,
    ColocatedViTEncoder,
)
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX
from megatron.core.parallel_state import get_colocated_encoder_tensor_model_parallel_group
from megatron.core.pipeline_parallel.colocated_encoder_comm import MergedEncoderBatch
from megatron.core.pipeline_parallel.colocated_schedule import IntraPacket
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_pg_rank,
    nvtx_range_pop,
    nvtx_range_push,
    unwrap_model,
)
from megatron.training import get_args, get_tokenizer, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args


def colocated_encoder_get_batch(data_iterator, image_token_index, img_seq_len):
    """Phase ①: fetch one (possibly merged) batch for the encoder producer (image + pack fields).

    优化 spec Task 1 起 batch 维与"一个 micro batch"解耦：非合并模式下一个 batch 就是
    一个 micro batch（batch_size = micro_batch_size）；合并模式（设计 A）下 dataloader 的
    batch_size 已提到 ``micro_batch_size * num_microbatches / num_producers``，一次
    ``next()`` 取回**本 rank 整个 iteration 的全部 micro batch**。本函数对 batch 维完全
    形状无关：取数、broadcast、labels 左移、pad 检查都按张量形状走，代码不变。

    只取 encoder 前传 + 打包所需：images（前传）、tokens/labels/num_image_tiles（打包给
    消费者）。**不生成 loss_mask/position_ids**（2026-08-13 用户确认：消费者在
    ``colocated_backbone_get_batch`` 里从 labels 的 IGNORE/pad 掩码本地重建，loss_mask
    本就不随包传）；attention_mask 恒 None（modulespec 指定）。与 ``train.py::get_batch``
    一致（复制），差异：去掉"中间 stage 不取数"守卫（colocated 里每个 rank 都是
    producer）、"last stage 不需要 images"分支，以及 loss_mask/position_ids 生成。

    Note: attn_mask_type in layer_specs.py sets the attention mask. Attention mask is None here.

    Returns:
        (tokens, labels, images, num_tiles)
    """
    imgs = None
    tokens = None
    labels = None
    num_tiles = None

    args = get_args()

    # Broadcast data on the ENCODER's own tensor model parallel group: this is the encoder
    # producer's data path, so every rank / group reference here must be the encoder's.
    # 取数与广播都走 **encoder 自己的**张量并行组：这是 encoder producer 的数据通路，取数
    # rank 的判断与广播的源 rank 必须同属这一个组（``is_colocated_dataloader_rank`` 用的也是
    # 它），否则建了 dataloader 的 rank 与广播源不是同一个 rank，广播会挂死。
    nvtx_range_push("get_data")
    encoder_tensor_group = get_colocated_encoder_tensor_model_parallel_group()
    if data_iterator is not None and get_pg_rank(encoder_tensor_group) == 0:  # encoder TP rank 0 取数
        data = next(data_iterator)
    else:
        data = None

    data_text = tensor_parallel.broadcast_data(
        ["tokens"], data, torch.int64, tp_group=encoder_tensor_group
    )["tokens"]
    labels = tensor_parallel.broadcast_data(
        ["labels"], data, torch.int64, tp_group=encoder_tensor_group
    )["labels"]

    imgs = tensor_parallel.broadcast_data(
        ["imgs"], data, torch.float32, tp_group=encoder_tensor_group
    )["imgs"]
    num_tiles = tensor_parallel.broadcast_data(
        ["num_tiles"], data, torch.int32, tp_group=encoder_tensor_group
    )["num_tiles"]

    # No image input (text-only sample) if the dataloader returned a size 1 image.
    if imgs.shape == torch.Size([1, 1]):
        # FSDP can hang with text-only samples. A workaround is to run a valid dummy image
        # through the vision model and then add image embeddings with a zero multiplier.
        if args.use_torch_fsdp2:
            imgs = torch.zeros((1, 3, args.img_h, args.img_w), dtype=torch.float32, device=data_text.device)
            num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)
        else:
            # Similar workaround is not needed without FSDP and we can use an empty image.
            imgs = torch.tensor([], dtype=torch.float32, device=data_text.device)
            num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)

    nvtx_range_pop("get_data")

    tokens_ = data_text.long()

    nvtx_range_push("index tokens")
    text_length = tokens_.shape[1]
    tokens = tokens_[:, :text_length].contiguous()
    labels = labels[:, 1 : text_length + 1].contiguous()  # left-shift the labels by one  # 取 columns [1, 2, ..., text_length]，这里对label进行了偏移（左移一位），真正的label

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    nvtx_range_pop("index tokens")

    # If context parallel is enabled, must shard inputs to CP ranks.
    # 只 pad tokens/labels；position_ids/loss_mask 不存在——消费者在
    # ``colocated_backbone_get_batch`` 里从 padded tokens/labels 重建，SP/CP 的 padding
    # 语义对齐与 packed_seq_params 留 Task 6.3（packed_seq_params 不随包传）。
    # Only tokens/labels are padded; position_ids/loss_mask are rebuilt by the consumer
    # from the padded tokens/labels; SP/CP padding-semantics alignment and
    # packed_seq_params are Task 6.3 (packed_seq_params is not sent in the packet).
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
        tokens, labels = [
            torch.nn.functional.pad(item, (0, mp_padding_needed))
            for item in (tokens, labels)
        ]

    return tokens, labels, imgs, num_tiles


def get_ltor_masks_and_position_ids(input_ids, target, pad_token):
    """Build masks and position id for left to right model (copied from train.py:155)."""
    seq_length = input_ids.shape[1]

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts

    return loss_mask, position_ids


def loss_func(loss_mask, output_tensor):
    """Weighted average of the per-token loss (copied from train.py:244)."""
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.contiguous().view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def colocated_backbone_get_batch(packet, pad_token):
    """Phase ② consumer: transform the forward packet into backbone inputs.

    转换逻辑（2026-08-13 用户确认）：从 ``ForwardPacket`` 读内容字段（image_embeddings/
    tokens/labels/num_image_tiles——labels 含 IGNORE/pad 掩码），**本地重建 loss_mask 与
    position_ids**（``get_ltor_masks_and_position_ids`` 逻辑，与 producer 侧 get_batch 的
    生成完全一致：loss_mask = (labels != pad) & (labels != IGNORE)、position_ids =
    arange）——loss_mask 不随包传；attention_mask 恒 None（LLaVA 约定）。SP/CP 场景的
    padding 语义对齐留 Task 6.3（与 position_ids 现状一致）。

    Returns:
        (image_embeddings, tokens, position_ids, attention_mask, labels, loss_mask,
        num_image_tiles)
    """
    loss_mask, position_ids = get_ltor_masks_and_position_ids(
        packet.tokens, packet.labels, pad_token
    )
    return (
        packet.image_embeddings,
        packet.tokens,
        position_ids,
        None,  # attention_mask
        packet.labels,
        loss_mask,
        packet.num_image_tiles,
    )


def _half_precision_forward_kwargs(chunk):
    """Return ``{"fp32_output": False}`` when the chunk is wrapped in ``Float16Module``.

    共置 encoder 的 pipeline 组只有一个成员 ⇒ ``Float16Module.forward`` 里
    ``is_pp_first_stage`` 与 ``is_pp_last_stage`` **同时**为真（module.py:491-499）：入参被
    转成 bf16（需要），出参又被升回 fp32（不能要）。encoder 的输出要经边界通信发出，
    communicator 的 dtype 是 ``config.pipeline_dtype``（bf16），升回 fp32 会让接收端按
    错误的元素宽度解析扁平缓冲区——这正是 4.x 那条 dtype 断言拦住的情况。
    ``fp32_output=False`` 是上游为此留的开关（module.py:470-473）。
    没有这层包装时（单测里的裸模块）不能传这个关键字，模块 forward 不认识它。
    """
    if isinstance(chunk, Float16Module) or isinstance(getattr(chunk, "module", None), Float16Module):
        return {"fp32_output": False}
    return {}


def _encoder_forward(data_iterator, encoder_chunk):
    """Phase ①: fetch the MERGED batch and run the encoder-only forward -> MergedEncoderBatch.

    优化 spec Task 1（设计 A）：dataloader 的 batch_size 已由 provider 提到合并粒度
    （``micro_batch_size * num_microbatches / num_producers``，colocated_dataloader_provider.py
    的 ``colocated_encoder_merged_batch_size``），因此本函数一次 ``next()`` 取回的是
    **本 rank 整个 iteration 的全部 micro batch**，`encoder_chunk` 也只调**一次**：

    - `colocated_encoder_get_batch` 对 batch 维完全形状无关（broadcast / 左移 labels /
      pad 检查都按张量形状走），代码不变，语义从"一个 micro batch"变为"一个合并批"；
    - 返回 ``MergedEncoderBatch``（裸张量集合，非 ``ForwardPacket``——合并批永不上
      线路，见其 docstring）。``image_embeddings`` 为
      ``[img_seq_len, merged_batch, h_lang]``（seq-first，batch 在 dim=1）。
      **切分回逐 microbatch 的 ForwardPacket 发生在 schedule**
      （``_colocated_encoder_forward``，经 ``ForwardPacket.split_merged_batch``）。
    - ``image_embeddings`` 保留 grad_fn（phase ④ 对**整块**做合并反传）；分离图在
      发送/组装时 detach。返回 ``(batch, None)`` 与 forward_step 契约一致。
    """
    args = get_args()
    image_token_index = getattr(args, "image_token_index", None)
    img_seq_len = getattr(args, "img_seq_len", None)
    # 一次取回合并批（images + 打包字段 tokens/labels/num_image_tiles）；
    # loss_mask/position_ids 不在 producer 侧生成（消费者在 colocated_backbone_get_batch 里重建）。
    tokens, labels, images, num_tiles = colocated_encoder_get_batch(
        data_iterator, image_token_index, img_seq_len
    )
    # 一次前传跑完合并批：[img_seq_len, merged_batch, h_lang]，保留 grad_fn。
    image_embeddings = encoder_chunk(
        images, **_half_precision_forward_kwargs(encoder_chunk)
    )
    return (
        MergedEncoderBatch(
            image_embeddings=image_embeddings,
            tokens=tokens,
            labels=labels,
            num_image_tiles=num_tiles,
        ),
        None,
    )


def _backbone_forward(data_iterator, backbone_chunk, packet=None, intra_packet=None):
    """Phase ②: backbone forward with the input already set by the schedule.

    phase ②（1F1B 内）：输入已由 schedule 设置好——consumer（``pre_process=True``）的包
    经 ``functools.partial`` 绑定到 forward_step_func 后由 schedule 传入（Task 4.3b，
    **不经模型属性**），非首 stage 的激活已 ``set_input_tensor``。组装在模型 forward
    内部（colocated_llava_model.py:655）。返回 ``(output, loss_func)``，与现有
    forward_step 契约一致。

    4.3j 重构：consumer 的展开 labels/loss_mask 写回 ``intra_packet``（输出盒子，
    schedule 在 forward_step 返回后读取做 backbone P2P 伴随发送）；非 consumer 的
    labels/loss_mask 由 schedule 经 ``intra_packet`` 闭包绑定传入（last stage 算
    loss）。模型不再持有交接状态。
    """
    # ``pre_process`` 在被包装的模块上，训练时 chunk 是 DDP(Float16Module(...))，
    # 直接取属性会落在包装类上而取不到。
    # pre_process lives on the wrapped module, not on the DDP / Float16Module wrapper.
    if get_attr_wrapped_model(backbone_chunk, "pre_process"):
        # Consumer (stage 0): assemble the schedule-provided packet and run the backbone.
        # consumer：使用 schedule 传入的包（ForwardPacket，partial 绑定），组装后跑 backbone。
        assert packet is not None, (
            "consumer forward step needs the packet bound by the schedule (Task 4.3b)"
        )
        # 转换：读包 + 本地重建 position_ids/loss_mask（loss_mask 不随包传，从 labels 的
        # IGNORE/pad 掩码重建，与 producer 侧 get_batch 的生成逻辑一致）。
        # Transform: read the packet + rebuild position_ids/loss_mask locally (loss_mask
        # is not sent in the packet; rebuilt from the IGNORE/pad mask of labels).
        (
            image_embeddings,
            tokens,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            num_image_tiles,
        ) = colocated_backbone_get_batch(packet, get_tokenizer().pad)
        # 4.3j：模型 forward 返回 3 元组——labels 是组装展开后的 new_labels（之前没有
        # 出口），现随返回值交还并写回 intra_packet，供 schedule 伴随发送（4.3h/4.3i
        # 定案的 backbone P2P 伴随传输）。
        output, loss_mask, new_labels = backbone_chunk(
            image_embeddings=image_embeddings,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )
        # 4.3j：consumer 的伴随传输是强制的（PP>1 时接收端 _recv_targets 无条件 irecv）——
        # 若 schedule 未绑定 intra_packet，属配置错误，须在源头大声失败而非下游静默死锁。
        assert intra_packet is not None, (
            "consumer forward step needs the intra_packet bound by the schedule (4.3j)"
        )
        intra_packet.labels = new_labels
        intra_packet.loss_mask = loss_mask
    else:
        # Non-first stage: the activation is injected via set_input_tensor; labels/loss_mask
        # come from the schedule's P2P accompaniment (4.3i, 4.3h 定案) bound through the
        # intra_packet (4.3j) — the consumer-assembled new_labels/new_loss_mask travel down
        # the pipeline; the last stage uses them for loss.
        # 非首 stage：激活已 set_input_tensor；labels/loss_mask 来自 schedule 的伴随传输
        #（4.3i：consumer 组装的 new_labels/new_loss_mask 沿流水下行，last stage 算 loss），
        # 4.3j 起经 intra_packet 闭包绑定传入，不再读取模型属性。
        assert intra_packet is not None, (
            "non-consumer forward step needs the intra_packet bound by the schedule (4.3j)"
        )
        output, loss_mask, _ = backbone_chunk(
            labels=intra_packet.labels, loss_mask=intra_packet.loss_mask
        )

    return output, partial(loss_func, loss_mask)


def colocated_forward_step(data_iterator, model, packet=None, intra_packet=None):
    """Colocated forward step: one function, branched by the chunk type in ``model``.

    注入给 train_step 的唯一 forward_step_func（签名 ``(data_iterator, model)``），按
    ``model`` 的 chunk 类型分支：
    - ``model=[encoder_chunk]``（phase ①，schedule 调，**list**）：encoder 分支，返回
      ``(MergedEncoderBatch, None)``。优化 spec Task 1（设计 A）起 schedule **每个
      iteration 只调一次**，返回的是**合并批的裸张量集合**（非 ``ForwardPacket``——
      合并批永不上线路，见 ``MergedEncoderBatch`` 的 docstring），由 schedule 经
      ``ForwardPacket.split_merged_batch`` 拆分回逐 microbatch 的包并打
      ``microbatch_id``；
    - ``model=backbone_chunk``（phase ②，1F1B 调，**已解包的单 chunk**——forward_step
      辅助函数在 schedules.py 内 ``model = model[0]``）：backbone 分支，输入已由 schedule
      设置好，返回 ``(output, loss_func)``。

    ``packet``（可选，仅 phase ② consumer 用）：schedule 用 ``functools.partial`` 绑定
    到本函数后传入（Task 4.3b，packet 走闭包、不经模型属性），backbone 分支的 consumer
    路径使用；encoder 分支忽略。默认 None 保持与 train_step 注入契约
    （``(data_iterator, model)``）兼容。

    ``intra_packet``（可选，4.3j）：schedule 与 ``colocated_forward_step`` 之间的
    内部交接载体（IntraPacket）——consumer 用它做输出盒子（``colocated_forward_step``
    写回展开 labels/loss_mask，schedule 在 forward_step 返回后读取伴随发送）；非
    consumer 用它做输入载体（schedule 伴随 recv 的 labels/loss_mask，last stage 算
    loss）。默认 None 保持契约兼容。
    """
    # 分派看**解包后**的类型，前传仍走最外层包装：训练时 chunk 是
    # ``DDP(Float16Module(module))``，Float16Module 负责输入/输出的 bf16 转换，绕过它会让
    # encoder 吃 fp32 图像、与 checkpoint 的精度口径不符。
    # Dispatch on the unwrapped type but keep calling the outermost wrapper: the training
    # chunk is DDP(Float16Module(module)) and the half-precision cast lives in the wrapper.
    chunk = model[0] if isinstance(model, (list, tuple)) else model
    inner_chunk = unwrap_model(chunk)
    if isinstance(inner_chunk, ColocatedViTEncoder):
        return _encoder_forward(data_iterator, chunk)
    if isinstance(inner_chunk, ColocatedGPTBackbone):
        return _backbone_forward(data_iterator, chunk, packet=packet, intra_packet=intra_packet)
    raise TypeError(
        f"colocated_forward_step expects the chunk to be ColocatedViTEncoder or "
        f"ColocatedGPTBackbone, got {type(inner_chunk)}"
    )


if __name__ == "__main__":
    # 入口与 ``train.py`` 同构，三处替换：dataloader provider 换成共置版（分片域=全 W、
    # 每个 rank 都取数）、forward_step_func 换成 ``colocated_forward_step``、去掉评估相关
    # 的回调（共置不支持评估，arguments.py 的共置校验块已断言 --eval-iters 0）。
    # 模型侧不需要替换：``model_provider`` 自己按 ``colocated_module`` 分派，共置与否由
    # ``--use-colocated-encoder`` 在 setup_model_and_optimizer 里决定
    # （``mpu.is_colocated_encoder_enabled()`` -> ``get_colocated_model``）。
    # The entry mirrors train.py with three substitutions: the colocated dataloader provider,
    # colocated_forward_step, and no evaluation callbacks. The model provider is unchanged - it
    # dispatches on colocated_module, and the colocated path is selected inside
    # setup_model_and_optimizer.
    # 延迟到入口再 import：这几个模块（Energon 数据、examples 侧 model/args、train.py 的
    # embedding rank 辅助函数）只有真正启动训练时才需要，import 它们的代价与副作用不该落在
    # "被 schedule 复用的实现层"上。
    # Imported here rather than at module scope: these are entry-only dependencies.
    from colocated_args import add_colocated_extra_args, validate_colocated_args
    from colocated_dataloader_provider import colocated_train_valid_test_dataloaders_provider
    from megatron.training.training import maybe_start_memory_snapshot_recording
    from model import model_provider
    from train import llava_embedding_ranks, llava_position_embedding_ranks

    colocated_train_valid_test_dataloaders_provider.is_distributed = True

    args = parse_and_validate_args(
        extra_args_provider=add_colocated_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    # encoder 的并行度参数由共置侧自己补默认值与校验：core 的 validate_args 不认识这几个字段。
    # The colocated encoder parallel sizes are defaulted and checked here - core's validate_args
    # does not know these fields.
    validate_colocated_args(args)
    full_config = pretrain_cfg_container_from_args(args)

    # env-gated CUDA memory snapshot: start recording BEFORE pretrain() so that
    # weights / optimizer states / DDP buckets carry allocation stacks. torch.distributed
    # is not initialized yet - the rank filter reads the RANK env var (set by torchrun).
    # 显存快照（环境开关驱动，默认无操作）：记录在 pretrain() 之前开启，权重/优化器/
    # DDP 桶的分配才带调用栈。此刻 torch.distributed 尚未初始化，rank 过滤读 RANK 环境变量。
    maybe_start_memory_snapshot_recording()

    pretrain(
        full_config,
        colocated_train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        colocated_forward_step,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
