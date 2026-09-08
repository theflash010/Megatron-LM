"""Task 7.5c：把 CLIP（ViT，全量副本）与 Mistral（按 PP 切分）合并成共置形态的 ckpt。

共置的 ``model`` 列表是**两个 chunk**——``[encoder_chunk, backbone_chunk]``（training.py 的
``get_colocated_model`` 按 ("encoder", "language_model") 顺序拼接），所以 legacy 格式下
保存/加载用的键是 ``model0`` / ``model1``（save 侧 checkpointing.py:1000-1003，load 侧
:1907-1912 的 ``'model%d' % i``），而不是单模型的 ``model``。这也是本脚本与上游
``examples/multimodal/combine_state_dicts.py`` 的根本区别，另外两处区别是：
  * PP>1 的目录命名是 ``mp_rank_{tp:02d}_{pp:03d}``（checkpointing.py:226-231），
    上游那个脚本只处理 PP=1 的 ``mp_rank_{tp:02d}``；
  * ViT 在共置形态下是**每个 rank 的完整副本**，所以要复制进**每一个** stage 的文件，
    而不是按 TP 切开分给不同文件。

两条刻意的设计（都为了避免静默错误）：
  1. **元数据显式写死，不继承输入文件**。上游脚本是 ``combined = first_input.copy()`` 再
     替换 ``model`` 键（combine_state_dicts.py:34），会把第一个输入的 ``checkpoint_version``
     一起带过来；这里显式设置并断言 ``checkpoint_version >= 2.0``——低于 2.0 时
     ``fix_query_key_value_ordering``（checkpointing.py:1913-1916）会走带
     ``assert len(model) == 1`` 的遗留路径，共置两 chunk 必崩。
  2. **打印每个输出文件里 ViT 部分的哈希**。ViT 在磁盘上有 P 份拷贝，若四份不一致，
     各 DP 副本的 encoder 从第一步就分叉，而梯度归约只同步梯度、不同步参数
     ⇒ 训练不报错、只静默错。四行哈希必须完全相同（运行期的组内校验见 Task 7.7）。

另外写入 ``colocated_chunk_modules`` 这条"下标 ↔ 组件名"的映射（2026-08-30 与用户讨论后
选定方案 A：保持上游扁平的 ``model{i}`` 键、另存身份元数据，而不是自造
``{"encoder": ..., "backbone": ...}`` 的嵌套格式并去改 ``load_checkpoint`` 的分派）。理由：
扁平布局**本来就兼容 VPP**（``model`` 列表是 ``[encoder] + backbone_chunks``，save/load 都按
全局下标枚举），嵌套的唯一真实收益是身份安全，而这条元数据用零格式偏离拿到了同样的
安全性——工具链（checkpoint_inspector、将来的 torch_dist 转换）也仍然认得这些文件。


``vision_projection`` 不在任何输入里（CLIP 预训练权重没有它，LLaVA 第一阶段才训它），
因此训练侧必须带 ``--allow-missing-vision-projection-checkpoint``，并且
``ColocatedViTEncoder`` 需要补上忽略该组键名的 load hook（Task 7.6）。

用法见 colocated/combine_colocated_checkpoints.sh。
"""

import argparse
import hashlib
import os

import torch

_ENCODER_CHUNK_KEY = "model0"
_BACKBONE_CHUNK_KEY = "model1"
_CHUNK_MODULES_KEY = "colocated_chunk_modules"
_ENCODER_PREFIX = "vision_model."
_BACKBONE_PREFIX = "language_model."
_MINIMUM_CHECKPOINT_VERSION = 2.0


def _checkpoint_file(checkpoint_dir, iteration, pipeline_rank=None, tensor_rank=0):
    """legacy 格式的文件路径：PP>1 时目录名带 pipeline rank（checkpointing.py:226-231）。"""
    directory = f"iter_{iteration:07d}"
    if pipeline_rank is None:
        leaf = f"mp_rank_{tensor_rank:02d}"
    else:
        leaf = f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}"
    return os.path.join(checkpoint_dir, directory, leaf, "model_optim_rng.pt")


def _load(path):
    assert os.path.isfile(path), f"缺输入文件：{path}"
    # NOTE: 需要 weights_only=False——legacy ckpt 里的 args 是 Namespace。
    return torch.load(path, weights_only=False, map_location="cpu")


def _tensor_hash(parameters):
    """对参数字典做一次确定性哈希（按键排序），用于跨 stage 比对 ViT 副本是否一致。"""
    digest = hashlib.sha256()
    for name in sorted(parameters):
        value = parameters[name]
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _prefixed(parameters, prefix, dtype=None):
    """给键加前缀（与两个 chunk 的子模块名对齐），可选地统一浮点 dtype。"""
    renamed = {}
    for name, value in parameters.items():
        if (
            dtype is not None
            and isinstance(value, torch.Tensor)
            and value.is_floating_point()
        ):
            value = value.to(dtype)
        renamed[f"{prefix}{name}"] = value
    return renamed


def combine(
    vision_checkpoint_dir,
    language_checkpoint_dir,
    output_dir,
    pipeline_parallel_size,
    iteration,
    vision_dtype,
):
    vision_state_dict = _load(_checkpoint_file(vision_checkpoint_dir, iteration))
    vision_parameters = _prefixed(
        vision_state_dict["model"], _ENCODER_PREFIX, dtype=vision_dtype
    )
    print(f"ViT: {len(vision_parameters)} 个键，dtype 统一为 {vision_dtype}")

    for pipeline_rank in range(pipeline_parallel_size):
        language_state_dict = _load(
            _checkpoint_file(language_checkpoint_dir, iteration, pipeline_rank)
        )
        checkpoint_version = language_state_dict.get("checkpoint_version")
        assert (
            checkpoint_version is not None
            and float(checkpoint_version) >= _MINIMUM_CHECKPOINT_VERSION
        ), (
            f"language ckpt 的 checkpoint_version={checkpoint_version}，低于 "
            f"{_MINIMUM_CHECKPOINT_VERSION} 会让加载侧走带 assert len(model)==1 的遗留路径"
        )

        # 元数据显式构造，不继承任何输入文件；不复制 optimizer / rng_state（本项目冷启动
        # 加载不读它们，留着只会误导）。args 取 language 侧那份——它记录的 pp=4 / num_layers
        # 与训练形态一致；仅在 --use-checkpoint-args 时才会被读到（本项目不开）。
        combined_state_dict = { #这里分成两个chunk，每个chunk的state dict直接赋值为检查点加载出的state dict。由于这里不是统一的LLaVA模型不需要对参数state dict的key额外加vision/language前缀
            _ENCODER_CHUNK_KEY: dict(vision_parameters), #encoder chunk
            _BACKBONE_CHUNK_KEY: _prefixed(language_state_dict["model"], _BACKBONE_PREFIX), #backbone chunk
            # 下标 i ↔ 组件名的显式映射：扁平的 model{i} 是**按位置**的，一旦
            # get_colocated_model 的拼接顺序变了（或 encoder 前面插入了别的 chunk），
            # 加载会静默错位——而且撞上 "if 'model%d' % i not in state_dict: continue"
            # （checkpointing.py:1910-1911）连报错都没有。加载后校验（Task 7.7）拿这条
            # 列表与各 chunk 自己声明的 colocated_module_name 逐位比对，把位置耦合变成
            # 按身份校验（Task 5.7 的原则）。上 VPP 后它自然变成
            # ["encoder", "language_model", "language_model", ...]，校验逻辑不用改。
            _CHUNK_MODULES_KEY: ["encoder", "language_model"],
            "checkpoint_version": float(checkpoint_version),
            "iteration": iteration,
            "args": language_state_dict.get("args"),
            "num_floating_point_operations_so_far": 0,
        }

        output_path = _checkpoint_file(output_dir, iteration, pipeline_rank)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(combined_state_dict, output_path)
        print(
            f"stage {pipeline_rank}: {_ENCODER_CHUNK_KEY}={len(combined_state_dict[_ENCODER_CHUNK_KEY])} 键 "
            f"{_BACKBONE_CHUNK_KEY}={len(combined_state_dict[_BACKBONE_CHUNK_KEY])} 键 "
            f"vit_sha256={_tensor_hash(combined_state_dict[_ENCODER_CHUNK_KEY])[:16]} -> {output_path}"
        )
        del language_state_dict, combined_state_dict

    latest_path = os.path.join(output_dir, "latest_checkpointed_iteration.txt")
    with open(latest_path, "w") as latest_file:
        latest_file.write(f"{iteration}\n")
    print(f"写入 {latest_path}（内容 {iteration}）；上面各 stage 的 vit_sha256 必须完全相同")


def main():
    parser = argparse.ArgumentParser(
        description="Combine a replicated ViT checkpoint and a PP-split language checkpoint "
        "into colocated two-chunk checkpoints (model0 = encoder, model1 = backbone)."
    )
    parser.add_argument("--vision-checkpoint", required=True, help="CLIP mcore ckpt 目录（TP=1）")
    parser.add_argument(
        "--language-checkpoint", required=True, help="Mistral mcore ckpt 目录（TP=1 / PP=P）"
    )
    parser.add_argument("--output", required=True, help="合并产物目录")
    parser.add_argument("--pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument(
        "--vision-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "keep"],
        help="ViT 浮点权重的目标 dtype。CLIP 转换产物是 fp32，而 language 侧是 bf16；"
        "统一成 bf16 省磁盘，数值上等价于加载时的那次 fp32->bf16 舍入。keep 表示不转换。",
    )
    args = parser.parse_args()

    vision_dtype = None if args.vision_dtype == "keep" else getattr(torch, args.vision_dtype)
    combine(
        args.vision_checkpoint,
        args.language_checkpoint,
        args.output,
        args.pipeline_parallel_size,
        args.iteration,
        vision_dtype,
    )


if __name__ == "__main__":
    main()



