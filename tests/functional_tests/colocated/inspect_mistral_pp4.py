"""Task 7.5a 产物核对：Mistral TP=1 / PP=4 的 mcore ckpt 是否符合共置形态的预期。

不是"转换没报错就算过"——这里逐 stage 读回文件并断言六件事（任一不符立即抛）：
  1. 顶层只有一个 ``model`` 键（saver 每次只存一个 local 模型，len(model)==1 走
     checkpointing.py:1001 的 "model" 分支）；
  2. 每个 stage 恰 ``EXPECTED_LAYERS_PER_STAGE`` 层，且层号是**本 stage 局部**的 0..7；
  3. ``embedding.*`` 只在 stage 0；``output_layer.*`` 与 ``decoder.final_layernorm.*``
     只在最后一个 stage；中间 stage 两者都没有；
  4. 参数名里不含 TE 融合命名（``layer_norm_weight`` / ``layer_norm_bias``）——本项目
     全链路 local 实现，出现即说明 saver 侧用了 transformer_engine；
  5. 全部浮点张量都是 bf16；
  6. 打印 ``checkpoint_version`` 与 ``args`` 里的并行度，供人工核对（版本 < 2.0 会让
     load 侧走带 assert len(model)==1 的遗留路径，共置两 chunk 必炸）。

用法：python colocated/inspect_mistral_pp4.py [ckpt_dir]
"""

import os
import sys

import torch

DEFAULT_CHECKPOINT_DIR = "/home/zn/zn_data/model/llava_colocated_pp4/mistral_pp4"
EXPECTED_PIPELINE_PARALLEL_SIZE = 4
EXPECTED_LAYERS_PER_STAGE = 8
TENSOR_ENGINE_FUSED_SUFFIXES = ("layer_norm_weight", "layer_norm_bias")


def _layer_indices(parameter_names):
    """从 ``decoder.layers.{i}.*`` 里抽出所有层号（本 stage 的局部下标）。"""
    prefix = "decoder.layers."
    indices = set()
    for name in parameter_names:
        if name.startswith(prefix):
            indices.add(int(name[len(prefix) :].split(".")[0]))
    return indices


def inspect(checkpoint_dir):
    iteration_dir = os.path.join(checkpoint_dir, "iter_0000001")
    assert os.path.isdir(iteration_dir), f"没有 {iteration_dir}"

    for pipeline_rank in range(EXPECTED_PIPELINE_PARALLEL_SIZE):
        path = os.path.join(
            iteration_dir, f"mp_rank_00_{pipeline_rank:03d}", "model_optim_rng.pt"
        )
        assert os.path.isfile(path), f"缺 stage {pipeline_rank} 的文件：{path}"
        state_dict = torch.load(path, weights_only=False, map_location="cpu")

        model_keys = [key for key in state_dict if key.startswith("model")]
        assert model_keys == ["model"], f"stage {pipeline_rank} 的模型键应只有 'model'，实际 {model_keys}"
        parameters = state_dict["model"]
        names = list(parameters)

        indices = _layer_indices(names)
        expected_indices = set(range(EXPECTED_LAYERS_PER_STAGE))
        assert indices == expected_indices, (
            f"stage {pipeline_rank} 的层号应是本地 {sorted(expected_indices)}，实际 {sorted(indices)}"
        )

        has_embedding = any(name.startswith("embedding.") for name in names)
        has_output_layer = any(name.startswith("output_layer.") for name in names)
        has_final_layernorm = any(name.startswith("decoder.final_layernorm.") for name in names)
        assert has_embedding == (pipeline_rank == 0), (
            f"stage {pipeline_rank} 的 embedding 存在性错了：{has_embedding}"
        )
        is_last_stage = pipeline_rank == EXPECTED_PIPELINE_PARALLEL_SIZE - 1
        assert has_output_layer == is_last_stage, (
            f"stage {pipeline_rank} 的 output_layer 存在性错了：{has_output_layer}"
        )
        assert has_final_layernorm == is_last_stage, (
            f"stage {pipeline_rank} 的 final_layernorm 存在性错了：{has_final_layernorm}"
        )

        fused = [
            name for name in names if name.endswith(TENSOR_ENGINE_FUSED_SUFFIXES)
        ]
        assert not fused, f"stage {pipeline_rank} 出现 TE 融合命名（saver 用了 TE？）：{fused[:5]}"

        dtypes = {
            value.dtype
            for value in parameters.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
        assert dtypes == {torch.bfloat16}, f"stage {pipeline_rank} 的浮点 dtype 应全是 bf16，实际 {dtypes}"

        checkpoint_args = state_dict.get("args")
        print(
            f"stage {pipeline_rank}: {len(names)} 个键、层号 {sorted(indices)}、"
            f"embedding={has_embedding}、output_layer={has_output_layer}、dtype={dtypes}"
        )
        print(
            f"  checkpoint_version={state_dict.get('checkpoint_version')} "
            f"iteration={state_dict.get('iteration')} "
            f"tp={getattr(checkpoint_args, 'tensor_model_parallel_size', None)} "
            f"pp={getattr(checkpoint_args, 'pipeline_model_parallel_size', None)} "
            f"num_layers={getattr(checkpoint_args, 'num_layers', None)} "
            f"padded_vocab_size={getattr(checkpoint_args, 'padded_vocab_size', None)} "
            f"transformer_impl={getattr(checkpoint_args, 'transformer_impl', None)}"
        )
        del state_dict

    print("全部 stage 核对通过")


if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT_DIR)
