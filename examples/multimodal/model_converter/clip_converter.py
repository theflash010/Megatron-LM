# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import torch

import clip


def convert(download_root, output_path, tensor_parallel_size, use_te):
    device = "cuda"

    model, _ = clip.load("ViT-L/14@336px", device=device, download_root=download_root) #获取CLIP模型，并且已经加载了TorchScript中的数据

    state_dict = model.state_dict() #获取CLIP模型的状态字典，当前是PyTorch格式，接下来需要转换为Megatron-LM格式
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 64 #单个attention head的hidden size
    hidden_dim = 1024
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels #lower bound 当前 head 的起始索引
        ub = (i + 1) * kv_channels #upper bound 当前 head 的结束索引
        indices.append(torch.arange(lb, ub, dtype=torch.int)) #获取Q的张量索引  #torch.arange(start, end) 是 PyTorch 的序列生成函数，生成从 start 到 end-1 的整数序列
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int)) #获取K的张量索引
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int)) #获取V的张量索引

    indices = torch.cat(indices)

    for name, tensor in state_dict.items():
        # Skip text model.
        if "visual" not in name:
            continue

        # Skip final layers not used in our model.
        if name == "visual.proj" or "ln_post" in name:
            continue

        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None #确定TP切分的维度

        if "class_embedding" in name: #开始名称转换和分片
            new_name = "class_token"
            # Our model uses class token that is expanded to input dimensions already.
            new_tensor = new_tensor.expand(1, 1, -1)
        elif "positional_embedding" in name:
            new_name = "position_embeddings.weight"
        elif "conv1" in name:
            new_name = "conv1.weight"
        elif "ln_pre.weight" in name:
            new_name = "ln_pre.weight"
        elif "ln_pre.bias" in name:
            new_name = "ln_pre.bias"
        elif "transformer.resblocks" in name:
            layer_idx = name.split(".")[3]
            base = f"decoder.layers.{layer_idx}"

            if "attn.in_proj_weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = new_tensor[indices] # 用前面构建的索引重排 QKV
                chunk_dim = 0 #这里的dim0是output_dim
            elif "attn.in_proj_bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.out_proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif "attn.out_proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif "ln_1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif "ln_1.bias" in name:
                new_name = f"{base}.input_layernorm.bias"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
            elif "mlp.c_fc.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif "mlp.c_fc.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif "mlp.c_proj.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif "mlp.c_proj.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            elif "ln_2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
            elif "ln_2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"

        assert new_name != "", f"unexpected layer name {name}"

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone() #将参数权重加入每个TP分片的new_state_dicts中

            # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
            extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2") #TE (Transformer Engine) 会为 Linear 层设置 _extra_state，用于存储 FP8 计算时的缩放因子等中间状态。 转换的权重在 Megatron 中可能用 TE 运行，也可能不用。不设置 _extra_state 会导致 TE 报错或行为异常
            is_extra_state_layer = any([l in new_name for l in extra_state_layers])
            if use_te and is_extra_state_layer:
                layer = new_name.split(".")[-2]
                if layer in extra_state_layers:
                    extra_state_name = (
                        new_name[: new_name.rfind(".") + 1] + "_extra_state"
                    )  # Replace the weight name.
                    new_state_dicts[i]["model"][extra_state_name] = None #兼容TE，随便赋值None

    for i in range(tensor_parallel_size): #把转换为Megatron格式并按TP分片的的参数权重保存到本地
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)


if __name__ == "__main__":
    # import debugpy
    # try:#使用异常处理适配多进程代码，这样只有一个进程会监听5678端口
    #     debugpy.listen(("localhost", 5678))
    #     print("Waiting for debugger attach")
    #     debugpy.wait_for_client()#强制等待vscode调试点击
    # except Exception as e:
    #     pass
    parser = argparse.ArgumentParser(
        description="""
Convert OpenAI CLIP VIT weights to megatron format.


Example usage:
python clip_converter.py --download-root /some/download/folder --output /some/output/folder --tensor-parallel-size 4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--download-root", type=str, required=True, help="Download folder for OpenAI CLIP weights"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")

    args = parser.parse_args()

    convert(args.download_root, args.output, args.tensor_parallel_size, args.use_te)

    print("done.")
