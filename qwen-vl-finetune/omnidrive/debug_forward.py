import torch
import os
import sys
import gc
import numpy as np
import torch.nn as nn
from transformers import AutoProcessor
from dataclasses import dataclass

# 确保能找到本地模块
sys.path.append(os.getcwd())

from omnidrive.modeling_omnidrive import OmniDriveVLA
from omnidrive.data.dataset import OmniDriveDataset
from omnidrive.data.collator import OmniDataCollator

@dataclass
class DataArguments:
    data_path: str
    stage: str = "stage2" 

def check_weight_stats(model, layer_name):
    """辅助函数：检查指定层的权重统计"""
    try:
        layer = dict(model.named_modules())[layer_name]
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            print(f"   🔎 [{layer_name}] Weight Stats: Min={w.min().item():.4e}, Max={w.max().item():.4e}, Mean={w.mean().item():.4e}")
            if w.abs().max() > 10.0:
                print(f"      ⚠️ 警告: 权重数值异常巨大！")
        else:
            print(f"   ⚪ [{layer_name}] has no weight attribute.")
    except KeyError:
        print(f"   ❌ [{layer_name}] not found in model.")

def debug_omni_drive():
    print("=" * 60)
    print("🚀 开始 OmniDrive-VLA 深度调试 (Final Fix)")
    print("=" * 60)

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct" 
    data_json = "/home/hjadmin/OmniDrive-VLA/final_expanded_dataset/expanded_train.json"
    ACTION_QUERY_TOKEN = "<|action_query|>"
    
    # 1. 加载 Processor
    print(f"\n[1/6] 初始化模型与处理器...")
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*256, max_pixels=1024*1024, trust_remote_code=True)
    processor.tokenizer.add_special_tokens({'additional_special_tokens': [ACTION_QUERY_TOKEN]})
    processor.image_processor.vision_token_rate = 1.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载模型
    print(f"   正在加载模型 (from_pretrained)...")
    model = OmniDriveVLA.from_pretrained(
        model_path,
        torch_dtype=torch.float32, 
        trust_remote_code=True
    ).to(device)
    
    model.resize_token_embeddings(len(processor.tokenizer))
    query_id = processor.tokenizer.convert_tokens_to_ids(ACTION_QUERY_TOKEN)
    model.action_query_token_id = query_id

    # =========================================================================
    # 🔥🔥🔥 核心修复环节：检测并清洗“幽灵权重” 🔥🔥🔥
    # =========================================================================
    print("\n🧐 [诊断] 检查 Action Head 权重状态 (加载后):")
    check_weight_stats(model, "action_head.vision_proj")
    check_weight_stats(model, "action_head.model.fc1")

    print("\n🧹 [修复] 正在强制重置 Action Head 所有权重...")
    
    def force_init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m, mean=0.0, std=0.02)

    # 这里的 apply 会递归地重置 action_head 下的所有层
    model.action_head.apply(force_init_weights)
    
    # 别忘了重置 Query Embeddings，它也可能被污染
    nn.init.trunc_normal_(model.action_query_embeddings, std=0.02)

    print("✅ [验证] 重置后的权重状态:")
    check_weight_stats(model, "action_head.vision_proj")
    check_weight_stats(model, "action_head.model.fc1")
    # =========================================================================

    model.train() 
    
    # 3. 准备数据
    print("\n[3/6] 准备 Stage 2 数据...")
    data_args = DataArguments(data_path=data_json, stage="stage2")
    dataset = OmniDriveDataset(data_json, processor, data_args)
    collator = OmniDataCollator(processor)
    
    # 找一个包含 Query 的样本
    sample = dataset[20] # 假设 index 20 有效
    if (sample['input_ids'] == query_id).sum() == 0:
        print("⚠️ 当前样本不包含 Action Query Token，尝试搜索...")
        for i in range(len(dataset)):
            if (dataset[i]['input_ids'] == query_id).sum() > 0:
                sample = dataset[i]
                print(f"   > 找到有效样本 Index: {i}")
                break

    batch = collator([sample])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # 4. 执行 Forward
    print("\n▶️ 执行 Stage 2 Forward (With Clean Weights)...")
    outputs = model(
        input_ids=batch['input_ids'],
        pixel_values=batch['pixel_values'],
        image_grid_thw=batch['image_grid_thw'],
        labels=batch['labels'],
        future_traj=batch['future_traj'],
        ego_status=batch['ego_status'],
        target_images=batch['target_images'],
        stage="stage2"
    )
    
    loss = outputs['loss']
    pred_traj = outputs['pred_traj']
    
    print(f"✅ Stage 2 Forward 完成 | Loss: {loss.item():.4f}")
    print(f"   Loss Components: {outputs['loss_dict']}")
    print(f"   Pred Traj First Row: {pred_traj[0].detach().cpu().numpy()}")

    if torch.isnan(loss):
        print("❌ Loss 依然是 NaN! 请检查 Dataset 输入是否包含 NaN。")
    else:
        print("🎉 成功解决 NaN 问题！")

if __name__ == "__main__":
    debug_omni_drive()