import os
import sys
import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

# ==============================================================================
# 1. 环境路径修复
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

local_utils_path = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-utils/src"
if local_utils_path not in sys.path:
    sys.path.insert(0, local_utils_path)

from omnidrive.evaluation.evaluate_nu import NuScenesEvaluator
from omnidrive.modeling_omnidrive import OmniDriveVLA
from omnidrive.data.dataset import OmniDriveDataset
from omnidrive.data.collator import OmniDataCollator

# ==============================================================================
# 2. 验证主逻辑
# ==============================================================================

class DataArgs:
    def __init__(self, data_path, val_data_path, stage="stage2"):
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.stage = stage

def run_full_validation(ckpt_path, val_json_path, gt_folder, batch_size=8):
    print(f"\n🚀 [Init] Loading Model from: {ckpt_path}")
    
    # --- Tokenizer 修复 ---
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True, fix_mistral_regex=True)
    processor = AutoProcessor.from_pretrained(ckpt_path)
    processor.tokenizer = tokenizer

    # --- 模型加载 ---
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = OmniDriveVLA.from_pretrained(ckpt_path, torch_dtype=dtype)
    model.cuda().eval()

    print("\n🔍 [Weight Verification] Checking Action Head weights...")
    for name, param in model.named_parameters():
        if "action_head" in name and "weight" in name and param.dim() > 1:
            print(f"   ✅ {name} | Mean: {param.data.mean().item():.6f} | Std: {param.data.std().item():.6f}")
            break

    # --- 数据集 ---
    print(f"\n📂 [Init] Loading Validation Data from: {val_json_path}")
    data_args = DataArgs(data_path=val_json_path, val_data_path=val_json_path, stage="stage2")
    val_dataset = OmniDriveDataset(data_path=val_json_path, processor=processor, data_args=data_args)
    collate_fn = OmniDataCollator(processor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    print(f"⚡ [Run] Starting inference on {len(val_dataset)} samples...")
    results_dict = {}
    total_action_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Inferencing"):
            # 数据上移
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            pixel_values = batch['pixel_values'].cuda().to(dtype)
            image_grid_thw = batch.get('image_grid_thw', None)
            if image_grid_thw is not None: image_grid_thw = image_grid_thw.cuda()
            ego_status = batch.get('ego_status', None)
            if ego_status is not None: ego_status = ego_status.cuda().to(dtype)
            
            # [关键修改] 传入 GT 轨迹以计算 Loss (用于验证是否匹配训练日志)
            future_traj = batch.get('future_traj', None)
            if future_traj is not None: future_traj = future_traj.cuda().to(dtype)

            sample_tokens = batch['sample_token']

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                ego_status=ego_status,
                future_traj=future_traj, # 传入 GT
                stage="stage2"
            )

            # 收集 Loss
            if 'loss_dict' in outputs and 'loss_action' in outputs['loss_dict']:
                total_action_loss += outputs['loss_dict']['loss_action'].item()
                total_batches += 1

            # 收集预测结果
            pred_trajs = outputs.get('pred_traj')
            batch_size_curr = input_ids.shape[0]
            for i in range(batch_size_curr):
                token = sample_tokens[i]
                traj = pred_trajs[i].detach().cpu()
                if traj.dim() == 1 and traj.shape[0] == 12:
                    traj = traj.view(6, 2)
                results_dict[token] = traj

    # --- 打印 Loss 对比 ---
    avg_loss = total_action_loss / total_batches if total_batches > 0 else 0.0
    print(f"\n📉 [Sanity Check] Avg Action Loss (Computed): {avg_loss:.4f} (Should be close to ~0.27)")

    # --- 运行 STP-3 评估 ---
    print(f"\n📊 [Eval] Computing STP-3 Metrics for {len(results_dict)} samples...")
    evaluator = NuScenesEvaluator(gt_folder=gt_folder)
    metrics = evaluator.compute(results_dict)

    print("\n" + "="*40)
    print("🏆 OmniDrive-VLA Final Results")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/checkpoints/omnidrive-stage2-e2e")
    parser.add_argument("--data", type=str, default="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/val_custom.json") # 默认跑全量
    parser.add_argument("--gt", type=str, default="/home/hjadmin/OmniDrive-VLA/data/data/metrics")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"❌ Checkpoint not found: {args.ckpt}")
    else:
        run_full_validation(args.ckpt, args.data, args.gt)