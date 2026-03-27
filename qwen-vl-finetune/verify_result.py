import torch
import json
import os
from omnidrive.modeling_omnidrive import OmniDriveVLA
from transformers import AutoProcessor
from omnidrive.data.dataset import OmniDriveDataset
from torch.utils.data import DataLoader

# 1. 加载最终模型
ckpt = "checkpoints/omnidrive-stage2-e2e"
print(f"Loading best model from {ckpt}...")
model = OmniDriveVLA.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

# 2. 验证权重状态
print(f"Heatmap Proj Std: {model.heatmap_proj.weight.std().item():.6f}")
print(f"Action Head Norm Mean: {model.action_head.input_norm_action.weight.mean().item():.6f}")

# 3. 跑一个样本看看
print("Running inference on one sample...")
# 这里简单读取验证数据的第一条
ds = OmniDriveDataset("/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/expanded_train_no_null.json", processor, class type("Args", (), {"stage": "stage2"})())
item = ds[0]

# 构造 Batch
input_ids = item['input_ids'].unsqueeze(0).cuda()
pixel_values = item['pixel_values'].unsqueeze(0).to(torch.bfloat16).cuda()
image_grid_thw = item['image_grid_thw'].unsqueeze(0).cuda()

with torch.no_grad():
    # 只需要预测 Action
    model.eval()
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        stage="stage2"
    )
    
    pred_traj = outputs['pred_traj'].float().cpu().numpy().reshape(-1, 2)
    gt_traj = item['future_traj'].float().cpu().numpy().reshape(-1, 2)

    # 打印对比
    print("-" * 30)
    print("Pred Traj (First 3 points):")
    print(pred_traj[:3])
    print("GT Traj (First 3 points):")
    print(gt_traj[:3])
    
    l2 = ((pred_traj - gt_traj)**2).sum(axis=1).mean()**0.5
    print(f"Sample L2 Error: {l2:.4f}")
    
    # 写入结果文件
    with open("final_validation_result.txt", "w") as f:
        f.write(f"Heatmap Proj Std: {model.heatmap_proj.weight.std().item()}\n")
        f.write(f"Sample L2 Error: {l2}\n")
        f.write(f"Pred: {pred_traj.tolist()}\n")
        f.write(f"GT: {gt_traj.tolist()}\n")

print("✅ Verification Done. Results saved to final_validation_result.txt")
