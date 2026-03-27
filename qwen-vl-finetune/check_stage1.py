import torch
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from dataclasses import dataclass

# 1. 确保项目路径在 sys.path 中 (根据你的环境调整)
PROJECT_ROOT = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 2. 尝试加载 Liger Kernel (可选优化)
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
    apply_liger_kernel_to_qwen2_5_vl(rope=True, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=True)
    print("⚡ [Liger Kernel] Applied!")
except ImportError:
    pass
except Exception as e:
    print(f"⚠️ Liger Kernel failed: {e}")

from omnidrive.modeling_omnidrive import OmniDriveVLA
from omnidrive.data.dataset import OmniDriveDataset
from transformers import AutoProcessor

# ================= 配置区 =================
# 检查点路径 (请确保该路径下有 config.json, model.safetensors 等)
CHECKPOINT_PATH = "checkpoints/omnidrive-stage1" 
# 验证集 JSON 路径
VAL_DATA_PATH = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/expanded_val_no_null.json"
# NuScenes 图片根目录
IMAGE_ROOT = "/home/hjadmin/OmniDrive-VLA/nuscenes" 
# 输出结果图片名
OUTPUT_IMG_NAME = "stage1_final_comparison4.png"

# 定义简单的参数类替代动态类型创建
@dataclass
class EvalDataArgs:
    stage: str = "stage1"

# ================= 工具函数 =================
def stitch_cameras_from_dict(image_dict, image_root):
    """
    将 Nuscenes 的 6 张相机图片拼成 2x3 的大图
    """
    if image_dict is None:
        return None

    # 布局: 前三后三
    cam_layout = [
        ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
        ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]
    ]
    
    # 单图尺寸 (缩小一点以便拼接显示)
    w, h = 400, 225 
    canvas = Image.new('RGB', (w * 3, h * 2), (0, 0, 0)) # 黑色背景
    
    for row_idx, row_cams in enumerate(cam_layout):
        for col_idx, cam_name in enumerate(row_cams):
            rel_path = image_dict.get(cam_name)
            
            if rel_path:
                full_path = os.path.join(image_root, rel_path)
                try:
                    img = Image.open(full_path).convert('RGB')
                    img = img.resize((w, h))
                    canvas.paste(img, (col_idx * w, row_idx * h))
                except Exception as e:
                    print(f"❌ Load Error ({cam_name}): {e}")
                    # 画个红叉占位
                    draw = ImageDraw.Draw(canvas)
                    rect = [col_idx*w, row_idx*h, (col_idx+1)*w, (row_idx+1)*h]
                    draw.rectangle(rect, outline="red", width=3)
                    draw.text((col_idx*w+10, row_idx*h+10), "MISSING", fill="red")
    
    return canvas

def add_label(img, text, color="white", bg="black"):
    """给图片添加文字标签"""
    if img is None: return None
    # 扩展图片高度加 Label 栏
    new_h = img.height + 40
    labeled_img = Image.new("RGB", (img.width, new_h), bg)
    labeled_img.paste(img, (0, 40))
    
    draw = ImageDraw.Draw(labeled_img)
    # 尝试加载字体，如果失败使用默认
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 5), text, fill=color, font=font)
    return labeled_img

# ================= 主流程 =================
def run_comparison():
    # 1. 加载模型
    print(f"🚀 Loading Model from: {CHECKPOINT_PATH}")
    try:
        model = OmniDriveVLA.from_pretrained(
            CHECKPOINT_PATH, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).cuda()
        model.eval() # 设为评估模式
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    # 2. 加载数据处理工具
    print(f"📂 Loading Processor & Dataset...")
    try:
        processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
        # 实例化参数对象 (修复点)
        data_args = EvalDataArgs(stage="stage1") 
        dataset = OmniDriveDataset(VAL_DATA_PATH, processor, data_args)
    except Exception as e:
        print(f"❌ Dataset/Processor load failed: {e}")
        return
    
    # 3. 搜索有效样本
    print("🔍 Searching for a sample with valid Future GT (next_1 != null)...")
    valid_idx = -1
    START_FROM = 150 # 跳过前 100 个
    
    for i in range(START_FROM, len(dataset)):
        raw_item = dataset.data[i]
        # 检查是否有 next_1 且不为空
        next_imgs = raw_item.get('images', {}).get('next_1')
        if next_imgs is not None and len(next_imgs) > 0:
            valid_idx = i
            print(f"✅ Found valid sample at Index: {i}")
            break
    
    if valid_idx == -1:
        print("❌ Error: Could not find any sample with 'next_1' images in the dataset!")
        return

    # 4. 获取数据并推理
    item_tensor = dataset[valid_idx]
    item_raw = dataset.data[valid_idx]

    print("🔮 Model Reasoning (Generating Next Frame)...")
    try:
        img_pred = model.generate_next_frame(
            input_ids=item_tensor['input_ids'].cuda(), 
            pixel_values=item_tensor['pixel_values'].to(torch.bfloat16).cuda(),
            image_grid_thw=item_tensor['image_grid_thw'].to(dtype=torch.long, device=model.device),
            future_traj=item_tensor['future_traj'].unsqueeze(0).to(torch.bfloat16).cuda(),
            ego_status=item_tensor['ego_status'].unsqueeze(0).to(torch.bfloat16).cuda(),
            guidance_scale=1.0, # 保持 1.0 看最真实效果
            num_inference_steps=50
        )
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return
    
    if img_pred is None:
        print("❌ Prediction returned None (Check model internal logic)")
        return

    # 5. 图片后处理与拼接
    print("🖼️ Processing Images...")
    
    # 获取输入图 (拼接)
    img_input = stitch_cameras_from_dict(item_raw['images']['current'], IMAGE_ROOT)
    img_input = add_label(img_input, "Input: Current Frame (6 Cams)", bg="darkgreen")

    # 获取 GT 图 (拼接)
    img_gt = stitch_cameras_from_dict(item_raw['images']['next_1'], IMAGE_ROOT)
    img_gt = add_label(img_gt, "Ground Truth: Next Frame (6 Cams)", bg="darkgreen")

    # 调整预测图尺寸以匹配 (预测图是 768x288 的单图，需要拉伸显示)
    # 我们将其宽度调整为和 input 一样 (1200)
    target_w = img_input.width
    target_h = int(img_pred.height * (target_w / img_pred.width))
    img_pred_resized = img_pred.resize((target_w, target_h), Image.Resampling.BILINEAR)
    img_pred_resized = add_label(img_pred_resized, "AI Prediction (World Model Output)", bg="darkblue")

    # 6. 最终组合
    print("📊 Composing Final Comparison...")
    # 最终高度 = 输入高 + 预测高 + GT 高
    final_h = img_input.height + img_pred_resized.height + img_gt.height
    final_img = Image.new('RGB', (target_w, final_h), "white")
    
    current_y = 0
    final_img.paste(img_input, (0, current_y))
    current_y += img_input.height
    
    final_img.paste(img_pred_resized, (0, current_y))
    current_y += img_pred_resized.height
    
    final_img.paste(img_gt, (0, current_y))
    
    # 保存
    final_img.save(OUTPUT_IMG_NAME)
    print(f"✨ ALL DONE! Comparison saved to: {OUTPUT_IMG_NAME}")
    print(f"   Structure: Top=Input | Mid=AI Prediction | Bottom=Ground Truth")

if __name__ == "__main__":
    run_comparison()