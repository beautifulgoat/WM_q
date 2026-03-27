import json
import torch
import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .simple_bev_nuscenes import SimpleBEVNuScenesBuilder

# --- 核心修复 1: 确保优先加载本地源码库 ---
local_utils_path = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-utils/src"
if local_utils_path not in sys.path:
    sys.path.insert(0, local_utils_path)

from qwen_vl_utils import process_vision_info

class OmniDriveDataset(Dataset):
    def __init__(self, data_path, processor, data_args):
        self.data_path = data_path
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.data_args = data_args

        self.image_root = getattr(data_args, "image_root", "/home/hjadmin/OmniDrive-VLA/nuscenes")
        self.nuscenes_version = getattr(data_args, "nuscenes_version", "v1.0-trainval")

        self.stage = getattr(data_args, 'stage', 'stage2')
        # 2x3 布局定义
        self.front_cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        self.back_cameras = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

        self.action_query_token = "<|action_query|>"
        # self.query_id = self.processor.tokenizer.convert_tokens_to_ids(self.action_query_token)
        self.image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        self.single_h, self.single_w = 144, 256
        # self.pano_h, self.pano_w = 512, 1344
        self.use_simple_bev = getattr(data_args, "use_simple_bev", True)
        if self.use_simple_bev:
            self.simple_bev_builder = SimpleBEVNuScenesBuilder(
                dataroot=self.image_root,
                version=self.nuscenes_version,
                final_dim=(448, 800),   # 官方 res_scale=2
                camera_order=[
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK_LEFT",
                    "CAM_BACK",
                    "CAM_BACK_RIGHT",
                ],
                verbose=False,
            )
        else:
            self.simple_bev_builder = None

    def _get_2x3_pano(self, item, timestep):
        """逻辑保持原样：将该时刻的 6 张图处理并拼成一张 2x3 的大图"""
        def process_group(cameras):
            row_imgs = []
            for cam in cameras:
                img_path = os.path.join(self.image_root, item["images"][timestep][cam])
                img = Image.open(img_path).convert("RGB")
                img = img.resize((self.single_w, self.single_h))
                row_imgs.append(np.array(img))
            return np.concatenate(row_imgs, axis=1)

        top_row = process_group(self.front_cameras)
        bottom_row = process_group(self.back_cameras)
        stitched_np = np.concatenate([top_row, bottom_row], axis=0)
        return Image.fromarray(stitched_np)

    def _format_past_trajectory(self, traj_list):
        """将过去轨迹坐标列表转换为字符串格式，辅助 LLM 理解运动趋势"""
        # traj_list shape: [[x, y], [x, y]...]
        # 取最近的几个点，保留 2 位小数
        traj_str = ", ".join([f"({p[0]:.2f}, {p[1]:.2f})" for p in traj_list])
        return f"[{traj_str}]"
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
    
        img_past_2 = self._get_2x3_pano(item, "past_2")
        img_past_1 = self._get_2x3_pano(item, "past_1")
        img_current = self._get_2x3_pano(item, "current")

        cmd_p2 = item['navigation']['past_2_cmd']
        cmd_p1 = item['navigation']['past_1_cmd']
        cmd_curr = item['navigation']['current_cmd']

        ego_stat = item['ego_status']
        ego_text = f"Velocity: {ego_stat['velocity']:.2f} m/s, Accel: {ego_stat['acceleration']:.2f} m/s², Yaw Rate: {ego_stat['yaw_rate']:.2f} rad/s"

        # 提取过去轨迹文本 (关键修改：加入 Prompt)
        past_traj_text = self._format_past_trajectory(item['trajectories']['past'])

        # 3. 构造 Messages
        # System Prompt: 设定专家角色，强调安全和轨迹预测
        system_content = (
            "You are an expert autonomous driving agent. "
            "Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0)."
            "Your task is to analyze the multi-view visual history, ego-motion status, and historical trajectory "
            "to predict a safe and precise future trajectory. "
            "Focus on traffic elements, road geometry, and navigation commands."
        )

        # User Prompt: 包含历史观测、状态和指令
        # 注意：不再包含 "question"，直接给出 "Context" 和 "Instruction"
        user_content_list = [
            {"type": "text", "text": f"Context History:\n- T-2: Cmd '{cmd_p2}'\n"},
            {"type": "image", "image": img_past_2}, # Image 1 插入此处
            
            {"type": "text", "text": f"\n- T-1: Cmd '{cmd_p1}'\n"},
            {"type": "image", "image": img_past_1}, # Image 2 插入此处
            
            {"type": "text", "text": f"\n- Current (T): Cmd '{cmd_curr}'\nObservation:\n"},
            {"type": "image", "image": img_current}, # Image 3 插入此处
            
            {"type": "text", "text": f"\n\nEgo Status: {ego_text}\nPast Trajectory: {past_traj_text}\n\nInstruction: Predict the future trajectory."}
        ]

        num_queries = 64
        assistant_content = self.action_query_token * num_queries

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content_list}, # 传入列表
            {"role": "assistant", "content": assistant_content}
        ]

        # --- 3. 提取视觉特征 (使用 process_vision_info) ---
        image_inputs, video_inputs = process_vision_info(messages)
        # --- 4. 构造 Prompt (区分 Stage 1 和 Stage 2) ---

        
        text_prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=4096, # 适当减小 max_length，因为去掉了长文本回复，省显存
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        
        vision_token_id = 151655 # image
        video_token_id = 151656  # video (不该出现)
        
        if (input_ids == video_token_id).sum() > 0:
            print("WARNING: Found Video Token ID in Image Mode, converting to Image Token ID.")
            input_ids[input_ids == video_token_id] = vision_token_id

        # --- 7. 后续处理 (Label, Trajectory, WM Target) ---
        labels = input_ids.clone()
        

        # --- 6. 其他数据准备 ---
        future_traj = torch.tensor(item["trajectories"]["future"], dtype=torch.float16).flatten()
        status = item["ego_status"]
        ego_vector = torch.tensor([status["velocity"], status["acceleration"], status["yaw_rate"]], dtype=torch.float16)



        # future_traj_norm = torch.nan_to_num(future_traj_norm, 0.0)
        # ego_vector_norm = torch.nan_to_num(ego_vector_norm, 0.0)


        wm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        target_image = wm_transform(self._get_2x3_pano(item, "next_1"))

        # if torch.isnan(future_traj_norm).any():
        #     print(f"⚠️ Warning: Found NaN in normalized future trajectory at idx {idx}!")
        #     # 简单修复：设为 0
        #     future_traj_norm = torch.nan_to_num(future_traj_norm, 0.0)

        # 确保 action_query_token 被包含在 input_ids 里
        if len(input_ids) < 100: 
             print(f"⚠️ Warning: Input IDs seems too short ({len(input_ids)}). Check prompt construction.")

        res = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "labels": labels,
            "future_traj": future_traj,
            "ego_status": ego_vector,
            "target_images": target_image,
            "sample_token": item['sample_token'],
        }
        # ---------------------------------------------------------
        # B. Simple-BEV sidecar 数据链：新增，不影响原主链
        # ---------------------------------------------------------
        if self.use_simple_bev:
            bev_triplet = self.simple_bev_builder.build_triplet(item["images"])
            res.update(bev_triplet)

        return res