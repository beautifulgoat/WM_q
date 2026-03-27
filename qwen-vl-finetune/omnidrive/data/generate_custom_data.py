import os
import json
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# 特殊处理：定义一个浮点数子类，用于控制 JSON 序列化时的显示格式
class PreciseFloat(float):
    def __repr__(self):
        return f"{self:.2f}"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate custom format dataset.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataroot", type=str, default="/home/hjadmin/OmniDrive-VLA/nuscenes")
    parser.add_argument("--info_path", type=str, default="/home/hjadmin/OmniDrive-VLA/data/data/cached_nuscenes_info.pkl")
    parser.add_argument("--split_path", type=str, default="/home/hjadmin/OmniDrive-VLA/FSDrive/create_data/full_split.json")
    parser.add_argument("--output", type=str, default="train_custom.json")
    return parser.parse_args()

def get_command_text(cmd_vec):
    if cmd_vec[0] > 0: return "turn right"
    elif cmd_vec[1] > 0: return "turn left"
    else: return "go forward"

def get_images_for_sample(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    return {cam: nusc.get('sample_data', sample['data'][cam])['filename'] for cam in cams}

def format_traj_to_precise_num(traj_array):
    """
    既要数字类型，又要强制补零显示。
    使用 PreciseFloat 对象包装每一个坐标。
    """
    return [[PreciseFloat(round(float(coord), 2)) for coord in point] for point in traj_array]

def calculate_ego_status_vad_style(info):
    """
    【修正版】严格按照 VAD/BEVFormer 的 can_bus 定义读取。
    NuScenes-VAD CanBus (18 dims) 映射:
    [0,1,2]: pos (x,y,z)
    [3,4,5,6]: orientation (quaternion)
    [7,8,9]: acceleration (a_x, a_y, a_z)  <-- VAD 用 [7]
    [10,11,12]: rotation_rate (w_x, w_y, w_z) <-- Yaw Rate 是 Z轴旋转, 即 [12]
    [13]: velocity (v)
    [14,15]: patch_angle (vector)
    [16,17]: ... (reserved/others)
    """
    can_bus = info['can_bus']
    
    # 1. 速度 (Velocity) - Index 13
    velocity = float(can_bus[13])
    
    # 2. 加速度 (Acceleration) - Index 7 (纵向加速度)
    acceleration = float(can_bus[7])
    
    # 3. 横摆角速度 (Yaw Rate) - Index 12 (Z轴旋转率)
    # 之前写错成 16 了，16 通常是空的或角度
    yaw_rate = float(can_bus[12])
    
    return {
        "velocity": round(velocity, 3),
        "acceleration": round(acceleration, 3),
        "yaw_rate": round(yaw_rate, 3)
        }

def main():
    args = parse_args()
    print(f"Loading info from {args.info_path}")
    with open(args.info_path, 'rb') as f: data_info = pickle.load(f)
    with open(args.split_path, 'r') as f: splits = json.load(f)
    tokens = splits[args.split]
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    output_data = []

    print("Generating data with Corrected VAD Ego Status (Indices: 13, 7, 12)...")
    for token in tqdm(tokens):
        try:
            s_curr = nusc.get('sample', token)
            # 确保序列连贯性
            if not s_curr['prev']: continue
            s_p1 = nusc.get('sample', s_curr['prev'])
            if not s_p1['prev']: continue
            if not s_curr['next']: continue
            
            t_p1 = s_curr['prev']
            t_p2 = s_p1['prev']
            t_n1 = s_curr['next']
            
            if any(t not in data_info for t in [token, t_p1, t_p2]): continue

            info_curr = data_info[token] 

            past_raw = info_curr['gt_ego_his_trajs'][:4] 
            fut_raw = info_curr['gt_ego_fut_trajs'][1:7]

            entry = {
                "sample_token": token,
                "images": {
                    "past_2": get_images_for_sample(nusc, t_p2),
                    "past_1": get_images_for_sample(nusc, t_p1),
                    "current": get_images_for_sample(nusc, token),
                    "next_1": get_images_for_sample(nusc, t_n1)
                },
                "navigation": {
                    "past_2_cmd": get_command_text(data_info[t_p2]['gt_ego_fut_cmd']),
                    "past_1_cmd": get_command_text(data_info[t_p1]['gt_ego_fut_cmd']),
                    "current_cmd": get_command_text(info_curr['gt_ego_fut_cmd'])
                },
                # 调用修正后的函数
                "ego_status": calculate_ego_status_vad_style(info_curr),
                "trajectories": {
                    "past": format_traj_to_precise_num(past_raw),
                    "future": format_traj_to_precise_num(fut_raw)
                }
            }
            output_data.append(entry)
        except Exception as e:
            continue

    print(f"Saving {len(output_data)} samples to {args.output}...")
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    main()