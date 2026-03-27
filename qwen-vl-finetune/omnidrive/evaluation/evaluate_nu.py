# evaluate_nu.py
import torch
import torch.nn as nn
import pickle
import os
import numpy as np
from skimage.draw import polygon

# ==============================================================================
# 1. 核心 Metric 计算类 (完全复刻 metric_stp3.py，去除了 Lightning 依赖)
# ==============================================================================

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)
    return bev_resolution, bev_start_position, bev_dimension

class PlanningMetric:
    def __init__(self, n_future=6):
        # 强制使用 CPU
        self.device = torch.device('cpu')
        
        # 严格照搬 gen_dx_bx 初始化逻辑
        dx, bx, _ = gen_dx_bx([-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0])
        dx, bx = dx[:2], bx[:2]
        self.dx = dx  # 移除 nn.Parameter，直接用 Tensor
        self.bx = bx

        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        self.bev_dimension = self.bev_dimension.numpy()

        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future

        # 状态追踪
        self.curr_obj_box_col = 0
        self.gt_collision = 0
        self.pred_collision = 0
        
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future).to(self.device)
        self.obj_box_col = torch.zeros(self.n_future).to(self.device)
        self.L2 = torch.zeros(self.n_future).to(self.device)
        self.total = 0

    def evaluate_single_coll(self, traj, segmentation, token=None):
        '''
        traj: (n_future, 2) [m]
        segmentation: (n_future, 200, 200)
        '''
        # 1. 构建车辆 Box
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        
        # 2. 坐标转换
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]] # Swap X/Y
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)

        # 3. 轨迹处理
        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs[:,:,[0,1]] = trajs[:,:,[1,0]] # Swap X/Y
        trajs = trajs / self.dx.to(traj.device)
        trajs = trajs.cpu().numpy() + rc 

        r = trajs[:,:,0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1) 

        c = trajs[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        # 4. 碰撞检测
        collision = np.full(n_future, False)
        for t in range(n_future):
            rr_t = r[t]
            cc_t = c[t]
            # 检查像素索引是否在范围内（冗余检查，前面已 clip，但保持源码逻辑）
            I = np.logical_and(
                np.logical_and(rr_t >= 0, rr_t < self.bev_dimension[0]),
                np.logical_and(cc_t >= 0, cc_t < self.bev_dimension[1]),
            )
            # 只有当 segmentation 对应位置有值时才算碰撞
            collision[t] = np.any(segmentation[t, rr_t[I], cc_t[I]].cpu().numpy())
            
        # (可选) 源码中用于统计 gt_collision 次数的逻辑，不影响 metric 计算，此处保留结构
        if collision.any() and len(collision)==6 and token:
             collision_time_steps = np.where(collision)[0]
             flag = True
             for ts in collision_time_steps:
                 if 0 in segmentation[ts].cpu().numpy(): # 简单的检查
                     if flag:
                         self.gt_collision += 1
                         flag = False

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation, token=None):
        '''
        STP3 核心碰撞逻辑
        '''
        B, n_future, _ = trajs.shape
        
        # 🚨 STP3 特有的轨迹 X 轴翻转
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            # 1. 检查 GT 是否碰撞
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], token=token)

            # (以下是点碰撞逻辑，metric_stp3.py 中包含，这里为了完整性保留)
            xx, yy = trajs[i,:,0], trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()
            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll)) # 排除 GT 碰撞的步
            ti = torch.arange(n_future, device=trajs.device)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            # 2. 检查预测 Box 碰撞
            m2 = torch.logical_not(gt_box_coll) # 只有 GT 没撞的时间步才有效
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i], token=None)
            
            # 3. 累加逻辑 (使用索引操作，同 metric_stp3.py)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask=None):
        if gt_trajs_mask is None:
            gt_trajs_mask = torch.ones_like(gt_trajs)
        #
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1))

    def update(self, trajs, gt_trajs, segmentation, token=None, gt_trajs_mask=None):
        assert trajs.shape == gt_trajs.shape
        
        # 转 CPU 计算
        trajs = trajs.cpu()
        gt_trajs = gt_trajs.cpu()
        segmentation = segmentation.cpu()
        if gt_trajs_mask is not None:
            gt_trajs_mask = gt_trajs_mask.cpu()

        # 1. 计算 L2 (注意：STP3 不在这里翻转坐标，UniAD 才翻)
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        
        # 2. 计算碰撞
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], segmentation, token=token)

        # 3. 更新累加器
        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total += len(trajs)
        self.curr_obj_box_col = obj_box_coll_sum # 保持源码一致

    def compute(self):
        if self.total == 0:
            return {'obj_col': 0.0, 'obj_box_col': 0.0, 'L2': 0.0}
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2' : self.L2 / self.total
        }


# ==============================================================================
# 2. 评测器类 (NuScenesEvaluator)
# ==============================================================================

class NuScenesEvaluator:
    def __init__(self, gt_folder="/home/hjadmin/OmniDrive-VLA/data/data/metrics"):
        print(f"📦 Loading NuScenes GT from {gt_folder}...")
        
        with open(os.path.join(gt_folder, 'gt_traj.pkl'), 'rb') as f:
            self.gt_trajs = pickle.load(f)
            
        with open(os.path.join(gt_folder, 'gt_traj_mask.pkl'), 'rb') as f:
            mask_data = pickle.load(f)
            self.gt_masks = dict(mask_data) if isinstance(mask_data, list) else mask_data

        with open(os.path.join(gt_folder, 'stp3_gt_seg.pkl'), 'rb') as f:
            self.gt_occ_map = pickle.load(f)
            # STP3 的地图预处理：翻转 X 和 Y
            # 注意：源码是在 compute 循环里翻转还是 init 里翻转？
            # 源码 evaluation.py 是在 init 阶段加载时就翻转了！这里我们也照做。
            print("🔄 Flipping GT Occupancy Maps (STP3 Standard)...")
            for token in self.gt_occ_map.keys():
                if not isinstance(self.gt_occ_map[token], torch.Tensor):
                    self.gt_occ_map[token] = torch.tensor(self.gt_occ_map[token])
                #
                self.gt_occ_map[token] = torch.flip(self.gt_occ_map[token], [-1])
                self.gt_occ_map[token] = torch.flip(self.gt_occ_map[token], [-2])
            
        self.metric_engine = PlanningMetric(n_future=6)

    def compute(self, results_dict):
        self.metric_engine.reset()
        count = 0
        
        for token, pred_traj in results_dict.items():
            if token not in self.gt_trajs: continue
            
            # --- 1. 获取并处理 GT Trajectory ---
            raw_traj = self.gt_trajs[token]
            gt_traj = torch.tensor(raw_traj).float().cpu() if not isinstance(raw_traj, torch.Tensor) else raw_traj.float().cpu()
            gt_traj = gt_traj.squeeze(0) 

            # --- 2. 获取并处理 GT Mask ---
            raw_mask = self.gt_masks[token]
            gt_mask = torch.tensor(raw_mask).float().cpu() if not isinstance(raw_mask, torch.Tensor) else raw_mask.float().cpu()
            gt_mask = gt_mask.squeeze(0)

            # --- 3. 获取 Occupancy (无需再次翻转，init里已翻转) ---
            occ = self.gt_occ_map[token].cpu()
            if occ.dim() == 4 and occ.shape[0] == 1:
                occ = occ.squeeze(0)
            
            # --- 4. 获取 Prediction ---
            pred_traj = pred_traj.detach().cpu().float().reshape(6, 2)

            # =========== 🚨 核心修复：复刻 evaluation.py 的切片逻辑 🚨 ===========
            # 源码逻辑：如果维度是奇数 (包含 t=0)，则切片取后面
            # 即使你的数据是 6 (偶数)，加上这个判断也是安全的，且能对齐逻辑
            
            # 注意：pred_traj 已经是我们 reshape 成 (6,2) 的，这里假设它是 pure future
            # 如果你的 pred_traj 包含当前帧，也需要处理。这里保持你的 (6,2) 假设。
            if pred_traj.shape[0] % 2 != 0: pred_traj = pred_traj[1:]
            
            if occ.shape[0] % 2 != 0: occ = occ[1:]
            if gt_traj.shape[0] % 2 != 0: gt_traj = gt_traj[1:]
            if gt_mask.shape[0] % 2 != 0: gt_mask = gt_mask[1:]
            
            # 确保 mask 维度以便广播
            if gt_mask.dim() == 1: gt_mask = gt_mask.unsqueeze(-1)

            # =========== 5. 更新 Metric (Batch=1) ===========
            self.metric_engine.update(
                pred_traj.unsqueeze(0),      
                gt_traj.unsqueeze(0),        
                occ.unsqueeze(0),            
                token=token, 
                gt_trajs_mask=gt_mask.unsqueeze(0)
            )
            count += 1

        print(f"✅ Computed metrics for {count} samples.")
        scores = self.metric_engine.compute()

        # =========== 6. 指标汇总 (复刻 evaluation.py) ===========
        # 1s: mean of first 2 steps (0.5, 1.0)
        ade_1s = scores["L2"][:2].mean().item()
        ade_2s = scores["L2"][:4].mean().item()
        ade_3s = scores["L2"][:6].mean().item()

        col_1s = scores["obj_box_col"][:2].mean().item()
        col_2s = scores["obj_box_col"][:4].mean().item()
        col_3s = scores["obj_box_col"][:6].mean().item()

        avg_l2 = (ade_1s + ade_2s + ade_3s) / 3.0
        avg_col = (col_1s + col_2s + col_3s) / 3.0

        return {
            "val_avg_L2": avg_l2,
            "val_avg_Collision": avg_col,
            "val_L2_1s": ade_1s,
            "val_L2_2s": ade_2s,
            "val_L2_3s": ade_3s,
            "val_Col_1s": col_1s, # Added
            "val_Col_2s": col_2s, # Added
            "val_Col_3s": col_3s
        }