# import torch
# import numpy as np
# from skimage.draw import polygon

# # 🛑 彻底移除 torchmetrics 和 nn.Module，变成普通的 Python 类
# # 这样它就绝对不会尝试去联网同步，彻底根除死锁
# class PlanningMetric:
#     def __init__(self, n_future=6):
#         # NuScenes BEV 参数初始化
#         # 强制使用 CPU，因为你的 evaluate_nu.py 里的输入都已经转到了 CPU
#         self.device = torch.device('cpu') 
        
#         dx = torch.tensor([-50.0, 50.0, 0.5])
#         bx = torch.tensor([-50.0, 50.0, 0.5])
        
#         # 计算 dx, bx
#         self.dx = torch.tensor([row[2] for row in [dx, bx]])
#         self.bx = torch.tensor([row[0] + row[2]/2.0 for row in [dx, bx]])
        
#         # 计算 BEV 维度
#         bev_resolution = torch.tensor([0.5, 0.5, 20.0])
#         bev_start_position = torch.tensor([-50.0 + 0.25, -50.0 + 0.25, -10.0 + 10.0])
#         # x: (-50, 50) -> 100 / 0.5 = 200
#         self.bev_dimension = torch.tensor([200, 200], dtype=torch.long)

#         self.W = 1.85
#         self.H = 4.084
#         self.n_future = n_future

#         # 初始化状态
#         self.reset()

#     def reset(self):
#         self.obj_col = torch.zeros(self.n_future).to(self.device)
#         self.obj_box_col = torch.zeros(self.n_future).to(self.device)
#         self.L2 = torch.zeros(self.n_future).to(self.device)
#         self.total = 0

#     def evaluate_single_coll(self, traj, segmentation, token=None):
#         '''
#         检测单条轨迹碰撞
#         traj: (n_future, 2) [m]
#         segmentation: (n_future, 200, 200)
#         '''
#         # 1. 构建车辆多边形 (Box)
#         pts = np.array([
#             [-self.H / 2. + 0.5, self.W / 2.],
#             [self.H / 2. + 0.5, self.W / 2.],
#             [self.H / 2. + 0.5, -self.W / 2.],
#             [-self.H / 2. + 0.5, -self.W / 2.],
#         ])
        
#         # 转换到像素坐标
#         pts = (pts - self.bx.numpy()) / (self.dx.numpy())
#         pts[:, [0, 1]] = pts[:, [1, 0]]
#         rr, cc = polygon(pts[:,1], pts[:,0])
#         rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)

#         # 2. 轨迹坐标转换
#         n_future, _ = traj.shape
#         trajs = traj.view(n_future, 1, 2)
#         trajs[:,:,[0,1]] = trajs[:,:,[1,0]] # XY 交换
#         trajs = trajs / self.dx.to(traj.device)
        
#         # 将车辆多边形叠加到轨迹点上
#         trajs = trajs.numpy() + rc 

#         r = trajs[:,:,0].astype(np.int32)
#         r = np.clip(r, 0, self.bev_dimension[0].item() - 1) 
#         c = trajs[:,:,1].astype(np.int32)
#         c = np.clip(c, 0, self.bev_dimension[1].item() - 1)

#         # 3. 碰撞检测
#         collision = np.full(n_future, False)
#         for t in range(n_future):
#             rr_t = r[t]
#             cc_t = c[t]
#             # 检查是否有任何车身像素落在障碍物(val>0)上
#             if np.any(segmentation[t, rr_t, cc_t].numpy()):
#                 collision[t] = True

#         return torch.from_numpy(collision).to(device=traj.device)

#     def evaluate_coll(self, trajs, gt_trajs, segmentation, token=None):
#         '''
#         FSDrive 核心碰撞计算逻辑
#         '''
#         B, n_future, _ = trajs.shape
        
#         # 🚨【关键步骤】坐标翻转 [-1, 1] 以对齐 NuScenes 地图方向
#         # 确保 tensor 在 CPU 上进行计算
#         flip_tensor = torch.tensor([-1, 1], device=trajs.device, dtype=trajs.dtype)
#         trajs = trajs * flip_tensor
#         gt_trajs = gt_trajs * flip_tensor

#         obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
#         obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

#         for i in range(B):
#             # 1. 计算 GT 是否本身就碰撞
#             gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], token=token)

#             # 2. 计算预测轨迹的 Box 碰撞
#             box_coll = self.evaluate_single_coll(trajs[i], segmentation[i], token=None)
            
#             # 3. 只有当 GT 没撞，而预测撞了，才算误报 (False Positive)
#             valid_collision = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))
            
#             obj_box_coll_sum += valid_collision.long()

#         return obj_coll_sum, obj_box_coll_sum
   
#     def compute_L2(self, trajs, gt_trajs, gt_trajs_mask=None):
#         if gt_trajs_mask is None:
#             gt_trajs_mask = torch.ones_like(gt_trajs)
            
#         # trajs: (B, 6, 2)
#         # gt_trajs: (B, 6, 2)
#         # gt_trajs_mask: (B, 6, 2)
        
#         # 1. 计算平方差: (B, 6, 2)
#         diff_sq = (trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2
        
#         # 2. 应用 Mask: (B, 6, 2)
#         # 既然 Mask 是 [1, 1]，这里就保留误差
#         masked_diff_sq = diff_sq * gt_trajs_mask
        
#         # 3. 求欧氏距离 (Euclidean Distance)
#         # 先对 xy 求和 (dim=-1)，再开方
#         # out: (B, 6)
#         l2_dist = torch.sqrt(masked_diff_sq.sum(dim=-1))
        
#         return l2_dist

#     def update(self, trajs, gt_trajs, segmentation, token=None, gt_trajs_mask=None):
#         assert trajs.shape == gt_trajs.shape
        
#         # 强制转换为 CPU 以避免任何 GPU 显存或同步问题，因为这是纯数学计算
#         trajs = trajs.cpu()
#         gt_trajs = gt_trajs.cpu()
#         segmentation = segmentation.cpu()
#         if gt_trajs_mask is not None:
#             gt_trajs_mask = gt_trajs_mask.cpu()

#         # 1. 计算 L2
#         L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        
#         # 2. 计算碰撞
#         obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], segmentation, token=token)

#         # 3. 更新状态
#         self.obj_col += obj_coll_sum
#         self.obj_box_col += obj_box_coll_sum
#         self.L2 += L2.sum(dim=0)
#         self.total += len(trajs)

#     def compute(self):
#         # 简单的平均值计算
#         if self.total == 0:
#             return {
#                 'obj_col': 0.0,
#                 'obj_box_col': 0.0,
#                 'L2': 0.0
#             }
            
#         return {
#             'obj_col': self.obj_col / self.total,
#             'obj_box_col': self.obj_box_col / self.total,
#             'L2' : self.L2 / self.total
#         }