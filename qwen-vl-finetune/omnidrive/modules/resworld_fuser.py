# omnidrive/modules/resworld_fuser.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenlearner import TokenLearner, TokenFuser

def warp_bev_features(feat_src, pose_dst, pose_src, pc_range=[-51.2, -51.2, 51.2, 51.2]):
    """保持不变：安全的 FP32 坐标对齐"""
    if pose_dst is None or pose_src is None:
        return feat_src 

    B, C, H, W = feat_src.shape
    device, dtype = feat_src.device, feat_src.dtype

    pose_src_inv = torch.inverse(pose_src.float())
    T_rel = torch.bmm(pose_src_inv, pose_dst.float()) 

    affine_matrix = torch.zeros((B, 2, 3), device=device, dtype=torch.float32)
    range_x, range_y = pc_range[2] - pc_range[0], pc_range[3] - pc_range[1]

    for b in range(B):
        R, T = T_rel[b, :2, :2], T_rel[b, :2, 3]
        affine_matrix[b, :2, :2] = R
        affine_matrix[b, 0, 2] = T[0] / (range_x / 2.0)
        affine_matrix[b, 1, 2] = T[1] / (range_y / 2.0)

    grid = F.affine_grid(affine_matrix, feat_src.size(), align_corners=False)
    warped_feat = F.grid_sample(feat_src.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return warped_feat.to(dtype)


class ResWorldFutureFeatureExtractor(nn.Module):
    def __init__(self, embed_dims=128, num_scenes=16,residual_fusion="single",residual_alpha=0.5,):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_scenes = num_scenes
        self.residual_fusion = residual_fusion
        self.residual_alpha = float(residual_alpha)


        # 1. Fused BEV 的卷积融合器
        self.bev_fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dims * 3, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dims, 200, 200))
        nn.init.normal_(self.pos_embed, std=0.02)

        # 2. 🚨 唯一共享的 TokenLearner (1套参数)
        self.res_tokenlearner = TokenLearner(num_scenes, embed_dims * 2) 
        
        # 3. 🚨 自注意力机制 (TransformerEncoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims, nhead=8, dim_feedforward=embed_dims*2, batch_first=True
        )
        self.res_latent_decoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. 唯一共享的 TokenFuser
        self.tokenfuser = TokenFuser(num_scenes, embed_dims)

    def set_residual_fusion(self, mode="single", alpha=None):
        if mode not in {"single", "alpha"}:
            raise ValueError(f"Unsupported residual_fusion mode: {mode}")
        self.residual_fusion = mode
        if alpha is not None:
            self.residual_alpha = float(alpha)

    def _get_pos_embed(self, B, H, W, device, dtype):
        pos = self.pos_embed.to(device=device, dtype=dtype)
        pos = F.interpolate(pos, size=(H, W), mode="bilinear", align_corners=False)
        pos = pos.expand(B, -1, -1, -1).reshape(B, self.embed_dims, H * W).permute(0, 2, 1)
        return pos.contiguous()

    def forward(self, feat_t0, feat_t1, feat_t2, pose_t0=None, pose_t1=None, pose_t2=None):
        # --- 设备与精度同步防御 ---
        active_device = feat_t0.device
        if active_device.type == 'cpu' and torch.cuda.is_available():
            active_device = torch.device('cuda', torch.cuda.current_device())
            feat_t0 = feat_t0.to(active_device)
        
        active_dtype = feat_t0.dtype
        self.to(device=active_device, dtype=active_dtype)
        
        feat_t1 = feat_t1.to(device=active_device, dtype=active_dtype)
        feat_t2 = feat_t2.to(device=active_device, dtype=active_dtype)
        
        if pose_t0 is not None: pose_t0 = pose_t0.to(active_device)
        if pose_t1 is not None: pose_t1 = pose_t1.to(active_device)
        if pose_t2 is not None: pose_t2 = pose_t2.to(active_device)

        B, C, H, W = feat_t0.shape
        N = H * W

        # --- 第一步：对齐坐标 ---
        aligned_feat_t1 = warp_bev_features(feat_t1, pose_t0, pose_t1)
        aligned_feat_t2 = warp_bev_features(feat_t2, pose_t0, pose_t2)

        # --- 第二步：生成 Fused BEV Feature (底层图) ---
        multi_frame_cat = torch.cat([feat_t0, aligned_feat_t1, aligned_feat_t2], dim=1) 
        fused_bev_feat = self.bev_fusion_conv(multi_frame_cat)

        # --- 第三步：绝对对齐原论文！利用共享 TokenLearner 提取 ---
        # 展平单帧特征
        flat_fused = fused_bev_feat.view(B, C, N).permute(0, 2, 1).contiguous()
        flat_t0 = feat_t0.view(B, C, N).permute(0, 2, 1).contiguous()
        flat_t1 = aligned_feat_t1.view(B, C, N).permute(0, 2, 1).contiguous()
        flat_t2 = aligned_feat_t2.view(B, C, N).permute(0, 2, 1).contiguous()
       
        # 扩展位置编码并拼接
        pos = self._get_pos_embed(B, H, W, active_device, active_dtype)
        fused_queries = torch.cat([flat_fused, pos], dim=-1) 

        _, shared_selected_masks = self.res_tokenlearner(fused_queries)  # [B, S, HW]
        shared_selected_masks = shared_selected_masks.to(dtype=flat_fused.dtype)

        # 从 Batch 维度拆分回 3 帧独立的 Token
        tokens_t0 = torch.einsum('bsn,bnc->bsc', shared_selected_masks, flat_t0)
        tokens_t1 = torch.einsum('bsn,bnc->bsc', shared_selected_masks, flat_t1)
        tokens_t2 = torch.einsum('bsn,bnc->bsc', shared_selected_masks, flat_t2)

        # --- 第四步：时序残差相减 ---
        res_01 = tokens_t0 - tokens_t1
        refined_res_01 = self.res_latent_decoder(res_01)

        if self.residual_fusion == "alpha":
            res_12 = tokens_t1 - tokens_t2
            refined_res_12 = self.res_latent_decoder(res_12)
            final_residual = refined_res_01 + self.residual_alpha * refined_res_12
        else:
            final_residual = refined_res_01

        # --- 第六步：融回 Fused BEV 生成 Future BEV Feature ---
        fused_flat_final = self.tokenfuser(final_residual, flat_fused) + flat_fused
        future_bev_feat = fused_flat_final.permute(0, 2, 1).view(B, C, H, W)

        return future_bev_feat