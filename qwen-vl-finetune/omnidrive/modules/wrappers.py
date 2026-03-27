import torch
import torch.nn as nn
import torch.nn.functional as F
from .action_heads import MLPResNet 
from .denoiser_sd import RossStableDiffusionXOmni
from diffusers import AutoencoderKL
from einops import rearrange
# from omnidrive.pure_sparseocc.pure_frontend import PureSparseOccFrontend
from accelerate.utils import set_module_tensor_to_device


class OmniWorldModel(nn.Module):
    def __init__(self, sd_path, condition_dim):
        super().__init__()
        # 1. 加载 VAE (冻结)
        self.vae = AutoencoderKL.from_pretrained(
            sd_path, 
            subfolder="vae",
            use_safetensors=True
        ).eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 2. 加载 UNet (ROSS 路径)
        # z_channel 设为 condition_dim * 2 外部做了 Concat(Vision, Traj)
        self.unet = RossStableDiffusionXOmni(
            unet_path=sd_path,
            z_channel=condition_dim, 
            mlp_depth=2,
            n_patches=3456 
        )
        
    def get_loss_map(self, target_images, wm_condition,action_query_embeds=None):
        """
        target_images: [B, 3, 512, 1344]
        wm_condition: [B, 4096, H, W] (拼接后的联合条件)
        """
        # 1. VAE 编码目标图像
        with torch.no_grad():
            latents = self.vae.encode(target_images).latent_dist.sample()
            latents = latents * 0.18215 

        # 2. 调用 UNet。因为 wm_condition 已经是拼接好的，
        # 所以直接传给 z，并将 z_a 设为 None
        return self.unet(z=wm_condition, target=latents, z_a=None,encoder_hidden_states=action_query_embeds)

    def forward(self, target_images, wm_visual_cond,action_query_embeds=None):
        # 构造拼接条件
        # traj_feat_map = traj_embed.unsqueeze(-1).unsqueeze(-1).expand_as(wm_visual_cond)
        # wm_condition = torch.cat([wm_visual_cond, traj_feat_map], dim=1)
        return self.get_loss_map(target_images, wm_visual_cond,action_query_embeds)



# class SparseOccWrapper(nn.Module):
#     """
#     Sparse Occupancy 视觉特征提取旁路封装类。
#     接收 8 帧多视图图像及相机位姿，提取出 600 个 3D 空间特征 Token，用于大模型的输入。
#     """
#     def __init__(self, ckpt_path=None, frozen=True,hidden_dim=3584):
#         super().__init__()
        
#         print("🚀 [SparseOccWrapper] 正在初始化纯 PyTorch 3D 视觉前端...")
#         # 1. 实例化我们脱壳后的特征提取器
#         self.sparse_occ_encoder = PureSparseOccFrontend()
#         self.frozen = frozen
        
#         if self.frozen:
#             for name, module in self.sparse_occ_encoder.named_modules():
#                 # 找到所有的 Batch Norm 层
#                 if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
#                     module.eval() # 设为评估模式
#                     module.track_running_stats = False # 物理切断统计追踪
#                     module.affine = True # 保持仿射变换(使用加载进来的 weight 和 bias)
                    
#                     # 确保不计算梯度
#                     if module.weight is not None:
#                         module.weight.requires_grad = False
#                     if module.bias is not None:
#                         module.bias.requires_grad = False
                        
#             # 对整个模型进行一次基础的梯度冻结（保险起见）
#             for param in self.sparse_occ_encoder.parameters():
#                 param.requires_grad = False
        
#         occ_dim = 1024
#         self.position_encoder = nn.Sequential(
#             nn.Linear(336, occ_dim),
#             nn.LayerNorm(occ_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(occ_dim, occ_dim),
#             nn.LayerNorm(occ_dim),
#             nn.ReLU(inplace=True),
#         )

#         self.projector = nn.Linear(occ_dim, hidden_dim)
#         self.can_bus_embed = nn.Sequential(
#             nn.Linear(40, occ_dim),
#             nn.ReLU(),
#             nn.Linear(occ_dim, hidden_dim)
#         )

#         self.future_decoder = OccFutureDecoder(hidden_dim=hidden_dim, num_classes=17)
#         # 2. 如果提供了 checkpoint，则加载它

#         self._initialize_alignment_layers()

#         # if ckpt_path is not None:
#         #     self._load_and_map_weights(ckpt_path)

#         # 3. 参数冻结机制（作为大模型的纯特征提取旁路，前期通常冻结）
#         if frozen:
#             self.sparse_occ_encoder.eval()
#             for p in self.sparse_occ_encoder.parameters(): p.requires_grad = False

    
#     def _initialize_alignment_layers(self):
#         print("✨ [SparseOccWrapper] 正在初始化对齐层与预测头...")
        
#         # 1. 初始化位置编码器 (MLP)
#         for m in self.position_encoder.modules():
#             if isinstance(m, nn.Linear):
#                 # 因为使用了 ReLU，推荐使用 Kaiming 初始化
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0)


#         for m in self.can_bus_embed.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None: nn.init.constant_(m.bias, 0)
#         # 2. 初始化 Projector (对齐到 LLM)
#         # 这是一个关键的桥梁，使用 Xavier 均匀分布保持方差稳定
#         nn.init.xavier_uniform_(self.projector.weight)
#         if self.projector.bias is not None:
#             nn.init.constant_(self.projector.bias, 0)

#         # 3. Future Decoder 的初始化已经在其类内部执行了
#         # 但我们可以在这里做最后的检查
#         if hasattr(self.future_decoder, 'init_weights'):
#             # 如果没在 __init__ 里调，这里强制调一次
#             pass

#     def forward_encoder(self, multiview_images, lidar2img, lidar2ego, can_bus,command):
#         """
#         修正后的接口，将离散的矩阵封装为 PureSparseOccFrontend 需要的格式
#         """
#         B = multiview_images.shape[0]
#         dtype = next(self.projector.parameters()).dtype
        
#         # 构造内部需要的 img_metas 格式
#         img_metas = [{
#         # 1. 【关键修复】将 [8, 6, 4, 4] 展平为 [48, 4, 4]，否则 Transformer 内部广播会崩溃
#         'lidar2img': lidar2img[b].reshape(-1, 4, 4).to(torch.float32).cpu().numpy(),
        
#         # 2. 【核心修复】linalg.inv 强制转换到 float32，避免 BFloat16 报错
#         # 且逆矩阵 lidar2ego -> ego2lidar 是原始 Transformer 要求的标准输入
#         'ego2lidar': torch.inverse(lidar2ego[b].to(torch.float32)).cpu().numpy(),
        
#         'img_shape': [(256, 704, 3)] * 6,
#         'num_fu_frame_real': 6
#         } for b in range(B)]

#         if hasattr(self, 'frozen') and self.frozen:
#             self.sparse_occ_encoder.eval()
#         # 1. 提取基础特征和 3D 坐标
#         # 确保你的 PureSparseOccFrontend.forward 返回的是整个 dict (包含 all_refine_pts)
#         outputs = self.sparse_occ_encoder(multiview_images, img_metas, can_bus)
        
#         # 【核心修正 3】原汁原味的 query_features_identity 必须留存
#         query_features_identity = outputs['query_features']     # [B, 600, 1024]
#         query_positions = outputs['all_refine_pts'][-1]         # [B, 600, 32/112, 3]
        
#         # 2. 空间位置编码与对齐
#         pos_embed = self.position_encoder(query_positions.flatten(-2).to(dtype))
#         query_features = query_features_identity + pos_embed
#         occ_tokens_aligned = self.projector(query_features)
        
#         # 3. 维度映射到 LLM 空间 (Alignment

#         can_bus_40 = torch.cat([can_bus, command], dim=-1).to(dtype) 
#         # 直接送入网络，不再需要后面的 .to(dtype)
#         can_bus_token = self.can_bus_embed(can_bus_40).unsqueeze(1)
        
#         # 4. 组装最终送给大模型的 Tokens (1个意图Token + 600个OCC Token)
#         # 注意：你在构建 LLM Prompt 的时候，应该预留 601 个空间！
#         final_llm_tokens = torch.cat([can_bus_token, occ_tokens_aligned], dim=1) # [B, 601, hidden_dim]
        
#         # print("[OCC DEBUG] identity_feat:", query_features_identity.shape,
#         # "mean=", query_features_identity.float().mean().item(),
#         # "std=", query_features_identity.float().std().item())

#         # print("[OCC DEBUG] llm_tokens:", final_llm_tokens.shape,
#         #     "mean=", final_llm_tokens.float().mean().item(),
#         #     "std=", final_llm_tokens.float().std().item())

#         # centers = query_positions.float().mean(dim=2)
#         # print("[OCC DEBUG] query_centers:", centers.shape,
#         #     "mean=", centers.mean().item(),
#         #     "std=", centers.std().item(),
#         #     "min=", centers.min().item(),
#         #     "max=", centers.max().item())
        
#         # if final_llm_tokens.shape[0] >= 2:
#         #     print("[OCC DEBUG] batch0-vs-batch1 token mean abs diff =",
#         #         (final_llm_tokens[0].float() - final_llm_tokens[1].float()).abs().mean().item())
#         # 返回对齐后的 Token 和 用于初始化的坐标（Ego系）
#         # query_positions.mean(-2) 得到 [B, 600, 3] 的中心点坐标
#         return final_llm_tokens, query_positions, query_features_identity
    
#     def forward_decoder(self, final_occ_hs, query_features_identity, query_positions, canbus_fu):
#         """
#         【核心修正 4】使用 Dataset 真实提取的 canbus_fu，并接收 LLM 隐状态及原始特征
#         Args:
#             final_occ_hs: [B, 600, hidden_dim] (LLM 输出的对应位置的 hidden_states)
#             query_features_identity: [B, 600, 1024] (Encoder 未加位置编码的原生特征)
#             query_positions: [B, 600, 112, 3] (Encoder 输出的 3D 坐标)
#             canbus_fu: [B, 6, 13] (Dataset 传过来的真实的未来动作)
#         """
#         return self.future_decoder(
#             query_features_identity=query_features_identity,
#             final_occ_hs=final_occ_hs,
#             query_positions=query_positions,
#             canbus_fu=canbus_fu
#         )

 


