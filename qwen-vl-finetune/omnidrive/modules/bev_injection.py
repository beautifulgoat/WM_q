# omnidrive/modules/bev_injection.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def Hidden_states_trans_inv_func(vision_hidden_states, hidden_states, img_mask):
    B = vision_hidden_states.shape[0]
    hidden_states_out = hidden_states.clone()
    for b in range(B):
        batch_mask = img_mask[b]
        hidden_states_out[b][batch_mask] = vision_hidden_states[b]
    return hidden_states_out

def Hidden_states_trans_func(hidden_states, bev_feat, img_mask):
    # bev_feat is [B, C, H, W]
    B, C, H, W = bev_feat.shape
    # Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]
    bev_feat_seq = bev_feat.view(B, C, -1).transpose(1, 2).contiguous()
    bev_feat_seq = bev_feat_seq.to(dtype=hidden_states.dtype, device=hidden_states.device)
    
    vision_hidden_states_list = []
    for b in range(B):
        batch_mask = img_mask[b]
        batch_vision_feat = hidden_states[b][batch_mask]
        vision_hidden_states_list.append(batch_vision_feat)
    
    vision_hidden_states = torch.stack(vision_hidden_states_list, dim=0)
    return vision_hidden_states, bev_feat_seq

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, F1, F2):
        B, N, _ = F1.shape
        N_2 = F2.shape[1]
        Q = self.q_proj(F1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) 
        K = self.k_proj(F2).view(B, N_2, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(F2).view(B, N_2, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(attn_output)
        return self.layernorm(F1 + out) 


class BEVLayerInjector(nn.Module):
    def __init__(self, qwen_model, bev_dim, hidden_dim=3584, num_layers=28, scale=4):
        super().__init__()
        
        # 【关键修复】：绝不能把传入的 qwen_model 挂载到 self 上，否则会导致 PyTorch 树形结构无限循环！
        # self.qwen_model = qwen_model  <-- 删掉这一行

        # 为每一层初始化注入用的 MLP 模块 (还原 VGGDrive 结构)
        self.prompt_tuning_mlp = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, bev_dim // scale),
                    nn.GELU(),
                    nn.Linear(bev_dim // scale, bev_dim // scale),
                ),
                nn.Sequential(
                    nn.Linear(bev_dim, bev_dim // scale),
                    nn.GELU(),
                    nn.Linear(bev_dim // scale, bev_dim // scale),
                ),
                CrossAttentionFusion(dim=bev_dim // scale),
                nn.Sequential(
                    nn.Linear(bev_dim // scale, bev_dim // scale),
                    nn.GELU(),
                    nn.Linear(bev_dim // scale, hidden_dim),
                )
            ])
            for _ in range(num_layers)
        ])
        
        # 全局上下文占位符
        self.current_bev_feat = None
        self.current_img_mask = None

        # 1. 临时获取 layers
        layers = self._get_layers(qwen_model)

        # 2. 直接遍历挂载 Hook
        self._register_hooks(layers)

    def _get_layers(self, qwen_model):
        """动态向下寻找 layers 属性，不绑定到 self"""
        if hasattr(qwen_model, "language_model"):
            if hasattr(qwen_model.language_model, "model") and hasattr(qwen_model.language_model.model, "layers"):
                return qwen_model.language_model.model.layers
            if hasattr(qwen_model.language_model, "layers"):
                return qwen_model.language_model.layers
        
        if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "layers"):
            return qwen_model.model.layers
            
        if hasattr(qwen_model, "layers"):
            return qwen_model.layers
            
        raise AttributeError(f"Cannot find Transformer 'layers' in the model. Please check transformers version.")

    def set_bev_context(self, bev_feat, img_mask):
        """在前向传播前调用，设置当次的 BEV 特征和掩码"""
        self.current_bev_feat = bev_feat
        self.current_img_mask = img_mask

    def clear_bev_context(self):
        """前向完成后清理内存"""
        self.current_bev_feat = None
        self.current_img_mask = None

    def _register_hooks(self, layers):
        # 直接遍历传入的 layers 挂载拦截器，而不是遍历 self 上的属性
        for idx, layer in enumerate(layers):
            layer.register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx):
        def hook(module, args, output):
            # 如果没给 BEV 特征，直接原样返回，不影响正常推理
            if self.current_bev_feat is None or self.current_img_mask is None:
                return output
            
            # Transformer layer 的 output 是一个 tuple，第 0 项是 hidden_states
            hidden_states = output[0]
            
            # --- 注入逻辑开始 (同 VGGDrive) ---
            vision_hidden_states, bev_3D_feat = Hidden_states_trans_func(
                hidden_states, self.current_bev_feat, self.current_img_mask
            )
            
            vision_hs = self.prompt_tuning_mlp[idx][0](vision_hidden_states)
            bev_hs = self.prompt_tuning_mlp[idx][1](bev_3D_feat) 
            
            enhanced_vision_hs = self.prompt_tuning_mlp[idx][2](vision_hs, bev_hs)
            enhanced_vision_hs = vision_hidden_states + self.prompt_tuning_mlp[idx][3](enhanced_vision_hs)
            
            # 拼回混合序列
            new_hidden_states = Hidden_states_trans_inv_func(enhanced_vision_hs, hidden_states, self.current_img_mask)
            # --- 注入逻辑结束 ---

            # 替换旧的 hidden_states，并返回新的 tuple 给下一层
            return (new_hidden_states,) + output[1:]
            
        return hook