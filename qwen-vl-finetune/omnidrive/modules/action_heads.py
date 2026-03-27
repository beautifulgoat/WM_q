"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
# from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
# 1. 轨迹维度: 你的数据是 6 个点，每个点 (x, y)，
ACTION_DIM = 2 
# 2. 动作分块大小: L1RegressionHead 通常一次预测整条轨迹，设为 1
NUM_ACTIONS_CHUNK = 6 
# 3. 本体感知维度: 你的数据是 [velocity, acceleration, yaw_rate]，维度是 3
PROPRIO_DIM = 3 
# 4. 训练时的忽略索引: PyTorch 默认通常是 -100
IGNORE_INDEX = -100 
ACTION_TOKEN_BEGIN_IDX = 0 
STOP_INDEX = 1
NUM_TOKENS = 256


def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


def check_tensor(name, x):
    if x is None:
        return
    if torch.isnan(x).any():
        print(f"❌ [NaN DETECTED] {name} contains NaN!")
        return True
    if torch.isinf(x).any():
        print(f"⚠️ [Inf DETECTED] {name} contains Inf!")
        return True
    # 打印部分统计信息，帮助判断数值范围
    if x.numel() > 0:
        print(f"   [DEBUG] {name}: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
    return False

# def check_tensor_verbose(name, x, block_idx=None):
#     if x is None: return False
#     prefix = f"[Block {block_idx}] " if block_idx is not None else ""
#     if torch.isnan(x).any() or torch.isinf(x).any():
#         print(f"💀 {prefix}!!! {name} CRASHED !!! | Min: {x.min().item():.4f} | Max: {x.max().item():.4f} | Mean: {x.mean().item():.4f}")
#         return True
#     # 仅针对有问题的前几层或报错层打印
#     if block_idx in [0, 1, 2]:
#         print(f"   {prefix}{name:<25} | Range: [{x.min().item():.4f}, {x.max().item():.4f}] | Mean: {x.mean().item():.4f}")
#     return False


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=1024,
        action_dim=ACTION_DIM,
        num_task_tokens=512,
        num_blocks=36,
        use_pro_version=True,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        print(f"[Init] L1RegressionActionHead with num_blocks={num_blocks}")
        self.model = MLPResNet(
            num_blocks=num_blocks, 
            input_dim=hidden_dim*action_dim, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            use_pro_version=use_pro_version
            )
        
        self.vision_proj = nn.Linear(input_dim, hidden_dim)

        self.input_norm_action = nn.LayerNorm(input_dim)
        self.input_norm_task = nn.LayerNorm(input_dim)

        self.apply(self._init_weights)
        print("🔧 Manually initializing gating_factors and other parameters...")
        for name, param in self.named_parameters():
            # 跳过 meta 设备 (防止 Tensor.item() 报错)
            if param.device.type == 'meta':
                continue
                
            if "gating_factor" in name:
                nn.init.zeros_(param) 
            elif "random_perturbations" in name:
                 nn.init.normal_(param, mean=0.0, std=0.02)
            
            # 只有在真实设备上才检查 NaN
            if torch.isnan(param).any():
                print(f"  ⚠️ Found NaN in {name} after init, forcing to Normal(0, 0.01)")
                nn.init.normal_(param, mean=0.0, std=0.01)

        print("✅ [Reset] Action Head weights initialized.")
        print("✅ [Reset] Action Head weights forced to Xavier/Normal initialization.")

    def _init_weights(self, m):
        """显式初始化逻辑"""
        if isinstance(m, nn.Linear):
            # 使用 Xavier Uniform 初始化 Linear 层，保证方差稳定
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm 默认为 1 和 0
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            # 其他 Parameter 比如 learnable embeddings
            nn.init.normal_(m, mean=0.0, std=0.02)

    def predict_action(
            self, 
            actions_hidden_states,
            task_hidden_states,
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        
        # print("\n🔍 --- Action Head Predict Start ---")
        
        # # 1. 检查输入
        if check_tensor("Input: actions_hidden_states", actions_hidden_states): pass
        if check_tensor("Input: task_hidden_states", task_hidden_states): pass
        target_dtype = self.input_norm_action.weight.dtype
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        if actions_hidden_states.dtype != target_dtype:
            actions_hidden_states = actions_hidden_states.to(target_dtype)
        
        if task_hidden_states.dtype != target_dtype:
            task_hidden_states = task_hidden_states.to(target_dtype)

        actions_hidden_states = self.input_norm_action(actions_hidden_states)
        task_hidden_states = self.input_norm_task(task_hidden_states)

        actions_hidden_states = self.vision_proj(actions_hidden_states)
        task_hidden_states = self.vision_proj(task_hidden_states)

        # if check_tensor("After Vision Proj (Action)", actions_hidden_states): pass

        if proprio is not None:
            proprio = proprio.reshape(batch_size, -1).to(target_dtype)
            # if check_tensor("Proprio Raw", proprio): pass
            
            proprio_features = proprio_projector(proprio)
            # if check_tensor("Proprio Features (Projected)", proprio_features): pass
            
            proprio_features = proprio_features.unsqueeze(dim=1)
        else:
            # print("⚠️ Proprio is None!")
            proprio_features = None


        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=target_dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)
        # print("➡️ Entering MLPResNet...")
        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states
            )
        
        if check_tensor("Output Action", action): pass
        print("✅ --- Action Head Predict End ---\n")

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        # if check_tensor("MLP Init (Pre-Blocks)", x): pass

        for i, block in enumerate(self.mlp_resnet_blocks):
            # x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
            x = block(x, h_t = h_t[:, i], h_a = h_a[:, i], p=p)
            # if check_tensor(f"Block {i} Output", x):
            #     # 如果这一层 NaN 了，我们再深入看看是不是 Attention Scores 炸了
            #     # (这里无法直接看内部，但可以通过定位层数来缩小范围)
            #     raise ValueError(f"NaN generated at Block {i}")
            # x = block(x, h_t = h_t[:, i+1:i+2, :], h_a = h_a[:, i+1:i+2, :], p=p)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x   



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        if torch.isnan(self.inv_freq).any():
            return None, None
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))
        # self.gating_factor = nn.Parameter(torch.tensor([0.0]))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )
        
        self.input_norm = nn.LayerNorm(dim)

    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None, block_idx=None):
        # --- 审计开始 ---
        # check_tensor_verbose("Input x", x, block_idx)
        
        g = torch.tanh(self.gating_factor)
        h_adapter = torch.cat((h_a, p), dim=1) if p is not None else h_a
        
        # 检查 RoPE Buffer
        cos_main, sin_main = self.rope(x.size(1), x.device, x.dtype)
        if cos_main is None:
            print(f"💀 [Block {block_idx}] CRITICAL: ROPE BUFFER IS NaN!")
        
        # 1. Projections
        q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        ks = self.k_self(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        vs = self.v_self(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        # check_tensor_verbose("Q projection", q, block_idx)
        # check_tensor_verbose("K_self projection", ks, block_idx)

        # 2. RoPE
        q, ks = apply_rope(q, ks, cos_main, sin_main)
        # check_tensor_verbose("Q after RoPE", q, block_idx)

        # 3. Attention Score (点积运算是最高危区)
        attn_scores = [torch.matmul(q.float(), ks.float().transpose(-2, -1))]
        # check_tensor_verbose("Self-Attention Scores", attn_scores[0], block_idx)

        # Adapter Cross-Attn
        ka = self.k_adapter(h_adapter).view(x.size(0), h_adapter.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        va = self.v_adapter(h_adapter).view(x.size(0), h_adapter.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores.append(torch.matmul(q.float(), ka.float().transpose(-2, -1)))
        # check_tensor_verbose("Adapter-Attention Scores", attn_scores[1], block_idx)

        # Task Cross-Attn
        kt = self.k_task(h_t).view(x.size(0), h_t.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        vt = self.v_task(h_t).view(x.size(0), h_t.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        score_t = torch.matmul(q.float(), kt.float().transpose(-2, -1)) * g.float()
        attn_scores.append(score_t)
        # check_tensor_verbose("Task-Attention Scores", attn_scores[2], block_idx)

        # 4. Softmax
        scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        max_val = scores.max(dim=-1, keepdim=True)[0]
        # check_tensor_verbose("Pre-Softmax MaxVal", max_val, block_idx)
        
        attn_weights = torch.softmax(scores - max_val, dim=-1)
        # check_tensor_verbose("Softmax Weights", attn_weights, block_idx)

        # 5. Output
        v_all = torch.cat([vs, va, vt], dim=2)
        out = torch.matmul(attn_weights, v_all.float()).to(x.dtype)
        out = self.o_proj(out.transpose(1, 2).reshape(x.shape))
        # check_tensor_verbose("Out Projection", out, block_idx)

        res = self.ffn(out + x)
        # check_tensor_verbose("Block Final Result", res, block_idx)
        return res
