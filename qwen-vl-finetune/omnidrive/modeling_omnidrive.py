# omnidrive/modeling_omnidrive.py

import os
import logging
from typing import Optional, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import UNet2DConditionModel, AutoencoderKL
from .modules.resworld_fuser import ResWorldFutureFeatureExtractor

# 内部模块
from .modules.wrappers import OmniWorldModel
from .modules.simple_bev_wrapper import SimpleBEVWrapper
from .modules.action_heads import L1RegressionActionHead
from .modules.bev_injection import BEVLayerInjector

logger = logging.getLogger(__name__)

def check_nan(tensor, name):
        if tensor is None: return
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"💀 [NaN DETECTED] {name} | Min: {tensor.min()} | Max: {tensor.max()}")
            return True
        return False

class OmniDriveVLA(Qwen2_5_VLForConditionalGeneration):
    """
    OmniDriveVLA: End-to-End Autonomous Driving VLA Model.
    Integrates Qwen2.5-VL backbone, Action Head, and World Model with multi-stage training support.
    """

    # --- Configuration Constants ---
    DEFAULT_SD_PATH = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/huggingface_cache/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    
    # Loss Weights Configuration
    WEIGHTS = {
        "qa": 0.0,
        "wm": 0.2,
        "action": 1.0,
        "slc": 1.0,
        "wm_back_scale": 0.03  # Scale factor for background region loss in WM
    }

    def __init__(self, config):
        super().__init__(config)
        
        # 1. Action Head (VLA-Adapter style)
        self.num_task_tokens = 512
        self.num_action_queries = 64
        self.action_head = L1RegressionActionHead(
            input_dim=config.hidden_size,
            hidden_dim=1024,
            action_dim=2,
            num_task_tokens=self.num_task_tokens,
            use_pro_version=True
        )
        
        # 2. Projectors
        self.proprio_projector = nn.Linear(3, 1024)
        
        self.intent_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Linear(2048, config.hidden_size)
        )

        sd_hidden_size = 768 
        self.semantic_projector = nn.Sequential(
            nn.Linear(config.hidden_size, 2048),
            nn.SiLU(),
            nn.Linear(2048, sd_hidden_size)
        )

        # self.simple_bev_sidecar = SimpleBEVWrapper(
        #     simple_bev_root="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev",
        #     ckpt_path="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev/checkpoints/8x5_5e-4_rgb12_22:43:46",
        #     encoder_type="res101",
        #     freeze=True,
        #     use_pre_decoder_feat=True,
        #     debug=True,
        # )
        self.simple_bev_sidecar = None
        self.simple_bev_enabled = False
        # 3. World Model
        self.world_model = OmniWorldModel(
            sd_path=self.DEFAULT_SD_PATH, 
            condition_dim=config.hidden_size
        )

        # 4. Special Embeddings & Tokens
        self.action_query_embeddings = nn.Parameter(
            torch.randn(1, self.num_action_queries, config.hidden_size)
        )
        nn.init.trunc_normal_(self.action_query_embeddings, std=0.02)   

        self.action_query_token = "<|action_query|>"        
        self.vision_token_id = getattr(config.vision_config, 'vision_token_id', 151655)
        # Fallback to hardcoded ID if not in config, matching your collator logic
        self.action_query_token_id = getattr(config, "action_query_token_id", 151665)
        self.bev_injection_mode = "current_only"  # "current_only" | "all_frames"
        # 5. Initialization
        self.post_init()
        self._check_and_reset_nan_weights()

    def init_simple_bev_sidecar(
        self,
        simple_bev_root: str,
        simple_bev_ckpt_path: str,
        encoder_type: str = "res101",
        freeze: bool = True,
        debug: bool = False,
    ):
        self.simple_bev_sidecar = SimpleBEVWrapper(
            simple_bev_root=simple_bev_root,
            ckpt_path=simple_bev_ckpt_path,
            device=str(self.device),
            encoder_type=encoder_type,
            freeze=freeze,
            use_pre_decoder_feat=True,
            debug=debug,
        )
        self.simple_bev_enabled = True

        print("✅ Simple-BEV sidecar initialized.")

        self.bev_injector = BEVLayerInjector(
            qwen_model=self, # self.model 指向底层 Qwen2_5_VLModel
            bev_dim=128,           # 【需确认】替换为实际的 C 维度大小
            hidden_dim=self.config.hidden_size,
            num_layers=self.config.num_hidden_layers
        )
        self.resworld_extractor = ResWorldFutureFeatureExtractor(
            embed_dims=128,   # 你 SimpleBEV 的通道数
            num_scenes=16,
            residual_fusion="single",
            residual_alpha=0.5,    
        )
        self.bev_injector.to(device=self.device, dtype=self.dtype)
        self.resworld_extractor.to(device=self.device, dtype=self.dtype)
        print("✅ Simple-BEV sidecar & BEV Injector initialized.")

    def set_bev_injection_mode(self, mode: str = "current_only"):
        if mode not in {"current_only", "all_frames"}:
            raise ValueError(f"Unsupported bev injection mode: {mode}")
        self.bev_injection_mode = mode

    def _build_current_frame_img_mask(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """
        当前数据集里 3 张图顺序是: T-2, T-1, Current(T)。
        Qwen 展开后，每张图通常对应一段连续的 vision tokens，中间夹着文本 token。
        因此这里取最后一段连续 vision token block，作为 Current(T) 的注入位置。
        """
        vision_mask = (input_ids == self.vision_token_id)
        current_mask = torch.zeros_like(vision_mask, dtype=torch.bool)

        for b in range(input_ids.shape[0]):
            vision_idx = torch.nonzero(vision_mask[b], as_tuple=False).flatten()
            if vision_idx.numel() == 0:
                continue

            block_breaks = torch.where((vision_idx[1:] - vision_idx[:-1]) > 1)[0]
            last_block_start = int(block_breaks[-1].item() + 1) if block_breaks.numel() > 0 else 0
            current_idx = vision_idx[last_block_start:]
            current_mask[b, current_idx] = True

        return current_mask

    def _build_bev_injection_mask(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        if self.bev_injection_mode == "all_frames":
            return (input_ids == self.vision_token_id)
        if self.bev_injection_mode == "current_only":
            return self._build_current_frame_img_mask(input_ids)
        raise ValueError(f"Unsupported bev injection mode: {self.bev_injection_mode}")

    # =========================================================================
    # Main Forward Pass
    # =========================================================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        future_traj: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        # 0. Context Setup
        if stage is None:
            stage = getattr(self, "training_stage", "stage1")
        
        if self.training:
            self._debug_forward_checker(input_ids, pixel_values, labels)

        batch_size = input_ids.shape[0]
        # Clean up kwargs for Qwen backbone
        kwargs.pop("num_items_in_batch", None)

        # 1. Embedding Injection (Action Query)
        inputs_embeds = self._inject_query_embeddings(input_ids, batch_size)
        bev_imgs = kwargs.pop("bev_imgs", None)
        bev_rots = kwargs.pop("bev_rots", None)
        bev_trans = kwargs.pop("bev_trans", None)
        bev_intrins = kwargs.pop("bev_intrins", None)
        bev_ego_pose = kwargs.pop("bev_ego_pose", None)
        simple_bev_out = None
        if bev_imgs is not None:
            simple_bev_out = self.simple_bev_sidecar(
                bev_imgs=bev_imgs,
                bev_rots=bev_rots,
                bev_trans=bev_trans,
                bev_intrins=bev_intrins,
                bev_ego_pose=bev_ego_pose,
            )
        if simple_bev_out is not None and getattr(self, "bev_injector", None) is not None:
            
            # ---> 新增：加工特征 <---
            # 这里假设 simple_bev_out 里能取到 ego_pose_seq (如果没有，传 None 也会正常运行不报错，只是不做 Warp)
            pose_seq = simple_bev_out.get("ego_pose_seq", None)
            pose_t2 = pose_seq[:, 0] if pose_seq is not None else None
            pose_t1 = pose_seq[:, 1] if pose_seq is not None else None
            pose_t0 = pose_seq[:, 2] if pose_seq is not None else None
            # 送入我们写的黑盒，吐出神装 Future BEV Feature
            future_bev_feature = self.resworld_extractor(
                feat_t0=simple_bev_out["bev_feat_t0"],
                feat_t1=simple_bev_out["bev_feat_t1"],
                feat_t2=simple_bev_out["bev_feat_t2"], # 🚨 加上 t2
                pose_t0=pose_t0,
                pose_t1=pose_t1,
                pose_t2=pose_t2                        # 🚨 加上 t2 位姿
            )

            img_mask = self._build_bev_injection_mask(input_ids)
            # 假设你注入的是当前帧 t0 的特征
            self.bev_injector.set_bev_context(future_bev_feature, img_mask)
        else:
            if getattr(self, "bev_injector", None) is not None:
                self.bev_injector.clear_bev_context()

        # 2. Backbone Forward (Qwen2.5-VL)
        outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True, 
            return_dict=True,
            **kwargs
        )

        if getattr(self, "bev_injector", None) is not None:
            self.bev_injector.clear_bev_context()

        # 3. Feature Extraction (Multiscale & Task-Specific)
        # task_hs: [B, Layers, 512, D], actions_hs: [B, Layers, 64, D]
        task_hidden_states, actions_hidden_states = self._extract_multiscale_features(
            outputs.hidden_states[1:], # Skip embedding layer
            input_ids,
            batch_size
        )

        safe_dtype = self.dtype
        
        
        # 1. 强转 Hidden States
        # 注意：这里会产生内存拷贝，但为了不 NaN 必须这么做
        actions_in_safe = actions_hidden_states.to(safe_dtype)
        task_in_safe = task_hidden_states.to(safe_dtype)
        
        # 2. 强转 Proprio 输入
        ego_status_safe = ego_status.to(safe_dtype) if ego_status is not None else None
        # 4. Action Head Prediction
        # Compute forward pass even in Stage 1 to ensure graph connectivity (though loss is 0)
        if check_nan(actions_hidden_states, "Input: actions_hidden_states"): pass
        if check_nan(task_hidden_states, "Input: task_hidden_states"): pass
        if check_nan(ego_status, "Input: ego_status"): pass
        pred_traj_output = self.action_head.predict_action(
            actions_hidden_states=actions_in_safe,
            task_hidden_states=task_in_safe,
            proprio=ego_status_safe,
            proprio_projector=self.proprio_projector,
            phase="Training" if self.training else "Inference"
        )
        pred_traj_output = pred_traj_output.reshape(batch_size, -1)
        if check_nan(pred_traj_output, "Output: pred_traj_output"):
            # 如果输出是 NaN，打印一下 Projector 的权重状态，看看是不是重置坏了
            print(f"  > Proprio Projector Weight Max: {self.proprio_projector.weight.max()}")
            print(f"  > Action Head First Layer Weight Max: {list(self.action_head.parameters())[0].max()}")
        # 5. Loss Computation
        total_loss, loss_dict, debug_outputs = self._compute_losses(
            outputs=outputs,
            actions_hidden_states=actions_hidden_states,
            all_hidden_states=outputs.hidden_states[1:], # Pass raw states for WM spatial alignment
            target_images=target_images,
            pred_traj=pred_traj_output,
            future_traj=future_traj,
            input_ids=input_ids,
            stage=stage,
            batch_size=batch_size,
            step=kwargs.get('step', 0)
        )

        # 6. Logging & Return
        self._update_logs(loss_dict)
        
        # Prepare return trajectory (GT for Stage 1 visual sanity, Pred for Stage 2)
        ret_traj = pred_traj_output if stage == "stage2" else (
            future_traj.clone() if future_traj is not None else torch.zeros(batch_size, 12, device=self.device)
        )

        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "loss_dict": loss_dict,
            "pred_traj": ret_traj,
            "debug_outputs": debug_outputs,
        }

    # =========================================================================
    # Inference Methods
    # =========================================================================
    @torch.no_grad()
    def generate_next_frame(
        self, 
        input_ids, 
        pixel_values, 
        image_grid_thw, 
        future_traj=None,
        ego_status=None, 
        num_inference_steps=50,
        guidance_scale=7.5
    ):
        """Inference pipeline for World Model generation."""
        self.eval()
        
        # Input standardisation
        if pixel_values.dim() == 5: pixel_values = pixel_values.squeeze(0)
        if image_grid_thw.dim() == 3: image_grid_thw = image_grid_thw.squeeze(0)
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        if ego_status is not None and ego_status.dim() == 1: ego_status = ego_status.unsqueeze(0)

        dummy_target = torch.zeros(1, 3, 512, 1344, dtype=self.dtype, device=self.device)
        
        # Reuse forward pass logic to get conditions
        outputs = self(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            future_traj=future_traj,
            ego_status=ego_status,
            target_images=dummy_target,
            stage="stage1" 
        )
        
        debug_data = outputs.get("debug_outputs", {})
        if not debug_data:
            logger.error("Forward pass did not return debug_outputs.")
            return None

        print(f"🎨 Generating Frame... Cond Shape: {debug_data['semantic_condition'].shape}")
        
        # Diffusion Sampling
        latents = self.world_model.unet.inference(
            z=debug_data['wm_spatial_condition'],
            encoder_hidden_states=debug_data['semantic_condition'],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=288, width=768
        )
        
        # Decode
        image = self.world_model.vae.decode(latents / 0.18215).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).astype(np.uint8))

    # =========================================================================
    # Internal Logic Methods
    # =========================================================================
    def _inject_query_embeddings(self, input_ids, batch_size):
        """Injects learnable Action Query embeddings into input."""
        inputs_embeds = self.get_input_embeddings()(input_ids)
        query_mask = (input_ids == self.action_query_token_id)
        
        if query_mask.any():
            try:
                num_queries = query_mask.sum()
                expected = batch_size * self.num_action_queries
                if num_queries == expected:
                    inputs_embeds[query_mask] = self.action_query_embeddings.expand(batch_size, -1, -1).reshape(-1, self.config.hidden_size)
                else:
                    # Log warning but don't crash (robustness)
                    if not hasattr(self, "_emb_warned"):
                        print(f"⚠️ [Embedding] Mismatch: {num_queries} vs {expected}. Skipping injection.")
                        self._emb_warned = True
            except Exception as e:
                print(f"⚠️ [Embedding] Injection Error: {e}")
        return inputs_embeds

    def _extract_multiscale_features(self, all_hidden_states, input_ids, batch_size):
        """Extracts and normalizes features for Action Head from all layers."""
        vision_mask = (input_ids == self.vision_token_id)
        query_mask = (input_ids == self.action_query_token_id)
        
        all_layer_task = []
        all_layer_action = []

        # Iterate over all layers
        for layer_hs in all_hidden_states:
            batch_task = []
            batch_action = []
            for b in range(batch_size):
                # 1. Vision Tokens -> Task Tokens (Interpolated)
                vis_tokens = layer_hs[b][vision_mask[b]]
                if vis_tokens.shape[0] == 0:
                    vis_tokens = torch.zeros(self.num_task_tokens, self.config.hidden_size, device=layer_hs.device, dtype=layer_hs.dtype)
                
                vis_tokens_t = vis_tokens.transpose(0, 1).unsqueeze(0)
                vis_resampled = F.interpolate(vis_tokens_t, size=self.num_task_tokens, mode='linear', align_corners=False)
                batch_task.append(vis_resampled.squeeze(0).transpose(0, 1))
                
                # 2. Action Query Tokens
                batch_action.append(layer_hs[b][query_mask[b]])
            
            all_layer_task.append(torch.stack(batch_task))
            all_layer_action.append(torch.stack(batch_action))

        # Stack -> [B, Layers, N, D]
        task_hidden_states = torch.stack(all_layer_task, dim=1)
        actions_hidden_states = torch.stack(all_layer_action, dim=1)

        # Pre-Head Normalization
        task_hidden_states = F.layer_norm(task_hidden_states, [task_hidden_states.size(-1)])
        actions_hidden_states = F.layer_norm(actions_hidden_states, [actions_hidden_states.size(-1)])
        
        return task_hidden_states, actions_hidden_states

    def _build_spatial_condition(self, all_hidden_states, input_ids, batch_size):
        """Builds the 2D spatial condition map for the World Model."""
        last_hidden_state = all_hidden_states[-1]
        vision_mask = (input_ids == self.vision_token_id)
        wm_conditions_list = []

        for b in range(batch_size):
            vis_tokens = last_hidden_state[b][vision_mask[b]]
            total_tokens = vis_tokens.shape[0]
            
            if total_tokens == 0:
                # Fallback: Zero tensor
                joint_cond = torch.zeros(1, self.config.hidden_size, 36, 96, device=vis_tokens.device, dtype=vis_tokens.dtype)
                wm_conditions_list.append(joint_cond)
                continue

            # Heuristic: Take last 1/3 (assumed front view/most relevant)
            start_idx = (total_tokens // 3) * 2
            curr_frame_feat = vis_tokens[start_idx:]
            
            # Reshape 1D -> 2D
            curr_frame_feat = curr_frame_feat.transpose(0, 1).unsqueeze(0) # [1, D, N]
            n = curr_frame_feat.shape[2]
            h_feat = int((n * 3 / 8) ** 0.5)
            w_feat = n // h_feat
            
            if h_feat * w_feat != n:
                curr_frame_feat = F.interpolate(curr_frame_feat, size=(h_feat * w_feat), mode='linear')
            
            curr_frame_feat_2d = curr_frame_feat.view(1, -1, h_feat, w_feat)
            
            # Align to target resolution (36x96)
            curr_frame_feat_aligned = F.interpolate(
                curr_frame_feat_2d, 
                size=(36, 96), 
                mode='bilinear', 
                align_corners=False
            )
            wm_conditions_list.append(curr_frame_feat_aligned)
        
        wm_condition_batch = torch.cat(wm_conditions_list, dim=0)
        
        # Apply normalization carefully on channel dimension
        # [B, C, H, W] -> [B, H, W, C] -> Norm -> [B, C, H, W]
        wm_condition_batch = wm_condition_batch.permute(0, 2, 3, 1)
        wm_condition_batch = F.layer_norm(wm_condition_batch, [wm_condition_batch.size(-1)])
        wm_condition_batch = wm_condition_batch.permute(0, 3, 1, 2)
        
        return wm_condition_batch
    
    

    # def _extract_target_features(self, target_images):
    #     """
    #     利用 Qwen Vision Tower 提取下一帧的全局特征。
    #     Args:
    #         target_images: [B, 3, 288, 768] (Dataset原始输出)
    #     Returns:
    #         target_feat: [B, Hidden_Dim] (Detached)
    #     """
    #     target_small = F.interpolate(target_images, size=(224, 224), mode='bilinear', align_corners=False)

    #     B = target_images.shape[0]
    #     grid_thw = torch.tensor([[1, 16, 16]] * B, device=target_images.device, dtype=torch.long)
        
        
    #     return F.adaptive_avg_pool2d(target_images, (16, 16)).flatten(1) # [B, 3*16*16] = [B, 768]
    
    def _compute_losses(self, outputs, actions_hidden_states, all_hidden_states, target_images, pred_traj, future_traj, input_ids, stage, batch_size, step):
        """Computes all loss components based on stage and data availability."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict = {}
        debug_outputs = {}

        # 1. QA Loss (Base LLM)
        raw_qa_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)
        if torch.isnan(raw_qa_loss): raw_qa_loss = torch.tensor(0.0, device=self.device)
        loss_dict['loss_qa'] = raw_qa_loss.detach()
        if self.WEIGHTS['qa'] > 0:
            total_loss = total_loss + self.WEIGHTS['qa'] * raw_qa_loss

        # 2. World Model Loss
        if target_images is not None:
            # A. Path A: Spatial Condition (Visual Features only, NO Trajectory Heatmap)
            wm_spatial_condition = self._build_spatial_condition(all_hidden_states, input_ids, batch_size)
            
            # B. Path B: Semantic Condition (Action Query Features -> Projector)
            action_query_feats = actions_hidden_states[:, -1, :, :] # Last layer
            semantic_condition = self.semantic_projector(action_query_feats)

            # Debug Logs
            if self.training and hasattr(self, "_debug_counter") and self._debug_counter < 4:
                self._log_condition_stats(wm_spatial_condition, semantic_condition)

            # C. Compute Diffusion Loss
            target_images_resized = F.interpolate(target_images, size=(288, 768), mode='bilinear', align_corners=False)
            raw_wm_loss_map = self.world_model.get_loss_map(
                target_images_resized, 
                wm_condition=wm_spatial_condition, 
                action_query_embeds=semantic_condition
            )
            
            # Weighted Foreground/Background Loss
            h_latent = raw_wm_loss_map.shape[2]
            front_loss = raw_wm_loss_map[:, :, :h_latent//2, :].mean()
            back_loss = raw_wm_loss_map[:, :, h_latent//2:, :].mean()
            
            loss_wm = front_loss + self.WEIGHTS['wm_back_scale'] * back_loss
            
            loss_dict.update({
                'loss_wm': loss_wm.detach(),
                'front_loss': front_loss.detach(),
                'back_loss': back_loss.detach()
            })
            debug_outputs.update({
                'wm_spatial_condition': wm_spatial_condition,
                'semantic_condition': semantic_condition
            })
            
            total_loss = total_loss + self.WEIGHTS['wm'] * loss_wm
        else:
            loss_dict['loss_wm'] = torch.tensor(0.0, device=self.device)

        # 3. Action Loss (Enabled only in Stage 2)
        if stage == "stage2" and future_traj is not None:
            # action_loss = F.l1_loss(pred_traj, future_traj)
            action_loss = F.smooth_l1_loss(pred_traj, future_traj, reduction='mean', beta=1.0)
            loss_dict['loss_action'] = action_loss.detach()
            total_loss = total_loss + self.WEIGHTS['action'] * action_loss
        else:
            loss_dict['loss_action'] = torch.tensor(0.0, device=self.device)

        # Console Log (Rank 0 only)
        self._log_step_console(loss_dict, total_loss)

        return total_loss, loss_dict, debug_outputs

    # =========================================================================
    # Utilities & Checks
    # =========================================================================
    def _update_logs(self, loss_dict):
        """Updates internal logging dict for Trainer callback."""
        self.training_logs = {
            "train/loss_wm": loss_dict.get('loss_wm', torch.tensor(0.0)).item(),
            "train/loss_action": loss_dict.get('loss_action', torch.tensor(0.0)).item(),
            "train/loss_qa": loss_dict.get('loss_qa', torch.tensor(0.0)).item(),
            "train/loss_wm_front": loss_dict.get('front_loss', torch.tensor(0.0)).item(),
            "train/loss_wm_back": loss_dict.get('back_loss', torch.tensor(0.0)).item(),
        }

    def _log_step_console(self, loss_dict, total_loss):
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            l_qa = loss_dict.get('loss_qa', torch.tensor(0.0)).item()
            l_wm = loss_dict.get('loss_wm', torch.tensor(0.0)).item()
            l_act = loss_dict.get('loss_action', torch.tensor(0.0)).item()
            print(f"\n[Step Loss] QA: {l_qa:.4f} | WM: {l_wm:.4f} | Action: {l_act:.4f} | Total: {total_loss.item():.4f}")

    def _log_condition_stats(self, wm_cond, sem_cond):
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"   👀 Semantic Condition: Mean={sem_cond.mean().item():.4f}, Std={sem_cond.std().item():.4f}")
            print(f"   > WM Condition: min={wm_cond.min().item():.4f}, max={wm_cond.max().item():.4f}")
            if sem_cond.std() == 0:
                print("   ⚠️ WARNING: Semantic condition is constant (collapsed or zero-init).")

    def _debug_forward_checker(self, input_ids, pixel_values, labels):
        if not hasattr(self, "_debug_counter"): self._debug_counter = 0
        if self._debug_counter >= 3: return

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\n{'='*20} [Step {self._debug_counter} Deep Check] {'='*20}")
            query_count = (input_ids == self.action_query_token_id).sum().item()
            print(f"1️⃣ Input Analysis:")
            print(f"   - Batch Size: {input_ids.shape[0]}")
            print(f"   - Action Query ID: {self.action_query_token_id}")
            print(f"   - Found Count: {query_count}")
            
            if query_count == 0:
                print("   ❌ CRITICAL: No Action Query Tokens found!")
            else:
                print("   ✅ Action Query Tokens detected.")

            if labels is not None:
                n_valid = (labels != -100).sum().item()
                print(f"   - Valid Labels: {n_valid}")
                if n_valid == 0:
                    print("   ✅ Labels are fully masked.")
                else:
                    print("   ⚠️ Labels contain valid targets.")
            
            self._debug_counter += 1

    def _check_and_reset_nan_weights(self):
        print("🛡️ Weight Health Check...")
        has_nan = False
        for name, param in self.named_parameters():
            if param.device.type == 'meta': continue
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"❌ Corrupted weight: {name}")
                has_nan = True
                if "action_head" in name or "projector" in name:
                     print(f"🔧 Resetting {name}...")
                     nn.init.zeros_(param)
        if not has_nan:
            print("✅ All weights healthy.")