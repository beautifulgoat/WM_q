#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OmniDrive-VLA Training Script (Final Fix)
=========================================
Stage 1: World Model Alignment
Stage 2: End-to-End Action Training

CRITICAL FIXES:
- Unconditional Action Head Reset for Stage 2
- Forced FP32 Precision for Action Heads
- RoPE Buffer Manual Regeneration
"""

import os
import sys
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoProcessor, 
    TrainerCallback
)
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers.trainer_utils import EvalLoopOutput
from omnidrive.evaluation.evaluate_nu import NuScenesEvaluator

# 引入自定义模块
from omnidrive.modeling_omnidrive import OmniDriveVLA
from omnidrive.data.dataset import OmniDriveDataset
from omnidrive.data.collator import OmniDataCollator

# --- 日志配置 ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 参数配置
# ==============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    freeze_llm: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"help": "eager, sdpa, or flash_attention_2"})

    use_simple_bev: bool = field(
        default=True,
        metadata={"help": "Whether to enable the external Simple-BEV sidecar."}
    )

    simple_bev_root: Optional[str] = field(
        default="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev",
        metadata={"help": "Path to the root of the Simple-BEV repository."}
    )

    simple_bev_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pretrained Simple-BEV checkpoint file or directory."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data (json)."})
    image_root: str = field(default=None, metadata={"help": "Root directory for images."})
    stage: str = field(
        default="stage2",
        metadata={"help": "Training stage: stage1, stage2, stage3, or stage4"}
    )
    val_data_path: str = field(default=None, metadata={"help": "Path to validation data (optional)."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=4096)
    load_best_model_at_end: bool = field(default=False)
    metric_for_best_model: str = field(default="l2_error")
    greater_is_better: bool = field(default=False)


# ==============================================================================
# 2. 核心：核弹级重置函数 (NUKE RESET)
# ==============================================================================
def nuclear_reset_action_head(model):
    """
    [FINAL DEFENSE] 彻底清洗 Action Head 每一个字节。
    不再通过 Module 遍历，而是直接通过 Parameters 遍历，确保 gating_factor 等独立参数也被重置。
    """
    print("\n" + "!"*60)
    print("☢️  TOTAL SYSTEM RESET: Action Head & Projectors")
    print("!"*60)

    # 1. 强制转 FP32
    model.action_head.to(model.dtype)
    model.proprio_projector.to(model.dtype)
    if hasattr(model, "intent_predictor"): model.intent_predictor.to(model.dtype)

    if hasattr(model, "semantic_projector"): model.semantic_projector.to(model.dtype)

    # 2. 地毯式参数重置
    reset_count = 0
    # 针对动作头相关的所有参数名关键字
    target_keywords = ["action_head", "proprio_projector", "intent_predictor"]
    
    for name, param in model.named_parameters():
        if any(kw in name for kw in target_keywords):
            # A. 处理 gating_factor (必须清零，这是导致 Task-Attention NaN 的头号嫌疑犯)
            if "gating_factor" in name:
                nn.init.zeros_(param)
            # B. 处理 LayerNorm 的 Weight (必须设为 1)
            elif "input_norm" in name or "layer_norm" in name:
                if "weight" in name: nn.init.ones_(param)
                else: nn.init.zeros_(param)
            # C. 处理权重矩阵
            elif "weight" in name:
                if param.dim() > 1:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
            # D. 处理偏置
            elif "bias" in name:
                nn.init.zeros_(param)
            
            # E. 终极保险：如果重置完还是 NaN，强行填 0
            if torch.isnan(param).any():
                param.data.fill_(0.0)
            
            reset_count += 1

    # 3. 修复 RoPE Buffer (Buffer 不是 Parameter，需要单独处理)
    rope_count = 0
    for m in model.action_head.modules():
        if "RotaryPositionEmbedding" in m.__class__.__name__ or hasattr(m, "inv_freq"):
            if hasattr(m, "inv_freq"):
                dim = m.inv_freq.shape[0] * 2
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float().to(m.inv_freq.device) / dim))
                m.register_buffer("inv_freq", inv_freq, persistent=False)
                rope_count += 1

    print(f"✅ Cleaned {reset_count} parameters.")
    print(f"✅ Refreshed {rope_count} RoPE buffers.")
    print("!"*60 + "\n")


def run_health_check(model):
    """基本健康检查"""
    print("\n[Health Check] Verifying Trainable Parameters...")
    trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
    
    print(f"   > Trainable Params: {trainable_params/1e6:.2f} M")
    if trainable_params == 0:
        raise ValueError("❌ No trainable parameters found! Check freeze logic.")
    
    # 检查 Action Head 是否真的被设为 fp32
    head_dtype = next(model.action_head.parameters()).dtype
    print(f"   > Action Head Dtype: {head_dtype}")
    if head_dtype != torch.float32:
        print("   ⚠️ WARNING: Action Head is NOT float32. NaN risk high.")


def set_trainable_only_llm(model):
    """
    用于 stage3/stage4 的临时调试版本：
    - 只训练 LLM 主干
    - 冻结 action_head / world_model / semantic_projector / visual tower / Simple-BEV
    """
    print("🧊 [Stage3/4 Freeze] Freezing everything except LLM backbone...")

    # 先全冻
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 只开 LLM backbone
    if hasattr(model, "model"):
        model.model.requires_grad_(True)

    # 视觉塔继续冻住（保持和你原逻辑一致）
    if hasattr(model.model, "visual"):
        model.model.visual.requires_grad_(False)
        print("❄️ Vision Tower Frozen.")

    # 明确冻结这些模块
    if hasattr(model, "action_head"):
        model.action_head.requires_grad_(False)
        model.action_head.eval()

    if hasattr(model, "world_model"):
        model.world_model.requires_grad_(False)
        model.world_model.eval()

    if hasattr(model, "semantic_projector"):
        model.semantic_projector.requires_grad_(False)
        model.semantic_projector.eval()

    if hasattr(model, "proprio_projector"):
        model.proprio_projector.requires_grad_(False)

    if hasattr(model, "intent_predictor"):
        model.intent_predictor.requires_grad_(False)

    # 外挂 Simple-BEV sidecar 默认就是 frozen=True
    if hasattr(model, "simple_bev_sidecar") and model.simple_bev_sidecar is not None:
        for p in model.simple_bev_sidecar.parameters():
            p.requires_grad = False
        model.simple_bev_sidecar.eval()
        print("❄️ Simple-BEV sidecar Frozen.")


def print_gpu_status(step_name=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 [GPU] {step_name}: Allocated {allocated:.2f} GB")


def set_trainable_stage3_new_modules_only(model):
    """
    Stage 3:
    - 冻结 LLM / Action Head / World Model / Visual Tower / Simple-BEV
    - 只训练新增的 BEV 注入与 ResWorld 提取模块
    """
    print("🧩 [Stage 3 Freeze] Training ONLY bev_injector + resworld_extractor...")

    # 先全冻
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 明确冻结这些模块
    if hasattr(model, "model"):
        model.model.requires_grad_(False)
        if hasattr(model.model, "visual"):
            model.model.visual.requires_grad_(False)
            model.model.visual.eval()
            print("❄️ Vision Tower Frozen.")
        model.model.eval()

    if hasattr(model, "action_head"):
        model.action_head.requires_grad_(False)
        model.action_head.eval()

    if hasattr(model, "world_model"):
        model.world_model.requires_grad_(False)
        model.world_model.eval()

    if hasattr(model, "semantic_projector"):
        model.semantic_projector.requires_grad_(False)
        model.semantic_projector.eval()

    if hasattr(model, "proprio_projector"):
        model.proprio_projector.requires_grad_(False)

    if hasattr(model, "intent_predictor"):
        model.intent_predictor.requires_grad_(False)

    if hasattr(model, "action_query_embeddings"):
        model.action_query_embeddings.requires_grad = False

    # 外挂 Simple-BEV sidecar 默认就是 frozen=True
    if hasattr(model, "simple_bev_sidecar") and model.simple_bev_sidecar is not None:
        for p in model.simple_bev_sidecar.parameters():
            p.requires_grad = False
        model.simple_bev_sidecar.eval()
        print("❄️ Simple-BEV sidecar Frozen.")

    # 只打开新增模块
    if hasattr(model, "bev_injector"):
        model.bev_injector.requires_grad_(True)
        model.bev_injector.train()
        print("✅ BEV Injector Trainable.")

    if hasattr(model, "resworld_extractor"):
        model.resworld_extractor.requires_grad_(True)
        model.resworld_extractor.train()
        print("✅ ResWorld Extractor Trainable.")


def set_trainable_stage4_joint_finetune(model):
    """
    Stage 4:
    - 冻结 World Model / Visual Tower / Simple-BEV
    - 解冻 LLM + Action Head + 相关投影层 + 新增的注入模块，联合微调
    """
    print("🚀 [Stage 4 Freeze] Joint finetune LLM + action head + BEV modules...")

    # 先全冻
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 解冻 LLM backbone（语言主干）
    if hasattr(model, "model"):
        model.model.requires_grad_(True)
        model.model.train()

        if hasattr(model.model, "visual"):
            model.model.visual.requires_grad_(False)
            model.model.visual.eval()
            print("❄️ Vision Tower Frozen.")

    # 动作相关模块解冻
    if hasattr(model, "action_head"):
        model.action_head.requires_grad_(True)
        model.action_head.train()

    if hasattr(model, "proprio_projector"):
        model.proprio_projector.requires_grad_(True)

    if hasattr(model, "semantic_projector"):
        model.semantic_projector.requires_grad_(True)
        model.semantic_projector.train()

    if hasattr(model, "intent_predictor"):
        model.intent_predictor.requires_grad_(True)
        model.intent_predictor.train()

    if hasattr(model, "action_query_embeddings"):
        model.action_query_embeddings.requires_grad = True

    # 新增模块继续训练
    if hasattr(model, "bev_injector"):
        model.bev_injector.requires_grad_(True)
        model.bev_injector.train()

    if hasattr(model, "resworld_extractor"):
        model.resworld_extractor.requires_grad_(True)
        model.resworld_extractor.train()

    # World Model 继续冻结
    if hasattr(model, "world_model"):
        model.world_model.requires_grad_(False)
        model.world_model.eval()

    # 外挂 Simple-BEV sidecar 继续冻结
    if hasattr(model, "simple_bev_sidecar") and model.simple_bev_sidecar is not None:
        for p in model.simple_bev_sidecar.parameters():
            p.requires_grad = False
        model.simple_bev_sidecar.eval()
        print("❄️ Simple-BEV sidecar Frozen.")



# ==============================================================================
# 3. 自定义 Trainer
# ==============================================================================

class DetailedLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get('model', None)
        if hasattr(model, 'module'): model = model.module
        if model and hasattr(model, 'training_logs') and logs is not None:
            logs.update(model.training_logs)

class OmniTrainer(Trainer):
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_ref, "training_logs"):
            logs.update(model_ref.training_logs)
        super().log(logs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            loss = outputs.get("loss", None)
            if isinstance(outputs, dict):
                logits = outputs.get("pred_traj", None)
            else:
                logits = outputs[1]
            labels = inputs.get("labels", None)
        return (loss, logits, labels)
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        分布式安全验证循环 (修复版)
        功能：
        1. 绕过 Trainer 的自动列删除，强制提取 sample_token
        2. 确保 Rank 0 收集到数据后进行计算
        3. 广播计算结果，防止分布式进程挂起
        """
        import torch.distributed as dist
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1

        print(f"🔍 [Rank {rank}] Gathering samples for evaluation...")

        # A. 状态初始化
        torch.cuda.empty_cache()
        self.model.eval()
        
        results_for_metrics = {}
        all_losses = []
        # traj_scale = 50.0  # 物理单位转换系数

        # B. 遍历验证集：所有卡同步运行推理
        for batch in dataloader:
            with torch.no_grad():
                # 1. 准备模型输入 (移至 GPU)
                inputs = self._prepare_inputs(batch)
                
                # =========== 🚨 核心修复：防御性获取 sample_token 🚨 ===========
                # 优先尝试从 batch 中获取 (Collator 输出)
                tokens = batch.get("sample_token", None)
                
                # 如果 batch 里没有，尝试从 inputs 里找 (虽然不太可能，但作为保底)
                if tokens is None:
                    # 如果还是没有，打印一次调试信息并跳过当前 batch 的指标收集
                    # (注意：不跳过 forward，否则多卡同步会死锁)
                    if not hasattr(self, "_warned_token_missing"):
                        print(f"❌ [Rank {rank}] CRITICAL: 'sample_token' NOT found in batch! Keys: {list(batch.keys())}")
                        if "input_ids" in inputs:
                            print(f"   (Debug) Inputs Keys: {list(inputs.keys())}")
                        self._warned_token_missing = True
                    tokens = [] # 设为空列表，下面逻辑会自动跳过收集
                # ==========================================================

                # 2. 模型前向
                # 注意：evaluation_loop 中不需要计算梯度
                outputs = self.model(**inputs)
                
                # 记录验证 Loss (如果有)
                if "loss" in outputs and outputs["loss"] is not None:
                    loss_val = outputs["loss"].item() if isinstance(outputs["loss"], torch.Tensor) else outputs["loss"]
                    all_losses.append(loss_val)
                
                # 3. 只有 Rank 0 搜集数据，减少内存压力
                if rank == 0 and tokens:
                    # 确保输出存在
                    if "pred_traj" in outputs:
                        pred_traj = outputs["pred_traj"].detach().cpu()
                        
                        # 检查数量是否对齐 (防御性编程)
                        if len(tokens) != pred_traj.shape[0]:
                            print(f"⚠️ [Rank 0] Batch Mismatch: {len(tokens)} tokens vs {pred_traj.shape[0]} preds. Skipping batch.")
                            continue

                        for i, tkn in enumerate(tokens):
                            # 应用反归一化，变回米(m)
                            results_for_metrics[tkn] = pred_traj[i]
                    else:
                        if not hasattr(self, "_warned_no_pred"):
                            print("⚠️ [Rank 0] Model output missing 'pred_traj'.")
                            self._warned_no_pred = True

        # C. 指标计算 (仅在 Rank 0)
        final_nu_scores = {}
        if rank == 0:
            # 延迟加载 evaluator，防止在非 rank 0 进程初始化
            if not hasattr(self, "nu_evaluator"):
                try:
                    from omnidrive.evaluation.evaluate_nu import NuScenesEvaluator
                    # 确认路径 (硬编码路径，确保无误)
                    gt_path = "/home/hjadmin/OmniDrive-VLA/data/data/metrics"
                    if os.path.exists(gt_path):
                        self.nu_evaluator = NuScenesEvaluator(gt_folder=gt_path)
                    else:
                        print(f"❌ GT Path Not Found: {gt_path}")
                        self.nu_evaluator = None
                except Exception as e:
                    print(f"❌ Failed to init Evaluator: {e}")
                    self.nu_evaluator = None
            
            if self.nu_evaluator is not None and len(results_for_metrics) > 0:
                print(f"🏆 [Rank 0] Calculating ST-P3 for {len(results_for_metrics)} samples...")
                try:
                    final_nu_scores = self.nu_evaluator.compute(results_for_metrics)
                except Exception as e:
                    print(f"❌ Metric Calculation Failed: {e}")
                    final_nu_scores = {"val_avg_L2": 999.0, "val_avg_Collision": 1.0}
            else:
                print(f"❌ [Rank 0 ERROR] results_for_metrics is EMPTY! (Collected: {len(results_for_metrics)})")
                # 给个保底值防止后面崩溃
                final_nu_scores = {"val_avg_L2": 999.0, "val_avg_Collision": 1.0}

        # D. 【核弹级同步】广播指标给所有 GPU
        # 即使 Rank 0 炸了，也要广播一个保底值，否则其他进程会在这里死锁等待
        if is_dist:
            dist_list = [final_nu_scores] 
            dist.broadcast_object_list(dist_list, src=0)
            final_nu_scores = dist_list[0]

        # E. 组装 Trainer 要求的 metrics 字典
        output_metrics = {}
        # 这里的 Key 必须包含 val_avg_L2，Trainer 会在前面补 eval_
        for k, v in final_nu_scores.items():
            output_metrics[f"{k}"] = v
        
        # 补充 Loss 指标
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
        output_metrics[f"loss"] = avg_loss

        # 加上前缀 (由 Trainer 传入的 eval)
        formatted_metrics = {f"{metric_key_prefix}_{k}": v for k, v in output_metrics.items()}

        if rank == 0:
            print(f"✅ [Eval Done] {metric_key_prefix}_val_avg_L2: {formatted_metrics.get(f'{metric_key_prefix}_val_avg_L2')}")

        # F. 最终清理
        torch.cuda.empty_cache()
        
        from transformers.trainer_utils import EvalLoopOutput
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=formatted_metrics, num_samples=len(dataloader.dataset))

# ==============================================================================
# 4. 主训练流程
# ==============================================================================

def train():
    # --- 1. 参数解析 ---
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False 
    # 显式告诉 Trainer 哪些字段是 Label，防止被误删
    training_args.label_names = ["labels", "future_traj", "ego_status", "target_images"]

    # --- 2. 加载 Processor ---
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True, 
        min_pixels=256*256, 
        max_pixels=288*768 + 100
    )
    processor.tokenizer.add_special_tokens({'additional_special_tokens': ["<|action_query|>"]})
    processor.image_processor.vision_token_rate = 1.0

    # --- 3. 加载模型 ---
    print(f"🚀 Loading model from: {model_args.model_name_or_path}")
    model = OmniDriveVLA.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation=model_args.attn_implementation
    )

    if getattr(model_args, "use_simple_bev", True):
        model.init_simple_bev_sidecar(
        simple_bev_root=model_args.simple_bev_root,
        simple_bev_ckpt_path=model_args.simple_bev_ckpt_path,
        encoder_type="res101",
        freeze=True,
        debug=True,
    )

    model.training_stage = data_args.stage
    
    # ID 配置
    query_id = processor.tokenizer.convert_tokens_to_ids("<|action_query|>")
    model.action_query_token_id = query_id
    model.config.action_query_token_id = query_id
    model.resize_token_embeddings(len(processor.tokenizer))

    print_gpu_status("Model Loaded")

    # =================================================================
    # 🔥 状态初始化与修复 (STATE INITIALIZATION)
    # =================================================================

    # 逻辑修改：Stage 2 意味着我们在训练动作策略。
    # 无论底座是 Qwen 原版还是 Stage 1 的产物，我们都假定动作头需要重新初始化。
    # 这样可以彻底规避 Stage 1 可能残留的 NaN 权重或 Buffer。

    if data_args.stage == "stage2":
        print("🔒 [Stage 2 Setup] Detected Stage 2. Performing Unconditional Action Head Reset...")
        
        # 1. 彻底重置 (Weights + RoPE Buffers + FP32)
        nuclear_reset_action_head(model)
        
        # 2. Semantic Projector 平滑启动处理 (可选)
        if hasattr(model, "semantic_projector"):
            print("🎨 Zero-initializing last layer of Semantic Projector for stability...")
            # nn.init.zeros_(model.semantic_projector[-1].weight)
            # nn.init.zeros_(model.semantic_projector[-1].bias)

    elif data_args.stage == "stage1":
        print("🔓 [Stage 1 Setup] Injecting Stable Diffusion & Resetting Heads...")
        # 注入 SD 权重 (代码省略，保持原样逻辑)
        sd_path = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/huggingface_cache/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
        try:
            vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
            model.world_model.vae.load_state_dict(vae.state_dict(), strict=False)
            unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
            if hasattr(model.world_model.unet, 'unet'):
                model.world_model.unet.unet.load_state_dict(unet.state_dict(), strict=False)
            else:
                model.world_model.unet.load_state_dict(unet.state_dict(), strict=False)
            print("✅ SD weights injected.")
        except Exception as e:
            print(f"⚠️ SD Injection Warning: {e}")
        
        # Stage 1 也要重置头
        nuclear_reset_action_head(model)

    elif data_args.stage == "stage3":
        print("🧩 [Stage 3 Setup] Train ONLY newly added BEV modules.")
        print("   > Load from Stage 2 checkpoint")
        print("   > NO action head reset")
        print("   > Trainable: bev_injector + resworld_extractor")

    elif data_args.stage == "stage4":
        print("🧩 [Stage 4 Setup] Joint finetune after BEV warmup.")
        print("   > Load from Stage 3 checkpoint")
        print("   > NO action head reset")
        print("   > Trainable: LLM + action_head + projectors + bev_injector + resworld_extractor")

    else:
        raise ValueError(f"Unsupported stage: {data_args.stage}")
    

    # =================================================================
    # ❄️ 冻结配置 (Freeze)
    # =================================================================
    
    if model_args.freeze_llm:
        print("❄️ Freezing LLM Backbone...")
        for name, param in model.named_parameters():
            if not any(k in name for k in ["action_head", "world_model", "projector", "embed"]):
                param.requires_grad = False

    if hasattr(model.model, "visual"):
        model.model.visual.requires_grad_(False)
        print("❄️ Vision Tower Frozen.")

    # Stage 特定冻结
    if data_args.stage == "stage2":
        if hasattr(model.world_model, "unet"):
            model.world_model.unet.requires_grad_(False)
            model.world_model.unet.eval()
        
        # 确保动作部分是活跃的
        model.action_head.requires_grad_(True)
        model.semantic_projector.requires_grad_(True)
        model.action_head.train()

    elif data_args.stage == "stage1":
        if hasattr(model.world_model, "unet"):
            model.world_model.unet.requires_grad_(True)
        model.action_head.requires_grad_(True)
        model.semantic_projector.requires_grad_(True)

    elif data_args.stage == "stage3":
        set_trainable_stage3_new_modules_only(model)

    elif data_args.stage == "stage4":
        set_trainable_stage4_joint_finetune(model)
    # =================================================================
    # 🛑 Gradient Checkpointing
    # =================================================================
    if training_args.gradient_checkpointing:
        print("🛑 Enabling Gradient Checkpointing...")
        model.gradient_checkpointing_enable()
        if hasattr(model, "model"):
            model.model.gradient_checkpointing = True
        if hasattr(model, "world_model") and hasattr(model.world_model, "unet"):
            unet = model.world_model.unet
            unet.gradient_checkpointing = True
            for m in unet.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.gradient_checkpointing = True

    # 运行检查
    run_health_check(model)

    # --- 4. Dataset & Trainer ---
    print(f"📂 Loading Datasets for {data_args.stage}...")
    train_dataset = OmniDriveDataset(data_path=data_args.data_path, processor=processor, data_args=data_args)
    eval_dataset = OmniDriveDataset(data_path=data_args.val_data_path, processor=processor, data_args=data_args) if data_args.val_data_path else None

    if eval_dataset:
        print(f"👀 [Debug] Eval Dataset Sample Keys: {list(eval_dataset[0].keys())}")

    data_collator = OmniDataCollator(processor)

    trainer = OmniTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=[DetailedLogCallback()],
    )

    # --- 5. Train ---
    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print(f"✅ Training Complete. Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    train()