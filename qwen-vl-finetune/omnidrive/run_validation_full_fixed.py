import os
import sys
import json
import argparse
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

# ==============================================================================
# 1. 环境路径修复
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

local_utils_path = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-utils/src"
if local_utils_path not in sys.path:
    sys.path.insert(0, local_utils_path)

from omnidrive.evaluation.evaluate_nu import NuScenesEvaluator
from omnidrive.modeling_omnidrive import OmniDriveVLA
from omnidrive.data.dataset import OmniDriveDataset
from omnidrive.data.collator import OmniDataCollator

try:
    from transformers.modeling_utils import load_sharded_checkpoint
except Exception:
    load_sharded_checkpoint = None

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


# ==============================================================================
# 2. 参数容器
# ==============================================================================
class DataArgs:
    def __init__(
        self,
        data_path: str,
        val_data_path: str,
        stage: str,
        image_root: Optional[str] = None,
        use_simple_bev: bool = False,
        nuscenes_version: str = "v1.0-trainval",
    ):
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.stage = stage
        self.image_root = image_root
        self.use_simple_bev = use_simple_bev
        self.nuscenes_version = nuscenes_version


# ==============================================================================
# 3. 工具函数
# ==============================================================================
def infer_stage_from_ckpt(ckpt_path: str, user_stage: Optional[str] = None) -> str:
    if user_stage is not None:
        return user_stage

    low = ckpt_path.lower()
    if "stage4" in low:
        return "stage4"
    if "stage3" in low or "simplebev" in low:
        return "stage3"
    if "stage2" in low:
        return "stage2"
    if "stage1" in low:
        return "stage1"
    return "stage2"


def find_single_weight_file(ckpt_path: str) -> Optional[str]:
    if os.path.isfile(ckpt_path):
        return ckpt_path

    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
    ]
    for name in candidates:
        full = os.path.join(ckpt_path, name)
        if os.path.exists(full):
            return full
    return None


def has_sharded_weights(ckpt_path: str) -> bool:
    if not os.path.isdir(ckpt_path):
        return False
    return (
        os.path.exists(os.path.join(ckpt_path, "model.safetensors.index.json"))
        or os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin.index.json"))
    )


def reload_full_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> Tuple[list, list]:
    """
    关键用途：
    - stage3/stage4 在 from_pretrained 时，新增模块(bev_injector/resworld_extractor)
      还没被创建，checkpoint 中这些权重会漏掉。
    - 因此在 init_simple_bev_sidecar() 之后，再把整个 checkpoint 重新 load 一遍。
    """
    print("\n🔁 [Reload] Reloading checkpoint weights after module initialization...")

    missing_keys = []
    unexpected_keys = []

    if has_sharded_weights(ckpt_path):
        if load_sharded_checkpoint is None:
            raise RuntimeError("Sharded checkpoint detected, but transformers.load_sharded_checkpoint is unavailable.")
        load_sharded_checkpoint(model, ckpt_path, strict=False, prefer_safe=True)
        print("✅ [Reload] Sharded checkpoint loaded with strict=False.")
        return missing_keys, unexpected_keys

    weight_file = find_single_weight_file(ckpt_path)
    if weight_file is None:
        raise FileNotFoundError(f"No model weight file found under: {ckpt_path}")

    if weight_file.endswith(".safetensors"):
        if safe_load_file is None:
            raise RuntimeError("safetensors file detected, but safetensors is unavailable.")
        state_dict = safe_load_file(weight_file)
    else:
        state_dict = torch.load(weight_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    incompatible = model.load_state_dict(state_dict, strict=False)
    if hasattr(incompatible, "missing_keys"):
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)

    print(f"✅ [Reload] Loaded from: {weight_file}")
    print(f"   > Missing keys: {len(missing_keys)}")
    print(f"   > Unexpected keys: {len(unexpected_keys)}")
    return missing_keys, unexpected_keys


def prepare_processor(ckpt_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path,
            use_fast=True,
            fix_mistral_regex=True,
            trust_remote_code=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path,
            use_fast=True,
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    special = "<|action_query|>"
    if special not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [special]})
    else:
        tokenizer.add_special_tokens({"additional_special_tokens": [special]})

    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "vision_token_rate"):
        processor.image_processor.vision_token_rate = 1.0

    return processor


def build_model(
    ckpt_path: str,
    processor,
    train_stage: str,
    use_simple_bev: bool,
    simple_bev_root: str,
    simple_bev_ckpt_path: str,
    device: torch.device,
):
    print(f"\n🚀 [Init] Loading Model from: {ckpt_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = OmniDriveVLA.from_pretrained(
        ckpt_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    query_id = processor.tokenizer.convert_tokens_to_ids("<|action_query|>")
    model.action_query_token_id = query_id
    model.config.action_query_token_id = query_id
    model.resize_token_embeddings(len(processor.tokenizer))
    model.training_stage = train_stage

    model.to(device)

    if use_simple_bev:
        print("\n🧩 [Init] Enabling Simple-BEV sidecar for stage3/stage4 evaluation...")
        model.init_simple_bev_sidecar(
            simple_bev_root=simple_bev_root,
            simple_bev_ckpt_path=simple_bev_ckpt_path,
            encoder_type="res101",
            freeze=True,
            debug=False,
        )
        reload_full_checkpoint_into_model(model, ckpt_path)

    model.eval()

    print("\n🔍 [Weight Verification] Checking Action Head weights...")
    for name, param in model.named_parameters():
        if "action_head" in name and "weight" in name and param.dim() > 1:
            print(f"   ✅ {name} | Mean: {param.data.float().mean().item():.6f} | Std: {param.data.float().std().item():.6f}")
            break

    if use_simple_bev:
        print("🔍 [Weight Verification] Checking stage3/stage4 extra modules...")
        found_extra = False
        for name, param in model.named_parameters():
            if name.startswith("bev_injector") or name.startswith("resworld_extractor"):
                print(f"   ✅ {name} | Mean: {param.data.float().mean().item():.6f} | Std: {param.data.float().std().item():.6f}")
                found_extra = True
                break
        if not found_extra:
            print("   ⚠️ No BEV extra-module parameters were found after reload.")

    return model, dtype


# ==============================================================================
# 4. 验证主逻辑
# ==============================================================================
def run_full_validation(
    ckpt_path: str,
    val_json_path: str,
    gt_folder: str,
    train_stage: str,
    batch_size: int = 8,
    num_workers: int = 8,
    image_root: str = "/home/hjadmin/OmniDrive-VLA/nuscenes",
    simple_bev_root: str = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev",
    simple_bev_ckpt_path: str = "/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev/checkpoints/8x5_5e-4_rgb12_22:43:46",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_simple_bev = train_stage in ["stage3", "stage4"]

    print("=" * 80)
    print(f"[Eval Config] train_stage={train_stage} | use_simple_bev={use_simple_bev} | device={device}")
    print("=" * 80)

    processor = prepare_processor(ckpt_path)
    model, dtype = build_model(
        ckpt_path=ckpt_path,
        processor=processor,
        train_stage=train_stage,
        use_simple_bev=use_simple_bev,
        simple_bev_root=simple_bev_root,
        simple_bev_ckpt_path=simple_bev_ckpt_path,
        device=device,
    )

    print(f"\n📂 [Init] Loading Validation Data from: {val_json_path}")
    data_args = DataArgs(
        data_path=val_json_path,
        val_data_path=val_json_path,
        stage=train_stage,
        image_root=image_root,
        use_simple_bev=use_simple_bev,
    )
    val_dataset = OmniDriveDataset(data_path=val_json_path, processor=processor, data_args=data_args)
    collate_fn = OmniDataCollator(processor)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"⚡ [Run] Starting inference on {len(val_dataset)} samples...")
    results_dict: Dict[str, torch.Tensor] = {}
    total_action_loss = 0.0
    total_batches = 0

    # 关键约定：
    # 这里统一用 stage="stage2" 做前向，是为了强制 modeling_omnidrive.py
    # 返回真实 pred_traj_output，而不是 stage3/stage4 分支下被 future_traj 覆盖的 GT。
    # 这不会改变 evaluate_nu.py 的指标计算逻辑；指标仍然完全由 NuScenesEvaluator.compute() 负责。
    model_forward_stage = "stage2"

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)

            image_grid_thw = batch.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            ego_status = batch.get("ego_status", None)
            if ego_status is not None:
                ego_status = ego_status.to(device=device, dtype=dtype)

            future_traj = batch.get("future_traj", None)
            if future_traj is not None:
                future_traj = future_traj.to(device=device, dtype=dtype)

            sample_tokens = batch["sample_token"]

            forward_kwargs = dict(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                ego_status=ego_status,
                future_traj=future_traj,
                stage=model_forward_stage,
            )

            if use_simple_bev:
                for key in ["bev_imgs", "bev_rots", "bev_trans", "bev_intrins", "bev_ego_pose"]:
                    if key not in batch:
                        raise KeyError(f"Missing BEV field in batch for {train_stage}: {key}")

                forward_kwargs.update(
                    bev_imgs=batch["bev_imgs"].to(device=device, dtype=torch.float32),
                    bev_rots=batch["bev_rots"].to(device=device, dtype=torch.float32),
                    bev_trans=batch["bev_trans"].to(device=device, dtype=torch.float32),
                    bev_intrins=batch["bev_intrins"].to(device=device, dtype=torch.float32),
                    bev_ego_pose=batch["bev_ego_pose"].to(device=device, dtype=torch.float32),
                )

            outputs = model(**forward_kwargs)

            if "loss_dict" in outputs and "loss_action" in outputs["loss_dict"]:
                loss_action = outputs["loss_dict"]["loss_action"]
                if isinstance(loss_action, torch.Tensor):
                    total_action_loss += loss_action.item()
                else:
                    total_action_loss += float(loss_action)
                total_batches += 1

            pred_trajs = outputs.get("pred_traj")
            if pred_trajs is None:
                raise RuntimeError("Model output missing 'pred_traj'.")

            batch_size_curr = input_ids.shape[0]
            for i in range(batch_size_curr):
                token = sample_tokens[i]
                traj = pred_trajs[i].detach().cpu()
                if traj.dim() == 1 and traj.shape[0] == 12:
                    traj = traj.view(6, 2)
                results_dict[token] = traj

    avg_loss = total_action_loss / total_batches if total_batches > 0 else 0.0
    print(f"\n📉 [Sanity Check] Avg Action Loss (Computed): {avg_loss:.6f}")

    print(f"\n📊 [Eval] Computing STP-3 Metrics for {len(results_dict)} samples...")
    evaluator = NuScenesEvaluator(gt_folder=gt_folder)
    metrics = evaluator.compute(results_dict)

    print("\n" + "=" * 40)
    print("🏆 OmniDrive-VLA Final Results")
    print("=" * 40)
    print(f"Checkpoint           : {ckpt_path}")
    print(f"Train Stage         : {train_stage}")
    print(f"Forward Stage       : {model_forward_stage} (forced for real pred_traj return)")
    print(f"Simple-BEV Enabled  : {use_simple_bev}")
    print(f"Avg Action Loss     : {avg_loss:.6f}")
    for k, v in metrics.items():
        if torch.is_tensor(v):
            v = v.item()
        print(f"{k:<20}: {float(v):.6f}")
    print("=" * 40)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, default="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/val_custom.json")
    parser.add_argument("--gt", type=str, default="/home/hjadmin/OmniDrive-VLA/data/data/metrics")
    parser.add_argument("--train_stage", type=str, default=None, choices=[None, "stage1", "stage2", "stage3", "stage4"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_root", type=str, default="/home/hjadmin/OmniDrive-VLA/nuscenes")
    parser.add_argument("--simple_bev_root", type=str, default="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev")
    parser.add_argument("--simple_bev_ckpt_path", type=str, default="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev/checkpoints/8x5_5e-4_rgb12_22:43:46")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    train_stage = infer_stage_from_ckpt(args.ckpt, args.train_stage)

    run_full_validation(
        ckpt_path=args.ckpt,
        val_json_path=args.data,
        gt_folder=args.gt,
        train_stage=train_stage,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_root=args.image_root,
        simple_bev_root=args.simple_bev_root,
        simple_bev_ckpt_path=args.simple_bev_ckpt_path,
    )
