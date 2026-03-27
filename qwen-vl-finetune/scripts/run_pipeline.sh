#!/bin/bash
set -e # 遇到错误立即停止

# ================= 配置区 =================
# 环境变量解决 OOM
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0,1  # 指定可见的两张卡
# export OMP_NUM_THREADS=4
# export QWEN_USE_FLASH_ATTN=false
# export USE_FLASH_ATTN=0
# export QWEN_2_VL_USE_FLASH_ATTENTION=False
# export USE_FLASH_ATTENTION=0
# export PYTORCH_SDPA_ENABLE_FLASH=0
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.7"
# 你的路径
ROOT_DIR="/home/hjadmin/OmniDrive-VLA"
DATA_TRAIN="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/train_custom.json"

# 建议切分一个小验证集 validation.json，如果没有就先用训练集代替跑通流程
DATA_VAL="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/val_custom.json" 
IMAGE_ROOT="$ROOT_DIR/nuscenes"
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
# BASE_MODEL="Qwen/Qwen3-VL-2B-Instruct"
# BASE_MODEL="Qwen/Qwen2-VL-2B-Instruct"


# 输出目录
DIR_STAGE1="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/checkpoints/checkpoints_e2e"
DIR_STAGE2="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/checkpoints/omnidrive-stage2-e2e"

DIR_STAGE1="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/checkpoints/checkpoints_stage1_world_model"
DIR_STAGE2="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/checkpoints/checkpoints_stage2_action"
DIR_STAGE3="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/checkpoints/omnidrive-stage3-simplebev"
DIR_STAGE4="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/checkpoints/omnidrive-stage4-simplebev"

SIMPLE_BEV_ROOT="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev"
SIMPLE_BEV_CKPT="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/simple_bev/checkpoints/8x5_5e-4_rgb12_22:43:46"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# ================= Stage 1: World Model Alignment =================
# echo "🚀 [Pipeline] Starting Stage 1: World Model Pretraining..."

# deepspeed --include localhost:0,1 --master_port $MASTER_PORT train_omnidrive.py \
#     --deepspeed scripts/ds_config_zero2.json \
#     --model_name_or_path $BASE_MODEL \
#     --data_path $DATA_TRAIN \
#     --stage "stage1" \
#     --image_root $IMAGE_ROOT \
#     --bf16 True \
#     --output_dir $DIR_STAGE1 \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --attn_implementation "sdpa" \
#     --eval_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 2 \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.05 \
#     --logging_steps 1 \
#     --tf32 True \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --report_to "tensorboard" 

# echo "✅ Stage 1 Finished. Checkpoint saved to $DIR_STAGE1"

# ================= Stage 2: End-to-End Training =================
# echo "🚀 [Pipeline] Starting Stage 2: End-to-End Action Training..."
# MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# # 注意：这里 model_name_or_path 指向 Stage 1 的输出目录
# deepspeed --include localhost:0,1 --master_port $MASTER_PORT train_omnidrive.py \
#     --deepspeed scripts/ds_config_zero2.json \
#     --model_name_or_path $DIR_STAGE1 \
#     --data_path $DATA_TRAIN \
#     --val_data_path $DATA_VAL \
#     --stage "stage2" \
#     --image_root $IMAGE_ROOT \
#     --bf16 True \
#     --tf32 True \
#     --output_dir $DIR_STAGE2 \
#     --eval_strategy "epoch" \
#     --max_grad_norm 1.0 \
#     --load_best_model_at_end True \
#     --metric_for_best_model "val_avg_L2" \
#     --greater_is_better False \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --attn_implementation "sdpa" \
#     --save_strategy "epoch" \
#     --save_total_limit 3 \
#     --learning_rate 1e-5 \
#     --warmup_ratio 0.03 \
#     --logging_steps 1 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --report_to "tensorboard" \

# echo "✅ Stage 2 Finished. Best model saved to $DIR_STAGE2"

# ================= Stage 3: Simple-BEV Warmup / Debug =================
echo "🚀 [Pipeline] Starting Stage 3: Simple-BEV Warmup..."
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --include localhost:0,1 --master_port $MASTER_PORT train_omnidrive.py \
    --deepspeed scripts/ds_config_zero2.json \
    --model_name_or_path $DIR_STAGE2 \
    --data_path $DATA_TRAIN \
    --val_data_path $DATA_VAL \
    --stage "stage3" \
    --image_root $IMAGE_ROOT \
    --use_simple_bev True \
    --simple_bev_root $SIMPLE_BEV_ROOT \
    --simple_bev_ckpt_path $SIMPLE_BEV_CKPT \
    --bf16 True \
    --tf32 True \
    --output_dir $DIR_STAGE3 \
    --eval_strategy "epoch" \
    --max_grad_norm 1.0 \
    --load_best_model_at_end True \
    --metric_for_best_model "val_avg_L2" \
    --greater_is_better False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --attn_implementation "sdpa" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard"

echo "✅ Stage 3 Finished. Checkpoint saved to $DIR_STAGE3"

# ================= Stage 4: Simple-BEV Follow-up =================
echo "🚀 [Pipeline] Starting Stage 4: Simple-BEV Follow-up..."
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --include localhost:0,1 --master_port $MASTER_PORT train_omnidrive.py \
    --deepspeed scripts/ds_config_zero2.json \
    --model_name_or_path $DIR_STAGE3 \
    --data_path $DATA_TRAIN \
    --val_data_path $DATA_VAL \
    --stage "stage4" \
    --image_root $IMAGE_ROOT \
    --use_simple_bev True \
    --simple_bev_root $SIMPLE_BEV_ROOT \
    --simple_bev_ckpt_path $SIMPLE_BEV_CKPT \
    --bf16 True \
    --tf32 True \
    --output_dir $DIR_STAGE4 \
    --eval_strategy "epoch" \
    --max_grad_norm 1.0 \
    --load_best_model_at_end True \
    --metric_for_best_model "val_avg_L2" \
    --greater_is_better False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --attn_implementation "sdpa" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard"

echo "✅ Stage 4 Finished. Checkpoint saved to $DIR_STAGE4"