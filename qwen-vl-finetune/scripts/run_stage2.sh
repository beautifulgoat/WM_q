#!/bin/bash

# 注意：这里加载 Stage 1 训练好的权重
PRETRAINED_PATH="checkpoints/omnidrive-stage1" 
DATA_PATH="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/expanded_train_no_null.json"
OUTPUT_DIR="checkpoints/omnidrive-stage2-e2e"

# 🔥 [修正点] 只使用 localhost:0
deepspeed --include localhost:0 --master_port 29500 train_omnidrive.py \
    --deepspeed scripts/ds_config_zero2.json \
    --model_name_or_path $PRETRAINED_PATH \
    --data_path $DATA_PATH \
    --stage "stage2" \
    --image_root "/home/hjadmin/OmniDrive-VLA/nuscenes" \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard"