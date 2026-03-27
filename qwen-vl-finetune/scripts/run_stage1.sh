export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_PATH="/home/hjadmin/OmniDrive-VLA/Qwen3-VL/qwen-vl-finetune/expanded_train_no_null.json"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="checkpoints/omnidrive-stage1"

# 🔥 [修正点] 只使用 localhost:0 (单卡)
# 💡 [建议] 单卡显存如果不构，可以将 per_device_train_batch_size 降为 2 或 1，
# 并相应增加 gradient_accumulation_steps (例如 16 或 32) 以保持总 Batch Size 不变。
deepspeed --include localhost:0 --master_port 29500 train_omnidrive.py \
    --deepspeed scripts/ds_config_zero2.json \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --stage "stage1" \
    --image_root "/home/hjadmin/OmniDrive-VLA/nuscenes" \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard"