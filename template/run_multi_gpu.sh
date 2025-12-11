TRAINPARAMS="
    --seed=42 \
    --output_dir="./trainer_output" \
    --num_train_epochs=5 \
    --logging_steps=20 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --max_grad_norm=1.0 \
    --learning_rate=5e-5 \
    --lr_scheduler_type="linear" \
    --weight_decay=1e-2 \
    --num_warmup_steps=0 \
    --mixed_precision="bf16" \
    --with_tracking=True \
    --report_to="tensorboard" \
    --checkpointing_steps=500 \
    --dataloader_pin_memory=True \
    --dataloader_persistent_workers=True \
    --dataloader_num_workers=4 \
    --dataloader_prefetch_factor=2 \
    --deepspeed="./ds_zero2.json""


MODELPARAMS="
    --model_name_or_path="./models/hfl-rbt3""


DATAPARAMS="
    --train_file="./ChnSentiCorp_htl_all.csv""


RANDOM_PORT=$(shuf -i 1024-65535 -n 1)
echo "Using Port: $RANDOM_PORT"

export CUDA_VISIBLE_DEVICES="1,4"
accelerate launch \
    --config_file="./default_config.yaml" \
    --num_processes=1 \
    --main_process_port=$RANDOM_PORT \
    train.py \
    $TRAINPARAMS \
    $MODELPARAMS \
    $DATAPARAMS