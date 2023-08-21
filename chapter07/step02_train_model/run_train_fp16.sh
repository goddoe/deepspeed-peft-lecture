#!/usr/bin/env bash

accelerate launch train_instruct_tuned_model.py\
    --model_name_or_path "EleutherAI/polyglot-ko-3.8b" \
    --max_length 256 \
    --r 4 \
    --lora_dropout 0.1 \
    --lora_alpha 64 \
    --evaluation_strategy "steps" \
    --eval_step 1000 \
    --eval_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_strategy "epoch" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --max_eval_sample 100 \
    --lr_scheduler_type "linear" \
    --fp16 true \
    --warmup_ratio 0.06 \
    --do_train \
    --report_to "tensorboard" \
    --dataset_path "./instruction_dataset" \
    --output_dir "./outputs/my_chatgpt_3.8b"



# --max_train_sample 100000 \
