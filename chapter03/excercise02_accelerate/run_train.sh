#!/usr/bin/env bash

accelerate launch train_sft_accelerate.py   \
    --seed 42 \
    --model_path "EleutherAI/polyglot-ko-1.3b" \
    --dataset_name heegyu/korquad-chat-v1 \
    --gradient_accumulation_steps 1 \
    --n_epoch 5 \
    --lr 1e-5 \
    --batch_size 1 \
    --weight_decay 0.1 \
    --warmpup_ratio 0.06 \
    --max_length 512 \
    --save_interval 500 \
    --verbose_interval 20 \
    --save_dir "outputs" \
    --tensorboard_log_interval 20 \
    --tensorboard_path "./tensorboard"

