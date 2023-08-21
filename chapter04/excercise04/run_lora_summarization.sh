#!/usr/bin/env bash

accelerate launch lora_summarization.py \
    --model_name_or_path "EleutherAI/polyglot-ko-3.8b" \
    --max_length 512 \
    --r 4 \
    --lora_dropout 0.1 \
    --lora_alpha 64 \
    --evaluation_strategy "steps" \
		--eval_accumulation_steps 1 \
    --adam_beta1 0.9 \
		--adam_beta2 0.95 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
		--save_strategy "epoch" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --lr_scheduler_type "linear" \
    --do_train \
    --report_to "tensorboard" \
    --dataset_name "daekeun-ml/naver-news-summarization-ko" \
    --output_dir "./outputs/lora"

# --max_train_sample 32 \
# --max_eval_sample 32 \

