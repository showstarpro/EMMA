#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /lpai/volumes/so-volume-ga/models/vicuna-7b-v1.5 \
    --version $PROMPT_VERSION \
    --data_path /lpai/dataset/llava-pre/0-1-0/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /lpai/dataset/llava-pre/0-1-0/LLaVA-Pretrain/images \
    --vision_tower /lpai/volumes/so-volume-ga/models/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /lpai/volumes/so-volume-ga/lhp/vicuna-7b-v1.5-pretrain/emma_base \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mm_text_select_layer -2 \
    --num_learnable_tokens 0 \
    --encoder_version v1 \
    --mm_text_select_feature patch  

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /lpai/volumes/so-volume-ga/models/vicuna-7b-v1.5 \
#     --version v1 \
#     --data_path /lpai/dataset/llava-ft/0-1-4/llava_ft/llava_v1_5_mix665k.json  \
#     --image_folder /lpai/dataset/llava-ft/0-1-4/llava_ft/data \
#     --vision_tower /lpai/volumes/so-volume-ga/models/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter /lpai/volumes/so-volume-ga/lhp/vicuna-7b-v1.5-pretrain/emma_base/mm_projector.bin \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-emma_base \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --text_module_path /lpai/volumes/so-volume-ga/lhp/vicuna-7b-v1.5-pretrain/emma_base 