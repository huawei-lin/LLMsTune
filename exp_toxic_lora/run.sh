#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES="0, 1"
export WANDB_PROJECT="H100_lora_llama_new_data"

master_port=`shuf -i 12000-30000 -n 1`

lora_r=256
lora_alpha=$(( lora_r * 2 ))
learning_rate="5e-5"
num_epoch=10
batch_size=8
world_size=2

total_batch_size=128
gradient_accumulation_steps=$(( total_batch_size / world_size / batch_size))
gradient_accumulation_steps=8
total_batch_size=$(( gradient_accumulation_steps * world_size * batch_size ))

run_name="e5_ne${num_epoch}_llama2_70b_qv_r${lora_r}_a${lora_alpha}_lr${learning_rate}_bs${total_batch_size}"


cd ..
torchrun --nproc_per_node=${world_size} --master_port=${master_port} train_lora.py \
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --data_path /home/hl3352/LLMs/stanford_alpaca/training_data/all_data_single_turn_merge_alpaca.jsonl \
    --resume_from_checkpoint ~/LLMs/stanford_alpaca/exp_toxic_lora/e10_llama2_70b_qv_r256_a512_lr1e-4_bs128/checkpoint-4596 \
    --output_dir ./exp_toxic_lora/${run_name}/ \
    --run_name  ${run_name}\
    --bf16 True \
    --num_train_epochs ${num_epoch} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_steps 300 \
    --save_strategy "epoch" \
    --lr_scheduler_type "constant_with_warmup" \
    --save_total_limit 10 \
    --learning_rate ${learning_rate} \
    --model_max_length 512 \
    --logging_steps 8 \
    --tf32 True \
    --ddp_find_unused_parameters False \
    --use_lora True \
    --load_in_4bit True \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules q_proj v_proj

    # --load_in_4bit True \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --deepspeed "./configs/default_offload_opt_param.json"
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --deepspeed "./configs/default_offload_opt_param.json"

    # --fsdp "full_shard auto_wrap offload" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --resume_from_checkpoint ./exp_toxic_lora/e5_toxic_llama2_lora/ \
    # --resume_from_checkpoint ./exp_toxic_lora/e10_llama2_qkvo_r8_a16_lr1e-4_ready/checkpoint-18000/ \
    # --resume_from_checkpoint ./exp_toxic_lora/e10_e20_llama2_qkvo_r8_a16_lr1e-4_ready/ \
    # --save_steps 2000 \
    # --eval_steps 20 \ #
    # --warmup_ratio 0.03 \
    # --model_name_or_path jeffwan/llama-7b-hf \
    # --model_name_or_path meta-llama/Llama-2-7b-hf \
    # --deepspeed "./configs/default_offload_opt_param.json" 
    # --lr_scheduler_type "cosine" \
