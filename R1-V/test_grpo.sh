cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export CUDA_LAUNCH_BLOCKING=1


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir /root/code/R1-V/src/r1-v/log/output/ \
    --model_name_or_path /root/code/Qwen2.5-VL-3B-Instruct/ \
    --dataset_name /root/code/GEOQA_R1V_Train_8K \
    --deepspeed local_scripts/zero2.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --fp16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2.5-VL-3B \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2   # 8 number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
