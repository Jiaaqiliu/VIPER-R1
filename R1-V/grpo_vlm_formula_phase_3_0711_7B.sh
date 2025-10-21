cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="/root/code/debug_log_2b_0707.txt"
export CUDA_LAUNCH_BLOCKING=1
export http_proxy=http://100.68.170.107:3128
export https_proxy=http://100.68.170.107:3128

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_vlm_formula_2.py \
    --output_dir /your/output/path/ \
    --model_name_or_path /your/model/path/ \
    --dataset_name json \
    --dataset_config /your/dataset/path/ \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 4096 \
    --max_completion_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name your_run_name \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4   # 8 number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
