cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/sft.py \
    --model_name_or_path /root/code/Qwen2.5-VL-3B-Instruct/  \
    --deepspeed local_scripts/zero2.json \
    --dataset_name /root/code/GEOQA_R1V_Train_8K \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --evaluation_strategy no \
    --fp16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir /root/code/R1-V/src/r1-v/log/output/