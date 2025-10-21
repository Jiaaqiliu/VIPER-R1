cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export http_proxy=http://100.68.170.107:3128
export https_proxy=http://100.68.170.107:3128

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/sft_vlm_formula.py \
    --model_name_or_path /fs-computility/ai4sData/shared/models/Qwen2.5-VL-7B-Instruct \
    --deepspeed local_scripts/zero3.json \
    --dataset_name json \
    --dataset_config /root/code/vlm_formula_all_data_5000.json \
    --learning_rate 2.0e-5 \
    --num_train_epochs 5 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --evaluation_strategy no \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir /fs-computility/ai4sData/shared/vlrl/vlm_sci/weights/Qwen2.5-VL-7B-Instruct-model1-5000 \
    --report_to wandb \
    --logging_dir /root/code/R1-V/src/r1-v/log/ \
    --run_name "sft_stage1_qwenvl_cot_eq_7B_5000" \
    --stage 1
