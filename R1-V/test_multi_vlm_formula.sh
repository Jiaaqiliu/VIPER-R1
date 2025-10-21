#!/bin/bash

set -e


#7B

# echo "Testing Model 0 (7B SFT)"
# python test_vlm_formula.py \
#   --model_path "/fs-computility/ai4sData/shared/vlrl/vlm_sci/weights/Qwen2.5-VL-7B-Instruct-model2-5000" \
#   --test_data_path "/root/code/vlm_formula_test_100_only_answer.json" \
#   --output_path "/root/code/test_results/vlm_formula_eval_result_stage2_7B_5000_100.json" \
#   --stage 2 \
#   --eval_mode "result_only"


echo "Testing Model 1 (7B SFT)"
python test_vlm_formula.py \
  --model_path "/fs-computility/ai4sData/shared/vlrl/vlm_sci/weights/Qwen2.5-VL-7B-Instruct-model2-1001" \
  --test_data_path "/root/code/vlm_formula_test_100_only_answer.json" \
  --output_path "/root/code/test_results/vlm_formula_eval_result_stage2_7B_1001_100.json" \
  --stage 2 \
  --eval_mode "result_only"


echo "Testing Model 2 (7B SFT)"
python test_vlm_formula.py \
  --model_path "/fs-computility/ai4sData/shared/vlrl/vlm_sci/weights/Qwen2.5-VL-7B-Instruct-model2" \
  --test_data_path "/root/code/vlm_formula_test_100_only_answer.json" \
  --output_path "/root/code/test_results/vlm_formula_eval_result_stage2_7B_638_100.json" \
  --stage 2 \
  --eval_mode "result_only"


# for job 0713
echo "Testing Model 3 (7B grpo)"
python test_vlm_formula.py \
  --model_path "/fs-computility/ai4sData/shared/vlrl/vlm_sci/weights0707/Qwen2.5-VL-7B-Instruct-model3-5000-grpo-0712" \
  --test_data_path "/fs-computility/ai4sData/shared/vlrl/autoFormula/vlm_formula_test_100_only_answer.json" \
  --output_path "/fs-computility/ai4sData/shared/vlrl/vlm_sci/test_results/vlm_formula_eval_result_grpo_7B_100_0713_sample_4.json" \
  --stage 3 \
  --eval_mode "result_only"


# 3B
# echo "Testing Model 4 (3B SFT)"
# python test_vlm_formula.py \
#   --model_path "/fs-computility/ai4sData/shared/vlrl/vlm_sci/weights/Qwen2.5-VL-3B-Instruct-model2-5000" \
#   --test_data_path "/root/code/vlm_formula_test_100_only_answer.json" \
#   --output_path "/root/code/test_results/vlm_formula_eval_result_stage2_3B_5000_100.json" \
#   --stage 2 \
#   --eval_mode "result_only"

echo "Testing Model 5 (3B SFT)"
python test_vlm_formula.py \
  --model_path "your_model_path" \
  --test_data_path "your_test_data_path" \
  --output_path "your_output_path" \
  --stage 2 \
  --eval_mode "result_only"

echo "Testing Model 6 (3B SFT)"
python test_vlm_formula.py \
  --model_path "your_model_path" \
  --test_data_path "your_test_data_path" \
  --output_path "your_output_path" \
  --stage 2 \
  --eval_mode "result_only"


echo "Testing Model 7 (3B Base)"
python test_vlm_formula.py \
  --model_path "your_model_path" \
  --test_data_path "your_test_data_path" \
  --output_path "your_output_path" \
  --stage 0 \
  --eval_mode "result_only"

echo "All tests finished!"