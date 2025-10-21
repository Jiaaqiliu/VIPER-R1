#!/bin/bash

set -e

echo "Starting SFT Phase 1..."
bash sft_vlm_formula_phase_1.sh

echo "Phase 1 finished. Starting SFT Phase 2..."
bash sft_vlm_formula_phase_2.sh

echo "All SFT phases finished!" 

echo "Testing Model 1 (SFT)"
python test_vlm_formula.py \
  --model_path "your_model_path" \
  --test_data_path "your_test_data_path" \
  --output_path "your_output_path" \
  --stage 2 \
  --eval_mode "result_only"