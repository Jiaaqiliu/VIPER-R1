# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import sympy


from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from open_r1.trainer import Qwen2_5_VLGRPOVLLMTrainerModified #Qwen2_5_VLGRPOTrainer, 
from open_r1.trainer import Qwen2_5_VLGRPOTrainer
# from open_r1.trainer import Qwen2_5_VLGRPOTrainer_2 as Qwen2_5_VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
# from peft import LoraConfig, TaskType

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'structural'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["structural", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'structural'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

CORE_VARIABLES = {sympy.Symbol('x'), sympy.Symbol('v'), sympy.Symbol('t')}

def get_structural_skeleton(expr: sympy.Expr) -> sympy.Expr:

    if expr in CORE_VARIABLES:
        return expr

    if isinstance(expr, sympy.Symbol):
        return sympy.Integer(1)
        
    if isinstance(expr, sympy.Number):
        return sympy.Integer(sympy.sign(expr))

    if isinstance(expr, sympy.Expr) and expr.args:
        new_args = [get_structural_skeleton(arg) for arg in expr.args]
        return expr.func(*new_args)
        
    return expr 



from typing import Dict, Any, List, Set

def _get_structural_term_set(expr: sympy.Expr) -> Set[sympy.Expr]:

    terms = expr.args if expr.is_Add else (expr,)
    
    structural_terms = {get_structural_skeleton(term) for term in terms}
    return structural_terms


def preprocess_formula(formula_str):
    # np.sin(x) -> sin(x)
    formula_str = re.sub(r'np\.sin', 'sin', formula_str)
    formula_str = re.sub(r'np\.cos', 'cos', formula_str)
    # np.random.normal(0,1) -> noise
    formula_str = re.sub(r'np\.random\.normal\(0,1\)', 'noise', formula_str)
    return formula_str

def structural_reward(completions: List[List[Dict[str, Any]]], solution: List[str], **kwargs) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")


    PARAM_SYMBOLS = [
        'x', 'v', 't', 'k', 'c', 'F', 'omega', 'beta', 'delta', 'alpha', 'eta', 'gamma', 'sigma', 'noise'
    ]
    local_dict = {name: sympy.Symbol(name) for name in PARAM_SYMBOLS}
    local_dict.update({
        'sin': sympy.sin,
        'cos': sympy.cos,
    })

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            content_match = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)

            if content_match and sol_match:
                # gen_formula_str = content_match[1]
                # gt_formula_str = sol_match.group(1).strip()
                gen_formula_str = preprocess_formula(content_match[1])
                gt_formula_str = preprocess_formula(sol_match.group(1).strip())

                gen_expr = sympy.parse_expr(gen_formula_str, local_dict=local_dict, evaluate=True)
                gt_expr = sympy.parse_expr(gt_formula_str, local_dict=local_dict, evaluate=True)

                gen_term_set = _get_structural_term_set(gen_expr)
                gt_term_set = _get_structural_term_set(gt_expr)
                
                intersection = gen_term_set.intersection(gt_term_set)
                union = gen_term_set.union(gt_term_set)


                if not union:
                    reward = 1.0
                else:
                    reward = len(intersection) / len(union)

        except Exception as e:
            reward = 0.0
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Partial Structural reward: {reward:.4f} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "structural": structural_reward,
}

PHYSICS_SYSTEM_PROMPT = (
    "The user provides visual and trajectory data of a physical phenomenon. The Assistant's task is to act as a physicist. "
    "First, think step-by-step about the underlying physical principles in <think> tags. "
    "Then, derive and state the final governing equation in <answer> tags. "
    "The equation should use symbolic placeholders for unknown parameters (e.g., k, c, F) "
    "and standard variables for the system (x, v, t)."
)

PHYSICS_QUESTION_TEMPLATE = (
    "{Question} Based on the provided data, derive the governing equation for the system. "
    "Output your reasoning in <think></think> and the final symbolic formula in <answer></answer> tags."
)



def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(
       script_args.dataset_name,  # "json"
       data_files=script_args.dataset_config  # 文件路径
    )


    def make_conversation_physics_stage3(example):

        return {
            "prompt": [

                {"role": "system", "content": [{"type": "text", "text": PHYSICS_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"}, 
                        {"type": "text", "text": PHYSICS_QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
            "solution": example.get("solution", "") 
        }

    # print("Using physics formula discovery prompt for 2-image input.")

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PHYSICS_QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    dataset = dataset.map(make_conversation_physics_stage3)
    # dataset = dataset.map(make_conversation_image)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    
    trainer_cls = Qwen2_5_VLGRPOTrainer if not training_args.use_vllm else Qwen2_5_VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)