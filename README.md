# PRISM-R1: Mimicking the Physicist's Eye

<div align="center">

### A VLM-centric Approach for Physics Formula Discovery

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://prism-r1.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**📄 Paper**](https://arxiv.org/abs/2025.XXXXX) | [**🌐 Project Page**](https://prism-r1.github.io/) | [**🤗 Dataset**](#-dataset) | [**💻 Code**](#-code)

</div>

## 📖 Abstract

Automated discovery of physical laws from observational data is a grand challenge in AI. Current methods, relying on symbolic regression or LLMs, are limited to uni-modal data and overlook the rich, visual phenomenological representations of motion that are indispensable to physicists. This "sensory deprivation" severely weakens their ability to interpret the inherent spatio-temporal patterns within dynamic phenomena.

To address this gap, we propose the VLM-based **Physics Reasoning and Inductive Synthesis Model (PRISM-R1)**, a multimodal framework that discovers fundamental symbolic formulas by methodically integrating visual perception, trajectory data, and symbolic reasoning through a process of inductive synthesis.

The model is trained via a curriculum of Motion Structure Induction (MSI), using supervised fine-tuning to interpret kinematic phase portraits and construct hypotheses guided by a Causal Chain of Thought (C-CoT), followed by Reward-Guided Symbolic Calibration (RGSC) to purify the formula's structure with reinforcement learning. During inference, the trained PRISM acts as an agent: it first posits a high-confidence symbolic ansatz, then proactively invokes an external symbolic regression tool to perform Symbolic Residual Realignment (SR²). This final step, analogous to a physicist's perturbation analysis, reconciles the theoretical model with empirical data.

To support this research, we introduce PhysSymbol, a new 5,000-instance multimodal corpus. Experiments show that PRISM-R1 consistently outperforms state-of-the-art VLM baselines in accuracy and interpretability, enabling more precise discovery of physical laws.

## 👥 Authors

**Jiaqi Liu**¹'³, **Songning Lai**², **Pengze Li**³'⁴, **Di Yu**³'⁵, **Wenjie Zhou**⁶'⁹, **Yiyang Zhou**¹, **Peng Xia**¹, **Zijun Wang**⁷, **Xi Chen**⁴, **Shixiang Tang**³, **Lei Bai**³, **Wanli Ouyang**³'⁸, **Mingyu Ding**¹, **Huaxiu Yao**¹, **Aoran Wang**³

¹ UNC-Chapel Hill  
² The Hong Kong University of Science and Technology (Guangzhou)  
³ Shanghai Artificial Intelligence Laboratory  
⁴ Fudan University  
⁵ Tsinghua University  
⁶ Nankai University  
⁷ UC Santa Cruz  
⁸ The Chinese University of Hong Kong  
⁹ Shanghai Innovation Institute  

**Contact**: jqliu@cs.unc.edu, wangaoran@pjlab.org.cn

## 🔥 Highlights

- **🎯 Novel Approach**: First VLM-based framework for physics formula discovery that integrates visual perception with symbolic reasoning
- **🏆 SOTA Performance**: 56.7% improvement in structural score and 45.4% improvement in accuracy over best baselines
- **🧠 Multi-Stage Training**: Motion Structure Induction (MSI) + Reward-Guided Symbolic Calibration (RGSC) pipeline
- **🤖 Agentic Design**: Symbolic Residual Realignment (SR²) with external tool integration
- **📊 New Benchmark**: PhysSymbol dataset with 5,000 multimodal instances

## 🚀 Key Results

| Metric | PRISM-R1 | Best Baseline | Improvement |
|--------|----------|---------------|-------------|
| Structural Score | **0.812** | 0.518 | **+56.7%** |
| Accuracy Score | **0.487** | 0.335 | **+45.4%** |
| Post-SR² MSE | **0.032** | 0.091 | **3× lower** |

## 🏗️ Framework Overview

<div align="center">
<img src="./docs/assets/images/Fig_overall.png" width="90%" alt="PRISM-R1 Framework Overview">
</div>

PRISM-R1 consists of three main stages:

1. **Motion Structure Induction (MSI)**: Two-step supervised fine-tuning for visual interpretation and hypothesis construction
2. **Reward-Guided Symbolic Calibration (RGSC)**: Reinforcement learning for formula structure refinement
3. **Symbolic Residual Realignment (SR²)**: Agentic tool use for empirical-theoretical reconciliation



## 🔜 Todo List

- [ ] **📝 Paper Release**: ArXiv publication (Expected: Q1 2025)
- [ ] **💻 Code Release**: Complete training and inference pipeline
  - [ ] MSI training scripts
  - [ ] RGSC implementation with GRPO
  - [ ] SR² agentic framework
  - [ ] Evaluation benchmarks
- [ ] **📊 Dataset Release**: PhysSymbol multimodal corpus
  - [ ] 5,000 annotated instances
  - [ ] Phase portrait visualizations
  - [ ] Trajectory data and symbolic equations
  - [ ] Data loading and preprocessing tools
- [ ] **🔧 Model Checkpoints**: Pre-trained PRISM-R1 weights
- [ ] **📚 Documentation**: Detailed usage tutorials and API docs
- [ ] **🧪 Demo**: Interactive online demonstration
- [ ] **🐳 Docker**: Containerized environment for easy setup

<!-- ## 🎯 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/PRISM-R1/PRISM-R1.git
cd PRISM-R1

# Install dependencies (Coming Soon)
pip install -r requirements.txt
```

### Training
```bash
# Stage 1: Motion Structure Induction
python src/training/train_msi.py --config configs/msi_config.yaml

# Stage 2: Reward-Guided Symbolic Calibration  
python src/training/train_rgsc.py --config configs/rgsc_config.yaml

# Stage 3: Full pipeline evaluation
python src/inference/evaluate.py --model_path checkpoints/prism_r1.pt
```

### Inference
```bash
# Run physics formula discovery on new data
python src/inference/discover.py \
    --image_path examples/phase_portrait.png \
    --trajectory_data examples/trajectory.csv \
    --output_dir results/
``` -->

## 📊 PhysSymbol Dataset

The PhysSymbol dataset contains 5,000 multimodal instances for physics formula discovery:

- **Visual Data**: Kinematic phase portraits (velocity vs. position)
- **Trajectory Data**: Time series of position, velocity, and acceleration
- **Symbolic Ground Truth**: Mathematical equations governing the dynamics
- **Reasoning Chains**: Causal Chain of Thought (C-CoT) explanations

### Dataset Statistics
- **Physics Terms**: 11 different types (harmonic, damping, driving forces, etc.)
- **Complexity Levels**: From simple harmonic motion to complex multi-scale dynamics
- **Visualization Types**: Phase space and temporal trajectory plots

## 🏆 Experiments

### Main Results
PRISM-R1 demonstrates significant improvements over state-of-the-art VLMs:

- **Claude-4-Sonnet**: 0.518 → **0.812** structural score (+56.7%)
- **GPT-4V**: 0.335 → **0.487** accuracy score (+45.4%)
- **Final MSE**: 3× reduction in prediction error after SR²

### Ablation Studies
- MSI alone: +475% improvement over base model
- MSI + RGSC: +746% total improvement
- SR² refinement: Additional 3× error reduction

## 📄 Citation

If you find our work useful, please consider citing:

```bibtex
@article{liu2025prismr1,
  title={Mimicking the Physicist's Eye: A VLM-centric Approach for Physics Formula Discovery},
  author={Liu, Jiaqi and Lai, Songning and Li, Pengze and Yu, Di and Zhou, Wenjie and Zhou, Yiyang and Xia, Peng and Wang, Zijun and Chen, Xi and Tang, Shixiang and Bai, Lei and Ouyang, Wanli and Ding, Mingyu and Yao, Huaxiu and Wang, Aoran},
  journal={arXiv preprint arXiv:2025},
  year={2025}
}
```
