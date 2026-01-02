# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository fine-tunes LFM2-350M (Liquid AI's language model) for browser control using GRPO (Group Relative Policy Optimization) reinforcement learning. It uses BrowserGym environments via OpenEnv for training agents to navigate and interact with websites.

## Commands

### Fine-tuning
```bash
# Full fine-tune
make fine-tune config=lfm2_350m.yaml

# LoRA fine-tune (parameter efficient)
make fine-tune config=lfm2_350m_lora.yaml
```

### Evaluation
```bash
make evaluation
```

### Development
```bash
# Install dependencies (uses uv)
uv sync

# Lint
uv run ruff check .
uv run ruff format .
```

## Architecture

The training framework has three components:
1. **GRPOTrainer** (trl library) - Runs on GPU for GRPO optimization
2. **vLLM server** - Generates rollouts with LFM2-350M (colocated on same GPU)
3. **BrowserGym environment** - Runs as a Hugging Face Space (CPU-based Docker container)

### Key Files
- `src/browser_control/fine_tune.py` - Main training loop with Modal GPU deployment
- `src/browser_control/config.py` - `FineTuningConfig` pydantic model for all hyperparameters
- `src/browser_control/modal_infra.py` - Modal serverless GPU setup
- `configs/*.yaml` - Training configurations (model, learning rate, LoRA settings, etc.)

### Training Flow
1. `rollout_func()` executes episodes using the language model
2. For each step: model observes accessibility tree (axtree) → generates action → environment returns reward
3. GRPO uses relative performance within rollout groups to update policy

### Config Structure
YAML configs in `configs/` control:
- Model selection (`model_name`)
- LoRA parameters (`use_peft`, `lora_r`, `lora_alpha`, `lora_target_modules`)
- vLLM settings (`use_vllm`, `vllm_mode`, `vllm_gpu_memory_utilization`)
- Training hyperparameters (`learning_rate`, `num_generations`, `max_steps`)

## Infrastructure

Uses Modal for serverless GPU (A100). Requires:
- `wandb-secret` Modal secret for W&B logging
- HuggingFace Space running BrowserGym at URL specified in config

## Subproject: greco/

Separate NLP project for Grammatical Error Correction quality estimation. Has its own dependencies in `greco/requirements.txt` (Python 2 required for m2scorer).
