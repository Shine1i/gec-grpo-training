# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEC (Grammatical Error Correction) fine-tuning using GRPO (Group Relative Policy Optimization) with neural rewards. Trains LFM2-700M to correct grammar using a composite reward signal.

## Commands

```bash
# Install dependencies
uv sync

# Test reward models locally
make test

# Local training smoke test (RTX 3090 / 24GB)
make local config=gec_typix_700m.yaml

# Modal deployment (A100)
make fine-tune config=gec_typix_700m.yaml
```

## Architecture

### Composite Reward Function
```
Reward = 0.6 × GRECO_GAIN + 0.3 × MPNet - 0.1 × HasEdit
```

- **GRECO_GAIN (60%)**: `GRECO(output) - GRECO(input)` - rewards improvement, not absolute quality
- **MPNet (30%)**: Semantic similarity preservation (`all-mpnet-base-v2`)
- **HasEdit (10%)**: Binary penalty (1 if edited, 0 if unchanged) - discourages unnecessary edits

### Key Files
- `src/gec/fine_tune.py` - GRPO training with TRL
- `src/gec/rewards.py` - Composite reward: `GECRewardModel`, `build_gec_reward_func`
- `src/gec/dataset.py` - Load HF dataset, extract `<Text>` tags
- `src/gec/config.py` - `GECConfig` pydantic model
- `configs/gec_typix_700m.yaml` - Training config

### Training Flow
1. Load dataset from HuggingFace (`moogin/typix-hq-grannar`)
2. GRPOTrainer generates N corrections per input
3. Reward function scores each with GRECO + MPNet + laziness
4. GRPO uses relative performance to update policy

## Infrastructure

- **Local**: `--local` flag for smoke testing on consumer GPU
- **Modal**: Serverless A100 deployment, requires `wandb-secret`

## Dependencies

- `transformers>=4.55.0` (required for LFM2)
- `trl>=0.25.1` (GRPO implementation)
- `sentence-transformers` (MPNet)
- GRECO loaded from `greco/models.py` with `use_fast=False`
