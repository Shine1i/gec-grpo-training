# W8A8 INT8 Quantization for LFM2 GEC Model

Quantizes LFM2-based GEC models using SmoothQuant + GPTQ for W8A8 (8-bit weights, 8-bit activations).

Optimized for throughput with multiple concurrent users on RTX 3090 (uses INT8 tensor cores).

## Setup

```bash
cd quantize_awq
uv sync
```

## Usage

```bash
# Default model path
uv run python quantize.py

# Custom model path
uv run python quantize.py --model-path /path/to/model

# Custom output directory
uv run python quantize.py --model-path /path/to/model --output-dir /path/to/output
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `../merged1.2b_models/merged-step-225` | Path to model to quantize |
| `--output-dir` | `{model_path}-W8A8` | Output directory |
| `--num-samples` | `512` | Calibration samples |
| `--max-seq-length` | `768` | Max sequence length |
| `--hf-token` | None | HuggingFace token for private datasets |
| `--seed` | `42` | Random seed |

## Notes

- Conv layers are kept in FP16 for vLLM compatibility (vLLM's LFM2 loader doesn't support quantized conv weights)
- Attention and FFN layers are quantized to INT8
- Config is auto-patched for vLLM 0.13.0 compatibility
