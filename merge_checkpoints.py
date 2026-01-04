#!/usr/bin/env python3
"""Merge LoRA checkpoints with base model for vLLM inference."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_checkpoint(base_model_name: str, checkpoint_path: Path, output_path: Path):
    """Load base model, merge LoRA adapter, save as float16."""
    print(f"Merging {checkpoint_path.name}...")

    # Load base model in float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep on CPU to avoid OOM
    )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model = model.merge_and_unload()

    # Save merged model
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    print(f"  Saved to {output_path}")

    # Free memory
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("modal_ckekpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("merged_models"),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="moogin/typix-mixed-epo",
    )
    args = parser.parse_args()

    # Find all checkpoints
    checkpoints = sorted(
        args.checkpoints_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}\n")

    for ckpt in checkpoints:
        step = ckpt.name.split("-")[1]
        output_path = args.output_dir / f"merged-step-{step}"

        if output_path.exists():
            print(f"Skipping {ckpt.name} (already merged)")
            continue

        merge_checkpoint(args.base_model, ckpt, output_path)

    print("\nDone! Merged models ready for vLLM.")


if __name__ == "__main__":
    main()
