#!/usr/bin/env python3
"""
W8A8 INT8 Quantization for GEC model using llm-compressor.
Uses stratified calibration data matching SFT training distribution.

W8A8 uses SmoothQuant + GPTQ for 8-bit weights and activations.
Optimized for throughput with multiple concurrent users (uses INT8 tensor cores).

Usage:
    cd quantize_awq
    uv sync
    uv run python quantize.py --model-path ../merged1.2b_models/merged-step-225
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor import oneshot


# System prompts (must match SFT training)
NEW_SYSTEM_PROMPT = "You are a writing assistant that helps improve text. Follow the user's instruction exactly. Output only the revised text."
PROFESSIONAL_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more professional and formal while retaining its essential content."
CONCISE_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more concise while preserving its core meaning."
FRIENDLY_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more friendly and approachable while maintaining its main points."


def transform_grammar_conversation(example):
    """Transform grammar dataset - just replace system prompt."""
    messages = example["messages"]
    transformed_messages = []

    for msg in messages:
        if msg["role"] == "system":
            transformed_messages.append({"role": "system", "content": NEW_SYSTEM_PROMPT})
        else:
            transformed_messages.append(msg)

    return {"messages": transformed_messages}


def transform_rewriting_conversation(example):
    """Transform explore-instruct-rewriting dataset."""
    messages = example["messages"]
    transformed_messages = []

    has_system = any(msg["role"] == "system" for msg in messages)
    if not has_system:
        transformed_messages.append({"role": "system", "content": NEW_SYSTEM_PROMPT})

    for msg in messages:
        if msg["role"] == "system":
            transformed_messages.append({"role": "system", "content": NEW_SYSTEM_PROMPT})
        elif msg["role"] == "user":
            content = msg["content"]
            if "\n" in content:
                split_pos = content.index("\n")
                prefix = content[:split_pos].rstrip(".")
                text = content[split_pos + 1:]
                colon = "" if prefix.endswith(":") else ":"
                new_content = f"{prefix}{colon}\n<Text>\n{text}\n</Text>"
            else:
                prefix = content.rstrip(".")
                colon = "" if prefix.endswith(":") else ":"
                new_content = f"{prefix}{colon}\n<Text>\n</Text>"
            transformed_messages.append({"role": "user", "content": new_content})
        else:
            transformed_messages.append(msg)

    return {"messages": transformed_messages}


def transform_rewrite_conversation(example):
    """Transform smol-rewrite dataset."""
    messages = example["messages"]
    transformed_messages = []

    system_content = None
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break

    instruction = system_content.replace("You're an AI assistant for text re-writing. ", "")

    for msg in messages:
        if msg["role"] == "system":
            transformed_messages.append({"role": "system", "content": NEW_SYSTEM_PROMPT})
        elif msg["role"] == "user":
            inst = instruction.rstrip(".")
            colon = "" if inst.endswith(":") else ":"
            new_content = f"{inst}{colon}\n<Text>\n{msg['content']}\n</Text>"
            transformed_messages.append({"role": "user", "content": new_content})
        else:
            transformed_messages.append(msg)

    return {"messages": transformed_messages}


def filter_by_system_prompt(example, target_prompt):
    """Filter dataset by system prompt content."""
    messages = example["messages"]
    for msg in messages:
        if msg["role"] == "system" and msg["content"] == target_prompt:
            return True
    return False


def create_stratified_calibration_dataset(
    num_samples: int = 512,
    seed: int = 42,
    hf_token: str | None = None,
):
    """
    Create stratified calibration dataset matching SFT training distribution.

    Distribution:
    - 60% grammar (typix-hq-grannar)
    - 40% rewriting:
      - 70% smol-rewrite (professional 50%, concise 35%, friendly 15%)
      - 30% explore-instruct-rewriting
    """
    print("Loading datasets...")

    # Load datasets
    dataset_grammar = load_dataset("moogin/typix-hq-grannar", token=hf_token)
    dataset_rewriting = load_dataset("HuggingFaceTB/smoltalk", "explore-instruct-rewriting")
    dataset_rewrite = load_dataset("HuggingFaceTB/smoltalk", "smol-rewrite")

    # Calculate sample sizes for stratified split
    grammar_samples = int(num_samples * 0.60)  # 60%
    rewrite_total = num_samples - grammar_samples  # 40%

    smol_rewrite_samples = int(rewrite_total * 0.70)  # 70% of rewriting
    explore_samples = rewrite_total - smol_rewrite_samples  # 30% of rewriting

    professional_samples = int(smol_rewrite_samples * 0.50)  # 50%
    concise_samples = int(smol_rewrite_samples * 0.35)  # 35%
    friendly_samples = smol_rewrite_samples - professional_samples - concise_samples  # 15%

    print(f"Stratified distribution for {num_samples} samples:")
    print(f"  Grammar: {grammar_samples}")
    print(f"  Professional: {professional_samples}")
    print(f"  Concise: {concise_samples}")
    print(f"  Friendly: {friendly_samples}")
    print(f"  Explore-rewriting: {explore_samples}")

    # Transform and sample grammar
    print("Processing grammar dataset...")
    grammar_data = dataset_grammar["train"].map(transform_grammar_conversation)
    grammar_sampled = grammar_data.shuffle(seed=seed).select(
        range(min(grammar_samples, len(grammar_data)))
    )

    # Transform and sample explore-instruct-rewriting
    print("Processing explore-instruct-rewriting dataset...")
    rewriting_data = dataset_rewriting["train"].map(transform_rewriting_conversation)
    rewriting_sampled = rewriting_data.shuffle(seed=seed).select(
        range(min(explore_samples, len(rewriting_data)))
    )

    # Filter and transform smol-rewrite by style
    print("Processing smol-rewrite datasets...")

    professional_data = dataset_rewrite["train"].filter(
        lambda x: filter_by_system_prompt(x, PROFESSIONAL_PROMPT)
    ).map(transform_rewrite_conversation)
    professional_sampled = professional_data.shuffle(seed=seed).select(
        range(min(professional_samples, len(professional_data)))
    )

    concise_data = dataset_rewrite["train"].filter(
        lambda x: filter_by_system_prompt(x, CONCISE_PROMPT)
    ).map(transform_rewrite_conversation)
    concise_sampled = concise_data.shuffle(seed=seed).select(
        range(min(concise_samples, len(concise_data)))
    )

    friendly_data = dataset_rewrite["train"].filter(
        lambda x: filter_by_system_prompt(x, FRIENDLY_PROMPT)
    ).map(transform_rewrite_conversation)
    friendly_sampled = friendly_data.shuffle(seed=seed).select(
        range(min(friendly_samples, len(friendly_data)))
    )

    # Combine all
    print("Combining datasets...")
    combined = concatenate_datasets([
        grammar_sampled,
        professional_sampled,
        concise_sampled,
        friendly_sampled,
        rewriting_sampled,
    ]).shuffle(seed=seed)

    print(f"Total calibration samples: {len(combined)}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="W8A8 INT8 quantization for GEC model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../merged1.2b_models/merged-step-225",
        help="Path to the model to quantize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {model_path}-W8A8)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=768,
        help="Maximum sequence length for calibration",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for private datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    output_dir = args.output_dir or f"{model_path}-W8A8"

    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Calibration samples: {args.num_samples}")
    print(f"Max sequence length: {args.max_seq_length}")
    print()

    # Step 1: Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Step 2: Create stratified calibration dataset
    print("\nCreating calibration dataset...")
    calibration_data = create_stratified_calibration_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        hf_token=args.hf_token,
    )

    # Step 3: Tokenize using chat template
    print("\nTokenizing calibration data...")

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer(
            text,
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
        )

    calibration_data = calibration_data.map(
        tokenize,
        remove_columns=calibration_data.column_names,
    )

    # Step 4: Configure W8A8 quantization recipe
    print("\nConfiguring W8A8 INT8 quantization (SmoothQuant + GPTQ) with LFM2 mappings...")
    # Only smooth attention layers (conv layers have dimension mismatch issues)
    sq_mappings = []
    attn_layers = {2, 5, 8, 10, 12, 14}
    for i in attn_layers:
        prefix = f"model.layers.{i}"
        sq_mappings.append([
            [f"{prefix}.self_attn.q_proj", f"{prefix}.self_attn.k_proj", f"{prefix}.self_attn.v_proj"],
            f"{prefix}.operator_norm",
        ])
    # Add FFN mappings for all layers
    for i in range(16):
        prefix = f"model.layers.{i}"
        sq_mappings.append([
            [f"{prefix}.feed_forward.w1", f"{prefix}.feed_forward.w3"],
            f"{prefix}.ffn_norm",
        ])
    print(f"  Created {len(sq_mappings)} SmoothQuant mappings (attn + FFN only, skipping conv)")

    # Ignore conv layers - vLLM's LFM2 loader can't handle quantized conv weights
    ignore_layers = ["lm_head", "re:.*conv.*"]

    recipe = [
        SmoothQuantModifier(
            smoothing_strength=0.8,
            mappings=sq_mappings,
        ),
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=ignore_layers,
        ),
    ]
    print(f"  Ignoring: {ignore_layers} (conv layers stay FP16 for vLLM compatibility)")

    # Step 5: Apply quantization
    print("\nApplying W8A8 quantization (this may take a while)...")
    oneshot(
        model=model,
        dataset=calibration_data,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
        output_dir=output_dir,
    )

    # Step 6: Patch config.json for vLLM compatibility
    print("\nPatching config.json for vLLM compatibility...")
    config_path = Path(output_dir) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove scale_dtype and zp_dtype fields that vLLM doesn't recognize
    qc = config.get("quantization_config", {})
    patched = False
    for group_name, group in qc.get("config_groups", {}).items():
        for section in ["input_activations", "weights"]:
            if group.get(section):
                if "scale_dtype" in group[section]:
                    del group[section]["scale_dtype"]
                    patched = True
                if "zp_dtype" in group[section]:
                    del group[section]["zp_dtype"]
                    patched = True

    if patched:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("  Removed scale_dtype/zp_dtype fields for vLLM 0.13.0 compatibility")
    else:
        print("  No patching needed")

    print("\nDone!")
    print(f"Quantized model saved to: {output_dir}")


if __name__ == "__main__":
    main()
