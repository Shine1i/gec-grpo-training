import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gec.config import GECConfig
from src.gec.dataset import load_gec_dataset, make_gec_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview model outputs for a few prompts."
    )
    parser.add_argument("--config-file-name", required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = GECConfig.from_yaml(file_name=args.config_file_name)

    dataset = load_gec_dataset(config.dataset_name, None)
    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    sampled = dataset.select(indices)

    sources = sampled["prompt"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=dtype
        ).to(device)
    model.eval()

    prompts = [
        tokenizer.apply_chat_template(
            make_gec_messages(src, config.system_prompt),
            add_generation_prompt=True,
            tokenize=False,
        )
        for src in sources
    ]

    completions = []
    for prompt_batch in batched(prompts, args.gen_batch_size):
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}
        input_lens = encoded["attention_mask"].sum(dim=1)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else 0,
                max_new_tokens=args.max_new_tokens or config.max_completion_length,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_lens = input_lens.repeat_interleave(args.num_generations)
        for seq, in_len in zip(generated, input_lens, strict=True):
            completion_ids = seq[int(in_len) :]
            completions.append(
                tokenizer.decode(completion_ids, skip_special_tokens=True)
            )

    for i in range(len(sources)):
        print(f"\n=== Output {i + 1} ===")
        base = i * args.num_generations
        for j in range(args.num_generations):
            print(completions[base + j])


if __name__ == "__main__":
    main()
