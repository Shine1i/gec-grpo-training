import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gec.config import GECConfig
from src.gec.dataset import load_gec_dataset, make_gec_messages
from src.gec.rewards import GECRewardModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune gain_epsilon by analyzing reward distributions."
    )
    parser.add_argument("--config-file-name", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--epsilons", type=str, default="0.01,0.02,0.03")
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--reward-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def collect_greco_scores(reward_model, sources, hypotheses, batch_size: int):
    scores = []
    for src_batch, hyp_batch in zip(
        batched(sources, batch_size), batched(hypotheses, batch_size), strict=True
    ):
        scores.append(reward_model.compute_greco_scores(src_batch, hyp_batch))
    return torch.cat(scores, dim=0)


def collect_source_greco_scores(reward_model, sources, batch_size: int):
    scores = []
    for src_batch in batched(sources, batch_size):
        scores.append(reward_model.compute_source_greco_scores(src_batch))
    return torch.cat(scores, dim=0)


def collect_semantic_scores(reward_model, sources, hypotheses, batch_size: int):
    scores = []
    for src_batch, hyp_batch in zip(
        batched(sources, batch_size), batched(hypotheses, batch_size), strict=True
    ):
        scores.append(reward_model.compute_semantic_similarity(src_batch, hyp_batch))
    return torch.cat(scores, dim=0)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = GECConfig.from_yaml(file_name=args.config_file_name)
    num_generations = args.num_generations or config.num_generations

    dataset = load_gec_dataset(config.dataset_name, None)
    if args.num_samples and len(dataset) > args.num_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    sources = dataset["prompt"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    repeated_sources = []

    max_new_tokens = args.max_new_tokens or config.max_completion_length

    for prompt_batch, source_batch in zip(
        batched(prompts, args.gen_batch_size),
        batched(sources, args.gen_batch_size),
        strict=True,
    ):
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        input_lens = encoded["attention_mask"].sum(dim=1)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else 0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_generations,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_lens = input_lens.repeat_interleave(num_generations)
        for seq, in_len in zip(generated, input_lens, strict=True):
            completion_ids = seq[int(in_len) :]
            completions.append(
                tokenizer.decode(completion_ids, skip_special_tokens=True)
            )

        repeated_sources.extend(
            [src for src in source_batch for _ in range(num_generations)]
        )

    reward_model = GECRewardModel(
        greco_model_name=config.greco_model_name,
        mpnet_model=config.mpnet_model,
        greco_weight=config.greco_weight,
        semantic_weight=config.semantic_weight,
        laziness_weight=config.laziness_weight,
        gain_epsilon=config.gain_epsilon,
    )

    unique_sources = list(dict.fromkeys(repeated_sources))
    unique_scores = collect_source_greco_scores(
        reward_model, unique_sources, args.reward_batch_size
    )
    source_greco_map = {
        source: score.item()
        for source, score in zip(unique_sources, unique_scores, strict=True)
    }
    source_greco = torch.tensor([source_greco_map[src] for src in repeated_sources])

    hyp_greco = collect_greco_scores(
        reward_model, repeated_sources, completions, args.reward_batch_size
    )
    greco_gain = hyp_greco - source_greco

    semantic_scores = collect_semantic_scores(
        reward_model, repeated_sources, completions, args.reward_batch_size
    )
    edit_penalties = reward_model.compute_edit_penalty(repeated_sources, completions)

    epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    num_prompts = len(sources)

    for epsilon in epsilons:
        improved = greco_gain > epsilon
        effective_gain = torch.where(improved, greco_gain, torch.zeros_like(greco_gain))
        non_improving_edits = (edit_penalties > 0) & (~improved)
        semantic_effective = torch.where(
            non_improving_edits,
            torch.zeros_like(semantic_scores),
            semantic_scores,
        )
        conditional_penalties = torch.where(
            non_improving_edits, edit_penalties, torch.zeros_like(edit_penalties)
        )

        rewards = (
            config.greco_weight * effective_gain
            + config.semantic_weight * semantic_effective
            - config.laziness_weight * conditional_penalties
        )

        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        frac_improved = improved.float().mean().item()
        frac_edits = edit_penalties.float().mean().item()
        frac_non_improving_edits = non_improving_edits.float().mean().item()
        frac_edited_positive = ((rewards > 0) & (edit_penalties > 0)).float().mean().item()

        grouped_rewards = rewards.view(num_prompts, num_generations)
        grouped_edits = edit_penalties.view(num_prompts, num_generations)
        has_copy = (grouped_edits == 0).any(dim=1)
        copy_rewards = grouped_rewards.clone()
        copy_rewards[grouped_edits > 0] = -float("inf")
        best_copy = copy_rewards.max(dim=1).values
        best_any = grouped_rewards.max(dim=1).values
        copy_best = (has_copy & (best_copy >= best_any - 1e-6)).float().mean().item()

        q10, q50, q90 = torch.quantile(rewards, torch.tensor([0.1, 0.5, 0.9])).tolist()

        print(f"\nEpsilon: {epsilon:.4f}")
        print(f"  reward_mean: {reward_mean:.4f} | reward_std: {reward_std:.4f}")
        print(f"  improved_frac: {frac_improved:.4f}")
        print(f"  edit_frac: {frac_edits:.4f}")
        print(f"  non_improving_edit_frac: {frac_non_improving_edits:.4f}")
        print(f"  edited_positive_frac: {frac_edited_positive:.4f}")
        print(f"  copy_best_frac: {copy_best:.4f}")
        print(f"  reward_quantiles: p10={q10:.4f} p50={q50:.4f} p90={q90:.4f}")


if __name__ == "__main__":
    main()
