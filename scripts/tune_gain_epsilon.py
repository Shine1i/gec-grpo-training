import argparse
import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gec.config import GECConfig
from src.gec.dataset import load_gec_dataset, make_gec_messages
from src.gec.rewards import GECRewardModel


def stratified_sample(
    ds: Dataset,
    size: int,
    clean_ratio: float = 0.14,
    seed: int = 42,
) -> Dataset:
    """
    Stratified sample with:
    - 14% clean samples
    - 50% wi_locness/fce combined, 50% lang8_replay by source
    """
    random.seed(seed)

    # Split by source and is_clean
    buckets = {
        ("wi_locness", True): [],
        ("wi_locness", False): [],
        ("fce", True): [],
        ("fce", False): [],
        ("lang8_replay", True): [],
        ("lang8_replay", False): [],
    }

    for i, x in enumerate(ds):
        source = x.get("source", "unknown")
        is_clean = x.get("is_clean", False)
        key = (source, is_clean)
        if key in buckets:
            buckets[key].append(i)

    # Shuffle all buckets
    for v in buckets.values():
        random.shuffle(v)

    # Calculate targets: 50% wi_locness+fce, 50% lang8_replay
    wi_fce_total = size // 2
    lang8_total = size - wi_fce_total

    # Within wi_fce: split evenly between wi_locness and fce
    wi_total = wi_fce_total // 2
    fce_total = wi_fce_total - wi_total

    # Clean ratio applied per source group
    total_clean = int(size * clean_ratio)
    wi_clean_target = int(total_clean * (wi_total / size))
    fce_clean_target = int(total_clean * (fce_total / size))
    lang8_clean_target = total_clean - wi_clean_target - fce_clean_target

    wi_error_target = wi_total - wi_clean_target
    fce_error_target = fce_total - fce_clean_target
    lang8_error_target = lang8_total - lang8_clean_target

    # Sample from each bucket
    selected = []
    selected += buckets[("wi_locness", True)][:wi_clean_target]
    selected += buckets[("wi_locness", False)][:wi_error_target]
    selected += buckets[("fce", True)][:fce_clean_target]
    selected += buckets[("fce", False)][:fce_error_target]
    selected += buckets[("lang8_replay", True)][:lang8_clean_target]
    selected += buckets[("lang8_replay", False)][:lang8_error_target]

    # Fill if we didn't get enough
    all_indices = set(range(len(ds)))
    remaining = list(all_indices - set(selected))
    random.shuffle(remaining)
    while len(selected) < size and remaining:
        selected.append(remaining.pop())

    random.shuffle(selected)

    # Log distribution
    final_clean = sum(1 for i in selected if ds[i].get("is_clean"))
    final_wi = sum(1 for i in selected if ds[i].get("source") == "wi_locness")
    final_fce = sum(1 for i in selected if ds[i].get("source") == "fce")
    final_lang8 = sum(1 for i in selected if ds[i].get("source") == "lang8_replay")
    print(f"Stratified sample: {len(selected)} total")
    print(f"  WI-LOCNESS: {final_wi} ({100*final_wi/len(selected):.1f}%)")
    print(f"  FCE: {final_fce} ({100*final_fce/len(selected):.1f}%)")
    print(f"  LANG8: {final_lang8} ({100*final_lang8/len(selected):.1f}%)")
    print(f"  Clean: {final_clean} ({100*final_clean/len(selected):.1f}%)")

    return ds.select(selected)


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
    parser.add_argument(
        "--greedy-per-prompt",
        type=int,
        default=1,
        help="Number of greedy candidates per prompt (0 or 1).",
    )
    parser.add_argument("--print-clean-samples", type=int, default=0)
    parser.add_argument("--print-dirty-samples", type=int, default=0)
    parser.add_argument("--use-clean-fields", action="store_true")
    parser.add_argument("--dirty-penalty-scale", type=float, default=0.2)
    parser.add_argument("--no-edit-penalty", type=float, default=0.05)
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
    if args.greedy_per_prompt < 0 or args.greedy_per_prompt > num_generations:
        raise ValueError("greedy_per_prompt must be between 0 and num_generations.")
    if args.greedy_per_prompt > 1:
        raise ValueError("greedy_per_prompt > 1 is not supported for greedy decoding.")
    sampled_per_prompt = num_generations - args.greedy_per_prompt

    dataset = load_gec_dataset(config.dataset_name, None)
    if args.num_samples and len(dataset) > args.num_samples:
        dataset = stratified_sample(dataset, args.num_samples, seed=args.seed)

    sources = dataset["prompt"]
    source_domains = dataset["source"] if "source" in dataset.column_names else None
    is_clean = dataset["is_clean"] if "is_clean" in dataset.column_names else None
    applied_edits = (
        dataset["applied_edits"] if "applied_edits" in dataset.column_names else None
    )
    if args.use_clean_fields and (is_clean is None or applied_edits is None):
        raise ValueError(
            "use_clean_fields requires dataset columns 'is_clean' and 'applied_edits'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, dtype=dtype
        ).to(device)
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
    repeated_sources = []
    repeated_is_clean = []
    repeated_applied_edits = []

    max_new_tokens = args.max_new_tokens or config.max_completion_length

    source_offset = 0
    for prompt_batch, source_batch in zip(
        batched(prompts, args.gen_batch_size),
        batched(sources, args.gen_batch_size),
        strict=True,
    ):
        if is_clean is not None:
            batch_is_clean = is_clean[source_offset : source_offset + len(source_batch)]
        else:
            batch_is_clean = None
        if applied_edits is not None:
            batch_applied_edits = applied_edits[
                source_offset : source_offset + len(source_batch)
            ]
        else:
            batch_applied_edits = None

        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}
        input_lens = encoded["attention_mask"].sum(dim=1)

        batch_size = len(source_batch)
        batch_completions = [[] for _ in range(batch_size)]

        with torch.no_grad():
            if args.greedy_per_prompt > 0:
                greedy_out = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for idx, (seq, in_len) in enumerate(
                    zip(greedy_out, input_lens, strict=True)
                ):
                    completion_ids = seq[int(in_len) :]
                    batch_completions[idx].append(
                        tokenizer.decode(completion_ids, skip_special_tokens=True)
                    )

            if sampled_per_prompt > 0:
                sampled_out = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k if args.top_k > 0 else 0,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=sampled_per_prompt,
                    pad_token_id=tokenizer.eos_token_id,
                )
                sample_input_lens = input_lens.repeat_interleave(sampled_per_prompt)
                for j, (seq, in_len) in enumerate(
                    zip(sampled_out, sample_input_lens, strict=True)
                ):
                    completion_ids = seq[int(in_len) :]
                    prompt_idx = j // sampled_per_prompt
                    batch_completions[prompt_idx].append(
                        tokenizer.decode(completion_ids, skip_special_tokens=True)
                    )

        for idx, items in enumerate(batch_completions):
            if len(items) != num_generations:
                raise ValueError(
                    "Incorrect number of completions for prompt "
                    f"{source_batch[idx]!r}: {len(items)} vs {num_generations}"
                )

        for idx, src in enumerate(source_batch):
            completions.extend(batch_completions[idx])
            repeated_sources.extend([src] * num_generations)
            if is_clean is not None:
                repeated_is_clean.extend([bool(batch_is_clean[idx])] * num_generations)
            if applied_edits is not None:
                repeated_applied_edits.extend(
                    [int(batch_applied_edits[idx])] * num_generations
                )
        source_offset += len(source_batch)

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
    if is_clean is not None:
        is_clean_tensor = torch.tensor(repeated_is_clean, dtype=torch.bool)
        is_clean_prompt = torch.tensor(is_clean, dtype=torch.bool)
    else:
        is_clean_tensor = None
        is_clean_prompt = None
    if applied_edits is not None:
        applied_edits_tensor = torch.tensor(repeated_applied_edits)
    else:
        applied_edits_tensor = None

    epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    num_prompts = len(sources)

    def _pick_samples(mask, count, rng):
        indices = [i for i, flag in enumerate(mask) if flag]
        if count <= 0 or not indices:
            return []
        if count >= len(indices):
            return indices
        return rng.sample(indices, count)

    def _print_samples(label, sample_indices):
        if not sample_indices:
            return
        print(f"\nSample {label} prompts ({len(sample_indices)}):")
        for idx in sample_indices:
            meta = []
            if is_clean is not None:
                meta.append(f"is_clean={bool(is_clean[idx])}")
            if applied_edits is not None:
                meta.append(f"applied_edits={int(applied_edits[idx])}")
            if source_domains is not None:
                meta.append(f"source={source_domains[idx]}")
            meta_str = ", ".join(meta)
            print(f"\n[{label} idx {idx}] {meta_str}")
            print(f"Input: {sources[idx]}")
            base = idx * num_generations
            for j in range(num_generations):
                pos = base + j
                completion = completions[pos]
                edit_flag = "COPY" if edit_penalties[pos].item() == 0 else "EDIT"
                print(
                    f"  - {edit_flag} | gain={greco_gain[pos].item():.4f} "
                    f"sem={semantic_scores[pos].item():.4f} "
                    f"hyp_greco={hyp_greco[pos].item():.4f}"
                )
                print(f"    {completion}")

    if args.print_clean_samples or args.print_dirty_samples:
        if is_clean is None:
            raise ValueError("print sample flags require dataset column 'is_clean'.")
        rng = random.Random(args.seed + 1)
        clean_indices = _pick_samples(is_clean, args.print_clean_samples, rng)
        dirty_indices = _pick_samples(
            [not flag for flag in is_clean], args.print_dirty_samples, rng
        )
        _print_samples("clean", clean_indices)
        _print_samples("dirty", dirty_indices)

    def summarize(
        label: str,
        rewards: torch.Tensor,
        improved: torch.Tensor,
        non_improving_edits: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
    ) -> None:
        all_rewards = rewards
        all_improved = improved
        all_non_improving = non_improving_edits

        grouped_rewards = all_rewards.view(num_prompts, num_generations)
        grouped_edits = edit_penalties.view(num_prompts, num_generations)
        subset_edit_penalties = edit_penalties

        if prompt_mask is not None:
            completion_mask = prompt_mask.repeat_interleave(num_generations)
            rewards = all_rewards[completion_mask]
            improved = all_improved[completion_mask]
            non_improving_edits = all_non_improving[completion_mask]
            subset_edit_penalties = edit_penalties[completion_mask]
            grouped_rewards = grouped_rewards[prompt_mask]
            grouped_edits = grouped_edits[prompt_mask]
        else:
            rewards = all_rewards
            improved = all_improved
            non_improving_edits = all_non_improving

        if rewards.numel() == 0:
            print(f"  {label}reward_mean: n/a (no samples)")
            return

        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item() if rewards.numel() > 1 else 0.0
        frac_improved = improved.float().mean().item()
        frac_edits = subset_edit_penalties.float().mean().item()
        frac_non_improving_edits = non_improving_edits.float().mean().item()
        frac_edited_positive = (
            ((rewards > 0) & (subset_edit_penalties > 0)).float().mean().item()
        )

        if grouped_rewards.numel() > 0:
            has_copy = (grouped_edits == 0).any(dim=1)
            copy_rewards = grouped_rewards.clone()
            copy_rewards[grouped_edits > 0] = -float("inf")
            best_copy = copy_rewards.max(dim=1).values
            best_any = grouped_rewards.max(dim=1).values
            copy_best = (
                (has_copy & (best_copy >= best_any - 1e-6)).float().mean().item()
            )
        else:
            copy_best = 0.0

        q10, q50, q90 = torch.quantile(
            rewards, torch.tensor([0.1, 0.5, 0.9], device=rewards.device)
        ).tolist()

        print(f"  {label}reward_mean: {reward_mean:.4f} | reward_std: {reward_std:.4f}")
        print(f"  {label}improved_frac: {frac_improved:.4f}")
        print(f"  {label}edit_frac: {frac_edits:.4f}")
        print(f"  {label}non_improving_edit_frac: {frac_non_improving_edits:.4f}")
        print(f"  {label}edited_positive_frac: {frac_edited_positive:.4f}")
        print(f"  {label}copy_best_frac: {copy_best:.4f}")
        print(
            f"  {label}reward_quantiles: p10={q10:.4f} p50={q50:.4f} p90={q90:.4f}"
        )

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

        print(f"\nEpsilon: {epsilon:.4f}")
        summarize("", rewards, improved, non_improving_edits)
        if is_clean_prompt is not None:
            summarize(
                "clean_subset_", rewards, improved, non_improving_edits, is_clean_prompt
            )
            summarize(
                "dirty_subset_",
                rewards,
                improved,
                non_improving_edits,
                ~is_clean_prompt,
            )

        if args.use_clean_fields and is_clean_tensor is not None:
            if applied_edits_tensor is None:
                raise ValueError(
                    "use_clean_fields requires dataset column 'applied_edits'."
                )
            penalty_scale = torch.where(
                is_clean_tensor,
                torch.ones_like(conditional_penalties),
                torch.full_like(conditional_penalties, args.dirty_penalty_scale),
            )
            conditional_penalties_clean = conditional_penalties * penalty_scale

            semantic_clean = torch.where(
                non_improving_edits & is_clean_tensor,
                torch.zeros_like(semantic_scores),
                semantic_scores,
            )
            no_edit = edit_penalties == 0
            dirty_has_edits = (~is_clean_tensor) & (applied_edits_tensor > 0)
            no_edit_penalty = torch.where(
                no_edit & dirty_has_edits,
                torch.full_like(conditional_penalties, args.no_edit_penalty),
                torch.zeros_like(conditional_penalties),
            )

            rewards_clean = (
                config.greco_weight * effective_gain
                + config.semantic_weight * semantic_clean
                - config.laziness_weight * conditional_penalties_clean
                - no_edit_penalty
            )
            summarize("clean_", rewards_clean, improved, non_improving_edits)


if __name__ == "__main__":
    main()
