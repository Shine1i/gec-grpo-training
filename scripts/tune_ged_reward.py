import argparse
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

from src.gec.config import GECConfig
from src.gec.dataset import load_gec_dataset, make_gec_messages

GED_BASELINES_ROOT = Path(__file__).resolve().parents[1] / "ged_baselines"
if GED_BASELINES_ROOT.exists():
    sys.path.insert(0, str(GED_BASELINES_ROOT))

try:
    from ged_baselines.token_ged.dataset import generate_dataset_for_inference
except ImportError as exc:
    raise ImportError(
        "ged_baselines not found. Ensure the repo exists at ./ged_baselines or install it."
    ) from exc


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

    for v in buckets.values():
        random.shuffle(v)

    wi_fce_total = size // 2
    lang8_total = size - wi_fce_total

    wi_total = wi_fce_total // 2
    fce_total = wi_fce_total - wi_total

    total_clean = int(size * clean_ratio)
    wi_clean_target = int(total_clean * (wi_total / size))
    fce_clean_target = int(total_clean * (fce_total / size))
    lang8_clean_target = total_clean - wi_clean_target - fce_clean_target

    wi_error_target = wi_total - wi_clean_target
    fce_error_target = fce_total - fce_clean_target
    lang8_error_target = lang8_total - lang8_clean_target

    selected = []
    selected += buckets[("wi_locness", True)][:wi_clean_target]
    selected += buckets[("wi_locness", False)][:wi_error_target]
    selected += buckets[("fce", True)][:fce_clean_target]
    selected += buckets[("fce", False)][:fce_error_target]
    selected += buckets[("lang8_replay", True)][:lang8_clean_target]
    selected += buckets[("lang8_replay", False)][:lang8_error_target]

    all_indices = set(range(len(ds)))
    remaining = list(all_indices - set(selected))
    random.shuffle(remaining)
    while len(selected) < size and remaining:
        selected.append(remaining.pop())

    random.shuffle(selected)

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
        description="Tune gain_epsilon using token-level GED reduction."
    )
    parser.add_argument("--config-file-name", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--epsilons", type=str, default="0.0,0.005,0.01")
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy-per-prompt", type=int, default=1)
    parser.add_argument(
        "--gain-mode",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        help="hard: gate by epsilon; soft: relu(gain - epsilon).",
    )

    parser.add_argument("--ged-model", type=str, default="gotutiyan/token-ged-electra-large-25cls")
    parser.add_argument("--ged-max-length", type=int, default=128)
    parser.add_argument("--ged-batch-size", type=int, default=32)
    parser.add_argument("--ged-use-rate", action="store_true")
    parser.add_argument(
        "--ged-use-slow",
        action="store_true",
        help="Force slow tokenizer (not supported; fast tokenizer required for word_ids).",
    )

    parser.add_argument("--ged-weights", type=str, default=None)
    parser.add_argument("--semantic-weights", type=str, default=None)
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument(
        "--semantic-mode",
        type=str,
        choices=["gated", "always"],
        default="gated",
    )
    parser.add_argument(
        "--semantic-drift-weight",
        type=float,
        default=0.0,
        help="Penalty weight for semantic drift (1 - similarity) above threshold.",
    )
    parser.add_argument(
        "--semantic-drift-threshold",
        type=float,
        default=0.05,
        help="Drift margin before penalty activates.",
    )
    parser.add_argument(
        "--clean-edit-penalty",
        type=float,
        default=0.0,
        help="Penalty for any edit on clean examples (requires is_clean).",
    )

    parser.add_argument("--print-clean-samples", type=int, default=0)
    parser.add_argument("--print-dirty-samples", type=int, default=0)
    parser.add_argument(
        "--print-token-labels",
        action="store_true",
        help="Print token-level GED labels for sampled prompts.",
    )
    parser.add_argument(
        "--token-label-limit",
        type=int,
        default=200,
        help="Max tokens to display per label line (after truncation).",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Print a compact summary per configuration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_float_list(value: str | None):
    if value is None:
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def predict_token_labels(
    srcs: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_len: int,
    batch_size: int,
):
    dataset = generate_dataset_for_inference(
        srcs=srcs,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred_labels = torch.argmax(logits, dim=-1)
            for temp_label, pred_label in zip(batch["labels"], pred_labels):
                word_level = pred_label[temp_label != -100].tolist()
                id2label = model.config.id2label
                word_level = [id2label[l] for l in word_level]
                predictions.append(word_level)
    return predictions


def label_is_correct(label: str) -> bool:
    if not isinstance(label, str):
        return False
    tag = label.strip().upper()
    return tag in {"CORRECT", "C"}


def error_rate(labels: list[str], use_rate: bool) -> float:
    if not labels:
        return 0.0
    errors = sum(1 for label in labels if not label_is_correct(label))
    if use_rate:
        return errors / max(len(labels), 1)
    return float(errors)


def compute_edit_penalty(sources: list[str], hypotheses: list[str]) -> torch.Tensor:
    penalties = []
    for src, hyp in zip(sources, hypotheses):
        penalties.append(1.0 if src.strip() != hyp.strip() else 0.0)
    return torch.tensor(penalties)


def split_for_ged(text: str) -> list[str]:
    # Match ged_baselines token splitting behavior.
    return text.split(" ")


def format_token_labels(
    text: str, labels: list[str], max_tokens: int
) -> tuple[str, bool]:
    tokens = split_for_ged(text)
    truncated = False
    if len(tokens) > len(labels):
        tokens = tokens[: len(labels)]
        truncated = True
    pairs = []
    for token, label in zip(tokens, labels, strict=False):
        if token == "":
            token = "<EMPTY>"
        pairs.append(f"{token}/{label}")
        if max_tokens and len(pairs) >= max_tokens:
            truncated = True
            break
    return " ".join(pairs), truncated


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else None

    default_ged_weight = getattr(config, "ged_weight", config.greco_weight)
    ged_weights = parse_float_list(args.ged_weights) or [default_ged_weight]
    semantic_weights = parse_float_list(args.semantic_weights) or [config.semantic_weight]
    weight_pairs = [(g, s) for g in ged_weights for s in semantic_weights]

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

    max_new_tokens = args.max_new_tokens or config.max_completion_length

    source_offset = 0
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
        encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}
        prompt_len = encoded["input_ids"].shape[1]

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
                for idx, seq in enumerate(greedy_out):
                    completion_ids = seq[prompt_len:]
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
                for j, seq in enumerate(sampled_out):
                    completion_ids = seq[prompt_len:]
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
                repeated_is_clean.extend(
                    [bool(is_clean[source_offset + idx])] * num_generations
                )
        source_offset += len(source_batch)

    if len(completions) != len(repeated_sources):
        raise ValueError(
            f"Mismatched completions and sources: {len(completions)} vs {len(repeated_sources)}"
        )

    ged_tokenizer = AutoTokenizer.from_pretrained(
        args.ged_model, use_fast=not args.ged_use_slow
    )
    if not getattr(ged_tokenizer, "is_fast", False):
        raise ValueError(
            "ged_baselines requires a fast tokenizer for word_ids; "
            "remove --ged-use-slow or use a model with a fast tokenizer."
        )
    ged_model = AutoModelForTokenClassification.from_pretrained(args.ged_model)

    unique_sources = list(dict.fromkeys(repeated_sources))
    source_labels = predict_token_labels(
        unique_sources,
        ged_model,
        ged_tokenizer,
        device=device,
        max_len=args.ged_max_length,
        batch_size=args.ged_batch_size,
    )
    source_labels_map = {
        src: labels for src, labels in zip(unique_sources, source_labels, strict=True)
    }
    source_err_map = {
        src: error_rate(labels, args.ged_use_rate)
        for src, labels in zip(unique_sources, source_labels, strict=True)
    }
    source_err = torch.tensor([source_err_map[src] for src in repeated_sources])

    hyp_labels = predict_token_labels(
        completions,
        ged_model,
        ged_tokenizer,
        device=device,
        max_len=args.ged_max_length,
        batch_size=args.ged_batch_size,
    )
    hyp_err = torch.tensor([error_rate(labels, args.ged_use_rate) for labels in hyp_labels])

    ged_gain = source_err - hyp_err

    compute_semantic = any(weight > 0 for weight in semantic_weights) or (
        args.semantic_drift_weight > 0
    )
    if compute_semantic:
        mpnet = SentenceTransformer(args.semantic_model, device=str(device))
        with torch.no_grad():
            src_embs = mpnet.encode(repeated_sources, convert_to_tensor=True)
            hyp_embs = mpnet.encode(completions, convert_to_tensor=True)
            semantic_scores = torch.nn.functional.cosine_similarity(
                src_embs, hyp_embs, dim=1
            ).cpu()
    else:
        semantic_scores = torch.zeros_like(ged_gain)
    semantic_drift = 1.0 - semantic_scores
    semantic_drift_penalty = torch.clamp(
        semantic_drift - args.semantic_drift_threshold, min=0.0
    )

    edit_penalties = compute_edit_penalty(repeated_sources, completions)
    if is_clean is not None:
        is_clean_prompt = torch.tensor(is_clean, dtype=torch.bool)
        is_clean_completion = torch.tensor(repeated_is_clean, dtype=torch.bool)
    else:
        is_clean_prompt = None
        is_clean_completion = None

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
            if source_domains is not None:
                meta.append(f"source={source_domains[idx]}")
            meta_str = ", ".join(meta)
            print(f"\n[{label} idx {idx}] {meta_str}")
            print(f"Input: {sources[idx]}")
            if args.print_token_labels:
                src_labels = source_labels_map.get(sources[idx], [])
                formatted, truncated = format_token_labels(
                    sources[idx], src_labels, args.token_label_limit
                )
                suffix = " [truncated]" if truncated else ""
                print(f"    src_labels: {formatted}{suffix}")
            base = idx * num_generations
            for j in range(num_generations):
                pos = base + j
                completion = completions[pos]
                edit_flag = "COPY" if edit_penalties[pos].item() == 0 else "EDIT"
                print(
                    f"  - {edit_flag} | gain={ged_gain[pos].item():.4f} "
                    f"semantic={semantic_scores[pos].item():.4f} "
                    f"src_err={source_err[pos].item():.4f} hyp_err={hyp_err[pos].item():.4f}"
                )
                print(f"    {completion}")
                if args.print_token_labels:
                    hyp_labels = hyp_labels[pos]
                    formatted, truncated = format_token_labels(
                        completion, hyp_labels, args.token_label_limit
                    )
                    suffix = " [truncated]" if truncated else ""
                    print(f"      hyp_labels: {formatted}{suffix}")

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
        grouped_rewards = rewards.view(num_prompts, num_generations)
        grouped_edits = edit_penalties.view(num_prompts, num_generations)
        subset_edit_penalties = edit_penalties

        if prompt_mask is not None:
            completion_mask = prompt_mask.repeat_interleave(num_generations)
            rewards = rewards[completion_mask]
            improved = improved[completion_mask]
            non_improving_edits = non_improving_edits[completion_mask]
            subset_edit_penalties = edit_penalties[completion_mask]
            grouped_rewards = grouped_rewards[prompt_mask]
            grouped_edits = grouped_edits[prompt_mask]

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

        if args.compact_output:
            print(
                f"  {label}reward_mean={reward_mean:.4f} reward_std={reward_std:.4f} "
                f"improved={frac_improved:.3f} edit={frac_edits:.3f} "
                f"non_impr_edit={frac_non_improving_edits:.3f} copy_best={copy_best:.3f}"
            )
        else:
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
        improved = ged_gain > epsilon
        if args.gain_mode == "soft":
            effective_gain = torch.clamp(ged_gain - epsilon, min=0.0)
        else:
            effective_gain = torch.where(
                improved, ged_gain, torch.zeros_like(ged_gain)
            )
        non_improving_edits = (edit_penalties > 0) & (~improved)
        if args.semantic_mode == "gated":
            semantic_effective = torch.where(
                non_improving_edits,
                torch.zeros_like(semantic_scores),
                semantic_scores,
            )
        else:
            semantic_effective = semantic_scores
        conditional_penalties = torch.where(
            non_improving_edits, edit_penalties, torch.zeros_like(edit_penalties)
        )

        print(f"\nEpsilon: {epsilon:.4f} (gain_mode={args.gain_mode})")
        for ged_weight, semantic_weight in weight_pairs:
            laziness_weight = config.laziness_weight
            weight_sum = ged_weight + semantic_weight + laziness_weight
            weight_line = (
                f"  Weights: ged={ged_weight:.2f} semantic={semantic_weight:.2f} "
                f"laziness={laziness_weight:.2f}"
            )
            extras = []
            if abs(weight_sum - 1.0) > 1e-3:
                extras.append(f"sum={weight_sum:.2f}")
            if args.semantic_drift_weight > 0:
                extras.append(
                    f"drift={args.semantic_drift_weight:.2f}@{args.semantic_drift_threshold:.2f}"
                )
            if args.clean_edit_penalty > 0:
                extras.append(f"clean_edit={args.clean_edit_penalty:.2f}")
            if extras:
                weight_line = f"{weight_line} ({', '.join(extras)})"
            print(weight_line)

            rewards = (
                ged_weight * effective_gain
                + semantic_weight * semantic_effective
                - laziness_weight * conditional_penalties
            )
            if args.semantic_drift_weight > 0:
                rewards = rewards - args.semantic_drift_weight * semantic_drift_penalty
            if args.clean_edit_penalty > 0 and is_clean_completion is not None:
                clean_edits = (is_clean_completion & (edit_penalties > 0)).float()
                rewards = rewards - args.clean_edit_penalty * clean_edits

            summarize("  ", rewards, improved, non_improving_edits)
            if is_clean_prompt is not None:
                summarize(
                    "  clean_subset_",
                    rewards,
                    improved,
                    non_improving_edits,
                    is_clean_prompt,
                )
                summarize(
                    "  dirty_subset_",
                    rewards,
                    improved,
                    non_improving_edits,
                    ~is_clean_prompt,
                )


if __name__ == "__main__":
    main()
