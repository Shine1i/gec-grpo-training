import random
import re
from typing import Optional

from datasets import Dataset, load_dataset


def extract_text_from_tags(content: str) -> str:
    """Extract text between <Text>...</Text> tags."""
    match = re.search(r"<Text>(.*?)</Text>", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def _stratified_sample(
    ds: Dataset,
    size: int,
    wi_locness_ratio: float = 0.625,
    clean_ratio: float = 0.28,
    seed: int = 42,
) -> Dataset:
    """
    Create stratified sample with specified ratios.

    Target distribution:
    - WI-LOCNESS: 62.5%, FCE: 37.5%
    - Clean: 28% overall
    """
    random.seed(seed)

    # Split by source and is_clean
    wi_clean = [i for i, x in enumerate(ds) if x.get("source") == "wi_locness" and x.get("is_clean")]
    wi_error = [i for i, x in enumerate(ds) if x.get("source") == "wi_locness" and not x.get("is_clean")]
    fce_clean = [i for i, x in enumerate(ds) if x.get("source") == "fce" and x.get("is_clean")]
    fce_error = [i for i, x in enumerate(ds) if x.get("source") == "fce" and not x.get("is_clean")]

    # Calculate target counts
    wi_total = int(size * wi_locness_ratio)  # 6250
    fce_total = size - wi_total  # 3750
    total_clean = int(size * clean_ratio)  # 2800

    # Distribute clean samples proportionally between sources
    wi_clean_target = int(total_clean * wi_locness_ratio)
    fce_clean_target = total_clean - wi_clean_target

    wi_error_target = wi_total - wi_clean_target
    fce_error_target = fce_total - fce_clean_target

    # Sample from each bucket (cap at available)
    random.shuffle(wi_clean)
    random.shuffle(wi_error)
    random.shuffle(fce_clean)
    random.shuffle(fce_error)

    selected_indices = (
        wi_clean[:min(wi_clean_target, len(wi_clean))] +
        wi_error[:min(wi_error_target, len(wi_error))] +
        fce_clean[:min(fce_clean_target, len(fce_clean))] +
        fce_error[:min(fce_error_target, len(fce_error))]
    )

    # If we didn't get enough, fill from remaining
    all_indices = set(range(len(ds)))
    selected_set = set(selected_indices)
    remaining = list(all_indices - selected_set)
    random.shuffle(remaining)

    while len(selected_indices) < size and remaining:
        selected_indices.append(remaining.pop())

    random.shuffle(selected_indices)

    # Log distribution
    final_wi = sum(1 for i in selected_indices if ds[i].get("source") == "wi_locness")
    final_clean = sum(1 for i in selected_indices if ds[i].get("is_clean"))
    print(f"Stratified sample: {len(selected_indices)} total")
    print(f"  WI-LOCNESS: {final_wi} ({100*final_wi/len(selected_indices):.1f}%)")
    print(f"  FCE: {len(selected_indices)-final_wi} ({100*(len(selected_indices)-final_wi)/len(selected_indices):.1f}%)")
    print(f"  Clean: {final_clean} ({100*final_clean/len(selected_indices):.1f}%)")

    return ds.select(selected_indices)


def load_gec_dataset(dataset_name: str, size: Optional[int] = None) -> Dataset:
    """
    Load GEC dataset from HuggingFace with stratified sampling.

    Expected format (chat):
    {
        "messages": [...],
        "source": "wi_locness" | "fce",
        "is_clean": bool
    }

    Returns Dataset with 'prompt' column containing extracted incorrect sentences.
    """
    ds = load_dataset(dataset_name, split="train")

    # Apply stratified sampling if size specified
    if size and size < len(ds):
        ds = _stratified_sample(ds, size)

    prompts = []
    references = []
    is_clean_flags = []
    sources = []

    for item in ds:
        messages = item.get("messages", [])
        is_clean = item.get("is_clean", False)
        source = item.get("source", "unknown")

        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

        if user_msg:
            incorrect_text = extract_text_from_tags(user_msg["content"])
            prompts.append(incorrect_text)
            is_clean_flags.append(is_clean)
            sources.append(source)

            if assistant_msg:
                references.append(assistant_msg["content"])
            else:
                references.append(None)

    return Dataset.from_dict({"prompt": prompts, "reference": references, "is_clean": is_clean_flags, "source": sources})


def make_gec_messages(incorrect_text: str, system_prompt: str) -> list[dict[str, str]]:
    """Create chat messages for GEC generation matching SFT format."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fix spelling, grammar, and punctuation. Make minimal changes and preserve meaning. If the text is already correct, return it unchanged. Return only the corrected text:\n<Text>\n{incorrect_text}\n</Text>"},
    ]
