import re
from typing import Optional

from datasets import Dataset, load_dataset


def extract_text_from_tags(content: str) -> str:
    """Extract text between <Text>...</Text> tags."""
    match = re.search(r"<Text>(.*?)</Text>", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def load_gec_dataset(dataset_name: str, size: Optional[int] = None) -> Dataset:
    """
    Load GEC dataset from HuggingFace.

    Expected format (chat):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Fix...\n<Text>incorrect text</Text>"},
            {"role": "assistant", "content": "corrected text"}
        ]
    }

    Returns Dataset with 'prompt' column containing extracted incorrect sentences.
    """
    ds = load_dataset(dataset_name, split="train")

    prompts = []
    references = []

    for item in ds:
        messages = item.get("messages", [])

        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

        if user_msg:
            incorrect_text = extract_text_from_tags(user_msg["content"])
            prompts.append(incorrect_text)

            if assistant_msg:
                references.append(assistant_msg["content"])
            else:
                references.append(None)

    if size:
        prompts = prompts[:size]
        references = references[:size]

    return Dataset.from_dict({"prompt": prompts, "reference": references})


def make_gec_messages(incorrect_text: str, system_prompt: str) -> list[dict[str, str]]:
    """Create chat messages for GEC generation."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Correct the following text:\n{incorrect_text}"},
    ]
