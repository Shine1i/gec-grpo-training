from difflib import SequenceMatcher
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# Handle both local and Modal paths for GRECO
greco_paths = [
    Path(__file__).parent.parent.parent / "greco",  # Local
    Path("/root/greco"),  # Modal
]
for greco_path in greco_paths:
    if greco_path.exists():
        sys.path.insert(0, str(greco_path))
        break

from models import GRECO


class GECRewardModel:
    """Composite reward model for GEC: GRECO + Semantic + Laziness penalty."""

    def __init__(
        self,
        greco_model_name: str = "mrqorib/grammaticality",
        mpnet_model: str = "sentence-transformers/all-mpnet-base-v2",
        greco_weight: float = 0.6,
        semantic_weight: float = 0.3,
        laziness_weight: float = 0.1,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load GRECO from HuggingFace
        print(f"Loading GRECO from {greco_model_name}...")
        self.greco = GRECO(lm="microsoft/deberta-v3-large", use_fast=True).to(
            self.device
        )
        checkpoint_path = hf_hub_download(greco_model_name, "pytorch_model.bin")
        self.greco.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device), strict=False
        )
        self.greco.eval()

        # Load MPNet for semantic similarity
        print(f"Loading MPNet from {mpnet_model}...")
        self.mpnet = SentenceTransformer(mpnet_model, device=str(self.device))

        # Reward weights
        self.greco_weight = greco_weight
        self.semantic_weight = semantic_weight
        self.laziness_weight = laziness_weight

    def compute_greco_scores(
        self, sources: list[str], hypotheses: list[str]
    ) -> torch.Tensor:
        """Compute GRECO quality scores."""
        with torch.no_grad():
            scores = self.greco.score(sources, hypotheses)
        return scores.cpu()

    def compute_semantic_similarity(
        self, sources: list[str], hypotheses: list[str]
    ) -> torch.Tensor:
        """Compute cosine similarity using MPNet embeddings."""
        with torch.no_grad():
            source_embs = self.mpnet.encode(sources, convert_to_tensor=True)
            hyp_embs = self.mpnet.encode(hypotheses, convert_to_tensor=True)
            similarities = F.cosine_similarity(source_embs, hyp_embs, dim=1)
        return similarities.cpu()

    def compute_laziness_penalty(
        self, sources: list[str], hypotheses: list[str], is_clean: list[bool] | None = None
    ) -> torch.Tensor:
        """
        Compute laziness penalty based on edit distance.
        Penalizes if correction is too similar to source (no real edits made).
        Skips penalty for clean samples where copying is correct behavior.
        """
        penalties = []
        for i, (src, hyp) in enumerate(zip(sources, hypotheses)):
            # Skip laziness penalty for clean samples
            if is_clean is not None and is_clean[i]:
                penalties.append(0.0)
                continue

            similarity = SequenceMatcher(None, src.lower(), hyp.lower()).ratio()
            # Penalize if model just copies input (similarity > 0.95)
            if similarity > 0.95:
                penalty = (similarity - 0.95) * 20  # Scale penalty
            else:
                penalty = 0.0
            penalties.append(penalty)
        return torch.tensor(penalties)

    def compute_rewards(
        self, sources: list[str], hypotheses: list[str], is_clean: list[bool] | None = None
    ) -> list[float]:
        """
        Compute composite reward.

        Reward = greco_weight * GRECO + semantic_weight * similarity - laziness_weight * penalty
        Skips laziness penalty for clean samples.
        """
        greco_scores = self.compute_greco_scores(sources, hypotheses)
        semantic_scores = self.compute_semantic_similarity(sources, hypotheses)
        laziness_penalties = self.compute_laziness_penalty(sources, hypotheses, is_clean)

        rewards = (
            self.greco_weight * greco_scores
            + self.semantic_weight * semantic_scores
            - self.laziness_weight * laziness_penalties
        )

        # Log first 3 samples every call for debugging
        is_clean_str = str(is_clean[:3]) if is_clean else "None"
        print(f"[Reward] GRECO: {greco_scores[:3].tolist()}, Semantic: {semantic_scores[:3].tolist()}, Laziness: {laziness_penalties[:3].tolist()}, is_clean: {is_clean_str}, Final: {rewards[:3].tolist()}")

        return rewards.tolist()


def _flatten_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part.get("content", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _completion_to_text(completion) -> str:
    if isinstance(completion, list):
        if not completion:
            return ""
        last_message = completion[-1]
        if isinstance(last_message, dict):
            return _flatten_message_content(last_message.get("content"))
    if isinstance(completion, dict):
        return _flatten_message_content(completion.get("content"))
    if completion is None:
        return ""
    return str(completion)


def build_gec_reward_func(reward_model: GECRewardModel):
    """Build a GRPO-compatible reward function using the composite GEC reward."""

    def reward_func(prompts, completions, source, is_clean=None, **kwargs) -> list[float]:
        completion_texts = [_completion_to_text(c) for c in completions]
        if isinstance(source, str):
            sources = [source] * len(completion_texts)
        else:
            sources = list(source)
        if len(sources) != len(completion_texts):
            raise ValueError(
                "Mismatched sources and completions: "
                f"{len(sources)} vs {len(completion_texts)}"
            )

        # Handle is_clean - expand to match completions if needed
        is_clean_list = None
        if is_clean is not None:
            if isinstance(is_clean, bool):
                is_clean_list = [is_clean] * len(completion_texts)
            else:
                is_clean_list = list(is_clean)

        return reward_model.compute_rewards(sources, completion_texts, is_clean_list)

    return reward_func
