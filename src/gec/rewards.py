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
        """Compute GRECO quality scores for hypotheses."""
        with torch.no_grad():
            scores = self.greco.score(sources, hypotheses)
        return scores.cpu()

    def compute_source_greco_scores(self, sources: list[str]) -> torch.Tensor:
        """Compute GRECO quality scores for source texts (used for gain calculation)."""
        with torch.no_grad():
            # GRECO expects (source, hypothesis) pairs - use source as both
            scores = self.greco.score(sources, sources)
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

    def compute_edit_penalty(
        self, sources: list[str], hypotheses: list[str]
    ) -> torch.Tensor:
        """
        Binary edit penalty: 1.0 if any edit was made, 0.0 if unchanged.
        Discourages unnecessary edits - model must justify changes via GRECO gain.
        """
        penalties = []
        for src, hyp in zip(sources, hypotheses):
            has_edit = 1.0 if src.strip() != hyp.strip() else 0.0
            penalties.append(has_edit)
        return torch.tensor(penalties)

    def compute_rewards(
        self, sources: list[str], hypotheses: list[str], is_clean: list[bool] | None = None
    ) -> list[float]:
        """
        Compute gain-based composite reward.

        Reward = greco_weight * GRECO_GAIN + semantic_weight * similarity - edit_weight * conditional_penalty

        Where GRECO_GAIN = GRECO(hypothesis) - GRECO(source)
        Edit penalty only applies when gain <= 0 (edits that don't improve quality).
        """
        # Cache source GRECO per unique source (same source has N completions in GRPO)
        unique_sources = list(dict.fromkeys(sources))
        unique_scores = self.compute_source_greco_scores(unique_sources)
        source_greco_map = {s: score.item() for s, score in zip(unique_sources, unique_scores)}
        source_greco = torch.tensor([source_greco_map[s] for s in sources])

        hyp_greco = self.compute_greco_scores(sources, hypotheses)
        greco_gain = hyp_greco - source_greco

        semantic_scores = self.compute_semantic_similarity(sources, hypotheses)
        edit_penalties = self.compute_edit_penalty(sources, hypotheses)

        # Only penalize edits that don't improve quality (gain <= 0)
        conditional_penalties = torch.where(
            greco_gain <= 0, edit_penalties, torch.zeros_like(edit_penalties)
        )

        rewards = (
            self.greco_weight * greco_gain
            + self.semantic_weight * semantic_scores
            - self.laziness_weight * conditional_penalties
        )

        # Log first 3 samples for debugging
        print(f"[Reward] SrcGRECO: {source_greco[:3].tolist()}, HypGRECO: {hyp_greco[:3].tolist()}, Gain: {greco_gain[:3].tolist()}, Semantic: {semantic_scores[:3].tolist()}, EditPen: {conditional_penalties[:3].tolist()}, Final: {rewards[:3].tolist()}")

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
