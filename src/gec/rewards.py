from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer

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

# Handle both local and Modal paths for token-level GED
ged_paths = [
    Path(__file__).parent.parent.parent / "ged_baselines",  # Local
    Path("/root/ged_baselines"),  # Modal
]
for ged_path in ged_paths:
    if ged_path.exists():
        sys.path.insert(0, str(ged_path))
        break

try:
    from ged_baselines.token_ged.dataset import generate_dataset_for_inference
except ImportError:
    generate_dataset_for_inference = None


def _label_is_correct(label: str) -> bool:
    if not isinstance(label, str):
        return False
    tag = label.strip().upper()
    return tag in {"CORRECT", "C"}


def _error_value(labels: list[str], use_rate: bool) -> float:
    if not labels:
        return 0.0
    errors = sum(1 for label in labels if not _label_is_correct(label))
    if use_rate:
        return errors / max(len(labels), 1)
    return float(errors)


class GECRewardModel:
    """Composite reward model for GEC: GRECO + Semantic + Laziness penalty."""

    def __init__(
        self,
        greco_model_name: str = "mrqorib/grammaticality",
        mpnet_model: str = "sentence-transformers/all-mpnet-base-v2",
        greco_weight: float = 0.6,
        semantic_weight: float = 0.3,
        laziness_weight: float = 0.1,
        gain_epsilon: float = 0.02,
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
        self.gain_epsilon = gain_epsilon

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
        Small gains below gain_epsilon are treated as no improvement to avoid GRECO noise.
        Edit penalty only applies when gain <= gain_epsilon (edits that don't improve quality).
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

        improved = greco_gain > self.gain_epsilon
        effective_gain = torch.where(improved, greco_gain, torch.zeros_like(greco_gain))
        non_improving_edits = (edit_penalties > 0) & (~improved)
        semantic_scores = torch.where(
            non_improving_edits,
            torch.zeros_like(semantic_scores),
            semantic_scores,
        )
        conditional_penalties = torch.where(
            non_improving_edits, edit_penalties, torch.zeros_like(edit_penalties)
        )

        rewards = (
            self.greco_weight * effective_gain
            + self.semantic_weight * semantic_scores
            - self.laziness_weight * conditional_penalties
        )

        # Log first 3 samples for debugging
        print(
            "[Reward] SrcGRECO: "
            f"{source_greco[:3].tolist()}, HypGRECO: {hyp_greco[:3].tolist()}, "
            f"Gain: {greco_gain[:3].tolist()}, EffGain: {effective_gain[:3].tolist()}, "
            f"Semantic: {semantic_scores[:3].tolist()}, "
            f"EditPen: {conditional_penalties[:3].tolist()}, Final: {rewards[:3].tolist()}"
        )

        return rewards.tolist()


class GEDRewardModel:
    """Token-level GED reward with semantic drift and clean-edit penalties."""

    def __init__(
        self,
        ged_model_name: str = "gotutiyan/token-ged-electra-large-bin",
        mpnet_model: str = "sentence-transformers/all-mpnet-base-v2",
        ged_weight: float = 0.6,
        semantic_weight: float = 0.0,
        laziness_weight: float = 0.1,
        gain_epsilon: float = 0.005,
        gain_mode: str = "soft",
        semantic_mode: str = "gated",
        semantic_drift_weight: float = 0.2,
        semantic_drift_threshold: float = 0.05,
        clean_edit_penalty: float = 0.0,
        ged_use_rate: bool = True,
        ged_max_length: int = 128,
        ged_batch_size: int = 32,
        device: str = "cuda",
    ):
        if generate_dataset_for_inference is None:
            raise ImportError(
                "ged_baselines is not available. Ensure the repo exists at ./ged_baselines "
                "or install it before using GED rewards."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load GED model (token-level)
        print(f"Loading GED model from {ged_model_name}...")
        self.ged_tokenizer = AutoTokenizer.from_pretrained(ged_model_name, use_fast=True)
        if not getattr(self.ged_tokenizer, "is_fast", False):
            raise ValueError(
                "GED tokenizers must be fast to support word_ids alignment."
            )
        self.ged_model = AutoModelForTokenClassification.from_pretrained(
            ged_model_name
        ).to(self.device)
        self.ged_model.eval()

        # Load MPNet for semantic similarity (optional)
        self.semantic_weight = semantic_weight
        self.semantic_drift_weight = semantic_drift_weight
        self.semantic_drift_threshold = semantic_drift_threshold
        if semantic_weight > 0 or semantic_drift_weight > 0:
            print(f"Loading MPNet from {mpnet_model}...")
            self.mpnet = SentenceTransformer(mpnet_model, device=str(self.device))
        else:
            self.mpnet = None

        # Reward weights and knobs
        self.ged_weight = ged_weight
        self.laziness_weight = laziness_weight
        self.gain_epsilon = gain_epsilon
        self.gain_mode = gain_mode
        self.semantic_mode = semantic_mode
        self.clean_edit_penalty = clean_edit_penalty
        self.ged_use_rate = ged_use_rate
        self.ged_max_length = ged_max_length
        self.ged_batch_size = ged_batch_size

    def _predict_token_labels(self, texts: list[str]) -> list[list[str]]:
        dataset = generate_dataset_for_inference(
            srcs=texts, tokenizer=self.ged_tokenizer, max_len=self.ged_max_length
        )
        loader = DataLoader(dataset, batch_size=self.ged_batch_size)
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.ged_model(**batch).logits
                pred_labels = torch.argmax(logits, dim=-1)
                for temp_label, pred_label in zip(batch["labels"], pred_labels):
                    word_level = pred_label[temp_label != -100].tolist()
                    id2label = self.ged_model.config.id2label
                    word_level = [id2label[l] for l in word_level]
                    predictions.append(word_level)
        return predictions

    def compute_semantic_similarity(
        self, sources: list[str], hypotheses: list[str]
    ) -> torch.Tensor:
        if self.mpnet is None:
            return torch.zeros(len(sources))
        with torch.no_grad():
            source_embs = self.mpnet.encode(sources, convert_to_tensor=True)
            hyp_embs = self.mpnet.encode(hypotheses, convert_to_tensor=True)
            similarities = F.cosine_similarity(source_embs, hyp_embs, dim=1)
        return similarities.cpu()

    def compute_edit_penalty(
        self, sources: list[str], hypotheses: list[str]
    ) -> torch.Tensor:
        penalties = []
        for src, hyp in zip(sources, hypotheses):
            penalties.append(1.0 if src.strip() != hyp.strip() else 0.0)
        return torch.tensor(penalties)

    def compute_rewards(
        self, sources: list[str], hypotheses: list[str], is_clean: list[bool] | None = None
    ) -> list[float]:
        # Cache GED source errors per unique source
        unique_sources = list(dict.fromkeys(sources))
        source_labels = self._predict_token_labels(unique_sources)
        source_err_map = {
            s: _error_value(labels, self.ged_use_rate)
            for s, labels in zip(unique_sources, source_labels)
        }
        source_err = torch.tensor([source_err_map[s] for s in sources])

        hyp_labels = self._predict_token_labels(hypotheses)
        hyp_err = torch.tensor(
            [_error_value(labels, self.ged_use_rate) for labels in hyp_labels]
        )

        ged_gain = source_err - hyp_err

        semantic_scores = self.compute_semantic_similarity(sources, hypotheses)
        edit_penalties = self.compute_edit_penalty(sources, hypotheses)

        improved = ged_gain > self.gain_epsilon
        if self.gain_mode == "soft":
            effective_gain = torch.clamp(ged_gain - self.gain_epsilon, min=0.0)
        else:
            effective_gain = torch.where(
                improved, ged_gain, torch.zeros_like(ged_gain)
            )

        non_improving_edits = (edit_penalties > 0) & (~improved)
        if self.semantic_mode == "gated":
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

        rewards = (
            self.ged_weight * effective_gain
            + self.semantic_weight * semantic_effective
            - self.laziness_weight * conditional_penalties
        )

        if self.semantic_drift_weight > 0:
            semantic_drift = 1.0 - semantic_scores
            drift_penalty = torch.clamp(
                semantic_drift - self.semantic_drift_threshold, min=0.0
            )
            rewards = rewards - self.semantic_drift_weight * drift_penalty

        if self.clean_edit_penalty > 0 and is_clean is not None:
            is_clean_tensor = torch.tensor(is_clean, dtype=torch.bool)
            clean_edits = (is_clean_tensor & (edit_penalties > 0)).float()
            rewards = rewards - self.clean_edit_penalty * clean_edits

        print(
            "[Reward] SrcGED: "
            f"{source_err[:3].tolist()}, HypGED: {hyp_err[:3].tolist()}, "
            f"Gain: {ged_gain[:3].tolist()}, EffGain: {effective_gain[:3].tolist()}, "
            f"Semantic: {semantic_scores[:3].tolist()}, "
            f"EditPen: {conditional_penalties[:3].tolist()}, Final: {rewards[:3].tolist()}"
        )

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
                if len(is_clean_list) != len(completion_texts):
                    if len(completion_texts) % len(is_clean_list) == 0:
                        repeat = len(completion_texts) // len(is_clean_list)
                        is_clean_list = [
                            flag for flag in is_clean_list for _ in range(repeat)
                        ]
                    else:
                        raise ValueError(
                            "Mismatched is_clean and completions: "
                            f"{len(is_clean_list)} vs {len(completion_texts)}"
                        )

        return reward_model.compute_rewards(sources, completion_texts, is_clean_list)

    return reward_func


def build_reward_model(config):
    """Factory to build the configured reward model."""
    reward_type = getattr(config, "reward_type", "greco").lower()
    if reward_type == "ged":
        return GEDRewardModel(
            ged_model_name=config.ged_model_name,
            mpnet_model=config.mpnet_model,
            ged_weight=config.ged_weight,
            semantic_weight=config.semantic_weight,
            laziness_weight=config.laziness_weight,
            gain_epsilon=config.gain_epsilon,
            gain_mode=config.gain_mode,
            semantic_mode=config.semantic_mode,
            semantic_drift_weight=config.semantic_drift_weight,
            semantic_drift_threshold=config.semantic_drift_threshold,
            clean_edit_penalty=config.clean_edit_penalty,
            ged_use_rate=config.ged_use_rate,
            ged_max_length=config.ged_max_length,
            ged_batch_size=config.ged_batch_size,
        )

    return GECRewardModel(
        greco_model_name=config.greco_model_name,
        mpnet_model=config.mpnet_model,
        greco_weight=config.greco_weight,
        semantic_weight=config.semantic_weight,
        laziness_weight=config.laziness_weight,
        gain_epsilon=config.gain_epsilon,
    )
