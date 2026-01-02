"""
Test script to verify GRECO and MPNet models work locally.
Run: python test_reward_models.py
"""

import torch
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Add greco to path
sys.path.insert(0, str(Path(__file__).parent / "greco"))

print("=" * 60)
print("Testing Reward Models for GEC GRPO")
print("=" * 60)

# Check GPU
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Test data - source sentences and different quality corrections
test_cases = [
    {
        "source": "He go to school yesterday.",
        "corrections": [
            "He went to school yesterday.",  # Perfect
            "He goes to school yesterday.",  # Partial fix (tense mismatch)
            "He go to school yesterday.",    # No change (lazy)
            "The cat sat on the mat.",       # Completely different (hallucination)
        ]
    },
    {
        "source": "Their going to the park tommorow.",
        "corrections": [
            "They're going to the park tomorrow.",  # Perfect
            "Their going to the park tomorrow.",    # Partial (only spelling)
            "They're going to the park tommorow.",  # Partial (only grammar)
            "Their going to the park tommorow.",    # No change
        ]
    },
]

# ============================================
# Test 1: GRECO Model
# ============================================
print("\n" + "=" * 60)
print("TEST 1: GRECO Quality Estimation")
print("=" * 60)

try:
    from huggingface_hub import hf_hub_download
    from models import GRECO

    print("\nLoading GRECO from mrqorib/grammaticality...")
    greco = GRECO(lm="microsoft/deberta-v3-large").to(device)

    print("Downloading checkpoint from HuggingFace...")
    checkpoint_path = hf_hub_download("mrqorib/grammaticality", "pytorch_model.bin")
    print(f"Checkpoint path: {checkpoint_path}")

    print("Loading weights...")
    greco.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    greco.eval()

    print("\nGRECO loaded successfully!")

    # Test scoring
    print("\n--- GRECO Scores ---")
    for i, case in enumerate(test_cases):
        source = case["source"]
        print(f"\nSource {i+1}: \"{source}\"")

        with torch.no_grad():
            scores = greco.score([source] * len(case["corrections"]), case["corrections"])

        for j, (corr, score) in enumerate(zip(case["corrections"], scores)):
            print(f"  [{score.item():.4f}] \"{corr}\"")

    print("\nGRECO score interpretation:")
    print("  - Higher = better quality correction")
    print("  - Typical range: 0.0 - 1.0+ (can exceed 1.0)")

except Exception as e:
    print(f"\nGRECO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# Test 2: MPNet Semantic Similarity
# ============================================
print("\n" + "=" * 60)
print("TEST 2: MPNet Semantic Similarity")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F

    print("\nLoading MPNet from sentence-transformers/all-mpnet-base-v2...")
    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

    print("MPNet loaded successfully!")

    # Test semantic similarity
    print("\n--- Semantic Similarity Scores ---")
    for i, case in enumerate(test_cases):
        source = case["source"]
        print(f"\nSource {i+1}: \"{source}\"")

        with torch.no_grad():
            source_emb = mpnet.encode([source], convert_to_tensor=True)
            corr_embs = mpnet.encode(case["corrections"], convert_to_tensor=True)
            similarities = F.cosine_similarity(
                source_emb.expand(len(case["corrections"]), -1),
                corr_embs,
                dim=1
            )

        for j, (corr, sim) in enumerate(zip(case["corrections"], similarities)):
            print(f"  [{sim.item():.4f}] \"{corr}\"")

    print("\nSemantic similarity interpretation:")
    print("  - Range: -1.0 to 1.0 (typically 0.5 - 1.0 for related text)")
    print("  - Higher = more similar meaning to source")
    print("  - We WANT high similarity (meaning preserved)")

except Exception as e:
    print(f"\nMPNet FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# Test 3: Laziness Penalty
# ============================================
print("\n" + "=" * 60)
print("TEST 3: Laziness Penalty (Edit Distance)")
print("=" * 60)

def compute_laziness_penalty(source: str, hypothesis: str) -> float:
    similarity = SequenceMatcher(None, source.lower(), hypothesis.lower()).ratio()
    if similarity > 0.95:
        return (similarity - 0.95) * 20
    return 0.0

print("\n--- Laziness Penalty Scores ---")
for i, case in enumerate(test_cases):
    source = case["source"]
    print(f"\nSource {i+1}: \"{source}\"")

    for corr in case["corrections"]:
        sim = SequenceMatcher(None, source.lower(), corr.lower()).ratio()
        penalty = compute_laziness_penalty(source, corr)
        print(f"  [sim={sim:.3f}, penalty={penalty:.4f}] \"{corr}\"")

print("\nLaziness penalty interpretation:")
print("  - Penalty > 0 when text is >95% similar to source")
print("  - Higher penalty = model is being lazy (not fixing errors)")

# ============================================
# Test 4: Composite Reward
# ============================================
print("\n" + "=" * 60)
print("TEST 4: Composite Reward (0.6*GRECO + 0.3*Semantic - 0.1*Laziness)")
print("=" * 60)

try:
    print("\n--- Composite Reward Scores ---")
    for i, case in enumerate(test_cases):
        source = case["source"]
        print(f"\nSource {i+1}: \"{source}\"")

        with torch.no_grad():
            greco_scores = greco.score([source] * len(case["corrections"]), case["corrections"])
            source_emb = mpnet.encode([source], convert_to_tensor=True)
            corr_embs = mpnet.encode(case["corrections"], convert_to_tensor=True)
            semantic_scores = F.cosine_similarity(
                source_emb.expand(len(case["corrections"]), -1),
                corr_embs,
                dim=1
            )

        for j, corr in enumerate(case["corrections"]):
            g = greco_scores[j].item()
            s = semantic_scores[j].item()
            l = compute_laziness_penalty(source, corr)

            composite = 0.6 * g + 0.3 * s - 0.1 * l

            print(f"  [{composite:.4f}] G={g:.3f} S={s:.3f} L={l:.3f} | \"{corr}\"")

    print("\nComposite reward interpretation:")
    print("  - Higher = better overall (quality + meaning - laziness)")
    print("  - Perfect correction should score highest")
    print("  - Lazy (no change) should be penalized")
    print("  - Hallucination should have low semantic score")

except Exception as e:
    print(f"\nComposite reward test FAILED: {e}")

print("\n" + "=" * 60)
print("TESTS COMPLETE")
print("=" * 60)
