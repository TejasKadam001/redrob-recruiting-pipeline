"""
semantic.py — Text building + embedding pipeline.

Two modes:
  1. GPU mode (Colab): bge-large-en-v1.5 + bge-reranker-large (cross-encoder)
  2. CPU mode (ranking step): load pre-computed embeddings from .npy files

The 5-minute CPU window is for RANKING only. Embeddings are pre-computed on GPU.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from src.config import (
    JD_FULL_TEXT, JD_CORE, JD_CHUNKS,
    EMBED_MODEL_GPU, EMBED_MODEL_CPU, CROSS_ENCODER_MODEL,
    BGE_QUERY_INSTRUCTION, BGE_DOC_INSTRUCTION,
    STAGE2_TFIDF_TOP, STAGE2_SEMANTIC_TOP, STAGE3_CROSSENCODER_TOP,
)


# ─────────────────────────────────────────────────────────────
# TEXT BUILDER — Produces rich, weighted candidate text
# ─────────────────────────────────────────────────────────────

def build_candidate_text(c: dict, mode: str = "full") -> str:
    """
    Build the canonical text representation of a candidate.

    mode="full"  → for embedding (all signals)
    mode="rerank" → for cross-encoder (richer context)
    mode="tfidf"  → for BM25 pre-filter (include all raw tokens)
    """
    profile = c["profile"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    certs = c.get("certifications", [])
    sig = c.get("redrob_signals", {})

    # Career descriptions are the most valuable signal — DOUBLE-WEIGHTED
    career_text = " ".join(
        f"{r.get('title', '')} at {r.get('company', '')} ({r.get('industry', '')}): "
        f"{r.get('description', '')}"
        for r in career
    )

    skill_text = " ".join(
        f"{s['name']} ({s.get('proficiency', '')})"
        for s in skills
    )

    cert_text = " ".join(
        f"{cert.get('name', '')} by {cert.get('issuer', '')}"
        for cert in certs
    )

    # Verified assessment topics — extra trust
    assessment_text = " ".join(sig.get("skill_assessment_scores", {}).keys())

    if mode == "tfidf":
        # Include everything for BM25
        parts = [
            profile.get("headline", ""),
            profile.get("summary", ""),
            career_text,
            skill_text,
            cert_text,
            assessment_text,
            profile.get("current_title", ""),
            profile.get("current_industry", ""),
        ]
    elif mode == "rerank":
        # For cross-encoder: include behavioral context too
        rr = sig.get("recruiter_response_rate", 0)
        github = sig.get("github_activity_score", -1)
        yoe = profile.get("years_of_experience", 0)

        parts = [
            f"Title: {profile.get('current_title', '')}",
            f"Experience: {yoe} years",
            f"Headline: {profile.get('headline', '')}",
            f"Summary: {profile.get('summary', '')}",
            f"Career: {career_text}",
            f"Skills: {skill_text}",
            f"Certifications: {cert_text}",
            f"Assessments: {assessment_text}",
            f"Location: {profile.get('location', '')} {profile.get('country', '')}",
            f"RecruiterResponseRate: {rr:.2f}",
            f"GitHub: {github}",
        ]
    else:
        # Standard embedding mode
        parts = [
            profile.get("headline", ""),
            profile.get("summary", ""),
            # Double-weight career text (most informative)
            career_text,
            career_text,
            skill_text,
            cert_text,
            assessment_text,
            f"{profile.get('current_title', '')} {profile.get('current_industry', '')}",
        ]

    return " ".join(p for p in parts if p.strip())


# ─────────────────────────────────────────────────────────────
# BM25 / TF-IDF PRE-FILTER
# ─────────────────────────────────────────────────────────────

def bm25_prefilter(
    candidates: list,
    texts: List[str],
    top_k: int = STAGE2_TFIDF_TOP,
) -> tuple:
    """
    Fast TF-IDF/BM25 pre-filter. Reduces 100K → top_k for embedding.
    Returns (filtered_candidates, filtered_texts, tfidf_scores).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"  [BM25] Vectorizing {len(texts):,} candidates...")

    jd_queries = list(JD_CHUNKS.values())
    all_docs = jd_queries + texts

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),         # Trigrams catch "learning to rank", "vector search"
        max_features=80_000,
        sublinear_tf=True,          # Log-scale TF → reduces domination by frequent terms
        analyzer="word",
        stop_words="english",
        min_df=2,                    # Only terms appearing in ≥2 candidates
    )
    matrix = vectorizer.fit_transform(all_docs)

    jd_vecs = matrix[: len(jd_queries)]
    cand_vecs = matrix[len(jd_queries):]

    # Multi-query: take MAX similarity across all JD representations
    scores = np.zeros(len(candidates))
    for i in range(len(jd_queries)):
        sim = cosine_similarity(jd_vecs[i : i + 1], cand_vecs).flatten()
        scores = np.maximum(scores, sim)

    top_idx = np.argsort(scores)[::-1][:top_k]
    return (
        [candidates[i] for i in top_idx],
        [texts[i] for i in top_idx],
        scores[top_idx],
    )


# ─────────────────────────────────────────────────────────────
# BI-ENCODER EMBEDDINGS (GPU or load pre-computed)
# ─────────────────────────────────────────────────────────────

def encode_with_biencoder(
    texts: List[str],
    model_name: str,
    batch_size: int = 256,
    use_instruction: bool = True,
    instruction: str = BGE_DOC_INSTRUCTION,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode texts with a bi-encoder (sentence-transformers).
    Returns L2-normalized float32 embeddings.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  [Encoder] Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)

    if use_instruction and "bge" in model_name.lower():
        # BGE models use instruction prefix for queries, not documents
        # For candidate docs we skip the instruction (per BGE paper)
        encode_texts = texts
    else:
        encode_texts = texts

    print(f"  [Encoder] Encoding {len(texts):,} texts (batch={batch_size})...")
    embeddings = model.encode(
        encode_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_jd(model_name: str, device: str = "cpu") -> np.ndarray:
    """Encode the JD with BGE query instruction."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)

    query_text = BGE_QUERY_INSTRUCTION + JD_CORE
    jd_emb = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return jd_emb[0].astype(np.float32)


def semantic_similarity(
    cand_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """Cosine similarity (fast with normalized vectors = dot product)."""
    return np.dot(cand_embeddings, jd_embedding).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# CROSS-ENCODER RERANKER (GPU only — runs on top-500 only)
# ─────────────────────────────────────────────────────────────

def cross_encode_top_candidates(
    candidates: list,
    top_k: int = STAGE3_CROSSENCODER_TOP,
    model_name: str = CROSS_ENCODER_MODEL,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Cross-encoder reranking: (JD, candidate_text) → relevance score.
    BAAI/bge-reranker-large is the best open-source reranker available.

    Returns scores for the first top_k candidates.
    """
    from sentence_transformers import CrossEncoder

    print(f"  [CrossEncoder] Loading {model_name} on {device}...")
    model = CrossEncoder(model_name, device=device, max_length=512)

    jd_text = JD_FULL_TEXT.strip()
    pairs = [
        (jd_text, build_candidate_text(c, mode="rerank"))
        for c in candidates[:top_k]
    ]

    print(f"  [CrossEncoder] Scoring {len(pairs)} pairs...")
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Normalize to [0, 1] with sigmoid
    scores_normalized = 1.0 / (1.0 + np.exp(-scores))
    return scores_normalized.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD PRE-COMPUTED ARTIFACTS
# ─────────────────────────────────────────────────────────────

def save_precomputed(
    artifacts_dir: str,
    candidate_ids: List[str],
    embeddings: np.ndarray,
    cross_scores: Optional[np.ndarray],
    top_indices: np.ndarray,
):
    """Save pre-computed GPU artifacts for offline CPU ranking."""
    p = Path(artifacts_dir)
    p.mkdir(parents=True, exist_ok=True)

    np.save(p / "embeddings.npy", embeddings)
    np.save(p / "top_indices.npy", top_indices)

    import json
    with open(p / "candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)

    if cross_scores is not None:
        np.save(p / "cross_scores.npy", cross_scores)

    print(f"  [Save] Artifacts saved to {artifacts_dir}/")
    print(f"         embeddings: {embeddings.shape}")
    print(f"         top_indices: {top_indices.shape}")
    if cross_scores is not None:
        print(f"         cross_scores: {cross_scores.shape}")


def load_precomputed(artifacts_dir: str) -> dict:
    """Load pre-computed GPU artifacts for CPU ranking step."""
    p = Path(artifacts_dir)
    import json

    result = {
        "embeddings": np.load(p / "embeddings.npy"),
        "top_indices": np.load(p / "top_indices.npy"),
    }
    with open(p / "candidate_ids.json") as f:
        result["candidate_ids"] = json.load(f)

    # Cross-encoder scores (optional)
    cross_path = p / "cross_scores.npy"
    if cross_path.exists():
        result["cross_scores"] = np.load(cross_path)
    else:
        result["cross_scores"] = None

    # JD embedding (bge-large, 1024-dim) — CRITICAL for correct cosine similarity
    # If missing, ranker falls back to bge-small (WRONG dims → crash)
    jd_path = p / "jd_embedding.npy"
    if jd_path.exists():
        result["jd_embedding"] = np.load(jd_path)
        print(f"         jd_embedding: {result['jd_embedding'].shape}  (bge-large ✅)")
    else:
        result["jd_embedding"] = None
        print("  [WARN] jd_embedding.npy not found — will use bge-small fallback (dimension mismatch risk!)")
        print("         Re-run colab_precompute.py to generate it.")

    print(f"  [Load] Pre-computed artifacts loaded from {artifacts_dir}/")
    print(f"         embeddings: {result['embeddings'].shape}")
    return result
