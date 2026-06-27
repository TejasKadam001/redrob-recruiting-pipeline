"""
ranker.py — Multi-stage ranking engine.
Orchestrates: hard-filter → BM25 → bi-encoder → cross-encoder → feature fusion.
"""

import numpy as np
from datetime import date
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.config import (
    STAGE_WEIGHTS, STAGE2_TFIDF_TOP, STAGE2_SEMANTIC_TOP,
    STAGE3_CROSSENCODER_TOP, FINAL_OUTPUT,
)
from src.honeypot import apply_honeypot_penalty, comprehensive_honeypot_check
from src.features import (
    build_feature_vector, compute_rule_score, compute_behavioral_multiplier,
    penalty_keyword_stuffer, penalty_wrong_domain,
    boost_hidden_gem,
)
from src.semantic import (
    build_candidate_text, bm25_prefilter,
    semantic_similarity, encode_jd, encode_with_biencoder,
    cross_encode_top_candidates,
)


# ─────────────────────────────────────────────────────────────
# STAGE 1 — HARD FILTER
# ─────────────────────────────────────────────────────────────

def stage1_hard_filter(candidates: list) -> Tuple[list, dict]:
    """
    Eliminate definite non-matches fast before expensive computation.
    Conservative — only removes clear honeypots and total ghosts.
    """
    passed, stats = [], {"honeypot": 0, "ghost": 0, "total_in": len(candidates)}
    today = date.today()

    for c in tqdm(candidates, desc="Stage1-filter", ncols=80):
        # --- Honeypot check (hard eliminate) ---
        is_hp, _ = comprehensive_honeypot_check(c)
        if is_hp:
            stats["honeypot"] += 1
            continue

        # --- Behavioral ghost: COMPLETELY unreachable (all 3 together) ---
        sig = c["redrob_signals"]
        try:
            last = date.fromisoformat(sig.get("last_active_date", "2020-01-01"))
            days_inactive = (today - last).days
        except Exception:
            days_inactive = 999

        rr = sig.get("recruiter_response_rate", 0.5)
        otw = sig.get("open_to_work_flag", False)

        if days_inactive > 450 and rr < 0.04 and not otw:
            stats["ghost"] += 1
            continue

        passed.append(c)

    stats["total_out"] = len(passed)
    print(f"  [Stage1] {stats['total_in']:,} → {stats['total_out']:,} "
          f"(removed {stats['honeypot']} honeypots, {stats['ghost']} ghosts)")
    return passed, stats


# ─────────────────────────────────────────────────────────────
# STAGE 2 — SEMANTIC MATCH (BM25 → Bi-encoder)
# ─────────────────────────────────────────────────────────────

def stage2_semantic_filter(
    candidates: list,
    precomputed: dict = None,
    model_name: str = None,
    device: str = "cpu",
) -> Tuple[list, np.ndarray]:
    """
    Two-pass semantic filtering: BM25 → bi-encoder embeddings.

    If precomputed artifacts exist (from Colab GPU run), use them directly.
    Otherwise compute on CPU (slower but still works).

    Returns (top_candidates, semantic_scores_for_top)
    """
    if precomputed is not None:
        return _stage2_from_precomputed(candidates, precomputed)
    return _stage2_compute(candidates, model_name, device)


def _stage2_from_precomputed(candidates: list, precomputed: dict) -> Tuple[list, np.ndarray]:
    """Use pre-computed embeddings from GPU run."""
    print("  [Stage2] Using pre-computed embeddings...")

    id_to_cand = {c["candidate_id"]: c for c in candidates}
    cand_ids   = precomputed["candidate_ids"]
    embeddings = precomputed["embeddings"]      # shape: (N, 1024) from bge-large

    # ── CRITICAL FIX: Load JD embedding saved by colab_precompute.py ──
    # Avoids dimension mismatch: bge-large(1024) vs bge-small(384)
    if "jd_embedding" in precomputed:
        jd_emb = precomputed["jd_embedding"]   # (1024,) from bge-large — CORRECT
        print("  [Stage2] JD embedding loaded from artifacts (bge-large dim).")
    else:
        # Fallback: CPU small model. Only works if candidates also use bge-small.
        from src.config import EMBED_MODEL_CPU
        print("  [Stage2] WARNING: jd_embedding.npy missing — using bge-small fallback.")
        print("           Re-run colab_precompute.py to generate jd_embedding.npy")
        jd_emb = encode_jd(EMBED_MODEL_CPU, device="cpu")

    sem_scores = semantic_similarity(embeddings, jd_emb)
    order      = np.argsort(sem_scores)[::-1][:STAGE2_SEMANTIC_TOP]

    top_candidates, top_scores = [], []
    for idx in order:
        cid = cand_ids[idx]
        if cid in id_to_cand:
            top_candidates.append(id_to_cand[cid])
            top_scores.append(float(sem_scores[idx]))

    print(f"  [Stage2] {len(top_candidates):,} candidates after semantic filter")
    return top_candidates, np.array(top_scores, dtype=np.float32)


def _stage2_compute(
    candidates: list,
    model_name: str,
    device: str,
) -> Tuple[list, np.ndarray]:
    """Compute embeddings on-the-fly (CPU mode, slower)."""
    from src.config import EMBED_MODEL_CPU
    if model_name is None:
        model_name = EMBED_MODEL_CPU

    print(f"  [Stage2] Building candidate texts...")
    texts = [build_candidate_text(c, mode="tfidf") for c in tqdm(candidates, ncols=80)]

    # BM25 pre-filter
    top_cands, top_texts, _ = bm25_prefilter(candidates, texts, top_k=STAGE2_TFIDF_TOP)

    # Bi-encoder on top 6000
    print(f"  [Stage2] Encoding {len(top_cands):,} candidates with {model_name}...")
    texts_emb = [build_candidate_text(c, mode="full") for c in top_cands]
    cand_embs = encode_with_biencoder(texts_emb, model_name, device=device)
    jd_emb = encode_jd(model_name, device=device)

    sem_scores = semantic_similarity(cand_embs, jd_emb)

    # Take top STAGE2_SEMANTIC_TOP
    top_idx = np.argsort(sem_scores)[::-1][:STAGE2_SEMANTIC_TOP]
    top_candidates = [top_cands[i] for i in top_idx]
    top_scores = sem_scores[top_idx]

    print(f"  [Stage2] {len(top_candidates):,} candidates after semantic filter")
    return top_candidates, top_scores


# ─────────────────────────────────────────────────────────────
# STAGE 3 — DEEP RANKING (Feature Fusion + Cross-encoder)
# ─────────────────────────────────────────────────────────────

def stage3_deep_rank(
    candidates: list,
    semantic_scores: np.ndarray,
    cross_scores: np.ndarray = None,
    today: date = None,
) -> List[dict]:
    """
    Final scoring using Reciprocal Rank Fusion (RRF).
    RRF combines ranks rather than raw scores, making it mathematically 
    robust against uncalibrated score distributions.
    
    RRF = w_ce/(k+Rank_ce) + w_sem/(k+Rank_sem) + w_rule/(k+Rank_rule)
    """
    if today is None:
        today = date.today()

    w_ce   = STAGE_WEIGHTS["cross_encoder"]
    w_sem  = STAGE_WEIGHTS["semantic_bge"]
    w_rule = STAGE_WEIGHTS["rule_features"]

    # If no cross-encoder, redistribute its weight to semantic
    if cross_scores is None:
        w_sem += w_ce
        w_ce = 0.0

    # 1. Gather all raw scores for independent ranking
    raw_data = []
    for i, c in enumerate(tqdm(candidates, desc="Stage3-features", ncols=80)):
        sem_raw = float(semantic_scores[i]) if i < len(semantic_scores) else 0.5
        sem_norm = (sem_raw + 1.0) / 2.0   # cosine sim [-1,1] → [0,1]

        if cross_scores is not None and i < len(cross_scores):
            ce_score = float(cross_scores[i])
        else:
            ce_score = sem_norm  # fallback to semantic

        features = build_feature_vector(c, today)
        rule_score = compute_rule_score(features)

        raw_data.append({
            "c": c,
            "sem": sem_norm,
            "ce": ce_score,
            "rule": rule_score,
            "features": features
        })

    # 2. Compute ranks for each dimension (1-indexed)
    rank_sem = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["sem"], reverse=True), 1)}
    rank_ce = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["ce"], reverse=True), 1)}
    rank_rule = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["rule"], reverse=True), 1)}

    results = []
    k = 60  # Standard RRF constant
    
    for item in raw_data:
        c = item["c"]
        cid = c["candidate_id"]
        
        r_sem = rank_sem[cid]
        r_ce = rank_ce[cid]
        r_rule = rank_rule[cid]

        # 3. Calculate Weighted RRF
        if cross_scores is None:
            # CPU fallback demo mode uses linear
            raw_score = w_sem * item["sem"] + w_rule * item["rule"]
        else:
            # Full Pipeline: Weighted RRF
            rrf_score = (w_sem * (1.0 / (k + r_sem))) + (w_ce * (1.0 / (k + r_ce))) + (w_rule * (1.0 / (k + r_rule)))
            # Multiply by (k+1) to scale the top rank back near 1.0
            raw_score = rrf_score * (k + 1)

        # 4. Apply Behavioral Multiplier (scales EV of engagement)
        beh_mult = compute_behavioral_multiplier(c, today)
        raw = raw_score * beh_mult

        # 5. Penalties (additive deductions)
        pen_stuffer = penalty_keyword_stuffer(c) * 0.25
        pen_domain  = penalty_wrong_domain(c)    * 0.15
        total_penalty = pen_stuffer + pen_domain
        raw -= total_penalty

        # 6. Hidden gem boost
        gem = boost_hidden_gem(c)
        raw += gem * 0.15

        # 7. Honeypot soft penalty
        raw = apply_honeypot_penalty(raw, c)

        # 8. Clamp
        final_score = max(0.0, min(1.0, raw))

        results.append({
            "candidate_id": cid,
            "score": final_score,
            "candidate": c,
            "_debug": {
                "sem_norm": round(item["sem"], 4),
                "ce_score": round(item["ce"], 4),
                "rule_score": round(item["rule"], 4),
                "rank_sem": r_sem,
                "rank_ce": r_ce,
                "rank_rule": r_rule,
                "beh_mult": round(beh_mult, 4),
                "penalty": round(total_penalty, 4),
                "gem_boost": round(gem, 4),
                "features": item["features"],
            },
        })

    # Sort: score descending, candidate_id ascending (tie-break per spec)
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return results


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    candidates: list,
    precomputed: dict = None,
    model_name: str = None,
    device: str = "cpu",
    use_cross_encoder: bool = False,
    cross_encoder_device: str = "cpu",
) -> List[dict]:
    """
    Full 3-stage pipeline from raw candidates list to ranked results.

    precomputed: dict from load_precomputed() — if provided, skips GPU embedding
    use_cross_encoder: only True on Colab GPU; too slow on CPU for 2000 candidates
    """
    import time
    t_start_total = time.time()

    print("\n═══ STAGE 1: Hard Filter ═══")
    t0 = time.time()
    stage1_candidates, _ = stage1_hard_filter(candidates)
    print(f"  [Time] Stage 1 completed in {time.time()-t0:.2f}s")

    print("\n═══ STAGE 2: Semantic Match ═══")
    t0 = time.time()
    stage2_candidates, sem_scores = stage2_semantic_filter(
        stage1_candidates,
        precomputed=precomputed,
        model_name=model_name,
        device=device,
    )
    print(f"  [Time] Stage 2 completed in {time.time()-t0:.2f}s")

    cross_scores = None
    if use_cross_encoder:
        from src.config import CROSS_ENCODER_MODEL
        print(f"\n═══ STAGE 2b: Cross-Encoder Rerank (top {STAGE3_CROSSENCODER_TOP}) ═══")
        t0 = time.time()
        cross_scores = cross_encode_top_candidates(
            stage2_candidates,
            top_k=STAGE3_CROSSENCODER_TOP,
            model_name=CROSS_ENCODER_MODEL,
            device=cross_encoder_device,
        )
        # Pad with 0.5 for candidates beyond top_k (not reranked)
        if len(cross_scores) < len(stage2_candidates):
            pad = np.full(len(stage2_candidates) - len(cross_scores), 0.50, dtype=np.float32)
            cross_scores = np.concatenate([cross_scores, pad])
        print(f"  [Time] Stage 2b completed in {time.time()-t0:.2f}s")

    print("\n═══ STAGE 3: Deep Feature Ranking ═══")
    t0 = time.time()
    results = stage3_deep_rank(
        stage2_candidates,
        sem_scores,
        cross_scores=cross_scores,
    )
    print(f"  [Time] Stage 3 completed in {time.time()-t0:.2f}s")

    bottom_idx = min(99, len(results) - 1)
    print(f"\n✅ Pipeline complete (Total: {time.time()-t_start_total:.2f}s). Top score: {results[0]['score']:.4f}  "
          f"Bottom-{bottom_idx+1} score: {results[bottom_idx]['score']:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# FINAL SUBMISSION FORMATTER
# ─────────────────────────────────────────────────────────────

def normalize_to_submission(results: List[dict]) -> List[dict]:
    """
    Normalize scores to [0.20, 0.99] range for submission.
    Ensures strict non-increasing scores and correct tie-breaking.
    """
    top100 = results[:100]
    scores = [r["score"] for r in top100]

    s_min, s_max = min(scores), max(scores)
    span = s_max - s_min if s_max > s_min else 1.0

    normalized = []
    for i, r in enumerate(top100):
        norm = 0.20 + (r["score"] - s_min) / span * 0.79
        normalized.append({**r, "submission_score": round(norm, 4)})

    # Enforce strict non-increasing (fix floating point)
    for i in range(1, len(normalized)):
        if normalized[i]["submission_score"] > normalized[i - 1]["submission_score"]:
            normalized[i]["submission_score"] = (
                normalized[i - 1]["submission_score"] - 0.0001
            )

    # Re-sort by submission_score desc, candidate_id asc
    normalized.sort(key=lambda x: (-x["submission_score"], x["candidate_id"]))
    return normalized
