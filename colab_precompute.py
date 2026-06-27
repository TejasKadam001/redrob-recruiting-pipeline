"""
colab_precompute.py — Run this ONCE on Google Colab (T4 GPU).
Generates all heavy embeddings + cross-encoder scores.
Saves artifacts/ folder which rank.py uses for fast CPU ranking.

COLAB SETUP (read before running):
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload candidates.jsonl to /content/
  3. Clone repo (Cell 1 below) — run `git pull` every session before the pipeline
  4. Do NOT manually upload an old project folder; that skips the reasoning fix

  If you already ran the 30-min pipeline and only reasoning is wrong:
    !cd /content/project && git pull
    !python regenerate_reasoning.py --submission /content/submission.csv \\
        --candidates /content/candidates.jsonl --out /content/submission_fixed.csv

STRATEGY (why this beats everyone):
  - bge-large-en-v1.5 (335M params) = best open bi-encoder for retrieval
  - bge-reranker-large = best open cross-encoder reranker
  - GPU makes this 20x faster than CPU
  - Pre-computed artifacts → rank.py runs in <60 seconds on CPU
  - Total Colab time: ~25-35 minutes for 100K candidates
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ════════════════════════════════════════════════════════════
# !pip install -q sentence-transformers scikit-learn tqdm torch transformers

import os, sys, json, time, gc
import numpy as np
from pathlib import Path
from datetime import date
from tqdm import tqdm

# Project root — MUST match where you cloned the repo (git pull before each run!)
PROJECT_ROOT = "/content/project"
if not Path(PROJECT_ROOT).joinpath("src", "reasoning.py").exists():
    PROJECT_ROOT = "/content"  # fallback if cloned flat into /content
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

CANDIDATES_PATH = "/content/candidates.jsonl"
ARTIFACTS_DIR   = "/content/artifacts"
Path(ARTIFACTS_DIR).mkdir(exist_ok=True)

print("✅ Setup complete")
print(f"   CUDA available: {__import__('torch').cuda.is_available()}")


# ════════════════════════════════════════════════════════════
# CELL 2 — Load all 100K candidates
# ════════════════════════════════════════════════════════════

def load_candidates(path):
    candidates = []
    opener = __import__("gzip").open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(candidates):,} candidates")
    return candidates

candidates = load_candidates(CANDIDATES_PATH)


# ════════════════════════════════════════════════════════════
# CELL 3 — Stage 1: Hard filter (honeypots + ghosts)
# ════════════════════════════════════════════════════════════

from src.honeypot import comprehensive_honeypot_check

def hard_filter(candidates):
    passed, n_hp, n_ghost = [], 0, 0
    today = date.today()
    for c in tqdm(candidates, desc="Hard filter"):
        is_hp, _ = comprehensive_honeypot_check(c)
        if is_hp:
            n_hp += 1
            continue
        sig = c["redrob_signals"]
        try:
            from datetime import date as d
            days = (today - d.fromisoformat(sig.get("last_active_date","2020-01-01"))).days
        except Exception:
            days = 999
        rr  = sig.get("recruiter_response_rate", 0.5)
        otw = sig.get("open_to_work_flag", False)
        if days > 450 and rr < 0.04 and not otw:
            n_ghost += 1
            continue
        passed.append(c)
    print(f"Hard filter: {len(candidates):,} → {len(passed):,}  "
          f"(removed {n_hp} honeypots, {n_ghost} ghosts)")
    return passed

stage1 = hard_filter(candidates)
del candidates  # free memory
gc.collect()


# ════════════════════════════════════════════════════════════
# CELL 4 — BM25 pre-filter: 25K → 6K
# ════════════════════════════════════════════════════════════

from src.semantic import build_candidate_text, bm25_prefilter

print("Building candidate texts for BM25...")
texts_bm25 = [build_candidate_text(c, mode="tfidf") for c in tqdm(stage1, ncols=80)]

stage2_cands, stage2_texts, bm25_scores = bm25_prefilter(
    stage1, texts_bm25, top_k=6000
)
print(f"BM25 top-6000 selected. Score range: {bm25_scores.min():.4f} – {bm25_scores.max():.4f}")
del texts_bm25
gc.collect()


# ════════════════════════════════════════════════════════════
# CELL 5 — Bi-encoder embeddings with bge-large (GPU)
# ════════════════════════════════════════════════════════════

from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL_GPU, BGE_QUERY_INSTRUCTION, JD_CORE

DEVICE = "cuda"
BATCH  = 256

print(f"Loading {EMBED_MODEL_GPU} on {DEVICE}...")
biencoder = SentenceTransformer(EMBED_MODEL_GPU, device=DEVICE)

# Build embedding texts (double-weight career descriptions)
print("Building embedding texts...")
embed_texts = [build_candidate_text(c, mode="full") for c in tqdm(stage2_cands, ncols=80)]

print(f"Encoding {len(embed_texts):,} candidates...")
t0 = time.time()
cand_embeddings = biencoder.encode(
    embed_texts,
    batch_size=BATCH,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Encoded in {time.time()-t0:.1f}s  shape={cand_embeddings.shape}")

# Encode JD with instruction prefix
jd_query = BGE_QUERY_INSTRUCTION + JD_CORE
jd_emb = biencoder.encode(
    [jd_query], normalize_embeddings=True, convert_to_numpy=True
)[0]
print(f"JD embedding shape: {jd_emb.shape}")

# Compute similarities
sem_scores = np.dot(cand_embeddings, jd_emb)

# Take top 2000
top2k_idx   = np.argsort(sem_scores)[::-1][:2000]
top2k_cands = [stage2_cands[i] for i in top2k_idx]
top2k_embs  = cand_embeddings[top2k_idx]
top2k_sems  = sem_scores[top2k_idx]
top2k_ids   = [c["candidate_id"] for c in top2k_cands]

print(f"Top-2000 sem score range: {top2k_sems.min():.4f} – {top2k_sems.max():.4f}")

# ── CRITICAL: Save JD embedding WITH artifacts so rank.py uses bge-large dims ──
np.save(f"{ARTIFACTS_DIR}/jd_embedding.npy", jd_emb.astype(np.float32))
print(f"Saved jd_embedding.npy  shape={jd_emb.shape}  (bge-large, 1024-dim)")

del biencoder, cand_embeddings, embed_texts
gc.collect()
__import__("torch").cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# CELL 6 — Cross-encoder reranking top-500 with bge-reranker-large (GPU)
# ════════════════════════════════════════════════════════════

from sentence_transformers import CrossEncoder
from src.config import CROSS_ENCODER_MODEL, JD_FULL_TEXT

RERANK_TOP = 500

print(f"Loading {CROSS_ENCODER_MODEL} on {DEVICE}...")
reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE, max_length=512)

pairs = [
    (JD_FULL_TEXT.strip(), build_candidate_text(c, mode="rerank"))
    for c in top2k_cands[:RERANK_TOP]
]

print(f"Cross-encoding {len(pairs)} pairs...")
t0 = time.time()
ce_scores_raw = reranker.predict(
    pairs,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Cross-encoder done in {time.time()-t0:.1f}s")

# Sigmoid normalize to [0,1]
ce_scores = (1.0 / (1.0 + np.exp(-ce_scores_raw))).astype(np.float32)

# Pad remaining 1500 candidates with 0.5 (not reranked)
ce_scores_full = np.full(len(top2k_cands), 0.50, dtype=np.float32)
ce_scores_full[:RERANK_TOP] = ce_scores

print(f"Cross-encoder score range (top-500): {ce_scores.min():.4f} – {ce_scores.max():.4f}")

del reranker, pairs, ce_scores_raw
gc.collect()
__import__("torch").cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# CELL 7 — Feature engineering + final fusion
# ════════════════════════════════════════════════════════════

from src.config import STAGE_WEIGHTS
from src.features import (
    build_feature_vector, compute_rule_score, compute_behavioral_multiplier,
    penalty_keyword_stuffer, penalty_wrong_domain,
    boost_hidden_gem,
)
from src.honeypot import apply_honeypot_penalty

w_ce   = STAGE_WEIGHTS["cross_encoder"]
w_sem  = STAGE_WEIGHTS["semantic_bge"]
w_rule = STAGE_WEIGHTS["rule_features"]

today = date.today()

# 1. Gather raw scores
raw_data = []
for i, c in enumerate(tqdm(top2k_cands, desc="Feature fusion", ncols=80)):
    sem_norm = (float(top2k_sems[i]) + 1.0) / 2.0
    ce  = float(ce_scores_full[i])
    feats = build_feature_vector(c, today)
    rule  = compute_rule_score(feats)
    
    raw_data.append({
        "c": c, "sem": sem_norm, "ce": ce, "rule": rule, "feats": feats
    })

# 2. Compute ranks
rank_sem = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["sem"], reverse=True), 1)}
rank_ce = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["ce"], reverse=True), 1)}
rank_rule = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["rule"], reverse=True), 1)}

# 3. Apply RRF
results = []
k = 60

for item in raw_data:
    c = item["c"]
    cid = c["candidate_id"]
    r_sem = rank_sem[cid]
    r_ce = rank_ce[cid]
    r_rule = rank_rule[cid]
    
    rrf_score = (w_sem * (1.0 / (k + r_sem))) + (w_ce * (1.0 / (k + r_ce))) + (w_rule * (1.0 / (k + r_rule)))
    raw = rrf_score * (k + 1)
    
    raw *= compute_behavioral_multiplier(c, today)
    
    # Penalties
    pen = (penalty_keyword_stuffer(c) * 0.25 + penalty_wrong_domain(c) * 0.15)
    raw -= pen
    
    raw += boost_hidden_gem(c) * 0.15
    raw = apply_honeypot_penalty(raw, c)
    final = max(0.0, min(1.0, raw))

    results.append({
        "candidate_id": c["candidate_id"],
        "score": final,
        "candidate": c,
        "_debug": {
            "sem_norm": round(sem_norm, 4),
            "ce_score": round(ce, 4),
            "rule_score": round(rule, 4),
            "beh_mult": round(compute_behavioral_multiplier(c, today), 4),
            "penalty": round(pen, 4),
            "gem_boost": round(boost_hidden_gem(c), 4),
            "features": feats,
        },
    })

results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
bottom_idx = min(99, len(results) - 1)
print(f"\nTop score:    {results[0]['score']:.4f}  ({results[0]['candidate_id']})")
if bottom_idx >= 0:
    print(f"Rank-{bottom_idx+1} score: {results[bottom_idx]['score']:.4f}  ({results[bottom_idx]['candidate_id']})")


# ════════════════════════════════════════════════════════════
# CELL 8 — Normalize scores + write submission CSV
# ════════════════════════════════════════════════════════════

import csv
import importlib
from src.ranker import normalize_to_submission

# Force reload after git pull — Colab caches old src.reasoning in sys.modules
import src.reasoning as _reasoning_mod
importlib.reload(_reasoning_mod)
from src.reasoning import REASONING_VERSION, generate_reasoning

print(f"Reasoning generator version: {REASONING_VERSION}")
if REASONING_VERSION != "2":
    raise RuntimeError(
        "Stale reasoning.py detected. Run:  %cd /content/project && git pull"
    )

results_norm = normalize_to_submission(results)

OUT_CSV = "/content/submission.csv"
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for rank, r in enumerate(results_norm[:100], 1):
        reasoning = generate_reasoning(
            c=r["candidate"],
            rank=rank,
            submission_score=r["submission_score"],
            debug=r.get("_debug", {}),
            today=today,
        )
        writer.writerow({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": f"{r['submission_score']:.4f}",
            "reasoning": reasoning,
        })

print(f"Submission written: {OUT_CSV}")

# Sanity check — old broken template always contained this exact phrase on rank 1
_sample = results_norm[0]
_sample_text = generate_reasoning(
    c=_sample["candidate"], rank=1,
    submission_score=_sample["submission_score"],
    debug=_sample.get("_debug", {}), today=today,
)
if "limited retrieval/ranking-specific work in career history" in _sample_text:
    raise RuntimeError(
        "Reasoning still using OLD template. Run:  %cd /content/project && git pull  "
        "then re-run this cell (or use regenerate_reasoning.py — no full pipeline needed)."
    )
print(f"Rank-1 reasoning preview: {_sample_text[:120]}...")


# NOTE: We intentionally do NOT use any generative LLM (local or API) for reasoning.
# The hackathon rules penalize model usage. Our reasoning is generated via
# engineered seeded-template logic in src/reasoning.py, which produces
# professional, specific, non-hallucinated output from structured candidate data.


# ════════════════════════════════════════════════════════════
# CELL 9 — Save artifacts for CPU reproduction
# ════════════════════════════════════════════════════════════

from src.semantic import save_precomputed

save_precomputed(
    artifacts_dir=ARTIFACTS_DIR,
    candidate_ids=top2k_ids,
    embeddings=top2k_embs,
    cross_scores=ce_scores_full,
    top_indices=np.arange(len(top2k_ids)),
)


# ════════════════════════════════════════════════════════════
# CELL 10 — Validate + audit
# ════════════════════════════════════════════════════════════

import subprocess
val = subprocess.run(["python", "validate_submission.py", OUT_CSV],
                     capture_output=True, text=True)
print(val.stdout or val.stderr)

# Honeypot audit
from src.honeypot import comprehensive_honeypot_check
hp_count = sum(1 for r in results_norm[:100]
               if comprehensive_honeypot_check(r["candidate"])[0])
print(f"Honeypot rate in top-100: {hp_count}/100  "
      f"({'✅ safe' if hp_count <= 10 else '❌ OVER LIMIT'})")

# Show top-10 cards
print("\n═══ TOP 10 CANDIDATES ═══")
for rank, r in enumerate(results_norm[:10], 1):
    c = r["candidate"]
    p = c["profile"]
    sig = c["redrob_signals"]
    print(f"\n#{rank}  {r['candidate_id']}  score={r['submission_score']:.4f}")
    print(f"   {p['current_title']} | {p['years_of_experience']}yrs | {p['current_company']}")
    print(f"   RR={sig['recruiter_response_rate']:.2f}  "
          f"Notice={sig['notice_period_days']}d  "
          f"GitHub={sig['github_activity_score']}")
    top_skills = [s['name'] for s in c.get('skills', [])[:4]]
    print(f"   Skills: {', '.join(top_skills)}")

print("\n\n🏆 DONE. Download /content/submission.csv")
print("📦 Download /content/artifacts/ for CPU reproduction")
