"""
app.py — HuggingFace Spaces sandbox for Redrob Candidate Ranker.

Accepts a JSON/JSONL file of ≤100 candidates, runs the full pipeline,
and outputs a ranked CSV + visual leaderboard.

Deploy to HuggingFace Spaces (Gradio SDK):
  1. Create a new Space: huggingface.co/new-space
  2. Select SDK: Gradio
  3. Upload this file + src/ folder + requirements_hf.txt
  4. Space URL is your sandbox_link
"""

import json
import csv
import io
import sys
import os
import functools
from datetime import date
from pathlib import Path

import gradio as gr

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import EMBED_MODEL_CPU, STAGE_WEIGHTS
from src.honeypot import comprehensive_honeypot_check
from src.features import (
    build_feature_vector, compute_rule_score,
    compute_behavioral_multiplier, penalty_keyword_stuffer,
    penalty_wrong_domain, boost_hidden_gem,
)
from src.honeypot import apply_honeypot_penalty
from src.reasoning import generate_reasoning
from src.ranker import normalize_to_submission
from src.semantic import build_candidate_text, bm25_prefilter, encode_with_biencoder, encode_jd, semantic_similarity

# ─────────────────────────────────────────────────────────────
# PIPELINE (CPU, small sample)
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading {EMBED_MODEL_CPU} (cached)...")
    return SentenceTransformer(EMBED_MODEL_CPU, device="cpu")

def run_pipeline_on_sample(candidates: list) -> list:
    """Full pipeline on ≤100 candidates. CPU-only, no GPU."""
    today = date.today()

    # Stage 1: Honeypot filter
    clean = []
    for c in candidates:
        is_hp, _ = comprehensive_honeypot_check(c)
        if not is_hp:
            clean.append(c)

    if not clean:
        return []

    # Stage 2: Semantic (CPU, small sample — no BM25 pre-filter needed)
    texts = [build_candidate_text(c, mode="full") for c in clean]
    model_name = EMBED_MODEL_CPU  # Fast small model for demo

    from src.config import JD_CORE, BGE_QUERY_INSTRUCTION
    import numpy as np

    model = get_model()
    cand_embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    jd_emb = model.encode(
        [BGE_QUERY_INSTRUCTION + JD_CORE],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]
    sem_scores = np.dot(cand_embs, jd_emb)

    # Stage 3: Feature fusion (RRF)
    w_sem  = STAGE_WEIGHTS["semantic_bge"] + STAGE_WEIGHTS["cross_encoder"]  # no CE on CPU demo
    w_rule = STAGE_WEIGHTS["rule_features"]

    raw_data = []
    for i, c in enumerate(clean):
        sem_norm = (float(sem_scores[i]) + 1.0) / 2.0
        feats = build_feature_vector(c, today)
        rule  = compute_rule_score(feats)
        raw_data.append({"c": c, "sem": sem_norm, "rule": rule, "feats": feats})

    rank_sem = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["sem"], reverse=True), 1)}
    rank_rule = {item["c"]["candidate_id"]: r for r, item in enumerate(sorted(raw_data, key=lambda x: x["rule"], reverse=True), 1)}

    results = []
    k = 60
    for item in raw_data:
        c = item["c"]
        cid = c["candidate_id"]
        
        rrf_score = (w_sem * (1.0 / (k + rank_sem[cid]))) + (w_rule * (1.0 / (k + rank_rule[cid])))
        raw = rrf_score * (k + 1)
        
        raw *= compute_behavioral_multiplier(c, today)
        
        pen = (penalty_keyword_stuffer(c) * 0.25 + penalty_wrong_domain(c) * 0.15)
        raw -= pen
        raw += boost_hidden_gem(c) * 0.15
        raw = apply_honeypot_penalty(raw, c)
        final = max(0.0, min(1.0, raw))

        results.append({
            "candidate_id": cid,
            "score": final,
            "candidate": c,
            "_debug": {
                "sem_norm": round(item["sem"], 4),
                "rule_score": round(item["rule"], 4),
                "beh_mult": round(compute_behavioral_multiplier(c, today), 4),
                "penalty": round(pen, 4),
                "features": item["feats"],
            },
        })

    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return normalize_to_submission(results)


# ─────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────

def rank_candidates(file):
    """Main Gradio handler: reads uploaded file, runs ranker, returns table + CSV."""
    if file is None:
        return "❌ Please upload a file.", None, None

    # Parse uploaded file
    try:
        content = Path(file.name).read_text(encoding="utf-8")
        # Try JSON array first
        try:
            data = json.loads(content)
            candidates = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # Try JSONL
            candidates = [json.loads(line) for line in content.splitlines() if line.strip()]
    except Exception as e:
        return f"❌ Could not parse file: {e}", None, None

    if len(candidates) > 150:
        return f"❌ Demo limited to 150 candidates. Uploaded {len(candidates)}.", None, None

    if len(candidates) == 0:
        return "❌ No candidates found in file.", None, None

    # Run pipeline
    try:
        results = run_pipeline_on_sample(candidates)
    except Exception as e:
        return f"❌ Pipeline error: {e}", None, None

    if not results:
        return "❌ All candidates were filtered (likely honeypots or empty file).", None, None

    top_n = min(100, len(results))
    today = date.today()

    # Build display table
    table_rows = []
    for rank, r in enumerate(results[:top_n], 1):
        c  = r["candidate"]
        p  = c["profile"]
        sig = c["redrob_signals"]
        dbg = r.get("_debug", {})

        is_hp, _ = comprehensive_honeypot_check(c)
        hp_flag  = "🍯 HP" if is_hp else ""

        reasoning = generate_reasoning(
            c=c, rank=rank,
            submission_score=r["submission_score"],
            debug=dbg, today=today,
        )

        table_rows.append([
            rank,
            r["candidate_id"],
            f"{r['submission_score']:.4f}",
            p.get("current_title", ""),
            f"{p.get('years_of_experience', 0):.1f}",
            p.get("location", ""),
            f"{sig.get('recruiter_response_rate', 0):.0%}",
            f"{sig.get('github_activity_score', -1):.0f}",
            f"{sig.get('notice_period_days', 0)}d",
            f"{dbg.get('sem_norm', 0):.3f}",
            f"{dbg.get('rule_score', 0):.3f}",
            hp_flag,
            reasoning[:120] + "…" if len(reasoning) > 120 else reasoning,
        ])

    headers = [
        "Rank", "Candidate ID", "Score", "Title", "YoE",
        "Location", "Response Rate", "GitHub", "Notice",
        "Semantic", "Rules", "HP?", "Reasoning",
    ]

    # Build CSV output
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for rank, r in enumerate(results[:top_n], 1):
        reasoning = generate_reasoning(
            c=r["candidate"], rank=rank,
            submission_score=r["submission_score"],
            debug=r.get("_debug", {}), today=today,
        )
        writer.writerow({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": f"{r['submission_score']:.4f}",
            "reasoning": reasoning,
        })
    csv_content = csv_buf.getvalue()

    # Write temp CSV for download
    tmp_csv = "/tmp/submission_preview.csv"
    with open(tmp_csv, "w", encoding="utf-8") as f:
        f.write(csv_content)

    # Stats summary
    honeypot_count = sum(
        1 for r in results[:top_n]
        if comprehensive_honeypot_check(r["candidate"])[0]
    )
    summary = (
        f"✅ Ranked {top_n} candidates\n"
        f"🍯 Honeypots in top-{top_n}: {honeypot_count} ({honeypot_count/top_n:.0%})\n"
        f"🏆 Top candidate: {results[0]['candidate_id']} "
        f"({results[0]['candidate']['profile']['current_title']}, "
        f"score={results[0]['submission_score']:.4f})\n"
        f"📊 Score range: {results[0]['submission_score']:.4f} → {results[top_n-1]['submission_score']:.4f}"
    )

    return summary, table_rows, tmp_csv


# ─────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Redrob Candidate Ranker",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:
    gr.Markdown("""
    # 🎯 Redrob Intelligent Candidate Ranker
    ### India Runs Data & AI Challenge — Candidate Discovery System

    Upload a JSON or JSONL file of candidates (≤150). The system will:
    - 🍯 **Detect honeypots** (10-check detector)
    - 🧠 **Semantically match** to the Senior ML Engineer JD (bge-small bi-encoder)
    - 📊 **Score 40+ features**: domain fit, skills, experience, behavioral signals, logistics
    - ✍️ **Generate reasoning** for each candidate
    """)
    
    gr.HTML("""
    <div style="background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; border: 1px solid #ffeeba; margin-bottom: 20px;">
        <h4 style="margin-top: 0; margin-bottom: 5px;">⚠️ CPU HEURISTIC DEMO</h4>
        <p style="margin: 0;">This UI demo uses a lightweight model (<strong>bge-small</strong>) and skips the cross-encoder for speed.<br>
        <strong>The final hackathon submission</strong> uses <strong>bge-large</strong> + <strong>Cross-Encoder reranking</strong> via Colab GPU precomputation.<br>
        <em>Both use Reciprocal Rank Fusion (RRF), but scores will differ slightly due to the models.</em></p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload candidates.json or candidates.jsonl (≤150 candidates)",
                file_types=[".json", ".jsonl"],
            )
            run_btn = gr.Button("🚀 Rank Candidates", variant="primary", size="lg")

            gr.Markdown("""
            ### 📋 Sample Input Format
            ```json
            [
              {
                "candidate_id": "CAND_0000001",
                "profile": { ... },
                "career_history": [ ... ],
                "skills": [ ... ],
                "redrob_signals": { ... }
              }
            ]
            ```
            Use `sample_candidates.json` from the hackathon bundle to test.
            """)

        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="📊 Summary",
                lines=5,
                interactive=False,
            )
            download_btn = gr.File(label="⬇️ Download submission.csv")

    results_table = gr.Dataframe(
        headers=[
            "Rank", "Candidate ID", "Score", "Title", "YoE",
            "Location", "Response Rate", "GitHub", "Notice",
            "Semantic", "Rules", "HP?", "Reasoning",
        ],
        label="🏆 Ranked Candidates",
        wrap=True,
        interactive=False,
    )

    run_btn.click(
        fn=rank_candidates,
        inputs=[file_input],
        outputs=[status_box, results_table, download_btn],
        show_progress=True,
    )

    gr.Markdown("""
    ---
    ### 🔬 Architecture
    ```
    Candidates → Honeypot Filter → Semantic Matching (bge-small) → Feature Fusion (40+ signals) → Ranked Output
    ```
    **Full submission pipeline** (Colab GPU): `bge-large-en-v1.5` + `bge-reranker-large` cross-encoder

    **Score formula**: `Final = RRF_Score × (k+1) × BehavioralMultiplier − Penalties` (where `RRF_Score = sum(weight_i / (60 + rank_i))`)

    **Behavioral signals used**: response rate, recency, open-to-work, notice period, GitHub activity, interview completion, profile trust
    """)

if __name__ == "__main__":
    demo.launch()
