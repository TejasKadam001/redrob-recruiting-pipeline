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
        return "Please upload a candidate file to begin.", None, None

    # Parse uploaded file
    try:
        content = Path(file.name).read_text(encoding="utf-8").strip()
        # Try JSON array or single object first
        try:
            data = json.loads(content)
            candidates = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # Try parsing concatenated multi-line JSON objects (like sequential JSON profiles)
            candidates = []
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                content_sub = content[pos:].lstrip()
                if not content_sub:
                    break
                pos += len(content[pos:]) - len(content_sub)
                
                try:
                    obj, index = decoder.raw_decode(content[pos:])
                    candidates.append(obj)
                    pos += index
                except json.JSONDecodeError:
                    # Fallback to line-by-line JSONL if raw_decode fails
                    try:
                        candidates = [json.loads(line) for line in content.splitlines() if line.strip()]
                        break
                    except Exception:
                        raise json.JSONDecodeError("Invalid JSON structure or concatenated stream", content, pos)
    except Exception as e:
        return f"Error parsing candidate dataset file: {e}", None, None

    if len(candidates) > 150:
        return f"Sandbox limit exceeded. Deployed instance supports up to 150 candidates. Uploaded dataset contains {len(candidates)} candidates.", None, None

    if len(candidates) == 0:
        return "No valid candidates identified in the uploaded file.", None, None

    # Run pipeline
    try:
        results = run_pipeline_on_sample(candidates)
    except Exception as e:
        return f"Execution pipeline error: {e}", None, None

    if not results:
        return "Zero candidates cleared the initial validation checks (potential data mismatch or honeypot detection).", None, None

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
        hp_flag  = "Flagged" if is_hp else "Clear"

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
            reasoning,
        ])

    headers = [
        "Rank", "Candidate ID", "Score", "Title", "YoE",
        "Location", "Response Rate", "GitHub", "Notice",
        "Semantic", "Rules", "HP Flag", "Evaluation Reasoning",
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
    import tempfile
    tmp_csv = os.path.join(tempfile.gettempdir(), "submission_preview.csv")
    with open(tmp_csv, "w", encoding="utf-8") as f:
        f.write(csv_content)

    # Stats summary
    honeypot_count = sum(
        1 for r in results[:top_n]
        if comprehensive_honeypot_check(r["candidate"])[0]
    )
    summary = (
        f"Evaluation status: Successfully processed {top_n} candidates\n"
        f"Honeypots identified in subset: {honeypot_count} ({honeypot_count/top_n:.0%})\n"
        f"Top Rank: {results[0]['candidate_id']} ({results[0]['candidate']['profile']['current_title']}, score={results[0]['submission_score']:.4f})\n"
        f"Score range: {results[0]['submission_score']:.4f} to {results[top_n-1]['submission_score']:.4f}"
    )

    return summary, table_rows, tmp_csv


# ─────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────

custom_css = """
/* Monochromatic Dark Theme inspired by openai.com/api */
:root, .dark {
    --background-fill-primary: #000000 !important;
    --background-fill-secondary: #000000 !important;
    --body-background-fill: #000000 !important;
    --border-color-accent: #1c1c1e !important;
    --border-color-primary: #1c1c1e !important;
    --block-border-color: #1c1c1e !important;
    --block-background-fill: #0c0c0e !important;
    --input-background-fill: #0c0c0e !important;
    --input-border-color: #1c1c1e !important;
    --input-text-color: #ffffff !important;
    --body-text-color: #e5e5e7 !important;
    --block-title-text-color: #ffffff !important;
    --block-label-text-color: #8e8e93 !important;
    
    --button-primary-background-fill: #ffffff !important;
    --button-primary-text-color: #000000 !important;
    --button-primary-background-fill-hover: #e5e5e7 !important;
    --button-primary-border-color: #ffffff !important;
    
    --button-secondary-background-fill: #000000 !important;
    --button-secondary-text-color: #ffffff !important;
    --button-secondary-border-color: #1c1c1e !important;
    --button-secondary-background-fill-hover: #0c0c0e !important;
    --button-secondary-border-color-hover: #ffffff !important;
    
    --table-border-color: #1c1c1e !important;
    --table-even-background-fill: #0c0c0e !important;
    --table-odd-background-fill: #0c0c0e !important;
    --table-row-focus: #1c1c1e !important;
    --table-header-background-fill: #0c0c0e !important;
    
    --checkbox-label-background-fill: #0c0c0e !important;
    --checkbox-label-border-color: #1c1c1e !important;
    --checkbox-border-color: #1c1c1e !important;
    --radio-circle-background-fill: #000000 !important;
    
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif !important;
}

body, html {
    background-color: #000000 !important;
    color: #e5e5e7 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    justify-content: center !important;
}

.gradio-container {
    width: 95% !important;
    max-width: 1500px !important;
    padding: 3rem 1.5rem !important;
    margin: 0 auto !important;
}

h1 {
    font-size: 3rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
    line-height: 1.15 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
    color: #ffffff !important;
}

h2 {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.03em !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    color: #ffffff !important;
}

h3 {
    font-size: 1.35rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
    color: #a1a1aa !important;
    margin-top: 1rem !important;
    margin-bottom: 1.5rem !important;
}

h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: #ffffff !important;
}

/* Component label and title styling */
.block-title, .gr-block-title, label > span, .block-label {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #a1a1aa !important;
}

/* Make table columns clean and thin */
.gr-dataframe table {
    border-collapse: collapse !important;
}

.gr-dataframe th {
    font-weight: 500 !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.07em !important;
    border-bottom: 1px solid #1c1c1e !important;
    color: #8e8e93 !important;
    padding: 12px 14px !important;
}

.gr-dataframe td {
    padding: 12px 14px !important;
    font-size: 0.8rem !important;
    border-bottom: 1px solid #0c0c0e !important;
}

/* Custom layout blocks */
.header-wrapper {
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #1c1c1e;
    padding-bottom: 2rem;
}

.info-banner {
    border: 1px solid #1c1c1e;
    background-color: #0c0c0e;
    padding: 1.25rem;
    border-radius: 4px;
    margin-bottom: 2rem;
}

.info-banner h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #ffffff;
}

.info-banner p {
    margin: 0;
    font-size: 0.8rem;
    color: #8e8e93;
    line-height: 1.5;
}

/* Hide theme toggle and footer if visible */
footer {
    display: none !important;
}
"""

with gr.Blocks(
    title="Redrob Candidate Ranker"
) as demo:
    gr.Markdown("""
    # Candidate Discovery and Ranking Engine
    ### Redrob Recruiting Pipeline — Technical Sandbox

    A high-throughput hybrid retrieval-and-ranking system designed to evaluate engineering candidate profiles against technical requirements.
    
    The engine processes raw profile data through a multi-stage funnel:
    - **Honeypot Filter**: Ten-point consistency check to eliminate adversarial profile configurations and inactive users.
    - **Semantic Retrieval**: CPU-fallback bi-encoder matching to evaluate semantic alignment with the job description.
    - **Feature Fusion**: Evaluates 40+ signals including domain fit, skills, experience, availability, and logistics.
    - **Reasoning Engine**: Generates precise, fact-based descriptions explaining rank placement.
    """)
    
    gr.HTML("""
    <div class="info-banner">
        <h4>System Configuration Note</h4>
        <p>This sandbox deployment runs in evaluation mode using a lightweight bi-encoder for real-time CPU processing. The production submission pipeline utilizes a deep bi-encoder combined with cross-encoder reranking pre-computed on a T4 GPU cluster to handle large-scale candidate sets.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Candidate Dataset (.json or .jsonl, limit: 150)",
                file_types=[".json", ".jsonl"],
            )
            run_btn = gr.Button("Evaluate and Rank", variant="primary", size="lg")

            gr.Markdown("""
            ### Dataset Schema
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
            Test files must conform to the specified Redrob schema. Use the provided sample candidate dataset to run a validation check.
            """)

        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Execution Summary",
                lines=5,
                interactive=False,
            )
            download_btn = gr.File(label="Download Submission CSV")

    results_table = gr.Dataframe(
        headers=[
            "Rank", "Candidate ID", "Score", "Title", "YoE",
            "Location", "Response Rate", "GitHub", "Notice",
            "Semantic", "Rules", "HP Flag", "Evaluation Reasoning",
        ],
        label="Evaluation Leaderboard",
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
    ### Methodological Details
    
    #### Pipeline Architecture
    ```
    Candidate Profiles → Honeypot Validation → Bi-Encoder Matching → Signal Synthesis (RRF) → Normalized Output
    ```
    
    #### RRF Synthesis and Calibration
    The final score is synthesized using Reciprocal Rank Fusion (RRF), which ensures robust aggregation of ranking signals across disparate scoring mechanisms without scale calibration issues.
    
    $$Score_{Final} = \\left( \\text{RRF\\_Score} \\times (k + 1) \\times \\text{BehavioralMultiplier} \\right) - \\text{Penalties}$$
    
    Where:
    - **RRF\\_Score** aggregates rank outputs from the semantic retriever and engineered rule models.
    - **BehavioralMultiplier** scales candidates based on outreach engagement probability (0.35x - 1.25x).
    - **Penalties** are applied for structural red flags (e.g., keyword stuffing, unrelated experience domains).
    
    *Developed for the Redrob Intelligent Candidate Discovery and Ranking Challenge.*
    """)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Default(primary_hue="neutral", secondary_hue="neutral", neutral_hue="neutral"),
        css=custom_css,
        js="""() => { document.documentElement.classList.add('dark'); }"""
    )

