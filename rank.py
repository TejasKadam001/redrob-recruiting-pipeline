#!/usr/bin/env python3
"""
rank.py — Main CLI entry point for CPU-only ranking step.

Usage (must complete in ≤5 minutes on CPU 16GB, no network):
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

If pre-computed artifacts exist (from colab_precompute.py):
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv \
                 --artifacts ./artifacts/

If no artifacts: runs CPU-only pipeline (slower but still works).
"""

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

from tqdm import tqdm

from src.ranker import run_full_pipeline, normalize_to_submission
from src.reasoning import generate_reasoning
from src.semantic import load_precomputed


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list:
    p = Path(path)
    print(f"Loading candidates from {p} ...")

    if not p.exists():
        print(f"ERROR: {p} not found.", file=sys.stderr)
        sys.exit(1)

    candidates = []
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        candidates = data if isinstance(data, list) else [data]
    else:
        # .jsonl (possibly gzipped)
        opener = __import__("gzip").open if p.suffix == ".gz" else open
        with opener(str(p), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    print(f"  Loaded {len(candidates):,} candidates.")
    return candidates


# ─────────────────────────────────────────────────────────────
# SUBMISSION WRITER
# ─────────────────────────────────────────────────────────────

def write_submission(results: list, out_path: str) -> None:
    today = date.today()
    rows = []

    for rank, r in enumerate(results, 1):
        reasoning = generate_reasoning(
            c=r["candidate"],
            rank=rank,
            submission_score=r["submission_score"],
            debug=r.get("_debug", {}),
            today=today,
        )
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": f"{r['submission_score']:.4f}",
            "reasoning": reasoning,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Submission written → {out_path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────
# QUICK AUDIT — Safety check before submission
# ─────────────────────────────────────────────────────────────

def audit_submission(results: list) -> bool:
    """Print a quick audit of the top-100 results. Returns True if clean."""
    from src.honeypot import comprehensive_honeypot_check

    print("\n─── Submission Audit ───")
    honeypots_found = 0
    warnings = []

    for rank, r in enumerate(results[:100], 1):
        c = r["candidate"]
        is_hp, flags = comprehensive_honeypot_check(c)
        if is_hp:
            honeypots_found += 1
            warnings.append(f"  ⚠ HONEYPOT rank {rank}: {r['candidate_id']} flags={[f[0] for f in flags]}")

        # Check scores are non-increasing
        if rank > 1:
            prev = results[rank - 2]["submission_score"]
            curr = r["submission_score"]
            if curr > prev + 0.0001:
                warnings.append(f"  ⚠ Score not decreasing at rank {rank}")

    hp_rate = honeypots_found / 100
    print(f"  Honeypot rate in top-100: {hp_rate:.1%}  (limit: 10%)")

    if warnings:
        for w in warnings:
            print(w)

    if hp_rate > 0.10:
        print("  ❌ DISQUALIFICATION RISK: honeypot rate > 10%!")
        return False

    print("  ✅ Audit passed.")
    return True


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl (or .json, .jsonl.gz)")
    parser.add_argument("--out", required=True,
                        help="Output CSV path, e.g. ./submission.csv")
    parser.add_argument("--artifacts", default=None,
                        help="Path to pre-computed GPU artifacts dir (from colab_precompute.py)")
    parser.add_argument("--model", default=None,
                        help="Override embedding model (default: BAAI/bge-small-en-v1.5 on CPU)")
    parser.add_argument("--device", default="cpu",
                        help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of candidates to output (default: 100)")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("  Redrob Candidate Ranker")
    print("=" * 60)

    # 1. Load candidates
    candidates = load_candidates(args.candidates)

    # 2. Load pre-computed artifacts (if available)
    precomputed = None
    if args.artifacts and Path(args.artifacts).exists():
        print(f"\nLoading pre-computed artifacts from {args.artifacts}/")
        precomputed = load_precomputed(args.artifacts)
    else:
        if args.artifacts:
            print(f"  (Artifacts dir '{args.artifacts}' not found — running CPU pipeline)")

    # 3. Run pipeline
    results = run_full_pipeline(
        candidates=candidates,
        precomputed=precomputed,
        model_name=args.model,
        device=args.device,
        use_cross_encoder=False,  # CPU: no cross-encoder (too slow for 2000 pairs)
    )

    # 4. Normalize scores
    results = normalize_to_submission(results)

    # 5. Audit
    clean = audit_submission(results)
    if not clean:
        print("\nFix issues before submitting!", file=sys.stderr)
        # Still write output so you can inspect it
        # (don't sys.exit — let user decide)

    # 6. Write CSV
    write_submission(results[:args.top_n], args.out)

    elapsed = time.time() - t_start
    print(f"\n⏱  Total runtime: {elapsed:.1f}s  "
          f"({'✅ within 5-min limit' if elapsed < 300 else '❌ OVER LIMIT'})")

if __name__ == "__main__":
    main()
