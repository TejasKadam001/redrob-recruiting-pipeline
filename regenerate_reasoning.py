#!/usr/bin/env python3
"""
regenerate_reasoning.py — Rewrite the reasoning column only.

Use this after git pull when ranks/scores are already correct but reasoning
text needs updating. Does NOT re-run embeddings or re-ranking (~5 min vs 30 min).

Usage:
  python regenerate_reasoning.py \\
      --submission output/submission.csv \\
      --candidates candidates.jsonl \\
      --out output/submission_fixed.csv
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

from src.features import build_feature_vector
from src.reasoning import REASONING_VERSION, generate_reasoning


def load_candidates(path: str) -> dict:
    p = Path(path)
    profiles = {}
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for c in items:
            profiles[c["candidate_id"]] = c
    else:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    profiles[c["candidate_id"]] = c
    return profiles


def main():
    parser = argparse.ArgumentParser(description="Regenerate reasoning column only")
    parser.add_argument("--submission", required=True, help="Existing CSV with ranks/scores")
    parser.add_argument("--candidates", required=True, help="candidates.jsonl or .json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    print(f"Reasoning generator version: {REASONING_VERSION}")

    with open(args.submission, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 100:
        print(f"WARNING: expected 100 rows, got {len(rows)}", file=sys.stderr)

    profiles = load_candidates(args.candidates)
    today = date.today()
    missing = []

    for r in rows:
        cid = r["candidate_id"]
        if cid not in profiles:
            missing.append(cid)
            continue
        c = profiles[cid]
        feats = build_feature_vector(c, today)
        r["reasoning"] = generate_reasoning(
            c=c,
            rank=int(r["rank"]),
            submission_score=float(r["score"]),
            debug={"features": feats},
            today=today,
        )

    if missing:
        print(f"ERROR: {len(missing)} candidate IDs not found in {args.candidates}", file=sys.stderr)
        sys.exit(1)

    broken = sum(1 for r in rows if "limited retrieval/ranking-specific work in career history" in r["reasoning"])
    if broken > 0:
        print(f"ERROR: {broken}/100 rows still have old broken reasoning phrase.", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)

    print(f"Written {len(rows)} rows -> {args.out}")
    print(f"Rank-1 preview: {rows[0]['reasoning']}")


if __name__ == "__main__":
    main()
