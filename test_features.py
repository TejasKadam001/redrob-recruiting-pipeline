"""
test_features.py — Feature Extraction Demonstration

This script demonstrates the extraction of the 40+ features used in the Redrob Candidate Ranker.
It loads a sample candidate and prints the extracted feature vector and the final rule-based score.

How to run:
    python test_features.py
"""

import json
from datetime import date
from src.features import build_feature_vector, compute_rule_score, compute_behavioral_multiplier

def run_test():
    print("="*60)
    print(" 🚀 Redrob Features Extraction Test")
    print("="*60)

    try:
        with open("sample_candidates.json", "r") as f:
            candidates = json.load(f)
    except Exception as e:
        print(f"Error loading sample_candidates.json: {e}")
        return

    if not candidates:
        print("No candidates found.")
        return

    # Pick the first valid candidate
    c = candidates[0]
    today = date.today()

    print(f"\nAnalyzing Candidate: {c.get('candidate_id', 'Unknown')}")
    print(f"Title: {c.get('profile', {}).get('current_title', 'Unknown')}")
    
    # Extract Features
    features = build_feature_vector(c, today)
    
    print("\n📊 Extracted Features:")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"  {key:<30}: {value:.4f}")
        else:
            print(f"  {key:<30}: {value}")

    # Compute Scores
    rule_score = compute_rule_score(features)
    beh_mult = compute_behavioral_multiplier(c, today)
    
    print("\n📈 Final Feature Scores:")
    print(f"  Rule Score (0.0 to 1.0):      {rule_score:.4f}")
    print(f"  Bayesian Behavior Multiplier: {beh_mult:.4f}")
    
    print("\n✅ Run successful.")

if __name__ == "__main__":
    run_test()
