"""Script to compute real empirical graph topology and narrator reliability score distributions."""
import sys
import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

def norm(t):
    if not t: return ""
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", str(t)).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def main():
    print("=" * 75)
    print("  COMPUTING REAL EMPIRICAL GRAPH TOPOLOGY & SCORE DISTRIBUTIONS")
    print("=" * 75)

    PROJECT = Path(".").resolve()
    NODE_NAMES = PROJECT / "output/stage2/node_names.json"
    SENTIMENT_CSV = PROJECT / "output/track_a_classifier/narrator_scores_sentiment.csv"
    GNN_PRED = PROJECT / "output/stage3/narrator_predictions.csv"
    ENC_PATH = PROJECT / "output/shamela/narrator_encyclopedia.json"
    ALIAS_PATH = PROJECT / "output/shamela/alias_index.json"

    # 1. Load Nodes
    id_to_name = json.load(open(NODE_NAMES, encoding="utf-8"))
    total_nodes = len(id_to_name)
    print(f"\n[1/3] Knowledge Graph Nodes: {total_nodes:,}")

    # 2. Load Sentiment & Classification Scores
    sent_df = pd.read_csv(SENTIMENT_CSV)
    annotated_nodes = (sent_df["critic_count"] > 0).sum()
    fallback_nodes = (sent_df["critic_count"] == 0).sum()

    annotated_pct = (annotated_nodes / total_nodes) * 100
    fallback_pct = (fallback_nodes / total_nodes) * 100

    print(f"\n[2/3] Table 5.5 Empirical Topology Verification:")
    print(f"  Total Graph Nodes (N):                {total_nodes:,}")
    print(f"  Explicitly Annotated Nodes (critic > 0): {annotated_nodes:,} ({annotated_pct:.2f}%)")
    print(f"  Unannotated / Fallback Nodes (critic = 0): {fallback_nodes:,} ({fallback_pct:.2f}%)")

    # 3. Classify Score Buckets for Table 5.7
    # Scores: 0.85 (thiqah), 0.70 (saduq), 0.40 (maqbul), 0.15 (daif), 0.05 (matruk), 0.632/0.30 (fallback)
    scores = sent_df["reliability_score"].values
    counts = {
        "thiqah (0.85)": np.sum(scores >= 0.80),
        "saduq (0.70)": np.sum((scores >= 0.55) & (scores < 0.80)),
        "maqbul (0.40)": np.sum((scores >= 0.30) & (scores < 0.55) & (sent_df["critic_count"] > 0)),
        "daif (0.15)": np.sum((scores >= 0.10) & (scores < 0.30) & (sent_df["critic_count"] > 0)),
        "matruk (0.05)": np.sum((scores < 0.10) & (sent_df["critic_count"] > 0)),
        "GNN Fallback (0.632/0.30)": fallback_nodes
    }

    print(f"\n[3/3] Table 5.7 Empirical Score Distribution Verification:")
    for tier, count in counts.items():
        pct = (count / total_nodes) * 100
        print(f"  {tier:<30s}: {count:>8,} nodes ({pct:6.2f}%)")

    # Save summary JSON
    summary = {
        "total_nodes": total_nodes,
        "annotated_nodes": int(annotated_nodes),
        "annotated_pct": annotated_pct,
        "fallback_nodes": int(fallback_nodes),
        "fallback_pct": fallback_pct,
        "tier_counts": {k: int(v) for k, v in counts.items()}
    }
    with open("output/final_evaluation/real_graph_distributions.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nResults saved to output/final_evaluation/real_graph_distributions.json")

if __name__ == "__main__":
    main()
