"""
Sentiment-Based Jumhūr Scoring (Multi-View Aggregation).

Splits biographical entries into individual critic statements, runs AraBERT
sentence-level classification, computes expected continuous scores, and aggregates
them into reliability and agreement consensus scores.
"""

import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parent.parent
ENC_PATH = PROJECT / "output/shamela/narrator_encyclopedia.json"
ALIAS_PATH = PROJECT / "output/shamela/alias_index.json"
NODE_NAMES_PATH = PROJECT / "output/stage2/node_names.json"
MODEL_DIR = PROJECT / "output/track_a_classifier/arabert_classifier"
OUT_CSV = PROJECT / "output/track_a_classifier/narrator_scores_sentiment.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_TO_SCORE = {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.70, 4: 0.85}
CLASS_SCORES = np.array([0.05, 0.15, 0.40, 0.70, 0.85], dtype=np.float32)

def norm(t):
    if not t: return ""
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def split_critic_statements(text: str) -> list[tuple[str, str]]:
    """Split biography text into (critic_name, statement_text) pairs."""
    # Pattern to match: "وقال أحمد بن حنبل:" or "وقال يحيى بن معين قال:" or "وقال النسائي:"
    # We clean diacritics to ensure matching is robust.
    text_clean = re.sub(r"[ً-ٌٍ-َّْٰ]", "", text)
    
    # Split using introduction verbs and colon or فقال introduction
    pattern = r'(?:^|\n|[\.\n])\s*(?:وقال|قال|ذكر|سئل|روى\s+عنه\s+فقال|عنه\s+قال)[:\s]+([^:.]+?)[:\s]+'
    matches = list(re.finditer(pattern, text_clean))
    
    if not matches:
        return []
        
    statements = []
    for idx, match in enumerate(matches):
        critic = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text_clean)
        statement = text_clean[start:end].strip()
        # Clean up statement punctuation
        statement = re.sub(r'^[:\s،,]+', '', statement)
        statement = re.sub(r'[\s،,.]+$', '', statement)
        if len(statement) > 3 and len(critic) < 50:
            statements.append((critic, statement))
            
    return statements

def main():
    print("=" * 65)
    print("  SENTIMENT-BASED JUMHŪR SCORING (MULTI-VIEW)")
    print("=" * 65)
    print(f"  Device: {DEVICE}")

    if not ENC_PATH.exists():
        print(f"  Error: encyclopedia not found at {ENC_PATH}")
        sys.exit(1)
        
    if not MODEL_DIR.exists():
        print(f"  Error: pre-trained AraBERT classifier not found at {MODEL_DIR}")
        sys.exit(1)

    print("\n[1/4] Loading models and dictionaries...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    enc = json.load(open(ENC_PATH, encoding="utf-8"))
    alias = json.load(open(ALIAS_PATH, encoding="utf-8"))
    id_to_name = json.load(open(NODE_NAMES_PATH, encoding="utf-8"))
    print(f"  Loaded {len(enc):,} encyclopedia profiles and {len(id_to_name):,} node names.")

    print("\n[2/4] Splitting biographies into critic statements...")
    entry_statements = {}
    total_quotes = 0
    
    for e in enc:
        eid = e.get("entry_id")
        text = e.get("full_text", "")
        if not eid or not text:
            continue
        quotes = split_critic_statements(text)
        if quotes:
            entry_statements[eid] = quotes
            total_quotes += len(quotes)
            
    print(f"  Extracted {total_quotes:,} statements from {len(entry_statements):,} biographies.")

    print("\n[3/4] Running AraBERT sentence-level scoring...")
    entry_scores = {}
    
    # Process in batch to speed up model inference
    all_sentences = []
    sentence_meta = [] # (eid, critic)
    
    for eid, quotes in entry_statements.items():
        for critic, statement in quotes:
            # Prepare input text: prepend critic context or score quote directly
            # Direct quote classification is standard
            all_sentences.append(statement[:150])
            sentence_meta.append(eid)
            
    batch_size = 32
    predicted_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_sentences), batch_size), desc="Scoring statements"):
            batch = all_sentences[i:i + batch_size]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            # Calculate expected score: sum(prob_c * score_c)
            batch_expected = np.sum(probs * CLASS_SCORES, axis=-1)
            predicted_scores.extend(batch_expected.tolist())

    # Map scores back to entries
    statements_by_entry = {}
    for idx, eid in enumerate(sentence_meta):
        statements_by_entry.setdefault(eid, []).append(predicted_scores[idx])

    # Aggregate: reliability = mean, agreement = 1 - var / 0.16
    entry_aggregated = {}
    for eid, scores in statements_by_entry.items():
        reliability = float(np.mean(scores))
        if len(scores) > 1:
            # Normalized variance consensus rate:
            # The expected continuous score is bounded between [0.05, 0.85].
            # Mathematically, the maximum possible variance for any bounded variable on [a, b]
            # occurs when the distribution is split 50/50 at the two boundaries a and b.
            # Max Variance = ((b - a) / 2)^2 = ((0.85 - 0.05) / 2)^2 = 0.40^2 = 0.16.
            # Dividing by 0.16 bounds the variance to [0.0, 1.0], meaning 1.0 - var/0.16
            # yields a normalized agreement index between 0.0 (total disagreement) and 1.0 (perfect consensus).
            var = float(np.var(scores))
            agreement = max(0.0, min(1.0, 1.0 - var / 0.16))
        else:
            agreement = 1.0
            
        entry_aggregated[eid] = {
            "reliability": reliability,
            "agreement": agreement,
            "count": len(scores)
        }

    print("\n[4/4] Mapping scores to Knowledge Graph node IDs...")
    
    # Build alias mappings (clean_name -> entry_id)
    name_to_eid = {}
    for ak, eids in alias.items():
        name_to_eid[norm(ak)] = eids[0]

    node_rows = []
    hits = 0
    for node_id_str, orig_name in id_to_name.items():
        nid = int(node_id_str)
        nkey = norm(orig_name)
        
        rel = 0.632 # default prior
        agreed = 1.0
        count = 0
        
        if nkey in name_to_eid:
            eid = name_to_eid[nkey]
            if eid in entry_aggregated:
                agg = entry_aggregated[eid]
                rel = agg["reliability"]
                agreed = agg["agreement"]
                count = agg["count"]
                hits += 1
                
        node_rows.append({
            "node_id": nid,
            "reliability_score": round(rel, 4),
            "agreement_score": round(agreed, 4),
            "critic_count": count
        })

    # Save to CSV
    df = pd.DataFrame(node_rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"  Saved aggregated database: {OUT_CSV}")
    print(f"  Coverage: {hits:,}/{len(id_to_name):,} matched node profiles ({hits/len(id_to_name)*100:.2f}%)")
    print(f"  Reliability: mean={df['reliability_score'].mean():.4f}, std={df['reliability_score'].std():.4f}")
    print(f"  Agreement:   mean={df['agreement_score'].mean():.4f}, std={df['agreement_score'].std():.4f}")

if __name__ == "__main__":
    main()
