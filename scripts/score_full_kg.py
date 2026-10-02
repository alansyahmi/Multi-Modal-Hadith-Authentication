"""
Score all 262K narrators in the full-scale KG using the trained AraBERT model.
The model was trained on grade-balanced biographical text and predicts [0, 1] scores.
"""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# ── Paths ──
MODEL_DIR = Path("output/track_a/arabert_regression_model")
NODE_NAMES_PATH = Path("output/stage2_full/node_names.json")
ENCYCLOPEDIA_PATH = Path("output/shamela/narrator_encyclopedia.json")
OUTPUT_DIR = Path("output/track_a")
OUTPUT_SCORES = OUTPUT_DIR / "full_kg_narrator_scores.json"
OUTPUT_METRICS = OUTPUT_DIR / "full_kg_scoring_stats.json"

MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  SCORE FULL-SCALE KG: 262K Narrators")
print("=" * 60)
print(f"  Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

# ── Load trained model ──
print(f"\n[1/4] Loading trained AraBERT model...")
if MODEL_DIR.exists():
    print(f"  Loading from fine-tuned checkpoint: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), num_labels=1,
    ).to(DEVICE)
else:
    print(f"  Fine-tuned model not found, loading base AraBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression",
    ).to(DEVICE)

model.eval()
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Load node names ──
print(f"\n[2/4] Loading graph node names...")
with open(NODE_NAMES_PATH, encoding="utf-8") as f:
    id_to_name: dict[str, str] = json.load(f)
print(f"  Narrators: {len(id_to_name):,}")

# Load encyclopedia for biographical text
print(f"  Loading encyclopedia for text lookups...")
with open(ENCYCLOPEDIA_PATH, encoding="utf-8") as f:
    enc: list[dict] = json.load(f)

import re
def norm(t): return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

enc_lookup: dict[str, str] = {}
for entry in enc:
    name = entry.get("name", "")
    text = entry.get("full_text", "")
    if name and text and len(text) > 50:
        key = norm(name)
        if key not in enc_lookup or len(text) > len(enc_lookup[key]):
            enc_lookup[key] = text

# ── Build texts ──
print(f"\n[3/4] Preparing texts for all narrators...")
node_ids = sorted(id_to_name.keys(), key=int)
texts: list[str] = []
node_ids_list: list[str] = []

for nid in node_ids:
    name = id_to_name[nid]
    name_norm = norm(name)
    bio_text = enc_lookup.get(name_norm, "")
    if bio_text and len(bio_text) > 50:
        texts.append(bio_text[:2000])
    else:
        texts.append(name)
    node_ids_list.append(nid)

print(f"  Texts prepared: {len(texts):,}")

# ── Batch inference ──
print(f"\n[4/4] Scoring all narrators (GPU inference)...")
start = time.time()
scores: dict[str, float] = {}

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
        batch_texts = texts[i:i + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts, truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(DEVICE)

        outputs = model(**encoded)
        batch_scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()

        for j, nid in enumerate(node_ids_list[i:i + BATCH_SIZE]):
            scores[nid] = float(round(batch_scores[j], 4))

elapsed = time.time() - start
score_vals = list(scores.values())

stats = {
    "n_narrators": len(scores),
    "score_mean": float(np.mean(score_vals)),
    "score_std": float(np.std(score_vals)),
    "score_min": float(min(score_vals)),
    "score_max": float(max(score_vals)),
    "inference_time_seconds": elapsed,
    "inference_speed": f"{len(scores)/elapsed:.0f} narrators/sec",
}

print(f"\n  Score distribution:")
print(f"    Mean:    {stats['score_mean']:.4f}")
print(f"    Std:     {stats['score_std']:.4f}")
print(f"    Min:     {stats['score_min']:.4f}")
print(f"    Max:     {stats['score_max']:.4f}")
print(f"  Time: {elapsed:.0f}s ({len(scores)/elapsed:.0f} narrators/sec)")

# Histogram
bins = [0.0, 0.15, 0.35, 0.60, 0.85, 1.01]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    cnt = sum(1 for s in score_vals if lo <= s < hi)
    pct = cnt/len(score_vals)*100
    bar = "#" * min(cnt // max(len(score_vals)//40, 1), 60)
    print(f"    [{lo:.2f}, {hi:.2f}): {cnt:>6d} ({pct:>5.1f}%) {bar}")

# ── Save ──
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_SCORES, "w", encoding="utf-8") as f:
    json.dump(scores, f, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_SCORES}")

with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(f"  Saved: {OUTPUT_METRICS}")

print(f"\n{'=' * 60}")
print(f"  FULL-SCALE SCORING COMPLETE")
print(f"  {len(scores):,} narrators scored in {elapsed:.0f}s")
print(f"{'=' * 60}")
