"""
Build narrator_scores_sentiment.csv:
1. Extract critic statements from bio texts (قال patterns)
2. Classify each statement with AraBERT
3. Jumhur vote → reliability score
4. Save with critic_count for quality filtering
"""
import sys, re, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_TO_SCORE = {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.70, 4: 0.85}
CLASS_NAMES = {0: "matruk", 1: "daif", 2: "maqbul", 3: "saduq", 4: "thiqah"}

def norm(t):
    if not t: return ""
    t = re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","a").replace("ة","ه").replace("ى","ي").strip()
    return t

# ── Load classifier ──
print("Loading AraBERT classifier...")
MODEL_DIR = "output/track_a_classifier/arabert_classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print(f"  Device: {DEVICE}")

# ── Load encyclopedia ──
print("Loading encyclopedia...")
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
print(f"  {len(enc):,} entries")

# ── Load alias to map entries to KG node IDs ──
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
with open("output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)

# Build entry_id → node_ids mapping
entry_to_nodes = defaultdict(list)
for nid_str, name in id_to_name.items():
    nkey = norm(name)
    if nkey in alias:
        eid = alias[nkey][0]
        entry_to_nodes[eid].append(int(nid_str))

# ── Extract critic statements ──
print("Extracting critic statements...")
# Grade terms to use as implicit votes when no explicit critic statement found
GRADE_TERMS = ["ثقة ثبت", "ثقة حافظ", "حافظ", "ثقة", "صدوق", "مقبول",
               "ضعيف", "متروك", "كذاب", "لين", "مجهول", "ليس به بأس",
               "لا بأس به", "صدوق يهم", "سيء الحفظ"]

CRITIC_RE = re.compile(
    r"(?:\(?\d+\)?\s*)?(?:و?ق[اا]ل\s+)([؀-ۿ\s]{5,80}?)\s*[:]\s*([؀-ۿ\s,]{3,200}?)(?:\.|$|\n)",
    re.MULTILINE,
)
COUNT = 0

entry_stats = {}
for e in enc:
    eid = e["entry_id"]
    text = e.get("full_text", "")
    if not text or len(text) < 30: continue

    statements = []

    # Method 1: Extract explicit "قال X: Y" critic statements
    for m in CRITIC_RE.finditer(text[:2000]):
        statement = m.group(2).strip()
        statement = re.sub(r"[ً-ٌٍ-َّْٰ]", "", statement).strip()
        if len(statement) >= 3:
            statements.append(statement)
            COUNT += 1

    # Method 2: If no critic statements found, use standalone grade terms as votes
    if len(statements) < 2:
        text_clean = re.sub(r"[ً-ٌٍ-َّْٰ]", "", text[:1000])
        for term in GRADE_TERMS:
            if term in text_clean:
                statements.append(term)
                COUNT += 1

    if statements:
        entry_stats[eid] = {"statements": statements, "name": e.get("name", "")}

print(f"  Extracted {COUNT:,} statements from {len(entry_stats):,} entries")

# ── Classify each statement ──
print("Classifying statements (jumhur vote)...")
BATCH = 64
results = []

for eid, data in entry_stats.items():
    stats = data["statements"]
    if len(stats) == 0: continue

    # Classify each statement
    with torch.no_grad():
        encoded = tokenizer(stats, truncation=True, padding=True, max_length=128, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded)
        preds = outputs.logits.argmax(-1).cpu().tolist()

    # Jumhur vote: count each class
    votes = defaultdict(int)
    for p in preds:
        votes[p] += 1

    # Majority class wins
    majority_class = max(votes, key=votes.get)
    majority_score = CLASS_TO_SCORE[majority_class]
    total_votes = len(preds)

    # Map to node IDs
    for nid in entry_to_nodes.get(eid, []):
        results.append({
            "node_id": nid,
            "reliability_score": majority_score,
            "critic_count": total_votes,
            "majority_class": CLASS_NAMES[majority_class],
        })

df = pd.DataFrame(results)
print(f"  Scored {len(df):,} narrator nodes")
print(f"  Critic count distribution:")
for c in sorted(df["critic_count"].value_counts().head(10).items()):
    print(f"    {c[0]} critics: {c[1]:,} narrators")

# ── Save ──
out_path = "output/track_a_classifier/narrator_scores_sentiment.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(df):,} rows)")
print(f"  With ≥2 critics: {(df['critic_count']>=2).sum():,}")
print(f"  With ≥5 critics: {(df['critic_count']>=5).sum():,}")
