"""
Build deep CAMeL BERT [CLS] embeddings for narrator-attributed matn texts.

Optimized for CPU: max_length=256 (4× faster than 512), batch_size=128,
torch.inference_mode, and configurable thread count.

Output: pyg_graph_bert.npz with 8 graph + 768 BERT = 776 feature dimensions.
"""
import sys, json, re, ast, time, os
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.stage3.sentiment import lexicon_sentiment_score

# ── Config ──
BERT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
BATCH_SIZE = 128          # Larger batch for CPU throughput
MAX_LENGTH = 256          # Half context = ~4× faster attention
MAX_MATNS_PER_NARRATOR = 5
MAX_CHARS_PER_NARRATOR = 1000
MAX_SANADSET_ROWS = 100_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU optimization
if not torch.cuda.is_available():
    N_CPUS = os.cpu_count() or 4
    torch.set_num_threads(min(N_CPUS, 16))
    print(f"CPU threads: {torch.get_num_threads()}")

print(f"Device: {DEVICE}")
print(f"Model: {BERT_MODEL}")
print(f"Batch size: {BATCH_SIZE}, Max length: {MAX_LENGTH}")

# ── Load graph ──
print("\n=== Loading graph ===")
graph = np.load(r"output/stage2/pyg_graph_large_graded.npz")
x = graph["x"]
edge_index = graph["edge_index"]
n_nodes = x.shape[0]
print(f"  Nodes: {n_nodes:,}")

with open(r"output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)
print(f"  Node names: {len(id_to_name):,}")

# ── Load Sanadset and build narrator→matns mapping ──
print(f"\n=== Loading Sanadset ({MAX_SANADSET_ROWS:,} rows) ===")
sanadset = pd.read_csv(r"output/stage1/sanadset_cleaned.csv", nrows=MAX_SANADSET_ROWS)
print(f"  Rows: {len(sanadset):,}")

matn_by_narrator: dict[str, list[str]] = defaultdict(list)
for _, row in tqdm(sanadset.iterrows(), total=len(sanadset), desc="Building narrator→matns"):
    matn = str(row.get("Matn", ""))
    if not matn or len(matn) < 20:
        continue
    try:
        chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
    except (ValueError, SyntaxError):
        continue
    for name in chain:
        name = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name.strip())
        if len(name) > 3:
            matn_by_narrator[name].append(matn)

print(f"  Narrators with matn data: {len(matn_by_narrator):,}")

# Count how many graph nodes we can cover
name_clean_to_node = {}
for node_id_str, name in id_to_name.items():
    name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
    if name_clean in matn_by_narrator:
        name_clean_to_node[name_clean] = int(node_id_str) - 1

n_covered = len(name_clean_to_node)
print(f"  Graph nodes matched: {n_covered:,}/{n_nodes:,} ({n_covered/n_nodes*100:.1f}%)")

# ── Pre-build texts (avoid repeated string concatenation in loop) ──
print(f"\n=== Pre-building narrator texts ===")
narrator_names = []
narrator_texts = []
for name in tqdm(name_clean_to_node, desc="Pre-building texts"):
    matns = matn_by_narrator[name][:MAX_MATNS_PER_NARRATOR]
    combined = " [SEP] ".join(matns)
    narrator_names.append(name)
    narrator_texts.append(combined[:MAX_CHARS_PER_NARRATOR])

print(f"  Texts prepared: {len(narrator_texts):,}")

# ── Load BERT model ──
print(f"\n=== Loading BERT model ===")
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE)
model.eval()
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Generate embeddings ──
print(f"\n=== Generating BERT embeddings ===")
n_batches = (n_covered + BATCH_SIZE - 1) // BATCH_SIZE
est_sec_per_batch = 8 if DEVICE.type == "cpu" else 0.5
est_total_min = n_batches * est_sec_per_batch / 60
print(f"  Batches: {n_batches}, ETA: ~{est_total_min:.0f} min")

bert_dim = 768
bert_embeddings = np.zeros((n_nodes, bert_dim), dtype=np.float32)

start_time = time.time()
for batch_start in tqdm(range(0, n_covered, BATCH_SIZE), desc="BERT batches"):
    batch_texts = narrator_texts[batch_start:batch_start + BATCH_SIZE]
    batch_node_names = narrator_names[batch_start:batch_start + BATCH_SIZE]

    inputs = tokenizer(
        batch_texts, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    for i, name in enumerate(batch_node_names):
        node_idx = name_clean_to_node[name]
        bert_embeddings[node_idx] = cls_embeddings[i]

elapsed = time.time() - start_time
print(f"  Done in {elapsed:.0f}s ({elapsed/n_covered:.2f}s/narrator, {elapsed/n_batches:.1f}s/batch)")

# ── Normalize BERT embeddings ──
print(f"\n=== Normalizing embeddings ===")
bert_norms = np.linalg.norm(bert_embeddings, axis=1, keepdims=True)
bert_norms = np.maximum(bert_norms, 1e-8)
bert_embeddings = bert_embeddings / bert_norms
print(f"  BERT embeddings L2-normalized (unit norm)")

# ── PCA reduction (optional: reduce 768-dim to 128-dim to avoid overwhelming graph features) ──
print(f"\n=== PCA: 768-dim → 128-dim for balanced GNN input ===")
from sklearn.decomposition import PCA
pca = PCA(n_components=128, random_state=42)
bert_reduced = pca.fit_transform(bert_embeddings)
variance = pca.explained_variance_ratio_.sum()
print(f"  PCA variance retained: {variance:.2%}")
print(f"  Reduced shape: {bert_reduced.shape}")

# Use the reduced version (128-dim instead of 768-dim)
expanded_x = np.hstack([x, bert_reduced.astype(np.float32)])
print(f"\n=== Final feature matrix ===")
print(f"  Shape: {expanded_x.shape}")
print(f"  Dims: 8 graph + 128 BERT-PCA = {expanded_x.shape[1]}")

# Also build lexicon features for comparison
print(f"\n=== Building lexicon features (for comparison) ===")
lexicon_dim = 7
lexicon_feats = np.zeros((n_nodes, lexicon_dim), dtype=np.float32)
for name_clean, node_idx in name_clean_to_node.items():
    matns = matn_by_narrator[name_clean][:10]
    feats = [lexicon_sentiment_score(m) for m in matns]
    if feats:
        avg = {k: np.mean([f[k] for f in feats]) for k in feats[0]}
        lexicon_feats[node_idx] = list(avg.values())

lexicon_x = np.hstack([x, lexicon_feats])
print(f"  Lexicon shape: {lexicon_x.shape}")

# ── Save ──
out_dir = Path(r"output/stage2")
print(f"\n=== Saving ===")
np.savez(out_dir / "pyg_graph_bert.npz",
         x=expanded_x, edge_index=edge_index,
         edge_attr=graph.get("edge_attr", np.zeros((0,))))
print(f"  BERT graph: output/stage2/pyg_graph_bert.npz")

np.savez(out_dir / "pyg_graph_lexicon.npz",
         x=lexicon_x, edge_index=edge_index,
         edge_attr=graph.get("edge_attr", np.zeros((0,))))
print(f"  Lexicon graph: output/stage2/pyg_graph_lexicon.npz")

# Save PCA for later use
import pickle
with open(out_dir / "bert_pca.pkl", "wb") as f:
    pickle.dump(pca, f)

# Save mapping
mapping_info = {
    "bert_model": BERT_MODEL,
    "bert_dim": bert_dim,
    "pca_dim": 128,
    "pca_variance": float(variance),
    "n_nodes": n_nodes,
    "n_nodes_with_matn": n_covered,
    "coverage": n_covered / n_nodes,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "device": str(DEVICE),
    "elapsed_seconds": elapsed,
}
with open(out_dir / "bert_embedding_info.json", "w", encoding="utf-8") as f:
    json.dump(mapping_info, f, indent=2, ensure_ascii=False)

print(f"\nDone. BERT + PCA features ready for GNN training.")
