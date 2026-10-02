"""
Semantic Sentiment Analysis for Matn (Hadith Content) Text.

Implements the "semantic sentiment analysis" component referenced in the
thesis title. Extracts per-hadith sentiment/severity features and aggregates
them to per-narrator features for the GNN.

Two modes:
  - Deep: CAMeL Arabic BERT [CLS] embeddings (requires transformers + GPU)
  - Lexicon: Rule-based severity scoring (always available, fast)

Output: per-narrator sentiment features appended to node feature matrix.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Arabic Sentiment / Severity Lexicon ─────────────────────────

# Threat/punishment terms (negative sentiment, high severity)
THREAT_TERMS: set[str] = {
    "نار", "جهنم", "عذاب", "وعيد", "لعنة", "غضب", "سخط",
    "خسران", "هلاك", "ويل", "عقاب", "خزي", "بئس", "شر",
    "فتنة", "بلاء", "نكال", "اثم", "وزر", "حرام", "حرم",
}

# Mercy/reward terms (positive sentiment)
MERCY_TERMS: set[str] = {
    "جنة", "رحمة", "مغفرة", "فضل", "نعيم", "بشارة", "أجر",
    "ثواب", "خير", "بركة", "سلام", "نور", "هدى", "رضوان",
    "فوز", "فلاح", "طيب", "عفاف", "حلال", "طاعة",
}

# Intensity amplifiers
INTENSITY_TERMS: set[str] = {
    "شديد", "عظيم", "كبير", "أليم", "مهين", "مقيت", "غليظ",
    "وبيل", "عقيم", "خالد", "مخلد", "أبد", "أبدا",
}


def lexicon_sentiment_score(text: str) -> dict[str, float]:
    """
    Compute lexicon-based sentiment features from Arabic matn text.

    Returns dict with:
      - severity_score: (threat_terms - mercy_terms) / total_words
      - threat_ratio: threat_terms / total_words
      - mercy_ratio: mercy_terms / total_words
      - intensity_ratio: intensity_terms / total_words
      - word_count: total word count
      - char_count: total character count
      - has_quran_quote: 1.0 if Quranic quotation detected
    """
    if not text or not isinstance(text, str):
        return {
            "severity_score": 0.0, "threat_ratio": 0.0, "mercy_ratio": 0.0,
            "intensity_ratio": 0.0, "word_count": 0, "char_count": 0,
            "has_quran_quote": 0.0,
        }

    # Normalize: remove diacritics, split on whitespace
    clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', text)
    words = clean.split()
    n_words = max(len(words), 1)

    threat_count = sum(1 for w in words if w in THREAT_TERMS)
    mercy_count = sum(1 for w in words if w in MERCY_TERMS)
    intensity_count = sum(1 for w in words if w in INTENSITY_TERMS)

    # Quranic quotation patterns
    quran_patterns = [
        r"قال الله", r"قوله تعالى", r"في كتابه", r"أنزل الله",
        r"وحي", r"جبريل", r"القرآن",
    ]
    has_quran = 1.0 if any(re.search(p, clean) for p in quran_patterns) else 0.0

    return {
        "severity_score": round((threat_count - mercy_count) / n_words, 4),
        "threat_ratio": round(threat_count / n_words, 4),
        "mercy_ratio": round(mercy_count / n_words, 4),
        "intensity_ratio": round(intensity_count / n_words, 4),
        "word_count": float(n_words),
        "char_count": float(len(clean)),
        "has_quran_quote": has_quran,
    }


def build_narrator_sentiment_features(
    matn_by_narrator: dict[str, list[str]],
    use_deep: bool = False,
) -> np.ndarray:
    """
    Aggregate sentiment features from all matns attributed to each narrator.

    Args:
        matn_by_narrator: {narrator_name: [matn_text_1, matn_text_2, ...]}
        use_deep: If True, use CAMeL BERT embeddings (requires transformers)

    Returns:
        np.ndarray of shape [N, F_sentiment] where N = len(matn_by_narrator)
    """
    all_features = []

    for name, matns in matn_by_narrator.items():
        # Compute lexicon features for each matn, then average
        matn_feats = [lexicon_sentiment_score(m) for m in matns if m]

        if not matn_feats:
            matn_feats = [lexicon_sentiment_score("")]

        # Average across all matns for this narrator
        avg_feat = {
            k: np.mean([m[k] for m in matn_feats])
            for k in matn_feats[0]
        }
        all_features.append(list(avg_feat.values()))

    feats = np.array(all_features, dtype=np.float32)

    # Normalize word_count and char_count with log scale
    for col_idx, key in enumerate(matn_feats[0].keys()):
        if key in ("word_count", "char_count"):
            vals = feats[:, col_idx]
            vals = np.log1p(vals)
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin > 1e-8:
                feats[:, col_idx] = (vals - vmin) / (vmax - vmin)

    return feats


def build_deep_embeddings(
    matn_by_narrator: dict[str, list[str]],
    model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
    batch_size: int = 32,
) -> Optional[np.ndarray]:
    """
    Build deep BERT [CLS] embeddings for narrator-attributed matn texts.

    Requires: pip install transformers

    Returns None if transformers is not available.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("  [sentiment] transformers not installed — skipping deep embeddings")
        return None

    print(f"  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    narrator_embeddings = []
    names = list(matn_by_narrator.keys())

    for i in range(0, len(names), batch_size):
        batch_names = names[i:i + batch_size]

        # Concatenate all matns for each narrator (truncate to 512 tokens)
        batch_texts = []
        for name in batch_names:
            combined = " ".join(matn_by_narrator[name][:5])  # Max 5 matns
            batch_texts.append(combined[:2000])  # Truncate chars

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            narrator_embeddings.append(cls_embeddings)

        if (i // batch_size) % 50 == 0:
            print(f"    Processed {i}/{len(names)} narrators...")

    embeddings = np.concatenate(narrator_embeddings, axis=0)
    print(f"  Deep embeddings: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_sentiment_features(
    graph_npz_path: Path,
    node_names_path: Path,
    sanadset_path: Path,
    use_deep: bool = False,
) -> np.ndarray:
    """
    Build per-narrator sentiment features and append to the graph node features.

    Process:
    1. Load graph node names
    2. For each narrator, find their matn texts from Sanadset
    3. Compute sentiment features (lexicon or deep)
    4. Return expanded node feature matrix

    Returns updated node features of shape [N, 8 + F_sentiment].
    """
    import pandas as pd

    print("\n=== BUILDING SENTIMENT FEATURES ===")
    print(f"  Mode: {'deep (BERT)' if use_deep else 'lexicon-based'}")

    # Load graph
    graph = np.load(graph_npz_path)
    x = graph["x"]

    with open(node_names_path, encoding="utf-8") as f:
        id_to_name = json.load(f)

    print(f"  Graph nodes: {len(id_to_name):,}")

    # Load Sanadset and build narrator→matns mapping
    print(f"  Loading Sanadset...")
    sanadset = pd.read_csv(sanadset_path, nrows=50000)

    import json, ast
    matn_by_narrator: dict[str, list[str]] = defaultdict(list)

    for _, row in sanadset.iterrows():
        matn = str(row.get("Matn", ""))
        if not matn or len(matn) < 20:
            continue

        try:
            chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
        except (ValueError, SyntaxError):
            continue

        for name in chain:
            name = name.strip()
            if len(name) > 3:
                matn_by_narrator[name].append(matn)

    print(f"  Narrators with matn data: {len(matn_by_narrator):,}")

    # Build sentiment features
    if use_deep:
        sent_feats = build_deep_embeddings(matn_by_narrator)
    else:
        sent_feats = build_narrator_sentiment_features(matn_by_narrator)

    print(f"  Sentiment features: {sent_feats.shape}")

    # Map sentiment features to graph nodes
    n_nodes = x.shape[0]
    n_sent = sent_feats.shape[1]
    expanded_x = np.zeros((n_nodes, x.shape[1] + n_sent), dtype=np.float32)
    expanded_x[:, :x.shape[1]] = x  # Copy original features

    # Match narrators to nodes
    matched = 0
    for node_id_str, name in id_to_name.items():
        node_id = int(node_id_str)
        idx = node_id - 1

        name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
        if name_clean in matn_by_narrator:
            # Use the sentiment features for this narrator
            narrator_idx = list(matn_by_narrator.keys()).index(name_clean)
            expanded_x[idx, x.shape[1]:] = sent_feats[narrator_idx]
            matched += 1

    print(f"  Sentiment features attached: {matched:,}/{n_nodes:,} ({matched/n_nodes*100:.1f}%)")

    return expanded_x
