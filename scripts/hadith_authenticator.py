"""
Interactive Hadith Authenticity Analyzer
========================================
Analyzes a hadith sanad (narrator chain) using the full multi-modal pipeline:
  - Per-narrator GNN reliability predictions
  - Chain-level statistics (mean, variance, weakest link)
  - Matn semantic sentiment analysis
  - Biographical data lookup (Ibn Ḥajar grades, death dates, ṭabaqa)
  - Final authenticity assessment

Usage:
  python scripts/hadith_authenticator.py                    # Interactive mode
  python scripts/hadith_authenticator.py --browse            # Browse evaluation dataset
  python scripts/hadith_authenticator.py --hadith-id 42     # Analyze specific hadith from eval set
  python scripts/hadith_authenticator.py --sanad "...text..." # Analyze pasted sanad
"""
import sys, json, re, ast, os, textwrap
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.stage3.sentiment import lexicon_sentiment_score
from rapidfuzz.fuzz import partial_ratio

# ── Paths ──
GRAPH_NODES_CSV = Path("output/stage2/graph_nodes.csv")
NODE_NAMES_JSON = Path("output/stage2/node_names.json")
PREDICTIONS_CSV = Path("output/stage3/narrator_predictions_clean.csv")
NAME_CLEANUP_JSON = Path("output/stage2/name_cleanup.json")
ENCYCLOPEDIA_JSON = Path("output/shamela/narrator_encyclopedia.json")
EVAL_CSV = Path("output/stage1/kutub_sitta_evaluation.csv")
SANADSET_CSV = Path("output/stage1/sanadset_cleaned.csv")
ALIGNMENT_JSON = Path("output/stage1/alignment_results.json")

# ── Ibn Ḥajar 12-tier grading system ──
GRADE_NAMES = {
    1: "صحابي (Companion)",
    2: "ثقة ثقة (Highly Trustworthy)",
    3: "ثقة (Trustworthy)",
    4: "صدوق (Truthful)",
    5: "صدوق سيء الحفظ (Truthful, Poor Memory)",
    6: "مقبول (Acceptable)",
    7: "مجهول الحال (Unknown Status)",
    8: "مجهول العين (Unknown Identity)",
    9: "لين الحديث (Soft in Hadith)",
    10: "ضعيف (Weak)",
    11: "متروك (Abandoned)",
    12: "كذاب / وضاع (Liar / Fabricator)",
}

CATEGORY_COLORS = {
    "ثقة": "\033[92m",       # Green
    "صدوق": "\033[94m",      # Blue
    "مقبول": "\033[93m",     # Yellow
    "ضعيف": "\033[91m",      # Red
    "متروك": "\033[91m",     # Red
    "RESET": "\033[0m",
}

# ── Load all data once ──
def load_data():
    """Load all reference data. Returns dict of dataframes/dicts."""
    data = {}

    if NODE_NAMES_JSON.exists():
        with open(NODE_NAMES_JSON, encoding="utf-8") as f:
            data["id_to_name"] = json.load(f)
        # Build reverse lookup: clean_name → node_id
        name_to_id = {}
        for nid_str, name in data["id_to_name"].items():
            name_clean = normalize_name(name)
            name_to_id[name_clean] = int(nid_str)
        data["name_to_id"] = name_to_id
        print(f"  Loaded {len(data['id_to_name']):,} node names")
    else:
        data["id_to_name"] = {}
        data["name_to_id"] = {}

    # Load GNN predictions
    pred_path = Path("output/stage3/narrator_predictions.csv")
    if pred_path.exists():
        data["predictions"] = pd.read_csv(pred_path)
        print(f"  Loaded {len(data['predictions']):,} GNN predictions")
    else:
        data["predictions"] = None

    # Load Sentiment aggregator results
    sent_path = Path("output/track_a_classifier/narrator_scores_sentiment.csv")
    if sent_path.exists():
        data["sentiment"] = pd.read_csv(sent_path)
        print(f"  Loaded {len(data['sentiment']):,} narrator sentiment profiles")
    else:
        data["sentiment"] = None

    emb_path = Path("output/stage3/narrator_embeddings.npy")
    if emb_path.exists():
        data["embeddings"] = np.load(emb_path)
        print(f"  Loaded {len(data['embeddings']):,} narrator structural embeddings")
    else:
        data["embeddings"] = None

    if ENCYCLOPEDIA_JSON.exists():
        with open(ENCYCLOPEDIA_JSON, encoding="utf-8") as f:
            raw_encyc = json.load(f)
        # Build lookup by normalized name
        encyc_by_name = {}
        for entry in raw_encyc:
            name = normalize_name(entry.get("name", ""))
            if name and name not in encyc_by_name:
                encyc_by_name[name] = entry
        data["encyclopedia"] = encyc_by_name
        print(f"  Loaded encyclopedia with {len(encyc_by_name):,} entries")
    else:
        data["encyclopedia"] = {}

    if EVAL_CSV.exists():
        data["eval_df"] = pd.read_csv(EVAL_CSV)
        print(f"  Loaded {len(data['eval_df']):,} evaluation hadiths")
    else:
        data["eval_df"] = None

    if SANADSET_CSV.exists():
        data["sanadset"] = pd.read_csv(SANADSET_CSV, nrows=200000)
        print(f"  Loaded {len(data['sanadset']):,} Sanadset rows")
    else:
        data["sanadset"] = None

    if ALIGNMENT_JSON.exists():
        with open(ALIGNMENT_JSON, encoding="utf-8") as f:
            data["alignment"] = pd.read_json(f)
        print(f"  Loaded {len(data['alignment']):,} alignment records")
    else:
        data["alignment"] = None

    return data


def normalize_name(name: str) -> str:
    """Strip Arabic diacritics and normalize alif variants."""
    name = re.sub(r'[ً-ٌٍ-َّْٰٓ]', '', name)
    name = name.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    name = name.replace('ى', 'ي').replace('ة', 'ه')
    # Remove <IDF> tags
    name = re.sub(r'<[^>]+>', '', name)
    return name.strip()

# ── Kinship patterns (same as fix_narrator_names.py) ──
KINSHIP_PATTERNS = [
    r'^أ?ب[iي]?[هةِ]{0,3}$',      # Abīhi (his father)
    r'^أ?خ[iي]?[هةِ]{0,3}$',      # Akhīhi (his brother)
    r'^أ?م[iي]?[هةِ]{0,3}$',      # Ummihi (his mother)
    r'^ا?بن[هةِ]{0,3}$',           # Ibnihi (his son)
    r'^ج[دذ][هةِ]{0,3}$',          # Jaddihi (his grandfather)
    r'^ع[مم][هةِ]{0,3}$',         # Ammihi (his uncle)
]
KINSHIP_RE = re.compile('|'.join(KINSHIP_PATTERNS))

BARE_PREFIXES = ['ابو', 'ابي', 'ابا', 'ابن', 'بنت', 'ام', 'اخو', 'اخي', 'اخا']

def is_valid_narrator_name(name: str) -> bool:
    """Check if a name is a real narrator (not a kinship ref or bare prefix)."""
    clean = normalize_name(name)
    # Kinship reference
    if KINSHIP_RE.match(clean):
        return False
    # Bare prefix
    if clean in BARE_PREFIXES:
        return False
    # Too short
    if len(clean) <= 2:
        return False
    return True


TRANSMISSION_VERBS = [
    'حدثنا', 'حدثني', 'حدثنى', 'حدثتنا', 'حدث',
    'أخبرنا', 'أخبرني', 'أخبرنى', 'أخبرتنا', 'أخبر',
    'أنبأنا', 'أنبأني', 'أنبأ',
    'سمعت', 'سمعنا', 'سمع',
    'قرأت', 'قرأنا',
]


def parse_chain(sanad_text: str) -> list[str]:
    """Extract narrator names from a sanad string.

    Handles three formats:
    1. Python-style list: ['Narrator A', 'Narrator B']
    2. <NAR> tagged format from Sanadset
    3. Raw Arabic hadith text with transmission verbs
    """
    if not sanad_text or not isinstance(sanad_text, str):
        return []

    # ── Format 1: Python list ──
    try:
        chain = ast.literal_eval(sanad_text)
        if isinstance(chain, list) and len(chain) > 0:
            names = []
            for n in chain:
                if isinstance(n, str) and is_valid_narrator_name(n):
                    names.append(normalize_name(n))
            return names
    except (ValueError, SyntaxError):
        pass

    # ── Format 2: <NAR> tags ──
    nars = re.findall(r'<NAR>(.*?)</NAR>', sanad_text)
    if nars:
        names = [normalize_name(n) for n in nars if len(n.strip()) > 2]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        if len(unique) >= 2:
            return unique

    # ── Format 3: Raw Arabic hadith text ──
    # Remove diacritics first for cleaner matching
    clean = re.sub(r'[ً-ٌٍ-َّْٰٓ]', '', sanad_text)

    # Split the text into segments at transmission verbs and عن
    # Pattern: any transmission verb OR standalone عن/أن
    split_pattern = r'\s*(?:' + '|'.join(TRANSMISSION_VERBS) + r')\s+|' \
                    r'\s*عن\s+|\s*أن\s+(?!رسول|الله|النبي|رسول)'

    segments = re.split(split_pattern, clean)

    # Extract potential narrator names from segments
    # A narrator name typically has 2-5 words (full name + lineage)
    names = []
    for seg in segments:
        seg = seg.strip()
        if not seg or len(seg) < 4:
            continue

        # Skip segments that contain hadith content markers
        if any(marker in seg for marker in [
            'قال', 'يقول', 'سمعت', 'صلى الله', 'عليه وسلم',
            'رضي الله', 'النبى', 'النبي', 'رسول', 'اللهم',
            'آية', 'سورة', 'باب', 'كتاب',
        ]):
            # But try to extract the name before قال/يقول
            before_said = re.split(r'\s+(?:قال|يقول|قَال|يَقُول|فقال|فَقَال)\s+', seg)
            if len(before_said) > 0:
                potential_name = before_said[0].strip()
                if 4 < len(potential_name) < 80:
                    names.append(normalize_name(potential_name))
            continue

        # Take first 3-4 words as the narrator name (typical Arabic name length)
        words = seg.split()
        if len(words) >= 2:
            # Try 2-word, 3-word, and 4-word combinations
            for n_words in [4, 3, 2]:
                if len(words) >= n_words:
                    candidate = ' '.join(words[:n_words])
                    candidate = re.sub(r'\s*[،,\.].*', '', candidate)  # Remove punctuation
                    candidate = re.sub(r'\s*\(.*?\)', '', candidate)    # Remove parenthetical
                    if 5 < len(candidate) < 80 and not any(
                        kw in candidate for kw in ['باب', 'كتاب', 'حديث', 'صلاة']
                    ):
                        names.append(normalize_name(candidate))
                        break

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for n in names:
        if n not in seen and len(n) > 2:
            seen.add(n)
            unique.append(n)

    return unique


def lookup_narrator(name: str, data: dict) -> dict:
    """Look up a narrator in our knowledge graph. Returns info dict."""
    result = {
        "name": name,
        "matched_name": None,
        "node_id": None,
        "predicted_score": None,
        "predicted_category": None,
        "ibn_hajar_grade": None,
        "ibn_hajar_rank": None,
        "death_year": None,
        "tabaqa": None,
        "match_type": "none",  # exact, fuzzy, none
    }

    # Exact match
    if name in data["name_to_id"]:
        node_id = data["name_to_id"][name]
        result["match_type"] = "exact"
        result["node_id"] = node_id
        # Get the original name
        for nid_str, orig_name in data["id_to_name"].items():
            if int(nid_str) == node_id:
                result["matched_name"] = orig_name
                break
    else:
        # Fuzzy match
        best_score = 0
        best_id = None
        best_orig_name = None
        for nid_str, orig_name in data["id_to_name"].items():
            orig_clean = normalize_name(orig_name)
            score = partial_ratio(name, orig_clean)
            if score > best_score and score >= 80:
                best_score = score
                best_id = int(nid_str)
                best_orig_name = orig_name
        if best_id is not None:
            result["match_type"] = "fuzzy"
            result["node_id"] = best_id
            result["matched_name"] = best_orig_name
            result["fuzzy_score"] = best_score

    # Get prediction (GNN)
    if result["node_id"] is not None and data["predictions"] is not None:
        pred_row = data["predictions"][data["predictions"]["node_id"] == result["node_id"]]
        if len(pred_row) > 0:
            score = float(pred_row["predicted_score"].iloc[0])
            result["predicted_score"] = score
            result["predicted_category"] = score_to_category(score)
            result["anomaly_score"] = float(pred_row["anomaly_score"].iloc[0]) if "anomaly_score" in pred_row.columns else 0.0
        else:
            result["predicted_score"] = 0.632
            result["predicted_category"] = "مقبول"
            result["anomaly_score"] = 0.0
    else:
        result["predicted_score"] = 0.632
        result["predicted_category"] = "مقبول"
        result["anomaly_score"] = 0.0

    # Get Sentiment aggregator results (Semantic view)
    if result["node_id"] is not None and data["sentiment"] is not None:
        sent_row = data["sentiment"][data["sentiment"]["node_id"] == result["node_id"]]
        if len(sent_row) > 0:
            result["semantic_reliability"] = float(sent_row["reliability_score"].iloc[0])
            result["semantic_agreement"] = float(sent_row["agreement_score"].iloc[0])
            result["critic_count"] = int(sent_row["critic_count"].iloc[0])
        else:
            result["semantic_reliability"] = 0.632
            result["semantic_agreement"] = 1.0
            result["critic_count"] = 0
    else:
        result["semantic_reliability"] = 0.632
        result["semantic_agreement"] = 1.0
        result["critic_count"] = 0

    # Narrator uncertainty: multi-view disagreement + consensus disagreement
    # U = |semantic_reliability - predicted_score| + (1.0 - semantic_agreement)
    result["uncertainty"] = abs(result["semantic_reliability"] - result["predicted_score"]) + (1.0 - result["semantic_agreement"])

    # Get biographical data
    if result["matched_name"]:
        bio_name = normalize_name(result["matched_name"])
        for entry_name, entry_data in data.get("encyclopedia", {}).items():
            if normalize_name(entry_name) == bio_name:
                result["ibn_hajar_grade"] = entry_data.get("grade")
                result["ibn_hajar_rank"] = entry_data.get("rank")
                result["death_year"] = entry_data.get("death_year")
                result["tabaqa"] = entry_data.get("tabaqa")
                break

    return result


def score_to_category(score: float) -> str:
    if score >= 0.85: return "ثقة"
    elif score >= 0.60: return "صدوق"
    elif score >= 0.35: return "مقبول"
    elif score >= 0.15: return "ضعيف"
    else: return "متروك"


def colored_cat(category: str) -> str:
    """Return color-coded category string."""
    c = CATEGORY_COLORS.get(category, "")
    r = CATEGORY_COLORS["RESET"]
    return f"{c}{category}{r}"


def analyze_chain(chain: list[str], data: dict, matn_text: str = None) -> dict:
    """Analyze a full narrator chain. Returns analysis dict."""
    results = []
    for name in chain:
        results.append(lookup_narrator(name, data))

    scores = [r["predicted_score"] for r in results if r["predicted_score"] is not None]
    matched = sum(1 for r in results if r["match_type"] != "none")

    analysis = {
        "narrators": results,
        "chain_length": len(chain),
        "matched_count": matched,
        "match_rate": matched / max(len(chain), 1),
        "mean_score": np.mean(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "score_variance": np.var(scores) if scores else None,
        "weakest_narrator": None,
        "strongest_narrator": None,
    }

    if scores:
        min_idx = np.argmin(scores)
        max_idx = np.argmax(scores)
        # Find the actual results with scores
        scored_results = [(i, r) for i, r in enumerate(results) if r["predicted_score"] is not None]
        if scored_results:
            analysis["weakest_narrator"] = scored_results[min_idx][1]["matched_name"] or scored_results[min_idx][1]["name"]
            analysis["strongest_narrator"] = scored_results[max_idx][1]["matched_name"] or scored_results[max_idx][1]["name"]

    # Matn sentiment
    if matn_text and len(matn_text) > 20:
        analysis["sentiment"] = lexicon_sentiment_score(matn_text)
    else:
        analysis["sentiment"] = None

    # Compute sequential edge compatibilities using GNN embeddings
    compatibilities = []
    if data.get("embeddings") is not None:
        for i in range(len(results) - 1):
            n1 = results[i]
            n2 = results[i+1]
            if n1["node_id"] is not None and n2["node_id"] is not None:
                id1 = n1["node_id"] - 1  # Convert to 0-based for numpy array
                id2 = n2["node_id"] - 1
                if 0 <= id1 < len(data["embeddings"]) and 0 <= id2 < len(data["embeddings"]):
                    z1 = data["embeddings"][id1]
                    z2 = data["embeddings"][id2]
                    dot = np.sum(z1 * z2)
                    sim = 1.0 / (1.0 + np.exp(-dot))
                    compatibilities.append(float(sim))
                else:
                    compatibilities.append(None)
            else:
                compatibilities.append(None)
    analysis["edge_compatibilities"] = compatibilities

    # Compute total chain uncertainty (disagreement of views + chain variance of semantic reliability)
    uncs = [r["uncertainty"] for r in results if r.get("uncertainty") is not None]
    if uncs:
        view_disagreement = np.mean(uncs)
        r_scores = [r["semantic_reliability"] for r in results if r.get("semantic_reliability") is not None]
        r_var = np.var(r_scores) if r_scores else 0.0
        analysis["chain_uncertainty"] = float(view_disagreement + r_var)
    else:
        analysis["chain_uncertainty"] = 0.0

    # Verdict
    analysis["verdict"] = compute_verdict(analysis)

    return analysis


def compute_verdict(analysis: dict) -> dict:
    """Compute overall authenticity assessment based PURELY on isnād (chain) analysis.

    Note: Matn (content) analysis is explicitly excluded from this project scope.
    The verdict is derived entirely from narrator reliability scores + chain structure."""
    scores = [r["predicted_score"] for r in analysis["narrators"] if r["predicted_score"] is not None]

    if not scores:
        return {"label": "غير معروف (Unknown)", "confidence": 0.0,
                "reason": "No narrator predictions available — insufficient data."}

    mean_s = analysis["mean_score"]
    min_s = analysis["min_score"]
    var_s = analysis["score_variance"]
    n = len(scores)

    # Multi-factor isnād-based verdict (no matn analysis per project scope)
    factors = []

    # Factor 1: Mean narrator reliability (weighted by chain length)
    if mean_s >= 0.80:
        factors.append(("HIGH", +2, f"Very high mean narrator reliability ({mean_s:.2f})"))
    elif mean_s >= 0.65:
        factors.append(("MED", +1, f"Good mean narrator reliability ({mean_s:.2f})"))
    elif mean_s >= 0.50:
        factors.append(("MED", 0, f"Moderate mean narrator reliability ({mean_s:.2f})"))
    else:
        factors.append(("LOW", -2, f"Low mean narrator reliability ({mean_s:.2f})"))

    # Factor 2: Weakest link (critical in ḥadīth criticism)
    if min_s >= 0.70:
        factors.append(("HIGH", +2, f"All narrators reliable — no weak links (min={min_s:.2f})"))
    elif min_s >= 0.50:
        factors.append(("MED", 0, f"Acceptable weakest narrator ({min_s:.2f})"))
    elif min_s >= 0.30:
        factors.append(("LOW", -2, f"Suspicious narrator in chain (min={min_s:.2f})"))
    else:
        factors.append(("LOW", -4, f"Critically weak narrator in chain (min={min_s:.2f})"))

    # Factor 3: Chain consistency — are all narrators similar reliability?
    if var_s is not None and n >= 3:
        if var_s < 0.01:
            factors.append(("HIGH", +1, f"Highly consistent chain (σ²={var_s:.4f})"))
        elif var_s < 0.05:
            factors.append(("MED", 0, f"Moderate variance in chain (σ²={var_s:.4f})"))
        else:
            factors.append(("LOW", -1, f"High variance — inconsistent narrator quality (σ²={var_s:.4f})"))

    # Factor 4: Chain length (longer chains are more scrutinized in rijal works)
    if n >= 6:
        factors.append(("HIGH", +1, f"Long chain ({n} narrators) — well-documented transmission path"))
    elif n <= 3:
        factors.append(("MED", 0, f"Short chain ({n} narrators) — potentially missing links"))

    # Factor 5: GNN Structural Anomaly Detection
    anomalies = [r["anomaly_score"] for r in analysis["narrators"] if r.get("anomaly_score") is not None]
    if anomalies:
        max_anom = max(anomalies)
        if max_anom > 0.40:
            factors.append(("LOW", -2, f"GNN structural feature anomaly signal (max anomaly={max_anom:.2f})"))
        elif max_anom > 0.20:
            factors.append(("MED", -1, f"Mild GNN structural anomaly feature detected (max anomaly={max_anom:.2f})"))
        else:
            factors.append(("HIGH", +1, f"GNN typical structural embedding signal (max anomaly={max_anom:.2f})"))

    # Factor 6: GNN Path/Edge Compatibility
    compats = [c for c in analysis.get("edge_compatibilities", []) if c is not None]
    if compats:
        min_comp = min(compats)
        if min_comp < 0.25:
            factors.append(("LOW", -2, f"GNN structural edge compatibility feature discount (min compatibility={min_comp:.2f})"))
        elif min_comp < 0.45:
            factors.append(("MED", -1, f"Weak GNN structural link compatibility signal (min compatibility={min_comp:.2f})"))
        else:
            factors.append(("HIGH", +1, f"Highly compatible GNN structural link compatibility signal (min compatibility={min_comp:.2f})"))

    # Factor 7: Multi-View Chain Uncertainty
    unc = analysis.get("chain_uncertainty", 0.0)
    if unc > 0.35:
        factors.append(("LOW", -2, f"High multi-view uncertainty / critic disagreement (U={unc:.3f})"))
    elif unc > 0.20:
        factors.append(("MED", -1, f"Moderate multi-view uncertainty / critic disagreement (U={unc:.3f})"))
    else:
        factors.append(("HIGH", +1, f"Low multi-view uncertainty (consistent views, U={unc:.3f})"))

    # Aggregate score (purely isnād-based)
    total = sum(w for _, w, _ in factors)
    max_possible = 9

    if total >= 5:
        label = "مستنتج: صحيح (Estimated Plausibility: Ṣaḥīḥ-aligned)"
        confidence = min(0.95, 0.60 + total / max_possible * 0.35)
    elif total >= 2:
        label = "مستنتج: حسن (Estimated Plausibility: Ḥasan-aligned)"
        confidence = 0.40 + total / max_possible * 0.30
    elif total >= -1:
        label = "مستنتج: ضعيف (Estimated Plausibility: Ḍaʿīf-aligned)"
        confidence = 0.25 + (total + 1) / 3 * 0.30
    else:
        label = "مستنتج: متروك / موضوع (Estimated Plausibility: Rejected/Fabricated)"
        confidence = 0.15 + abs(total) / max_possible * 0.25

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "score": total,
        "max_score": max_possible,
        "factors": factors,
    }


# ── Display ──
SEP = "─" * 68
THIN = "─" * 50

def display_analysis(analysis: dict, matn_text: str = None, show_sentiment: bool = True):
    """Pretty-print the full analysis."""

    # Header
    print(f"\n╔{'═'*66}╗")
    print(f"║ {'HADITH AUTHENTICITY ANALYSIS':^64} ║")
    print(f"╚{'═'*66}╝")

    # Narrator chain
    print(f"\n┌── NARRATOR CHAIN ({analysis['chain_length']} narrators, "
          f"{analysis['matched_count']}/{analysis['chain_length']} matched) {'──'*10}")
    for i, r in enumerate(analysis["narrators"]):
        match_marker = ""
        if r["match_type"] == "fuzzy":
            match_marker = f" (≈ {r.get('fuzzy_score', 0):.0f}% match)"
        elif r["match_type"] == "none":
            match_marker = " ⚠ NOT FOUND"

        score_str = ""
        if r["predicted_score"] is not None:
            cat = colored_cat(r["predicted_category"])
            anom_str = f" (GNN Anomaly: {r.get('anomaly_score', 0.0):.2f})" if "anomaly_score" in r else ""
            score_str = f" → Fused: {r['predicted_score']:.3f} [{cat}]{anom_str}"
        else:
            score_str = " → (no prediction)"

        print(f"  {i+1}. {r['matched_name'] or r['name']}{match_marker}{score_str}")

        # Bio details if available
        bio_parts = []
        if r["ibn_hajar_grade"]:
            bio_parts.append(f"Ibn Ḥajar: {r['ibn_hajar_grade']}")
        if r.get("semantic_reliability") is not None and r.get("critic_count", 0) > 0:
            bio_parts.append(f"Jumhūr Semantic: {r['semantic_reliability']:.3f} (Agreement: {r['semantic_agreement']:.0%}, Critics: {r['critic_count']})")
        if r["death_year"]:
            bio_parts.append(f"D. {r['death_year']}")
        if r["tabaqa"]:
            bio_parts.append(f"Ṭabaqa: {r['tabaqa']}")
        if bio_parts:
            print(f"     └─ {' | '.join(bio_parts)}")

        # Print edge compatibility to next narrator if available
        if i < len(analysis["narrators"]) - 1:
            compats = analysis.get("edge_compatibilities", [])
            if i < len(compats) and compats[i] is not None:
                print(f"     ├─── [GNN Link Compatibility: {compats[i]:.2%}]")

    # Chain Statistics
    print(f"\n┌── CHAIN STATISTICS {'─'*50}")
    print(f"  Mean Score:     {analysis['mean_score']:.3f}" if analysis['mean_score'] else "  Mean Score: N/A")
    if analysis['weakest_narrator']:
        print(f"  Weakest Link:   {analysis['weakest_narrator']} ({analysis['min_score']:.3f})")
    if analysis['strongest_narrator']:
        print(f"  Strongest Link: {analysis['strongest_narrator']} ({analysis['max_score']:.3f})")
    if analysis['score_variance'] is not None:
        print(f"  Variance:       {analysis['score_variance']:.4f} "
              f"({'consistent' if analysis['score_variance'] < 0.02 else 'diverse'})")
    
    unc = analysis.get("chain_uncertainty", 0.0)
    unc_label = "Low (Highly consistent views)" if unc <= 0.15 else \
                "Medium (Moderate view disagreement)" if unc <= 0.30 else \
                "High (Severe view disagreement / high path variance)"
    print(f"  Uncertainty:    {unc:.3f} ({unc_label})")

    # Verdict (purely isnād-based — no matn analysis per project scope)
    v = analysis["verdict"]
    color = "\033[92m" if "صحيح" in v["label"] else \
            "\033[94m" if "حسن" in v["label"] else \
            "\033[93m" if "ضعيف" in v["label"] else "\033[91m"
    reset = "\033[0m"

    print(f"\n╔{'═'*66}╗")
    print(f"║ {'ESTIMATED GRADING PLAUSIBILITY':^64} ║")
    print(f"╚{'═'*66}╝")
    print(f"║ {color}{v['label']:^64}{reset} ║")
    print(f"║ {'Confidence: ' + str(int(v['confidence']*100)) + '%':^64} ║")
    print(f"╠{'═'*66}╣")
    print(f"║ Factors:                                              ║")
    for level, weight, reason in v["factors"]:
        wstr = f"+{weight}" if weight > 0 else str(weight)
        print(f"║   [{wstr}] {reason[:53]:<53s} ║")
    print(f"╚{'═'*66}╝")


# ── Interactive Modes ──

def interactive_mode(data: dict):
    """Main interactive loop."""
    print(f"\n{'='*70}")
    print("  INTERACTIVE HADITH AUTHENTICITY ANALYZER")
    print(f"{'='*70}")
    print("\n  Enter a sanad (narrator chain) to analyze, or type:")
    print("    :browse     — Browse the evaluation dataset")
    print("    :example    — Show a random example from the dataset")
    print("    :help       — Show help")
    print("    :quit       — Exit")
    print()

    while True:
        try:
            user_input = input("  SANAD> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith(":"):
            cmd = user_input[1:].lower()
            if cmd in ("q", "quit", "exit"):
                print("  Goodbye!")
                break
            elif cmd in ("h", "help"):
                print_help()
            elif cmd in ("b", "browse"):
                browse_mode(data)
            elif cmd in ("e", "example"):
                random_example(data)
            else:
                print(f"  Unknown command: {user_input}")
            continue

        # Analyze the input as a sanad
        chain = parse_chain(user_input)
        if len(chain) < 2:
            print(f"\n  ⚠ Could not parse a valid chain (found {len(chain)} narrators).")
            print("  Try pasting a full sanad text or use :browse to select one.")
            continue

        print(f"\n  Parsed {len(chain)} narrators. Analyzing...")

        analysis = analyze_chain(chain, data)
        display_analysis(analysis)


def browse_mode(data: dict):
    """Browse and select from the evaluation dataset."""
    if data["eval_df"] is None:
        print("  Evaluation dataset not available.")
        return

    df = data["eval_df"]
    print(f"\n  Evaluation Dataset: {len(df):,} hadiths")
    print(f"  Books: {', '.join(df['book_key'].unique())}")
    print(f"  Grades: {', '.join(df['grade_normalised'].dropna().unique())}")
    print()

    # Show summary
    for book in sorted(df["book_key"].unique()):
        bdf = df[df["book_key"] == book]
        print(f"  {book:<15s}: {len(bdf):>4d} hadiths")

    while True:
        try:
            sel = input("\n  Enter hadith # (0-1199), book name, or :back: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if sel.lower() in (":back", ":b", ""):
            break
        if sel.lower() in (":q", ":quit"):
            sys.exit(0)

        # Try as index
        try:
            idx = int(sel)
            if 0 <= idx < len(df):
                analyze_eval_entry(idx, data)
                continue
            else:
                print(f"  Index out of range (0-{len(df)-1})")
                continue
        except ValueError:
            pass

        # Try as book name
        matching = df[df["book_key"].str.contains(sel.lower(), na=False)]
        if len(matching) > 0:
            print(f"  Found {len(matching)} matches. Showing first 5:")
            for i, (qidx, row) in enumerate(matching.head(5).iterrows()):
                matn_preview = str(row.get("Arabic_Text", ""))[:80]
                print(f"    [{qidx}] {row['Book']} — {row['Grade']}")
                print(f"         {matn_preview}...")
            continue

        print(f"  No matches for '{sel}'")


def analyze_eval_entry(qidx: int, data: dict):
    """Analyze a specific entry from the evaluation dataset."""
    if data["eval_df"] is None:
        print("  Evaluation dataset not available.")
        return

    row = data["eval_df"].iloc[qidx]
    info = {
        "Book": row["Book"],
        "Grade": row["Grade"],
        "grade_normalised": row.get("grade_normalised", ""),
        "book_key": row["book_key"],
        "Arabic_Text": row["Arabic_Text"],
    }

    print(f"\n{'─'*70}")
    print(f"  Hadith #{qidx} — {info['Book']} ({info['book_key']})")
    print(f"  Canonical Grade: {info['Grade']} ({info['grade_normalised']})")
    print(f"{'─'*70}")

    # Try to find in alignment → Sanadset
    chain = None
    matn_text = None

    if data["alignment"] is not None:
        match = data["alignment"][
            (data["alignment"]["query_idx"] == qidx) &
            data["alignment"]["is_match"] &
            ~data["alignment"]["quarantine"]
        ]
        if not match.empty:
            corpus_idx = int(match.iloc[0]["corpus_idx"])
            if corpus_idx < len(data["sanadset"]):
                sanad_row = data["sanadset"].iloc[corpus_idx]
                sanad_text = str(sanad_row.get("Sanad", ""))
                matn_text = str(sanad_row.get("Matn", ""))
                chain = parse_chain(sanad_text)

    if chain is None or len(chain) < 2:
        # Fallback: try to parse the Arabic_Text for narrator names
        chain = parse_chain(str(row.get("Arabic_Text", "")))
        matn_text = None

    if chain and len(chain) >= 2:
        analysis = analyze_chain(chain, data, matn_text)

        # Show matn snippet
        if matn_text:
            snippet = matn_text[:300]
            print(f"\n  Matn: {snippet}...")

        display_analysis(analysis, matn_text)
    else:
        print(f"  Could not extract narrator chain for this hadith.")
        print(f"  Arabic text: {str(row.get('Arabic_Text', ''))[:200]}...")


def random_example(data: dict):
    """Show a random example from the evaluation dataset."""
    if data["eval_df"] is None:
        print("  No evaluation dataset available.")
        return

    import random
    qidx = random.randint(0, len(data["eval_df"]) - 1)
    analyze_eval_entry(qidx, data)


def print_help():
    """Print help text."""
    print(f"""
  ┌── HADITH AUTHENTICITY ANALYZER HELP ─────────────────────
  │
  │  INPUT FORMATS:
  │    • Paste a Python-style sanad list:
  │      ['بخاري', 'عمر بن حفص', 'أبي هريرة']
  │
  │    • Paste raw Arabic sanad text:
  │      حدثنا عمر بن حفص عن أبيه عن أبي هريرة
  │
  │    • Use <NAR> tags:
  │      <NAR>أبو هريرة</NAR> عن <NAR>سعيد بن المسيب</NAR>
  │
  │  COMMANDS:
  │    :browse  — Browse evaluation dataset by book/grade
  │    :example — Show a random hadith with full analysis
  │    :help    — Show this help
  │    :quit    — Exit
  │
  │  VERDICT FACTORS:
  │    • Mean narrator reliability score
  │    • Weakest link in the chain
  │    • Chain consistency (score variance)
  │    • Matn semantic sentiment (if provided)
  │
  └──────────────────────────────────────────────────────────
""")


# ── Main ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hadith Authenticity Analyzer")
    parser.add_argument("--browse", action="store_true", help="Browse evaluation dataset")
    parser.add_argument("--hadith-id", type=int, help="Analyze specific hadith from eval set")
    parser.add_argument("--sanad", type=str, help="Analyze a pasted sanad text")
    parser.add_argument("--matn", type=str, help="Optional matn text for sentiment analysis")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   Multi-Modal Hadith Authenticity Analyzer                  ║")
    print("║   ʿIlm ar-Rijāl · GNN Reliability · Semantic Sentiment     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\nLoading knowledge base...")
    data = load_data()
    print()

    if args.hadith_id is not None:
        analyze_eval_entry(args.hadith_id, data)
    elif args.sanad:
        chain = parse_chain(args.sanad)
        if len(chain) < 2:
            print(f"Could not parse a valid chain (found {len(chain)} narrators)")
            sys.exit(1)
        print(f"Parsed {len(chain)} narrators. Analyzing...")
        analysis = analyze_chain(chain, data, args.matn)
        display_analysis(analysis, args.matn)
    elif args.browse:
        browse_mode(data)
    else:
        interactive_mode(data)
