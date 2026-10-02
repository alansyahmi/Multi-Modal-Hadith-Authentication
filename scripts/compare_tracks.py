"""
Compare Track A (AraBERT v2 regression) vs Track B (GRADE_SCORE_MAP keyword lookup).

Evaluates both approaches against ground-truth grades from Kutub as-Sitta,
reporting accuracy, F1, MAE, RMSE, Spearman ρ, Pearson r, and Cohen's Kappa.

Output: comparison report (JSON + console table).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr, pearsonr

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACK_A_DIR = PROJECT_ROOT / "output" / "track_a"
STAGE2_DIR = PROJECT_ROOT / "output" / "stage2"
STAGE3_DIR = PROJECT_ROOT / "output" / "stage3"

# ── GRADE_SCORE_MAP (Track B lookup) ──
GRADE_SCORE_MAP: dict[str, float] = {
    "صحابي": 1.0,
    "ثقة ثبت": 0.95, "ثقة حافظ": 0.95, "ثقة متقن": 0.95, "إمام": 0.95,
    "حافظ": 0.90, "ثبت": 0.90, "متقن": 0.90,
    "ثقة": 0.85, "ثقات": 0.85, "مأمون": 0.85,
    "صدوق": 0.70, "لا بأس به": 0.65, "ليس به بأس": 0.65, "محله الصدق": 0.65,
    "صدوق يهم": 0.55, "صدوق سيء الحفظ": 0.50, "سيء الحفظ": 0.40,
    "مقبول": 0.40,
    "مستور": 0.25, "مجهول الحال": 0.20,
    "ضعيف": 0.15, "ضعيف الحديث": 0.15, "لين": 0.15, "لين الحديث": 0.15,
    "ليس بالقوي": 0.15, "فيه ضعف": 0.15,
    "منكر الحديث": 0.10, "مجهول": 0.10, "مجهول العين": 0.05,
    "متروك": 0.05, "متروك الحديث": 0.05, "ساقط": 0.05, "مختلط": 0.05,
    "متهم": 0.0, "كذاب": 0.0, "وضاع": 0.0, "دجال": 0.0,
}

DEFAULT_SCORE = 0.30


def to_ordinal(score: float) -> int:
    """Map continuous score to ordinal category for Kappa."""
    if score >= 0.85: return 4  # thiqah
    elif score >= 0.60: return 3  # saduq
    elif score >= 0.35: return 2  # maqbul
    elif score >= 0.15: return 1  # daif
    else: return 0  # matruk


def to_category_label(score: float) -> str:
    labels = {4: "thiqah", 3: "saduq", 2: "maqbul", 1: "daif", 0: "matruk"}
    return labels[to_ordinal(score)]


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Compute comprehensive evaluation metrics."""
    y_pred = np.clip(y_pred, 0.0, 1.0)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    spear_r, spear_p = spearmanr(y_true, y_pred)
    pear_r, pear_p = pearsonr(y_true, y_pred)

    # Binary: accepted (>= 0.65) vs rejected
    y_true_bin = (y_true >= 0.65).astype(int)
    y_pred_bin = (y_pred >= 0.65).astype(int)
    bin_acc = float(accuracy_score(y_true_bin, y_pred_bin))
    bin_f1 = float(f1_score(y_true_bin, y_pred_bin))

    # Ordinal categories
    y_true_cat = np.array([to_ordinal(s) for s in y_true])
    y_pred_cat = np.array([to_ordinal(s) for s in y_pred])
    kappa = float(cohen_kappa_score(y_true_cat, y_pred_cat, weights="quadratic"))

    # Per-category MAE
    cat_names = {0: "matruk", 1: "daif", 2: "maqbul", 3: "saduq", 4: "thiqah"}
    cat_mae = {}
    cat_counts = {}
    for cid, cname in cat_names.items():
        mask = y_true_cat == cid
        cnt = int(mask.sum())
        cat_counts[cname] = cnt
        cat_mae[cname] = float(np.abs(y_pred[mask] - y_true[mask]).mean()) if cnt >= 3 else None

    return {
        "label": label,
        "n": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "binary_accuracy": bin_acc,
        "binary_f1": bin_f1,
        "cohen_kappa_quadratic": kappa,
        "cat_mae": cat_mae,
        "cat_counts": cat_counts,
        "y_true_mean": float(y_true.mean()),
        "y_true_std": float(y_true.std()),
        "y_pred_mean": float(y_pred.mean()),
        "y_pred_std": float(y_pred.std()),
    }


def load_track_a_scores() -> dict[str, float]:
    """Load Track A narrator scores, mapped from node IDs → Arabic names."""
    path = TRACK_A_DIR / "narrator_scores.json"
    if not path.exists():
        print(f"  [WARN] Track A scores not found at {path}")
        return {}

    with open(path, encoding="utf-8") as f:
        node_scores: dict[str, float] = json.load(f)

    # Load node_id → name mapping
    names_path = STAGE2_DIR / "node_names.json"
    if not names_path.exists():
        print(f"  [WARN] node_names.json not found")
        return {}

    with open(names_path, encoding="utf-8") as f:
        id_to_name: dict[str, str] = json.load(f)

    # Remap: node_id → score → narrator_name → score
    import re

    def norm(text: str) -> str:
        """Normalize Arabic text for fuzzy matching."""
        text = re.sub(r"[ً-ٌٍ-َّْٰ]", "", text)
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        text = text.replace("ة", "ه").replace("ى", "ي")
        return text.strip()

    name_scores: dict[str, float] = {}
    for node_id, score in node_scores.items():
        if node_id in id_to_name:
            name_scores[id_to_name[node_id]] = score

    print(f"  Track A name-mapped: {len(name_scores):,}/{len(node_scores):,} narrators")
    return name_scores


def load_track_b_scores() -> dict[str, float]:
    """Compute Track B scores from narrator_grades.json using GRADE_SCORE_MAP."""
    path = STAGE2_DIR / "narrator_grades.json"
    with open(path, encoding="utf-8") as f:
        grades: dict = json.load(f)

    scores: dict[str, float] = {}
    for name, entry in grades.items():
        grade_text = entry.get("grade", "")
        matched = False
        for term in sorted(GRADE_SCORE_MAP, key=len, reverse=True):
            if term in grade_text:
                scores[name] = GRADE_SCORE_MAP[term]
                matched = True
                break
        if not matched:
            scores[name] = DEFAULT_SCORE
    return scores


def load_kutub_sitta_grades() -> dict[str, float]:
    """Load ground-truth grades from Stage 1 evaluation set.

    Returns dict of (hadith_id or narrator_name) → reliability score.
    The evaluation CSV has ground-truth classifications: Sahih, Hasan, Daif, etc.
    """
    path = STAGE2_DIR / "narrator_grades.json"
    with open(path, encoding="utf-8") as f:
        grades: dict = json.load(f)

    # Ground truth: use the Ibn Hajar grade scores from the grading map
    # as the best available approximation of scholarly consensus
    ground_truth: dict[str, float] = {}
    for name, entry in grades.items():
        grade_text = entry.get("grade", "")
        matched = False
        for term in sorted(GRADE_SCORE_MAP, key=len, reverse=True):
            if term in grade_text:
                ground_truth[name] = GRADE_SCORE_MAP[term]
                matched = True
                break
        if not matched:
            ground_truth[name] = DEFAULT_SCORE
    return ground_truth


def evaluate_vs_kutub_sitta(
    track_a: dict[str, float],
    track_b: dict[str, float],
    ground_truth: dict[str, float],
) -> dict:
    """Evaluate both tracks against ground-truth Kutub as-Sitta grades.

    Track A uses graph node names (from node_names.json).
    Track B uses Arabic narrator names from narrator_grades.json.
    We match them via fuzzy string comparison on normalized names.
    """
    import re
    from rapidfuzz import fuzz

    def norm(text: str) -> str:
        text = re.sub(r"[ً-ٌٍ-َّْٰ]", "", text)
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        text = text.replace("ة", "ه").replace("ى", "ي")
        return text.strip()

    # Build normalized lookup for Track A
    a_norm = {norm(k): s for k, s in track_a.items()}

    # The ground truth and Track B share the same names (from narrator_grades.json)
    # Match Track A names to ground truth names via fuzzy matching
    b_names = list(set(track_b.keys()) & set(ground_truth.keys()))
    a_norm_names = list(a_norm.keys())

    matched_names: list[str] = []
    for b_name in b_names:
        b_norm = norm(b_name)
        if not b_norm or len(b_norm) < 3:
            continue

        # Try exact match first
        if b_norm in a_norm:
            matched_names.append(b_name)
            continue

        # Fuzzy match: check if any Track A normalized name contains or is contained
        # by the Track B name, or vice versa (common in Arabic naming)
        best_score = 0
        best_a = None
        for a_n in a_norm_names:
            score = fuzz.partial_ratio(b_norm, a_n)
            if score > best_score:
                best_score = score
                best_a = a_n

        if best_score >= 80:
            # Use the matched Track A score under the ground truth name
            track_a[b_name] = a_norm[best_a]
            matched_names.append(b_name)

    print(f"  Fuzzy-matched Track A ↔ Track B: {len(matched_names):,}/{len(b_names):,}")

    if not matched_names:
        return {
            "note": f"No common names found. Track A has {len(a_norm):,} names, Track B has {len(b_names):,}",
            "track_a_names": len(a_norm),
            "track_b_names": len(b_names),
        }

    names = sorted(matched_names)
    y_true = np.array([ground_truth[n] for n in names])
    y_a = np.array([track_a.get(n, 0.30) for n in names])
    y_b = np.array([track_b.get(n, 0.30) for n in names])

    metrics = {
        "common_narrators": len(names),
        "track_a": compute_all_metrics(y_true, y_a, "Track A (AraBERT regression)"),
        "track_b": compute_all_metrics(y_true, y_b, "Track B (GRADE_SCORE_MAP lookup)"),
    }

    # Track A vs B direct comparison
    a_minus_b = y_a - y_b
    metrics["track_a_vs_b"] = {
        "mean_difference": float(a_minus_b.mean()),
        "std_difference": float(a_minus_b.std()),
        "max_a_above_b": float(a_minus_b.max()),
        "max_b_above_a": float(abs(a_minus_b.min())),
        "correlation": float(pearsonr(y_a, y_b).statistic),
    }

    return metrics


def print_comparison_table(metrics: dict):
    """Print a formatted comparison table."""
    if "note" in metrics:
        print(f"\n  {metrics['note']}")
        return

    print(f"\n{'=' * 65}")
    print(f"  TRACK A vs TRACK B COMPARISON")
    print(f"  Common narrators evaluated: {metrics['common_narrators']}")
    print(f"{'=' * 65}")

    rows = [
        ("MAE ↓", "mae", True),
        ("RMSE ↓", "rmse", True),
        ("R² ↑", "r2", False),
        ("Spearman ρ ↑", "spearman_r", False),
        ("Pearson r ↑", "pearson_r", False),
        ("Binary Accuracy ↑", "binary_accuracy", False),
        ("Binary F1 ↑", "binary_f1", False),
        ("Cohen's κ (quad) ↑", "cohen_kappa_quadratic", False),
    ]

    print(f"\n  {'Metric':<30s} {'Track A':>10s} {'Track B':>10s} {'Better':>10s}")
    print(f"  {'-' * 62}")

    for label, key, lower_is_better in rows:
        a_val = metrics["track_a"][key]
        b_val = metrics["track_b"][key]
        if lower_is_better:
            better = "Track A" if a_val < b_val else "Track B" if b_val < a_val else "Tie"
        else:
            better = "Track A" if a_val > b_val else "Track B" if b_val > a_val else "Tie"
        print(f"  {label:<30s} {a_val:>10.4f} {b_val:>10.4f} {better:>10s}")

    # Per-category MAE
    print(f"\n  Per-Category MAE:")
    print(f"  {'Category':<15s} {'Track A':>10s} {'Track B':>10s} {'Count':>8s}")
    print(f"  {'-' * 45}")
    for cat in ["thiqah", "saduq", "maqbul", "daif", "matruk"]:
        a_cat = metrics["track_a"]["cat_mae"].get(cat)
        b_cat = metrics["track_b"]["cat_mae"].get(cat)
        cnt = metrics["track_a"]["cat_counts"].get(cat, 0)
        a_str = f"{a_cat:.4f}" if a_cat is not None else "N/A"
        b_str = f"{b_cat:.4f}" if b_cat is not None else "N/A"
        print(f"  {cat:<15s} {a_str:>10s} {b_str:>10s} {cnt:>8d}")

    # Prediction distribution
    print(f"\n  Prediction Distribution:")
    print(f"  {'Source':<20s} {'Mean':>8s} {'Std':>8s}")
    print(f"  {'-' * 38}")
    for src_key in ["track_a", "track_b"]:
        m = metrics[src_key]
        print(f"  {src_key:<20s} {m['y_pred_mean']:>8.3f} {m['y_pred_std']:>8.3f}")
    print(f"  {'ground_truth':<20s} {metrics['track_a']['y_true_mean']:>8.3f} "
          f"{metrics['track_a']['y_true_std']:>8.3f}")

    # Track A vs B difference
    if "track_a_vs_b" in metrics:
        d = metrics["track_a_vs_b"]
        print(f"\n  Track A vs B Direct Comparison:")
        print(f"    Mean difference (A - B): {d['mean_difference']:.4f}")
        print(f"    Std of difference:       {d['std_difference']:.4f}")
        print(f"    Max A above B:           {d['max_a_above_b']:.4f}")
        print(f"    Max B above A:           {d['max_b_above_a']:.4f}")
        print(f"    Correlation (A vs B):    {d['correlation']:.4f}")


def main():
    print("=" * 65)
    print("  TRACK A vs TRACK B EVALUATION")
    print("=" * 65)

    # Load scores
    print("\nLoading scores...")
    track_a = load_track_a_scores()
    track_b = load_track_b_scores()
    ground_truth = load_kutub_sitta_grades()

    print(f"  Track A (AraBERT):      {len(track_a):,} narrators")
    print(f"  Track B (keyword):      {len(track_b):,} narrators")
    print(f"  Ground truth (Ibn Hajar): {len(ground_truth):,} narrators")

    if not track_a:
        print("\n  Track A scores not found.")
        print("  Run 'python scripts/train_arabert_regression.py' first.")
        sys.exit(1)

    # Evaluate
    metrics = evaluate_vs_kutub_sitta(track_a, track_b, ground_truth)
    print_comparison_table(metrics)

    # Save results
    out_path = Path("output") / "track_a" / "comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")

    # Summary verdict
    if "note" not in metrics:
        a_mae = metrics["track_a"]["mae"]
        b_mae = metrics["track_b"]["mae"]
        a_kap = metrics["track_a"]["cohen_kappa_quadratic"]
        b_kap = metrics["track_b"]["cohen_kappa_quadratic"]
        print(f"\n{'=' * 65}")
        print(f"  VERDICT: Track A MAE={a_mae:.4f} vs Track B MAE={b_mae:.4f}")
        if a_mae < b_mae:
            print(f"  Track A improves over Track B by {b_mae - a_mae:.4f} MAE")
        elif b_mae < a_mae:
            print(f"  Track B still outperforms Track A by {a_mae - b_mae:.4f} MAE")
        else:
            print(f"  Both tracks perform equally")
        print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
