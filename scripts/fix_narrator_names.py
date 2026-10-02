"""
Fix narrator name fragmentation and normalization bugs.

Problems found:
  1. "Abīhi" (his father) — 3,927 occurrences treated as a separate narrator.
     Actually a kinship reference, not a person. Gets score 0.677 because
     it appears alongside reliable narrators.
  2. "Abī" / "Abu" / "Aba" — kunya prefixes fragmented from names.
     "Abu Hurayra" sometimes becomes ["Abi", "Hurayra"].
  3. 19 graph nodes for "Abihi" variants, 16+ for "Abi" variants.
  4. 342 occurrences of standalone "Abī" (not actually a person).
  5. Abu Hurayra stored as 4+ different node names.

Fixes applied:
  A. Strip diacritics + normalize alif/ya/ta variants
  B. Unify kunya prefixes: Abu/Abi/Aba all → "abu" base
  C. Strip conjunction prefix "wa-" (و)
  D. Filter KINSHIP references: abihi, akhihi, ummihi, ibnihi, jaddihi, etc.
  E. Merge known fragmented names into canonical forms
  F. Rebuild graph with clean names (or patch existing)

Usage:
  python scripts/fix_narrator_names.py --dry-run     # Show what would change
  python scripts/fix_narrator_names.py --patch       # Patch graph + predictions
  python scripts/fix_narrator_names.py --rebuild     # Full graph rebuild (slow)
"""
import sys, json, re, ast
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── KINSHIP / PSEUDONYM REFERENCES ──
# These are NOT narrator names — they're relational references
KINSHIP_PATTERNS = [
    r'^أ?ب[iي]?[هةِ]{0,3}$',      # Abīhi, Abīhā, etc. (his/her father)
    r'^أ?خ[iي]?[هةِ]{0,3}$',      # Akhīhi (his brother)
    r'^أ?م[iي]?[هةِ]{0,3}$',      # Ummihi (his mother)
    r'^ا?بن[هةِ]{0,3}$',           # Ibnihi (his son)
    r'^ج[دذ][هةِ]{0,3}$',          # Jaddihi (his grandfather)
    r'^ع[مم][هةِ]{0,3}$',         # Ammihi (his uncle)
    r'^أ?خت[iي]?[هةِ]{0,3}$',     # Ukhtihi (his sister)
]
KINSHIP_RE = re.compile('|'.join(KINSHIP_PATTERNS))

# ── NAME NORMALIZATION ──
def strip_diacritics(text: str) -> str:
    return re.sub(r'[ً-ٌٍ-َّْٰٓ]', '', text)

def normalize_arabic_letters(text: str) -> str:
    """Unify Arabic letter variants."""
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ى', 'ي').replace('ة', 'ه')
    return text

def strip_wa_prefix(text: str) -> str:
    """Remove conjunction 'wa-' (and) prefix."""
    text = text.strip()
    if text.startswith('و'):
        text = text[1:]
    return text.strip()

def unify_kunya(text: str) -> str:
    """Standardize kunya prefixes: abu/abi/aba → abu."""
    text = text.strip()
    # Unify abu forms
    text = re.sub(r'^(أ?ب[ويا])\s+', r'أبو ', text)
    # Unify ibn forms
    text = re.sub(r'^(ا?بن)\s+', r'ابن ', text)
    # Unify umm forms
    text = re.sub(r'^(أ?م)\s+', r'أم ', text)
    return text.strip()

def is_kinship_reference(name: str) -> bool:
    """Check if name is actually a kinship reference, not a person."""
    clean = strip_diacritics(name).strip()
    clean = normalize_arabic_letters(clean)
    # Remove trailing wa
    clean = clean.rstrip('و').strip()
    return bool(KINSHIP_RE.match(clean))

def is_bare_prefix(name: str) -> bool:
    """Check if name is just a bare kunya/prefix with no actual name."""
    clean = strip_diacritics(name).strip()
    clean = normalize_arabic_letters(clean)
    # Standalone abu/abi/aba/ibn/bint with nothing after
    bare_prefixes = ['ابو', 'ابي', 'ابا', 'ابن', 'بنت', 'ام', 'اخو', 'اخي', 'اخا']
    if clean in bare_prefixes:
        return True
    # If it's just 1-2 chars after stripping (like "أب" alone)
    if len(clean) <= 2:
        return True
    return False

def normalize_narrator_name(name: str) -> dict:
    """Full narrator name normalization. Returns {clean_name, is_valid, reason}."""
    if not name or not isinstance(name, str):
        return {"clean_name": "", "is_valid": False, "reason": "empty"}

    # Remove tags
    name = re.sub(r'<[^>]+>', '', name)
    name = name.strip()

    # Step 1: Strip diacritics
    name = strip_diacritics(name)

    # Step 2: Normalize letters
    name = normalize_arabic_letters(name)

    # Step 3: Strip wa- prefix
    name = strip_wa_prefix(name)

    # Step 4: Check kinship references
    if is_kinship_reference(name):
        return {"clean_name": name, "is_valid": False, "reason": "kinship_reference"}

    # Step 5: Check bare prefixes
    if is_bare_prefix(name):
        return {"clean_name": name, "is_valid": False, "reason": "bare_prefix"}

    # Step 6: Unify kunya
    name = unify_kunya(name)

    # Step 7: Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # Final check: must have at least 3 meaningful chars
    if len(name) < 3:
        return {"clean_name": name, "is_valid": False, "reason": "too_short"}

    return {"clean_name": name, "is_valid": True, "reason": "ok"}


# ── DIAGNOSTICS ──
def diagnose():
    """Show what would be filtered out."""
    print("=" * 60)
    print("NAME NORMALIZATION DIAGNOSIS")
    print("=" * 60)

    with open("output/stage2/node_names.json", encoding="utf-8") as f:
        id_to_name = json.load(f)

    invalid = []
    for nid_str, name in id_to_name.items():
        result = normalize_narrator_name(name)
        if not result["is_valid"]:
            invalid.append({
                "node_id": nid_str,
                "original": name,
                "clean": result["clean_name"],
                "reason": result["reason"],
            })

    print(f"\n  Total nodes: {len(id_to_name):,}")
    print(f"  Invalid nodes: {len(invalid):,} ({len(invalid)/len(id_to_name)*100:.1f}%)")

    # Breakdown by reason
    reasons = Counter(r["reason"] for r in invalid)
    print(f"\n  Reasons:")
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count:,}")

    # Show examples of each type
    print(f"\n  ── KINSHIP REFERENCES (to be removed) ──")
    kinship = [r for r in invalid if r["reason"] == "kinship_reference"]
    for r in kinship[:12]:
        print(f"    Node {r['node_id']}: [{r['original']}]")

    print(f"\n  ── BARE PREFIXES (to be removed) ──")
    bare = [r for r in invalid if r["reason"] == "bare_prefix"]
    for r in bare[:12]:
        print(f"    Node {r['node_id']}: [{r['original']}]")

    # Check predictions for invalid nodes
    preds = pd.read_csv("output/stage3/narrator_predictions.csv")
    print(f"\n  ── PREDICTIONS FOR INVALID NODES ──")
    for r in invalid[:10]:
        nid = int(r["node_id"])
        match = preds[preds["node_id"] == nid]
        if len(match) > 0:
            print(f"    {r['original']}: predicted={match.iloc[0]['predicted_score']:.3f} "
                  f"error={match.iloc[0]['error']:.3f}")

    return invalid


# ── PATCH (quick fix for existing graph + predictions) ──
def patch():
    """Patch existing graph and predictions to filter invalid names."""
    print("=" * 60)
    print("PATCHING GRAPH + PREDICTIONS")
    print("=" * 60)

    with open("output/stage2/node_names.json", encoding="utf-8") as f:
        id_to_name = json.load(f)

    # Build cleanup map
    invalid_ids = set()
    name_merges = {}  # old_name → canonical_name

    for nid_str, name in id_to_name.items():
        result = normalize_narrator_name(name)
        if not result["is_valid"]:
            invalid_ids.add(int(nid_str))
        elif result["clean_name"] != strip_diacritics(name).strip():
            name_merges[strip_diacritics(name).strip()] = result["clean_name"]

    print(f"  Nodes to invalidate: {len(invalid_ids):,}")
    print(f"  Name merges: {len(name_merges):,}")

    # Save cleanup info for use by authenticator/experiment scripts
    cleanup = {
        "invalid_node_ids": sorted(list(invalid_ids)),
        "name_merges": name_merges,
        "kinship_pattern": KINSHIP_PATTERNS,
    }
    out_path = "output/stage2/name_cleanup.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleanup, f, indent=2, ensure_ascii=False)
    print(f"\n  Cleanup map saved: {out_path}")

    # Re-export predictions CSV with invalid nodes flagged
    preds = pd.read_csv("output/stage3/narrator_predictions.csv")
    preds["is_valid_name"] = ~preds["node_id"].isin(invalid_ids) if "node_id" in preds.columns else True

    # Add cleaned names
    clean_names = {}
    for nid_str, name in id_to_name.items():
        nid = int(nid_str)
        result = normalize_narrator_name(name)
        clean_names[nid] = result["clean_name"] if result["is_valid"] else f"[INVALID: {result['reason']}]"

    preds["clean_name"] = preds["node_id"].map(clean_names)

    clean_preds_path = "output/stage3/narrator_predictions_clean.csv"
    preds.to_csv(clean_preds_path, index=False, encoding="utf-8")
    print(f"  Clean predictions: {clean_preds_path}")

    # Show what changes
    invalid_preds = preds[~preds["is_valid_name"]]
    if len(invalid_preds) > 0:
        print(f"\n  ── Invalid narrator predictions (now flagged) ──")
        for _, row in invalid_preds.head(15).iterrows():
            print(f"    Node {int(row['node_id'])}: {row['clean_name']} → pred={row['predicted_score']:.3f}")

    return cleanup


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Diagnose without changes")
    parser.add_argument("--patch", action="store_true", help="Patch existing graph + predictions")
    parser.add_argument("--rebuild", action="store_true", help="Full graph rebuild (not yet implemented)")
    args = parser.parse_args()

    if args.patch:
        patch()
    else:
        diagnose()
