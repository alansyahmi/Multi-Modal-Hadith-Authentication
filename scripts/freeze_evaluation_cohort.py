"""
Build and freeze the evaluation cohort with multi-critic ground truth.

This script links:
  1. `output/stage2/graph_nodes.csv` (Graph topological nodes)
  2. `output/hadith_viz_match/matches_cohort.csv` (Entity resolution links)
  3. `Dataset/hadith_viz/rijal_shamela_cards.json` (Hadith-Viz verified metadata)
  4. `Dataset/hadith_viz/shamela_rijal2.json` (Multi-scholar evaluations & citations)

Outputs:
  - `output/hadith_viz_match/frozen_evaluation_cohort.csv`
      Canonical 1,353 source-supported cohort containing:
        - node_id, canonical_name, death_year, tabaqa
        - hadith_viz_id, hadith_viz_name
        - primary_consensus_grade, grade_source
        - multi_scholar_citations_count
        - split (train: 70%, val: 15%, test: 15% stratified)
  - `FYP2/cohort_distribution.md`
      Statistical summary for Chapter 3 thesis writeup.

Usage:
    python scripts/freeze_evaluation_cohort.py
"""

import os
import csv
import json
from collections import Counter
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "hadith_viz_match")
HV_DIR = os.path.join(ROOT, "Dataset", "hadith_viz")

# 5-class canonical label map (0: matruk, 1: daif, 2: maqbul, 3: saduq, 4: thiqah/sahabi)
TIER_TO_CLASS = {
    "sahabi": (4, "sahabi"),
    "thiqat": (4, "thiqah"),
    "saduq": (3, "saduq"),
    "saduq_weak": (2, "maqbul"),
    "daif": (1, "daif"),
    "matruk": (0, "matruk"),
}

TAQRIB_TO_CLASS = {
    "صحابي": (4, "sahabi"),
    "ثقة": (4, "thiqah"),
    "صدوق": (3, "saduq"),
    "مقبول": (2, "maqbul"),
    "مجهول": (2, "maqbul"),
    "ضعيف": (1, "daif"),
    "متروك": (0, "matruk"),
}


def main():
    print("[Cohort Freeze] Loading datasets...")
    cards = json.load(open(os.path.join(HV_DIR, "rijal_shamela_cards.json"), encoding="utf-8"))["narrators"]
    cards_by_name = {c["name"]: c for c in cards}

    evals = json.load(open(os.path.join(HV_DIR, "shamela_rijal2.json"), encoding="utf-8"))["narrators"]
    evals_by_name = {e["name"]: e for e in evals}

    node_rows = {r["node_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "output", "stage2", "graph_nodes.csv"), encoding="utf-8"))}

    matches = list(csv.DictReader(open(os.path.join(OUT_DIR, "matches_cohort.csv"), encoding="utf-8-sig")))
    curated_matches = [m for m in matches if m["status"] in ("exact", "disambiguated") and m["pool"] == "curated"]

    print(f"[Cohort Freeze] Total source-supported candidates: {len(curated_matches)}")

    records = []
    for m in curated_matches:
        nid = m["node_id"]
        cname = m["canonical_name"]
        hv_name = m["hv_name"]
        gn = node_rows.get(nid, {})
        card = cards_by_name.get(hv_name, {})
        ev = evals_by_name.get(hv_name, {})

        # Determine death year
        d_year = m.get("hv_death") or card.get("death_year") or ""

        # Determine multi-scholar ground truth
        # Priority 1: Hadith-Viz curated consensus tier
        hv_tier = card.get("grade_tier")
        if not hv_tier and card.get("is_companion"):
            hv_tier = "sahabi"

        grade_class = None
        grade_name = None
        source_type = None

        if hv_tier and hv_tier in TIER_TO_CLASS:
            grade_class, grade_name = TIER_TO_CLASS[hv_tier]
            source_type = "hadith_viz_consensus"
        elif gn.get("grade") and gn["grade"] in TAQRIB_TO_CLASS:
            grade_class, grade_name = TAQRIB_TO_CLASS[gn["grade"]]
            source_type = "taqrib_classical"
        else:
            # Unannotated narrator
            grade_class = -1
            grade_name = "unrated"
            source_type = "unrated"

        jt_count = len(ev.get("jarh_tadil", []))
        bios_count = len(card.get("verified_bios", []))
        citations_count = len(ev.get("sources_cited", []))

        records.append({
            "node_id": nid,
            "canonical_name": cname,
            "normalized_name": m["norm"],
            "hv_name": hv_name,
            "hv_shamela_id": m.get("hv_shamela_id") or card.get("shamela_id", ""),
            "death_year_ah": d_year,
            "tabaqa": gn.get("tabaqa", ""),
            "is_companion": 1 if card.get("is_companion") else 0,
            "ground_truth_class": grade_class,
            "ground_truth_label": grade_name,
            "ground_truth_source": source_type,
            "hadith_viz_tier": hv_tier or "",
            "taqrib_grade": gn.get("grade", ""),
            "total_evaluations_count": card.get("total_evaluations", 0),
            "jarh_tadil_statements_count": jt_count,
            "biographies_count": bios_count,
            "sources_cited_count": citations_count,
        })

    # Stratified 70/15/15 split on valid graded nodes
    graded_records = [r for r in records if r["ground_truth_class"] >= 0]
    ungraded_records = [r for r in records if r["ground_truth_class"] < 0]

    labels = [r["ground_truth_class"] for r in graded_records]
    train_idx, val_test_idx = train_test_split(
        range(len(graded_records)),
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    val_labels = [labels[i] for i in val_test_idx]
    val_sub_idx, test_sub_idx = train_test_split(
        range(len(val_test_idx)),
        test_size=0.50,
        random_state=42,
        stratify=val_labels,
    )

    val_idx = [val_test_idx[i] for i in val_sub_idx]
    test_idx = [val_test_idx[i] for i in test_sub_idx]

    for i in train_idx: graded_records[i]["split"] = "train"
    for i in val_idx: graded_records[i]["split"] = "val"
    for i in test_idx: graded_records[i]["split"] = "test"
    for r in ungraded_records: r["split"] = "unrated"

    all_records = graded_records + ungraded_records
    all_records.sort(key=lambda x: int(x["node_id"]))

    # Save CSV
    out_csv = os.path.join(OUT_DIR, "frozen_evaluation_cohort.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader()
        w.writerows(all_records)

    print(f"[Cohort Freeze] Saved {len(all_records)} cohort profiles to: {out_csv}")

    # Generate Statistical Documentation for Chapter 3
    class_counts = Counter(r["ground_truth_label"] for r in graded_records)
    split_counts = Counter(r["split"] for r in all_records)
    src_counts = Counter(r["ground_truth_source"] for r in all_records)

    md_lines = [
        "# Frozen Evaluation Cohort Specification (Week 3 Freeze)", "",
        "_Generated by `scripts/freeze_evaluation_cohort.py`. This artifact defines the frozen experimental dataset for PO1, PO2, and PO3._", "",
        "## 1. Cohort Overview",
        f"- **Total Source-Supported Cohort Nodes**: {len(all_records):,}",
        f"- **Graded Evaluation Nodes**: {len(graded_records):,} ({len(graded_records)/len(all_records)*100:.1f}%)",
        f"- **Unrated / Exploration Nodes**: {len(ungraded_records):,} ({len(ungraded_records)/len(all_records)*100:.1f}%)", "",
        "## 2. Multi-Critic Ground Truth Sources",
        "| Ground Truth Source | Count | Pct | Methodological Justification |",
        "|---|---:|---:|---|",
        f"| `hadith_viz_consensus` | {src_counts['hadith_viz_consensus']:,} | {src_counts['hadith_viz_consensus']/len(all_records)*100:.1f}% | Multi-scholar consensus across classical works (Ibn Ma'īn, Ahmad, Bukhārī, etc.) |",
        f"| `taqrib_classical` | {src_counts['taqrib_classical']:,} | {src_counts['taqrib_classical']/len(all_records)*100:.1f}% | Canonical *Taqrīb al-Tahdhīb* baseline |",
        f"| `unrated` | {src_counts['unrated']:,} | {src_counts['unrated']/len(all_records)*100:.1f}% | Retained for graph topology; excluded from supervised loss |", "",
        "## 3. Class Distribution (5-Class Canonical Scale)",
        "| Class ID | Canonical Label | Count | Proportion | Stratified Train (70%) | Stratified Val (15%) | Stratified Test (15%) |",
        "|:---:|:---|---:|---:|---:|---:|---:|",
    ]

    for cid, cname in [(4, "thiqah/sahabi"), (3, "saduq"), (2, "maqbul"), (1, "daif"), (0, "matruk")]:
        cnt = class_counts.get(cname.split("/")[0], 0) + (class_counts.get("sahabi", 0) if "sahabi" in cname else 0)
        tr = sum(1 for r in graded_records if r["split"] == "train" and r["ground_truth_class"] == cid)
        va = sum(1 for r in graded_records if r["split"] == "val" and r["ground_truth_class"] == cid)
        te = sum(1 for r in graded_records if r["split"] == "test" and r["ground_truth_class"] == cid)
        md_lines.append(f"| {cid} | `{cname}` | {cnt:,} | {cnt/len(graded_records)*100:.1f}% | {tr:,} | {va:,} | {te:,} |")

    md_lines.extend([
        f"| **Total** | **All Graded** | **{len(graded_records):,}** | **100.0%** | **{split_counts['train']:,}** | **{split_counts['val']:,}** | **{split_counts['test']:,}** |", "",
        "## 4. Multi-Source Evidence Depth",
        f"- Nodes with individual classical `jarh_tadil` critic statements: **{sum(1 for r in all_records if r['jarh_tadil_statements_count'] > 0):,}**",
        f"- Nodes with cross-referenced classical biographical texts: **{sum(1 for r in all_records if r['biographies_count'] > 0):,}**",
        f"- Average citations per narrator profile: **{sum(r['total_evaluations_count'] for r in all_records) / len(all_records):.1f} classical opinions**",
    ])

    report_path = os.path.join(ROOT, "FYP2", "cohort_distribution.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[Cohort Freeze] Specification written to: {report_path}")
    print("\n" + "\n".join(md_lines[:35]))


if __name__ == "__main__":
    main()
