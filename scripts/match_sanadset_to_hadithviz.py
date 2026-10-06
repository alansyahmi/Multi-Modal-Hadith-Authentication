"""
Hierarchical entity resolution: SanadSet narrator names -> Hadith-Viz records.

Tiers (no fuzzy string similarity is used):
  T1  Normalized full-name match (diacritics stripped, alef/ya/ta-marbuta folded,
      ibn/abu forms folded, <IDF> tags cleaned, leading conjunctions stripped if needed).
      Unique candidate -> `exact`.
  T2  Several candidates share the name -> disambiguate by:
      (a) Taqrib tabaqa <-> Hadith-Viz death-year consistency (cohort nodes only),
      (b) Graph teacher/student transmission overlap against Hadith-Viz top_teachers/top_students,
      (c) A single candidate surviving -> `disambiguated`.
  Otherwise -> `ambiguous` (multiple survivors) or `unmatched`.

Outputs (output/hadith_viz_match/):
  matches_cohort.csv   per-node result for the 2,172 graded graph nodes
  attrition.json       funnel counts
  audit_sample.csv     stratified sample for manual precision audit (stratified by length x pool)
Also writes FYP2/attrition_table.md for Chapter 3.

Usage: python scripts/match_sanadset_to_hadithviz.py [--audit-n 100]
"""
import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HV = os.path.join(ROOT, "Dataset", "hadith_viz")
OUT = os.path.join(ROOT, "output", "hadith_viz_match")

_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_PUNCT = re.compile(r"[^\u0621-\u064A0-9 ]")


def norm(s: str) -> str:
    if not s:
        return ""
    # Strip <IDF> annotations (e.g. حماد<IDF> وهو </IDF>ابن زيد -> حماد ابن زيد)
    s = re.sub(r"<IDF>.*?</IDF>", " ", s)
    s = _DIAC.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = _PUNCT.sub(" ", s)
    toks = s.split()
    out = []
    for t in toks:
        if t in ("ابن", "بن"):
            t = "بن"
        elif t in ("ابو", "ابي", "ابا"):
            t = "ابو"
        out.append(t)
    return " ".join(out)


# Ibn Hajar tabaqa -> loose death-year window (AH), used ONLY as a consistency filter
TABAQA_WINDOWS = {
    "الاولي": (0, 110), "الثاني": (0, 125), "الثالث": (60, 140), "الرابع": (80, 160),
    "الخامس": (90, 170), "السادس": (100, 190), "السابع": (130, 215), "الثامن": (150, 235),
    "التاسع": (170, 255), "العاشر": (190, 275), "الحادي": (210, 290),
}


def tabaqa_window(t):
    n = norm(t or "")
    for k, w in TABAQA_WINDOWS.items():
        if k in n.split():
            return w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- Hadith-Viz indices
    shamela = json.load(open(os.path.join(HV, "rijal_shamela_cards.json"), encoding="utf-8"))["narrators"]
    v2 = json.load(open(os.path.join(HV, "rijal_v2_cards.json"), encoding="utf-8"))["rows"]
    
    # v2 columns: ['norm', 'display_name', 'candidate_count', 'source_tier', 'agreed_death_year_ah', 'agreed_kunya', 'agreed_nisba']
    idx = defaultdict(list)  # norm -> list of candidate dicts
    for r in v2:
        norm_key = norm(r[0])
        disp_name = r[1].strip() if r[1] and r[1].strip() else r[0]
        idx[norm_key].append({
            "src": "v2",
            "name": disp_name,
            "tier": r[3],
            "death": r[4],
            "kunya": r[5],
            "nisba": r[6],
        })

    sh_idx = defaultdict(list)
    for c in shamela:
        sh_idx[norm(c["name"])].append(c)

    # ---- SanadSet-derived names
    node_names = json.load(open(os.path.join(ROOT, "output", "stage2", "node_names.json"), encoding="utf-8"))
    uniq_norm = {norm(v) for v in node_names.values() if norm(v)}
    n_mentions = len(node_names)

    funnel = Counter()
    funnel["name_strings_node_names"] = n_mentions
    funnel["unique_normalized_names"] = len(uniq_norm)
    funnel["unique_norm_in_hv_v2"] = sum(1 for u in uniq_norm if u in idx)
    funnel["unique_norm_in_hv_curated"] = sum(1 for u in uniq_norm if u in sh_idx)

    # ---- Load transmission graph topology for context-aware tie breaking
    edges_path = os.path.join(ROOT, "output", "stage2", "graph_edges.csv")
    node_to_name = {}
    node_teachers = defaultdict(set)
    node_students = defaultdict(set)
    
    rows = list(csv.DictReader(open(os.path.join(ROOT, "output", "stage2", "graph_nodes.csv"), encoding="utf-8")))
    for r in rows:
        node_to_name[r["node_id"]] = r["canonical_name"]

    if os.path.exists(edges_path):
        with open(edges_path, encoding="utf-8") as ef:
            for er in csv.DictReader(ef):
                # source transmits from target: target is teacher, source is student
                node_teachers[er["source_id"]].add(node_to_name.get(er["target_id"], ""))
                node_students[er["target_id"]].add(node_to_name.get(er["source_id"], ""))

    funnel["cohort_graph_nodes"] = len(rows)
    results = []
    for r in rows:
        raw_name = r["canonical_name"]
        n = norm(raw_name)
        
        # Step 1: candidate lookup in curated pool, then v2 pool
        cands = sh_idx.get(n, [])
        pool = "curated"
        if not cands:
            cands = idx.get(n, [])
            pool = "v2"

        # Step 1b: Conjunction / particle fallback if no candidates found
        # (e.g., 'و ابو كريب' -> 'ابو كريب', or 'ومحمد بن جعفر' -> 'محمد بن جعفر')
        if not cands:
            alt_n = None
            if n.startswith("و "):
                alt_n = n[2:].strip()
            else:
                toks = n.split()
                if toks and toks[0].startswith("و") and len(toks[0]) > 2 and toks[0] not in ("وهب", "وكيع", "واصل", "ورقاء", "وهبان", "وليد"):
                    alt_n = " ".join([toks[0][1:]] + toks[1:])
            
            if alt_n:
                cands = sh_idx.get(alt_n, [])
                pool = "curated"
                if not cands:
                    cands = idx.get(alt_n, [])
                    pool = "v2"
                if cands:
                    n = alt_n

        status, pick, ncand = "unmatched", None, len(cands)

        if len(cands) == 1:
            status, pick = "exact", cands[0]
        elif len(cands) > 1:
            w = tabaqa_window(r["tabaqa"])
            def death_of(c):
                return c.get("death_year") if pool == "curated" else c.get("death")
            
            # Disambiguation Filter A: Tabaqa / Death-Year Window
            keep = [c for c in cands if w and death_of(c) is not None and w[0] <= death_of(c) <= w[1]]
            if len(keep) == 1:
                status, pick = "disambiguated", keep[0]
            else:
                target_pool = keep if keep else cands
                # Disambiguation Filter B: Graph Transmission Overlap against top_teachers/students
                t_names = {norm(name) for name in node_teachers.get(r["node_id"], set()) if name}
                s_names = {norm(name) for name in node_students.get(r["node_id"], set()) if name}

                def overlap_score(cand):
                    score = 0
                    if pool == "curated":
                        c_t = {norm(t[0]) for t in cand.get("top_teachers", []) if isinstance(t, list) and t}
                        c_s = {norm(s[0]) for s in cand.get("top_students", []) if isinstance(s, list) and s}
                        score += len(t_names.intersection(c_t)) * 2
                        score += len(s_names.intersection(c_s)) * 2
                    return score

                scored = [(overlap_score(c), c) for c in target_pool]
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                    status, pick = "disambiguated", scored[0][1]
                else:
                    status = "ambiguous"

        results.append({
            "node_id": r["node_id"],
            "canonical_name": raw_name,
            "norm": n,
            "name_tokens": len(n.split()),
            "tabaqa": r["tabaqa"],
            "pool": pool if cands else "",
            "n_candidates": ncand,
            "status": status,
            "hv_name": (pick or {}).get("name", ""),
            "hv_death": (pick or {}).get("death_year", (pick or {}).get("death", "")) if pick else "",
            "hv_shamela_id": (pick or {}).get("shamela_id", "") if pick else "",
        })

    sc = Counter(x["status"] for x in results)
    identity_confirmed = sc["exact"] + sc["disambiguated"]
    src_supported = sum(1 for x in results if x["status"] in ("exact", "disambiguated") and x["pool"] == "curated")
    funnel.update({
        "cohort_exact": sc["exact"],
        "cohort_disambiguated": sc["disambiguated"],
        "cohort_ambiguous": sc["ambiguous"],
        "cohort_unmatched": sc["unmatched"],
        "cohort_identity_confirmed": identity_confirmed,
        "cohort_source_supported_curated": src_supported,
    })

    with open(os.path.join(OUT, "matches_cohort.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    json.dump(dict(funnel), open(os.path.join(OUT, "attrition.json"), "w", encoding="utf-8"), indent=2)

    # ---- Stratified audit sample: stratified by (Name-Length Type x Pool)
    # Stratum 1: Single-token Curated (High ambiguity risk)
    # Stratum 2: Single-token V2
    # Stratum 3: Multi-token Curated (Standard high-precision)
    # Stratum 4: Multi-token V2
    random.seed(args.seed)
    strata = defaultdict(list)
    for x in results:
        if x["status"] in ("exact", "disambiguated"):
            length_cat = "single_token" if x["name_tokens"] == 1 else "multi_token"
            strata[(length_cat, x["pool"])].append(x)

    per_stratum = max(1, args.audit_n // len(strata))
    sample = []
    for (length_cat, pool_name), items in sorted(strata.items()):
        n_sample = min(per_stratum, len(items))
        selected = random.sample(items, n_sample)
        for s in selected:
            s_dict = dict(s)
            s_dict["stratum"] = f"{length_cat}_{pool_name}"
            sample.append(s_dict)

    with open(os.path.join(OUT, "audit_sample.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "sanadset_name", "tabaqa", "tokens", "status", "pool", "stratum", "hv_name", "hv_death", "hv_shamela_id", "CORRECT(1/0)"])
        for x in sample:
            w.writerow([
                x["node_id"],
                x["canonical_name"],
                x["tabaqa"],
                x["name_tokens"],
                x["status"],
                x["pool"],
                x["stratum"],
                x["hv_name"],
                x["hv_death"],
                x["hv_shamela_id"],
                "",
            ])

    # ---- Chapter 3 table
    single_confirmed = sum(1 for x in results if x["status"] in ("exact", "disambiguated") and x["name_tokens"] == 1)
    multi_confirmed = sum(1 for x in results if x["status"] in ("exact", "disambiguated") and x["name_tokens"] > 1)

    md = [
        "# Dataset Attrition Table (Entity Resolution, Week 3)", "",
        "_Generated by `scripts/match_sanadset_to_hadithviz.py`. Counts are reproducible._", "",
        "| Stage | Count | Note |",
        "|---|---:|---|",
        f"| SanadSet-derived narrator name strings (`node_names.json`) | {n_mentions:,} | Raw isnād chain tokens |",
        f"| Unique normalized names | {len(uniq_norm):,} | Orthographic folding & clean |",
        f"| ...with a Hadith-Viz v2 (248K) name hit | {funnel['unique_norm_in_hv_v2']:,} | Master index lookup |",
        f"| ...with a Hadith-Viz curated (3,047) name hit | {funnel['unique_norm_in_hv_curated']:,} | Curated multi-critic lookup |",
        f"| Graded cohort nodes (Stage 2 graph) | {len(rows):,} | Graph nodes evaluated in FYP |",
        f"| Cohort: exact unique match | {sc['exact']:,} | Deterministic 1:1 match |",
        f"| Cohort: disambiguated (tabaqa & graph topology) | {sc['disambiguated']:,} | Resolved via death-year / edges |",
        f"| Cohort: ambiguous (excluded) | {sc['ambiguous']:,} | Multiple candidates unresolved |",
        f"| Cohort: unmatched (excluded) | {sc['unmatched']:,} | Incomplete / pronouns / unindexed |",
        f"| **Identity-confirmed Cohort** | **{identity_confirmed:,}** | ({multi_confirmed:,} multi-token, {single_confirmed:,} single-token) |",
        f"| **Source-supported Cohort** | **{src_supported:,}** | Confirmed & present in multi-critic set |", "",
        "### Audit Sample Composition (`audit_sample.csv`)",
        f"- Total sample size: {len(sample)} instances across 4 strata (Single-token Curated, Single-token V2, Multi-token Curated, Multi-token V2).",
        "- All empty display names resolved by falling back to indexed canonical surface forms.",
        "- Precision score will be computed once hand-labelling of `audit_sample.csv` is completed.", "",
    ]
    open(os.path.join(ROOT, "FYP2", "attrition_table.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
