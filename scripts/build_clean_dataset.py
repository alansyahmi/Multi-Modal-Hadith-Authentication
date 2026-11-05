"""
Build Clean Cohort & Graph (Phase 1 of Recovery Plan).

Performs strict, leak-free cohort assembly and graph reconstruction:
1. Merges duplicate surface nodes by unique Hadith-Viz identity (`hv_name`).
2. Applies strict exclusion criteria:
   - Non-companion (`is_companion == False`)
   - Multi-token canonical names only (excludes ambiguous single-token matches)
   - Valid Ibn Hajar grade (`ibn_hajar_grade`)
   - >= 1 earlier critic appraisal statement
3. Extracts clean text exclusively from earlier classical critics (strictly excluding Ibn Hajar blocks & summary grades).
4. Maps Ibn Hajar grade to 3 canonical tiers:
   - 0: below_saduq (weak / cautionary / unverified)
   - 1: saduq (truthful / acceptable)
   - 2: thiqah (trustworthy / authoritative)
5. Attaches Dual Consensus flag (Ibn Hajar == Dhahabi) for Chapter 6/7 sub-analyses.
6. Rebuilds graph: merges nodes, removes self-loops, and separates Topology (5) vs. Metadata (3) features.

Outputs:
  - `output/cleaned_cohort/clean_narrators_cohort.csv`
  - `output/cleaned_cohort/clean_texts.json`
  - `output/cleaned_cohort/clean_graph.npz`
  - `output/cleaned_cohort/cohort_summary.json`
"""

import sys
import json
import os
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "output/cleaned_cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from src.canonical_data_pipeline import (
    map_ibn_hajar_tier,
    map_dhahabi_tier,
    TIER_TO_INT,
    is_earlier_critic,
    has_late_compiler_leak,
    EXCLUDE_LATE_SCHOLARS
)


def parse_tabaqa_float(t_str: str) -> float:
    t = str(t_str).strip()
    order = {
        "أول": 1, "اول": 1, "الأول": 1,
        "ثاني": 2, "الثاني": 2,
        "ثالث": 3, "الثالث": 3,
        "رابع": 4, "الرابع": 4,
        "خامس": 5, "الخامس": 5,
        "سادس": 6, "السادس": 6,
        "سابع": 7, "السابع": 7,
        "ثامن": 8, "الثامن": 8,
        "تاسع": 9, "التاسع": 9,
        "عاشر": 10, "العاشر": 10, "عا": 10,
        "حادي": 11, "الحادي": 11,
        "ثاني عشر": 12, "الثاني عشر": 12,
    }
    for k, v in order.items():
        if k in t:
            return round(v / 12.0, 4)
    return 0.5


def main():
    print("=" * 65)
    print("PHASE 1: CLEANING COHORT & RECONSTRUCTING LEAK-FREE GRAPH")
    print("=" * 65)

    # 1. Load source datasets
    cards_path = ROOT / "Dataset/hadith_viz/rijal_shamela_cards.json"
    evals_path = ROOT / "Dataset/hadith_viz/shamela_rijal2.json"
    cohort_path = ROOT / "output/hadith_viz_match/frozen_evaluation_cohort.csv"
    nodes_csv = ROOT / "output/stage2/graph_nodes.csv"
    edges_csv = ROOT / "output/stage2/graph_edges.csv"

    print("Loading raw Hadith-Viz cards and classical evaluations...")
    cards = json.load(open(cards_path, encoding="utf-8"))["narrators"]
    evals = json.load(open(evals_path, encoding="utf-8"))["narrators"]
    evals_by_name = {e["name"]: e for e in evals}
    cards_by_name = {c["name"]: c for c in cards}

    cohort_raw = pd.read_csv(cohort_path)
    nodes_df = pd.read_csv(nodes_csv).set_index("node_id")
    edges_df = pd.read_csv(edges_csv)

    print(f"  Raw cohort rows: {len(cohort_raw)}")
    print(f"  Graph nodes: {len(nodes_df)} | Raw edges: {len(edges_df)}")

    # 2. Filter cohort by strict eligibility criteria
    cohort_raw["n_tokens"] = cohort_raw["canonical_name"].fillna("").str.split().str.len()
    eligible_raw = cohort_raw[cohort_raw["n_tokens"] > 1].copy()

    # Map raw nodes to canonical Hadith-Viz identities
    identity_to_nodes = defaultdict(list)
    identity_records = {}

    for _, r in eligible_raw.iterrows():
        hv_name = r["hv_name"]
        ev = evals_by_name.get(hv_name)
        if not ev:
            continue

        # Exclude companions
        if r.get("is_companion") == 1 or ev.get("is_companion"):
            continue

        ih_grade_raw = ev.get("ibn_hajar_grade")
        ih_tier = map_ibn_hajar_tier(ih_grade_raw)
        if not ih_tier:
            continue

        # Extract earlier critics exclusively (strictly exclude Ibn Hajar blocks & grades)
        earlier_statements = []
        for b in ev.get("jarh_tadil", []):
            sch = b.get("scholar", "").strip()
            # Strictly exclude later synthesis authorities (Ibn Hajar, Dhahabi, Mizzi, Mughultay, etc.)
            if not is_earlier_critic(sch):
                continue
            for s in b.get("evaluations", []):
                txt = s.get("text", "").strip()
                if txt and len(txt) > 5 and not has_late_compiler_leak(txt):
                    earlier_statements.append({"scholar": sch, "text": txt})

        if len(earlier_statements) == 0:
            continue

        nid = int(r["node_id"])
        identity_to_nodes[hv_name].append(nid)

        if hv_name not in identity_records:
            dh_grade_raw = ev.get("dhahabi_grade")
            dh_tier = map_dhahabi_tier(dh_grade_raw)
            is_consensus = bool(ih_tier and dh_tier and (ih_tier == dh_tier))

            # Biographical metadata from card/node
            death_yr = r.get("death_year_ah")
            tabaqa = r.get("tabaqa", "")

            identity_records[hv_name] = {
                "hv_name": hv_name,
                "canonical_name": r["canonical_name"],
                "shamela_id": ev.get("shamela_id"),
                "ibn_hajar_grade": ih_grade_raw,
                "dhahabi_grade": dh_grade_raw,
                "tier_label": ih_tier,
                "tier_class": TIER_TO_INT[ih_tier],
                "dhahabi_tier": dh_tier,
                "is_dual_consensus": is_consensus,
                "earlier_critic_statements_count": len(earlier_statements),
                "earlier_statements": earlier_statements,
                "death_year_ah": death_yr,
                "tabaqa": tabaqa,
                "node_ids": [nid],
            }
        else:
            identity_records[hv_name]["node_ids"].append(nid)

    print(f"\n[Filtering Done]")
    print(f"  Unique consolidated identities: {len(identity_records)}")
    print(f"  Total graph nodes covered: {sum(len(d['node_ids']) for d in identity_records.values())}")

    # Build DataFrame
    records_list = list(identity_records.values())
    df_clean = pd.DataFrame([
        {
            "identity_id": i,
            "hv_name": d["hv_name"],
            "canonical_name": d["canonical_name"],
            "shamela_id": d["shamela_id"],
            "tier_label": d["tier_label"],
            "tier_class": d["tier_class"],
            "ibn_hajar_grade": d["ibn_hajar_grade"],
            "dhahabi_grade": d["dhahabi_grade"],
            "dhahabi_tier": d["dhahabi_tier"],
            "is_dual_consensus": int(d["is_dual_consensus"]),
            "earlier_critic_statements_count": d["earlier_critic_statements_count"],
            "death_year_ah": d["death_year_ah"],
            "tabaqa": d["tabaqa"],
            "graph_node_ids": ",".join(map(str, sorted(d["node_ids"]))),
        }
        for i, d in enumerate(records_list)
    ])

    print("\n--- Clean 3-Tier Distribution (Primary Ground Truth) ---")
    tier_counts = df_clean["tier_label"].value_counts()
    for t_name, cnt in tier_counts.items():
        print(f"  Tier {TIER_TO_INT[t_name]} ({t_name:<11}): {cnt:3d} ({cnt / len(df_clean):.1%})")

    consensus_counts = df_clean[df_clean["is_dual_consensus"] == 1]["tier_label"].value_counts()
    print(f"\n--- Dual-Synthesizer Consensus Subset (N={df_clean['is_dual_consensus'].sum()}) ---")
    for t_name, cnt in consensus_counts.items():
        print(f"  Tier {TIER_TO_INT[t_name]} ({t_name:<11}): {cnt:3d} ({cnt / df_clean['is_dual_consensus'].sum():.1%})")

    # 3. Assemble clean earlier-critic texts
    print("\nAssembling earlier-critic text files...")
    clean_texts = {}
    for d in records_list:
        snippets = []
        for s in d["earlier_statements"]:
            snippets.append(f"قال {s['scholar']}: {s['text']}")
        full_text = "\n".join(snippets)
        clean_texts[d["hv_name"]] = {
            "identity_id": int(df_clean.loc[df_clean["hv_name"] == d["hv_name"], "identity_id"].values[0]),
            "hv_name": d["hv_name"],
            "tier_label": str(d["tier_label"]),
            "tier_class": int(d["tier_class"]),
            "earlier_critic_text": full_text,
            "statements_count": int(len(d["earlier_statements"])),
        }

    # 4. Reconstruct Condensed Isnād Graph
    # Map raw node_id -> consolidated identity_id
    raw_node_to_identity = {}
    for _, row in df_clean.iterrows():
        ident_id = int(row["identity_id"])
        nids = list(map(int, row["graph_node_ids"].split(",")))
        for nid in nids:
            raw_node_to_identity[nid] = ident_id

    # Remap edges
    # In raw edges: source_id = student, target_id = teacher
    new_edges = []
    self_loops_dropped = 0
    external_edges = 0

    for _, r in edges_df.iterrows():
        s_raw = int(r["source_id"])
        t_raw = int(r["target_id"])
        s_new = raw_node_to_identity.get(s_raw)
        t_new = raw_node_to_identity.get(t_raw)

        # Both endpoints within our clean cohort
        if s_new is not None and t_new is not None:
            if s_new == t_new:
                self_loops_dropped += 1
                continue
            new_edges.append((s_new, t_new))
        else:
            external_edges += 1

    print(f"\n[Graph Edge Remapping]")
    print(f"  Raw edges: {len(edges_df):,}")
    print(f"  Edges between clean cohort identities: {len(new_edges):,}")
    print(f"  Self-loops eliminated: {self_loops_dropped}")

    # Compute node features on condensed graph
    n_nodes = len(df_clean)
    in_deg = np.zeros(n_nodes, dtype=np.float32)
    out_deg = np.zeros(n_nodes, dtype=np.float32)
    appearances = np.zeros(n_nodes, dtype=np.float32)

    for s, t in new_edges:
        out_deg[s] += 1  # Student cited teacher
        in_deg[t] += 1   # Teacher cited by student

    # Total appearances across all matched chains from raw nodes
    for ident_id, row in df_clean.iterrows():
        nids = list(map(int, row["graph_node_ids"].split(",")))
        app = sum(nodes_df.loc[nid, "total_appearances"] for nid in nids if nid in nodes_df.index)
        appearances[ident_id] = app

    # Build 5 pure topological features
    def log_scale(v):
        lv = np.log1p(v)
        mi, ma = lv.min(), lv.max()
        return np.zeros_like(lv) if ma - mi < 1e-8 else (lv - mi) / (ma - mi)

    x_appear = log_scale(appearances)
    x_in_deg = log_scale(in_deg)
    x_out_deg = log_scale(out_deg)
    x_net_flow = (out_deg - in_deg) / (out_deg + in_deg + 1.0)
    # Average position: mean normalized position from collector end
    avg_pos = np.zeros(n_nodes, dtype=np.float32)
    for ident_id, row in df_clean.iterrows():
        nids = list(map(int, row["graph_node_ids"].split(",")))
        positions = [nodes_df.loc[nid, "avg_chain_position"] for nid in nids if nid in nodes_df.index]
        avg_pos[ident_id] = np.mean(positions) if positions else 0.5
    mi_p, ma_p = avg_pos.min(), avg_pos.max()
    x_pos = (avg_pos - mi_p) / (ma_p - mi_p) if ma_p - mi_p > 1e-8 else np.zeros_like(avg_pos)

    x_topology = np.stack([x_appear, x_in_deg, x_out_deg, x_net_flow, x_pos], axis=1).astype(np.float32)

    # Build 3 biographical metadata features
    has_death = (df_clean["death_year_ah"].fillna("").astype(str).str.strip().str.len() > 0).astype(np.float32).values
    has_tabaqa = (df_clean["tabaqa"].fillna("").astype(str).str.strip().str.len() > 0).astype(np.float32).values
    tabaqa_val = df_clean["tabaqa"].apply(parse_tabaqa_float).values.astype(np.float32)

    x_metadata = np.stack([has_death, has_tabaqa, tabaqa_val], axis=1).astype(np.float32)
    x_combined_feats = np.concatenate([x_topology, x_metadata], axis=1).astype(np.float32)

    # Edge index: create undirected edges for bidirectional GCN message passing
    if new_edges:
        src_edges = np.array([e[0] for e in new_edges], dtype=np.int64)
        tgt_edges = np.array([e[1] for e in new_edges], dtype=np.int64)
        # Undirected: concatenate (src, tgt) with (tgt, src) and drop duplicates
        undir_src = np.concatenate([src_edges, tgt_edges])
        undir_tgt = np.concatenate([tgt_edges, src_edges])
        edge_pairs = set(zip(undir_src, undir_tgt))
        edge_index = np.array([[p[0] for p in edge_pairs], [p[1] for p in edge_pairs]], dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    y_labels = df_clean["tier_class"].values.astype(np.int64)

    print(f"\n[Condensed Graph Specifications]")
    print(f"  Nodes: {n_nodes} (Unique historical identities)")
    print(f"  Undirected transmission edges: {edge_index.shape[1]}")
    print(f"  Topology features (X_topo): {x_topology.shape} (appear, in_deg, out_deg, net_flow, chain_pos)")
    print(f"  Metadata features (X_meta): {x_metadata.shape} (has_death, has_tabaqa, tabaqa_num)")
    print(f"  Combined features (X_all):  {x_combined_feats.shape}")
    print(f"  Labels (Y):                 {y_labels.shape} (3 canonical tiers)")

    # 5. Save artifacts
    cohort_csv_out = OUT_DIR / "clean_narrators_cohort.csv"
    texts_json_out = OUT_DIR / "clean_texts.json"
    graph_npz_out = OUT_DIR / "clean_graph.npz"
    summary_json_out = OUT_DIR / "cohort_summary.json"

    df_clean.to_csv(cohort_csv_out, index=False, encoding="utf-8-sig")
    print(f"\nSaved clean cohort metadata: {cohort_csv_out}")

    with open(texts_json_out, "w", encoding="utf-8") as f:
        json.dump(clean_texts, f, indent=2, ensure_ascii=False)
    print(f"Saved clean earlier-critic texts: {texts_json_out}")

    np.savez(
        graph_npz_out,
        x_topo=x_topology,
        x_meta=x_metadata,
        x_all=x_combined_feats,
        edge_index=edge_index,
        y=y_labels,
        is_consensus=df_clean["is_dual_consensus"].values.astype(bool),
        identities=df_clean["hv_name"].values,
    )
    print(f"Saved condensed graph archive: {graph_npz_out}")

    summary = {
        "total_unique_identities": n_nodes,
        "class_distribution_3_tier": {
            "tier_0_below_saduq": int((y_labels == 0).sum()),
            "tier_1_saduq": int((y_labels == 1).sum()),
            "tier_2_thiqah": int((y_labels == 2).sum()),
        },
        "dual_consensus_count": int(df_clean["is_dual_consensus"].sum()),
        "dual_consensus_distribution": {
            "tier_0_below_saduq": int(((y_labels == 0) & df_clean["is_dual_consensus"]).sum()),
            "tier_1_saduq": int(((y_labels == 1) & df_clean["is_dual_consensus"]).sum()),
            "tier_2_thiqah": int(((y_labels == 2) & df_clean["is_dual_consensus"]).sum()),
        },
        "undirected_edges_count": int(edge_index.shape[1]),
        "features": {
            "topology_dim": int(x_topology.shape[1]),
            "metadata_dim": int(x_metadata.shape[1]),
            "combined_dim": int(x_combined_feats.shape[1]),
        },
    }
    with open(summary_json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved cohort summary metrics: {summary_json_out}")

    print("\nPhase 1 Data Preparation successfully completed!")


if __name__ == "__main__":
    main()
