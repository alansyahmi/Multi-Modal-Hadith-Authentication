"""
Knowledge Graph Builder — constructs a directed multigraph of hadith narrators.

Nodes: Canonical narrators with attributes (grade, tabaqa, death year, stats)
Edges: Transmission relationships with attributes (method, book, position)

Process:
1. Load aligned Sanadset chains (1,030 matched hadiths)
2. Parse chains → (narrator, position) tuples
3. Build edge list: consecutive narrator pairs in each chain
4. Aggregate duplicate edges → frequency weight
5. Attach narrator bio data from grades JSON
6. Compute node statistics (degree, frequency, avg position)

Output: node table, edge table, adjacency representation
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    ALIGNMENT_RESULTS,
    SANADSET_CLEANED,
    NARRATOR_GRADES,
    GRADE_SCORE_MAP,
    DEFAULT_GRADE_SCORE,
    METHOD_ENCODING,
    MAX_CHAIN_LENGTH,
    GRAPH_NODES_CSV,
    GRAPH_EDGES_CSV,
    GRAPH_METADATA_JSON,
)


@dataclass
class NarratorNode:
    """A narrator vertex in the knowledge graph."""
    node_id: int
    canonical_name: str                 # Clean normalized name
    grade: str = ""                      # Ibn Hajar grade label
    grade_score: float = DEFAULT_GRADE_SCORE
    tabaqa: str = ""                     # Generation/era
    death_info: str = ""                 # Death year text
    total_appearances: int = 0           # In how many chains
    in_degree: int = 0                   # Times appearing as target (student)
    out_degree: int = 0                  # Times appearing as source (teacher)
    avg_chain_position: float = 0.0      # Mean position across all chains
    book_set: set[int] = field(default_factory=set)  # Unique books appeared in


@dataclass
class TransmissionEdge:
    """A transmission edge in the knowledge graph."""
    source_id: int
    target_id: int
    method: str = "unknown"
    method_code: int = 2
    book_name: str = ""
    position: int = 0                    # Position from collector end
    frequency: int = 1                   # Aggregated frequency


class KnowledgeGraph:
    """Directed multigraph of hadith narrator transmissions."""

    def __init__(self) -> None:
        self.nodes: dict[str, NarratorNode] = {}  # canonical_name → node
        self.edges: list[TransmissionEdge] = []
        self._node_id_counter: int = 0
        self._name_to_id: dict[str, int] = {}

    def add_narrator(self, name: str) -> int:
        """Get or create a node ID for a narrator name."""
        if name not in self._name_to_id:
            self._node_id_counter += 1
            nid = self._node_id_counter
            self._name_to_id[name] = nid
            self.nodes[name] = NarratorNode(node_id=nid, canonical_name=name)
        return self._name_to_id[name]

    def add_edge(
        self,
        source_name: str,
        target_name: str,
        method: str = "unknown",
        book_name: str = "",
        position: int = 0,
    ) -> None:
        """Add a transmission edge (aggregated later)."""
        src_id = self.add_narrator(source_name)
        tgt_id = self.add_narrator(target_name)
        self.edges.append(TransmissionEdge(
            source_id=src_id,
            target_id=tgt_id,
            method=method,
            method_code=METHOD_ENCODING.get(method, 2),
            book_name=book_name,
            position=position,
        ))

    def aggregate_edges(self) -> None:
        """Aggregate duplicate edges → frequency-weighted."""
        buckets: dict[tuple[int, int, int], TransmissionEdge] = {}
        for e in self.edges:
            key = (e.source_id, e.target_id, e.method_code)
            if key in buckets:
                buckets[key].frequency += 1
            else:
                buckets[key] = e
        self.edges = list(buckets.values())
        print(f"  Aggregated {len(buckets):,} unique edges")

    def compute_node_stats(self) -> None:
        """Compute in/out degree, frequency, and avg position."""
        in_deg: dict[int, int] = defaultdict(int)
        out_deg: dict[int, int] = defaultdict(int)
        appearances: dict[int, int] = defaultdict(int)
        pos_sum: dict[int, float] = defaultdict(float)
        pos_count: dict[int, int] = defaultdict(int)

        for e in self.edges:
            freq = e.frequency
            out_deg[e.source_id] += freq
            in_deg[e.target_id] += freq
            appearances[e.source_id] += freq
            appearances[e.target_id] += freq
            pos_sum[e.target_id] += e.position * freq
            pos_count[e.target_id] += freq

        for name, node in self.nodes.items():
            nid = node.node_id
            node.in_degree = in_deg[nid]
            node.out_degree = out_deg[nid]
            node.total_appearances = appearances[nid]
            if pos_count[nid] > 0:
                node.avg_chain_position = pos_sum[nid] / pos_count[nid]

    def attach_bio_data(self, grades_json: Path) -> int:
        """Attach Ibn Hajar grades and bio data to matching nodes."""
        with open(grades_json, encoding="utf-8") as f:
            grades_data = json.load(f)

        from rapidfuzz.fuzz import partial_ratio
        from src.preprocessing.arabic_normalizer import normalise_hadith_text

        attached = 0
        for name, node in self.nodes.items():
            name_norm = normalise_hadith_text(name, for_matching=True)

            # Try exact match first
            if name in grades_data:
                gd = grades_data[name]
            elif name_norm in grades_data:
                gd = grades_data[name_norm]
            else:
                # Fuzzy match
                best = None
                best_score = 0
                for gname, gd in grades_data.items():
                    gname_norm = normalise_hadith_text(gname, for_matching=True)
                    score = partial_ratio(name_norm, gname_norm)
                    if score > best_score and score >= 85:
                        best_score = score
                        best = gd
                gd = best

            if gd and gd.get("grade"):
                node.grade = gd["grade"]
                node.grade_score = GRADE_SCORE_MAP.get(gd["grade"], DEFAULT_GRADE_SCORE)
                node.tabaqa = gd.get("tabaqa", "")
                node.death_info = gd.get("death_info", "")
                attached += 1

        print(f"  Attached bio data to {attached}/{len(self.nodes)} nodes")
        return attached

    def export_tables(self, nodes_path: Path, edges_path: Path) -> None:
        """Export node and edge tables as CSV."""
        # Nodes
        node_rows = []
        for name, node in self.nodes.items():
            node_rows.append({
                "node_id": node.node_id,
                "canonical_name": node.canonical_name,
                "grade": node.grade,
                "grade_score": round(node.grade_score, 3),
                "tabaqa": node.tabaqa,
                "death_info": node.death_info,
                "total_appearances": node.total_appearances,
                "in_degree": node.in_degree,
                "out_degree": node.out_degree,
                "avg_chain_position": round(node.avg_chain_position, 2),
            })
        pd.DataFrame(node_rows).to_csv(nodes_path, index=False, encoding="utf-8")
        print(f"  Nodes: {nodes_path} ({len(node_rows):,} rows)")

        # Edges
        edge_rows = []
        for e in self.edges:
            edge_rows.append({
                "source_id": e.source_id,
                "target_id": e.target_id,
                "method": e.method,
                "method_code": e.method_code,
                "position": e.position,
                "frequency": e.frequency,
            })
        pd.DataFrame(edge_rows).to_csv(edges_path, index=False, encoding="utf-8")
        print(f"  Edges: {edges_path} ({len(edge_rows):,} rows)")

    def get_stats(self) -> dict:
        """Return summary statistics."""
        grades = [n.grade_score for n in self.nodes.values()]
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "avg_grade_score": round(sum(grades) / len(grades), 3) if grades else 0,
            "nodes_with_grade": sum(1 for n in self.nodes.values() if n.grade),
            "nodes_with_death": sum(1 for n in self.nodes.values() if n.death_info),
            "max_in_degree": max((n.in_degree for n in self.nodes.values()), default=0),
            "max_out_degree": max((n.out_degree for n in self.nodes.values()), default=0),
            "avg_chain_position": round(
                sum(n.avg_chain_position for n in self.nodes.values()) / max(len(self.nodes), 1), 2
            ),
        }


def build_knowledge_graph(
    alignment_path: Path = ALIGNMENT_RESULTS,
    sanadset_path: Path = SANADSET_CLEANED,
    grades_path: Path = NARRATOR_GRADES,
) -> KnowledgeGraph:
    """
    Build the full knowledge graph from Stage 1 aligned data.

    Returns a KnowledgeGraph with nodes, edges, and bio enrichment.
    """
    print("=" * 50)
    print("BUILDING NARRATOR KNOWLEDGE GRAPH")
    print("=" * 50)

    # Load Stage 1 data
    with open(alignment_path, encoding="utf-8") as f:
        alignment = json.load(f)

    # Load Sanadset with appropriate nrows
    sanadset = pd.read_csv(sanadset_path)

    matched = [a for a in alignment if a.get("is_match")]
    print(f"\n  Matched hadiths: {len(matched):,}")

    # Build graph
    kg = KnowledgeGraph()

    for a in matched:
        cidx = a["corpus_idx"]
        if cidx is None:
            continue

        # Get chain
        try:
            row = sanadset.loc[cidx]
        except KeyError:
            continue

        try:
            chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
        except (ValueError, SyntaxError):
            continue

        if len(chain) < 2:
            continue

        book = str(row.get("Book", ""))
        chain = chain[:MAX_CHAIN_LENGTH]  # Truncate

        # Add edges for consecutive narrator pairs
        # Chain flows: collector → ... → Prophet
        # Position 0 = collector end (last narrator in chain)
        for i in range(len(chain) - 1):
            source = chain[i].strip()      # Earlier narrator (closer to Prophet)
            target = chain[i + 1].strip()  # Later narrator (student)
            position = i                    # 0-indexed from collector end
            method = "ambiguous"            # Default; can be refined with sanad parser

            if source and target and len(source) > 2 and len(target) > 2:
                kg.add_edge(source, target, method=method, book_name=book, position=position)

    print(f"  Raw edges: {len(kg.edges):,}")
    print(f"  Unique narrators: {len(kg.nodes):,}")

    # Aggregate
    kg.aggregate_edges()

    # Compute stats
    kg.compute_node_stats()

    # Attach bio
    if grades_path.exists():
        kg.attach_bio_data(grades_path)

    # Export
    kg.export_tables(GRAPH_NODES_CSV, GRAPH_EDGES_CSV)

    # Summary
    stats = kg.get_stats()
    print(f"\n=== Graph Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save metadata
    with open(GRAPH_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return kg
