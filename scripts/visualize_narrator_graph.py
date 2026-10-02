"""
Rich Narrator Knowledge Graph Visualization
============================================
Generates an interactive force-directed graph of the hadith narrator
network, styled by reliability predictions and transmission patterns.

Outputs:
  - Interactive HTML (pyvis) with zoom, drag, tooltips, search
  - Static high-res PNG (matplotlib)
  - Subgraph JSON for external tools

Usage:
  python scripts/visualize_narrator_graph.py                        # Default: top narrators
  python scripts/visualize_narrator_graph.py --center "أبو هريرة"   # Subgraph around a narrator
  python scripts/visualize_narrator_graph.py --n-narrators 50       # Limit to N narrators
  python scripts/visualize_narrator_graph.py --output mygraph.html  # Custom output path
"""
import sys, json, re, os
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# ── Config ──
OUTPUT_DIR = Path("output/stage2/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color map for reliability categories (matching the image style)
CATEGORY_COLORS = {
    "ثقة":       "#1b9e77",  # Green — trustworthy
    "صدوق":      "#377eb8",  # Blue — truthful
    "مقبول":     "#ff7f00",  # Orange — acceptable
    "ضعيف":      "#e41a1c",  # Red — weak
    "متروك":     "#984ea3",  # Purple — abandoned
    "unknown":   "#999999",  # Grey — no data
}

# Node size ranges
SIZE_RANGE = (8, 45)    # Min, max node size
LABEL_THRESHOLD = 18    # Only label nodes this size and above

# Sahabah (Companions) to use as default centers
KEY_SAHABAH = [
    "أبو هريرة", "عائشة", "ابن عباس", "عبد الله بن عمر",
    "أنس بن مالك", "جابر بن عبد الله", "أبو سعيد الخدري",
    "عبد الله بن مسعود", "علي بن أبي طالب", "عمر بن الخطاب",
    "أبو موسى الأشعري", "أبي بن كعب", "عثمان بن عفان",
    "أبو بكر الصديق", "أبو ذر الغفاري", "سعد بن أبي وقاص",
    "عبد الله بن عمرو بن العاص", "زيد بن ثابت",
    "أبو أيوب الأنصاري", "معاذ بن جبل",
]


def load_all():
    """Load graph data, predictions, and metadata."""
    print("Loading graph data...")

    data = {}

    # Graph
    g = np.load("output/stage2/pyg_graph_large_graded.npz")
    data["x"] = g["x"]           # (42021, 8) node features
    data["edge_index"] = g["edge_index"]  # (2, 147329)
    data["edge_attr"] = g.get("edge_attr", None)

    # Node names
    with open("output/stage2/node_names.json", encoding="utf-8") as f:
        id_to_name = json.load(f)
    data["id_to_name"] = id_to_name
    data["name_to_id"] = {}
    for nid_s, name in id_to_name.items():
        nid = int(nid_s) - 1  # 1-based → 0-based
        clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
        data["name_to_id"][clean] = nid

    # Predictions
    pred_path = "output/stage3/narrator_predictions.csv"
    if Path(pred_path).exists():
        preds = pd.read_csv(pred_path)
        # Map node_id (1-based) → predicted_score
        data["pred_map"] = {}
        for _, row in preds.iterrows():
            data["pred_map"][int(row["node_id"]) - 1] = float(row["predicted_score"])
    else:
        # Fallback: use col 0 from features (ground truth grade)
        data["pred_map"] = {i: data["x"][i, 0] for i in range(len(data["x"]))}

    print(f"  Nodes: {len(data['x']):,}")
    print(f"  Edges: {data['edge_index'].shape[1]:,}")
    print(f"  Named: {len(id_to_name):,}")
    print(f"  Predictions: {len(data['pred_map']):,}")

    return data


def score_to_category(score):
    if score >= 0.85: return "ثقة"
    elif score >= 0.60: return "صدوق"
    elif score >= 0.35: return "مقبول"
    elif score >= 0.15: return "ضعيف"
    else: return "متروك"


def build_nx_graph(data, center_names=None, max_nodes=150, hops=3):
    """Build a NetworkX graph, optionally centered on specific narrators.

    If center_names is None: takes the top max_nodes narrators by degree.
    Otherwise: BFS from center narrators up to `hops` steps.
    """
    x = data["x"]
    ei = data["edge_index"]
    ea = data["edge_attr"]

    # First pass: compute all degrees
    degrees = Counter()
    edge_list = []  # (src, dst, method, position, freq)
    for j in range(ei.shape[1]):
        src = int(ei[0, j]) - 1  # 1-based → 0-based
        dst = int(ei[1, j]) - 1
        if 0 <= src < len(x) and 0 <= dst < len(x):
            degrees[src] += 1
            degrees[dst] += 1
            method = ea[j, 0] if ea is not None else 0
            position = ea[j, 1] if ea is not None else 0
            freq = ea[j, 2] if ea is not None else 1
            edge_list.append((src, dst, method, position, freq))

    # Select nodes
    if center_names:
        # BFS from center narrators
        center_ids = set()
        for name in center_names:
            name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
            if name_clean in data["name_to_id"]:
                center_ids.add(data["name_to_id"][name_clean])

        if not center_ids:
            print(f"  WARNING: None of {center_names} found in graph. Falling back to top-degree.")
            center_ids = set()

        if center_ids:
            # BFS from centers
            visited = set(center_ids)
            frontier = set(center_ids)
            for _ in range(hops):
                next_frontier = set()
                for nid in frontier:
                    for src, dst, *_ in edge_list:
                        if src == nid and dst not in visited:
                            next_frontier.add(dst)
                            visited.add(dst)
                        elif dst == nid and src not in visited:
                            next_frontier.add(src)
                            visited.add(src)
                frontier = next_frontier
                if len(visited) >= max_nodes:
                    break
            selected = set(list(visited)[:max_nodes])
            print(f"  BFS from {len(center_ids)} centers → {len(selected)} nodes (hops={hops})")
        else:
            selected = None
    else:
        selected = None

    # Fallback: top-degree nodes
    if selected is None:
        top_nodes = [n for n, _ in degrees.most_common(max_nodes)]
        selected = set(top_nodes)
        print(f"  Top {max_nodes} nodes by degree")

    # Find node with max degree for normalization
    max_degree = max(degrees.values()) if degrees else 1

    # Ensure all values are native Python types (not numpy)
    def py_val(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    # Build NX graph
    G = nx.DiGraph()
    for nid in selected:
        name = data["id_to_name"].get(str(nid + 1), f"N{nid}")
        name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name) if name else f"N{nid}"
        deg = int(degrees.get(nid, 0))
        pred = float(data["pred_map"].get(nid, 0.3))
        cat = score_to_category(pred)
        grade_score = float(x[nid, 0]) if nid < len(x) else 0.3

        G.add_node(nid,
                   label=name_clean,
                   title=name,
                   degree=deg,
                   predicted_score=pred,
                   category=cat,
                   grade_score=grade_score,
                   color=CATEGORY_COLORS.get(cat, CATEGORY_COLORS["unknown"]),
                   size=float(SIZE_RANGE[0] + (SIZE_RANGE[1] - SIZE_RANGE[0]) * (deg / max_degree)),
                   border_color="#333333" if pred >= 0.35 else "#e41a1c",
        )

    # Add edges between selected nodes
    for src, dst, method, position, freq in edge_list:
        if src in selected and dst in selected:
            m = float(method)
            p = float(position)
            f = float(freq)
            method_label = "مباشر" if m < 0.3 else ("محتمل" if m < 0.7 else "غير معروف")
            edge_color = "#2ca02c" if m < 0.3 else ("#ff7f00" if m < 0.7 else "#d62728")
            G.add_edge(src, dst,
                       method=m,
                       position=p,
                       frequency=f,
                       title=f"{method_label} | pos={p:.2f} | freq={f:.0f}",
                       color=edge_color,
                       width=float(max(0.5, min(5, np.log1p(f) * 0.8))),
                       arrows="to",
            )

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def export_pyvis(G, output_path, physics=True):
    """Export as interactive HTML using pyvis."""
    from pyvis.network import Network

    net = Network(
        height="900px", width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
    )

    # Configure physics
    if physics:
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.015,
              "springLength": 120,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.2
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 300,
              "updateInterval": 25
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "vertical"
            },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true
          },
          "nodes": {
            "font": { "size": 11, "face": "Tahoma", "strokeWidth": 2, "strokeColor": "#ffffff" }
          }
        }
        """)

    for nid, ndata in G.nodes(data=True):
        score = ndata.get("predicted_score", 0)
        label = ndata.get("label", str(nid))[:25]
        title = (
            f"<b>{ndata.get('title', label)}</b><br>"
            f"Reliability: {score:.3f} ({ndata.get('category', '?')})<br>"
            f"Appearances: {ndata.get('degree', 0):,}<br>"
            f"Grade Score: {ndata.get('grade_score', 0):.2f}"
        )
        net.add_node(
            nid,
            label=label if ndata.get("size", 10) >= LABEL_THRESHOLD else "",
            title=title,
            color=ndata.get("color", "#999"),
            size=ndata.get("size", 10),
            borderWidth=2 if ndata.get("predicted_score", 0) >= 0.60 else 1,
            borderWidthSelected=4,
        )

    for src, dst, edata in G.edges(data=True):
        net.add_edge(
            src, dst,
            title=edata.get("title", ""),
            color=edata.get("color", "#555"),
            width=edata.get("width", 1),
            arrows="to",
        )

    net.write_html(str(output_path))
    print(f"  Interactive HTML: {output_path}")


def export_matplotlib(G, output_path):
    """Export as high-res static PNG via matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(24, 18), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Spring layout for nice spacing
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Draw edges colored by transmission method
    edge_colors = [edata.get("color", "#555") for _, _, edata in G.edges(data=True)]
    edge_widths = [edata.get("width", 0.5) for _, _, edata in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.35, arrows=True,
                           arrowsize=6, arrowstyle='->')

    # Draw nodes colored by reliability
    node_colors = [ndata.get("color", "#999") for _, ndata in G.nodes(data=True)]
    node_sizes = [ndata.get("size", 10) * 8 for _, ndata in G.nodes(data=True)]

    # Only label large nodes
    labels = {}
    for nid, ndata in G.nodes(data=True):
        if ndata.get("size", 10) >= LABEL_THRESHOLD:
            labels[nid] = ndata.get("label", str(nid))[:20]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85,
                           edgecolors="#ffffff", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                            font_size=6, font_color='white',
                            font_family='Tahoma')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CATEGORY_COLORS["ثقة"],
               markersize=12, label='ثقة (Thiqah — Trustworthy)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CATEGORY_COLORS["صدوق"],
               markersize=12, label='صدوق (Ṣadūq — Truthful)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CATEGORY_COLORS["مقبول"],
               markersize=12, label='مقبول (Maqbūl — Acceptable)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CATEGORY_COLORS["ضعيف"],
               markersize=12, label='ضعيف (Ḍaʿīf — Weak)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CATEGORY_COLORS["متروك"],
               markersize=12, label='متروك (Matrūk — Abandoned)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              facecolor='#2a2a3e', edgecolor='#555', labelcolor='white')

    ax.set_title("Hadith Narrator Knowledge Graph\n"
                 "Node: Reliability Score | Size: Narrator Appearances | "
                 "Edge Color: Transmission Method",
                 color='white', fontsize=14, pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Static PNG: {output_path}")


def export_gephi(G, output_path):
    """Export as GEXF for Gephi."""
    nx.write_gexf(G, str(output_path))
    print(f"  Gephi GEXF: {output_path}")


# ── Main ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Narrator Knowledge Graph Visualization")
    parser.add_argument("--center", type=str, nargs="*",
                        help="Center graph around specific narrator name(s)")
    parser.add_argument("--n-narrators", type=int, default=150,
                        help="Max number of narrators to include")
    parser.add_argument("--hops", type=int, default=3,
                        help="BFS hops from center narrators")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename prefix (default: auto-generated)")
    parser.add_argument("--no-png", action="store_true", help="Skip static PNG export")
    parser.add_argument("--no-html", action="store_true", help="Skip interactive HTML export")
    parser.add_argument("--gephi", action="store_true", help="Also export GEXF for Gephi")
    args = parser.parse_args()

    print("\n╔════════════════════════════════════════════════════╗")
    print("║   Narrator Knowledge Graph Visualization          ║")
    print("╚════════════════════════════════════════════════════╝\n")

    data = load_all()

    # Determine center narrators
    center = args.center if args.center else KEY_SAHABAH[:8]  # Default: top 8 Sahaba

    # Filename prefix
    if args.output:
        prefix = args.output
    else:
        if args.center:
            prefix = f"narrator_graph_{'_'.join(args.center[0].split()[:2])}"
        else:
            prefix = "narrator_graph_top"

    # Build graph
    print(f"\nBuilding graph (center={'auto' if not args.center else center}, "
          f"max_nodes={args.n_narrators}, hops={args.hops})...")
    G = build_nx_graph(data, center_names=center, max_nodes=args.n_narrators, hops=args.hops)

    # Category distribution
    cat_counts = Counter(ndata["category"] for _, ndata in G.nodes(data=True))
    print(f"\nCategory distribution:")
    for cat in ["ثقة", "صدوق", "مقبول", "ضعيف", "متروك"]:
        n = cat_counts.get(cat, 0)
        bar = "█" * (n // 2)
        print(f"  {cat:<10s}: {n:>4d}  {bar}")

    # Exports
    print(f"\nExporting...")
    if not args.no_html:
        html_path = OUTPUT_DIR / f"{prefix}.html"
        export_pyvis(G, html_path)

    if not args.no_png:
        png_path = OUTPUT_DIR / f"{prefix}.png"
        export_matplotlib(G, png_path)

    if args.gephi:
        gexf_path = OUTPUT_DIR / f"{prefix}.gexf"
        export_gephi(G, gexf_path)

    print(f"\nDone! Outputs in: {OUTPUT_DIR}")
