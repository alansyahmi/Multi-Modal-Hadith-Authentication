import networkx as nx
import matplotlib.pyplot as plt

def draw_networkx_pipeline():
    G = nx.DiGraph()

    # Define the nodes (Phases)
    nodes = [
        ("A1", "Acquire Texts"), ("A2", "Preprocessing"), ("A3", "Segmentation"),
        ("B1", "AraBERT NER"), ("B2", "Extract Mentions"), ("B3", "Knowledge Graph"), ("B4", "Disambiguation"),
        ("C1", "Map Labels"), ("C2", "Aggregate"), ("C3", "Chain Score"), ("C4", "Grade"), ("C5", "Evaluate")
    ]
    G.add_nodes_from([n[0] for n in nodes])
    
    # Define Edges
    edges = [
        ("A1", "A2"), ("A2", "A3"), ("A3", "B1"),
        ("B1", "B2"), ("B2", "B3"), ("B3", "B4"), ("B4", "C1"),
        ("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5")
    ]
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)  # or nx.shell_layout for circular
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
    
    # Custom labels to show full names
    labels = {n[0]: n[1] for n in nodes}
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    plt.title("Hadith Analysis Pipeline (NetworkX)")
    plt.show()

draw_networkx_pipeline()