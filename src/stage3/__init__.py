"""
Stage 3: Probabilistic Trust Scoring Using Graph Neural Networks & Semantic Fusion.

Predicts narrator reliability scores by fusing:
  - Graph transmission topology (directed isnād chains, degree, flow, tabaqa)
  - Zero-leakage topological node features (8 clean graph features)
  - Semantic AraBERT reliability embeddings from classical rijāl critic statements (PO1)

Outputs calibrated per-narrator reliability classifications and scores.
"""

