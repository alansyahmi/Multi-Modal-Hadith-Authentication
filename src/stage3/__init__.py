"""
Stage 3: Probabilistic Trust Scoring Using Graph Neural Networks.

Trains a GCN to predict narrator reliability scores from:
  - Graph structure (transmission chains)
  - Node features (Ibn Hajar grades, chain position, degree)
  - Semantic sentiment embeddings from matn text (optional)

Outputs per-narrator reliability scores for the full eval set.
"""
