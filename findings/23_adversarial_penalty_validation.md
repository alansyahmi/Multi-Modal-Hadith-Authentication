---
name: adversarial-penalty-validation
description: Validated that geo/temporal penalties work but GNN path penalty is a no-op
metadata:
  type: finding
---

# Adversarial Penalty Validation

Generated adversarial hadith chains with injected anomalies to validate penalty mechanisms for Table 5.9.

## Results

| Mechanism | Chains | Clean Score | Anomaly Score | Penalty | Status |
|-----------|:-----:|:-----------:|:-------------:|:-------:|:------:|
| **Geospatial** (injected >3,000km gaps) | 20 | 0.4519 | 0.4021 | **15.7%** | ✅ Works |
| **Temporal** (cross-era chain splice) | 30 | 0.4706 | 0.3823 | **18.8%** | ✅ Works |
| **GNN Path** (inserted random narrator) | 20 | 0.4414 | 0.4433 | **0.0%** | ❌ No-op |

## Key Finding

The GNN path compatibility penalty (threshold < 0.25) **cannot fire** on the current embeddings. All 262,497 node embeddings produce sigmoid-normalized similarities between 0.50–0.60 when paired randomly. Even deliberately inserting a random, unrelated narrator into a clean chain yields similarities > 0.25.

This means the GNN penalty mechanism in the evaluation pipeline is effectively dead code with the current GAE model. The 0.25 threshold was set too aggressively for the learned embedding space.

## Implications

- The ablation study's identical results across all configs are explained: only geo/temp penalties exist, and they don't fire on clean Kutub as-Sittah chains
- If the GNN penalty were removed, results would be unchanged
- The temporal penalty (inqita') also never fires on canonical data — classical compilers already verified generation gaps

## Script

`scripts/generate_adversarial_set.py` — run with `python scripts/generate_adversarial_set.py`
