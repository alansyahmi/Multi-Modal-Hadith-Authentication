# Finding 10: Recommendations & Next Steps

## Summary of Current State

| Component | Status | Key Metric |
|---|---|---|
| Encyclopedia | ✅ Complete | 15,448 entries, 10,920 with grades |
| Alias Index | ✅ Complete | 24,545 forms, 84% KG coverage |
| Family Graph | ✅ Complete | 13,592 father-son pairs |
| Track A (AraBERT) | ✅ Trained | Spearman ρ = 0.85 |
| GNN (Full KG) | ✅ Trained | R² = +0.014, 262K narrators |
| Chain Score (Mean) | ❌ Invalid | 99.9% "Uncertain" |
| **Chain Score (Weakest-Link)** | **✅ Best** | **15.9% Daif, meaningful variance** |
| Multi-Sanad Convergence | ✅ Complete | 81.6% with std ≤ 0.02 |
| Consensus Tier | ❌ Not Differentiated | GNN range too narrow |

## Recommended Thesis Experiments (Priority Order)

### 1. AraBERT Scoring for All Narrators (Track A Completion)
**Current issue**: AraBERT covers only ~6K narrators with bio text.
**Fix**: Score all 262K KG narrators using the trained AraBERT model. Use the alias index to map each KG name to bio text. For narrators without bio, fall back to GNN.
**Expected impact**: Wider score range → more Hasan/Sahih chains in weakest-link evaluation.

### 2. 5-Class Classification (Replace Regression)
**Current issue**: Regression head (MSE loss) compresses predictions toward the mean (pred std = 0.07 vs true std = 0.30).
**Fix**: Replace the 1-output regression head with 5-class classification using ordinal weighted cross-entropy loss.
**Expected impact**: Better separation between grade categories → more variance in narrator scores.

### 3. Edge Risk Modifiers (§3.4.2.2)
**Current issue**: The thesis describes tadlīs (0.80), ikhtilāṭ (0.50) modifiers but they're not implemented.
**Fix**: Extract tadlīs/ikhtilāṭ indicators from biographical text and apply as multiplicative modifiers to narrator scores.
**Expected impact**: More fine-grained chain scoring reflecting known risk factors.

### 4. Calibrated Thresholds for Weakest-Link
**Current issue**: The GNN's max score is ~0.70, so no chain reaches "Sahih" (0.85) threshold.
**Fix**: Re-calibrate grade thresholds to match the GNN's output distribution using isotonic regression on the Kutub as-Sitta evaluation set.
**Expected impact**: Meaningful grade distribution across all 5 categories.

### 5. Re-run Consensus Tier with Weakest-Link
**Current issue**: Consensus analysis with mean-based scoring shows no differentiation.
**Fix**: Re-run `sahih_vs_mutawatir_experiment.py` with weakest-link chain scores.
**Expected impact**: Ṣaḥīḥ al-Bukhārī chains should score higher than Sunan Ibn Mājah chains.

## Data Pipeline Improvements

### 6. Improve Tahdhib Parser Coverage
**Current issue**: Only 7,892/10,000+ entries extracted from 35 raw volumes.
**Fix**: Replace line-based entry detection with section-aware parsing using the Shamela page structure.
**Expected impact**: ~2,000+ more biographical entries → more AraBERT training data.

### 7. Shamela Narrator Database Integration
**Current issue**: The user found narrator ID 510 on Shamela that's not in our 2 books.
**Fix**: Scrape Shamela's narrator database as a third supplementary source.
**Expected impact**: Additional biographical entries for previously unknown narrators.

## Model Architecture Improvements

### 8. GNN with Edge Risk Features
**Current issue**: GNN uses 8 node features but no edge features for tadlīs/ikhtilāṭ.
**Fix**: Create a GAT (Graph Attention Network) variant that attends over edge features.
**Expected impact**: Better propagation of reliability signals through the graph.

### 9. Multi-Modal Fusion
**Current issue**: Track A (text) and GNN (graph) are trained independently.
**Fix**: Joint training: GNN node features include AraBERT embeddings as input, trained end-to-end.
**Expected impact**: Combined text + graph signal outperforms either alone.

## Quick Wins (1-2 hours each)

| # | Task | Est. Time | Impact |
|---|---|---|---|
| 1 | Weakest-link + AraBERT hybrid scoring | 1 hr | Wider score range |
| 4 | Calibrate thresholds to GNN distribution | 30 min | Meaningful grade distribution |
| 5 | Re-run consensus with weakest-link | 30 min | Thesis-validating result |
