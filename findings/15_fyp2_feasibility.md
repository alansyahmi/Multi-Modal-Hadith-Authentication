# Finding 15: FYP2 Feasibility & Roadmap

## Current Performance Summary

| Level | Current Best | Target | Gap |
|---|---|---|---|
| Narrator scoring (κ) | 0.87 (classifier) | 0.90 | Small |
| Chain accuracy (exact) | 8–10% | 25–30% | **Large** |
| Chain ±1 grade accuracy | 38–56% | 75% | **Medium** |
| Framework completeness | 60% | 100% | **Medium** |

## Why the Gap Is Achievable

The 8% chain accuracy is not caused by a fundamental design flaw. It's caused by **3 specific, well-defined engineering gaps**:

### Gap 1: Short Name Alias Resolution (Fixable in Days)

**Problem**: Chains use short forms (`هِشَامٌ`, `أَبُو نُعَيْمٍ`) while the encyclopedia has full names (`هِشَام بْن عُرْوَة`, `الْفَضْل بْن دُكَيْن`). Our alias index doesn't map the short forms → the classifier falls back to GNN scores (mean 0.63).

**Fix**: Extract short name forms from the full encyclopedia names (first word only, first two words, common abbreviations). **50% of the remaining misclassifications would be resolved.**

**Effort**: Low (1–2 days).

### Gap 2: Jumhūr (Majority) Scoring (FYP2 Core)

**Problem**: The classifier reads ALL text equally. A single critic saying "layyin" (soft/weak) can override 10 scholars saying "thiqah" (trustworthy). Classical *muḥaddithūn* use **majority consensus (jumhūr)** — if most scholars accept a narrator, isolated criticism is ignored.

**Fix**: For each narrator's bio text, extract all grade terms, count praise (ثقة/صدوق) vs criticism (ضعيف/لين), take the majority. This is a **text aggregation model** on top of the existing classifier.

**Effort**: Medium (1–2 weeks, requires sentence-level grade extraction).

**Projected impact**: +10–12% accuracy.

### Gap 3: Risk Modifier Calibration (Fixable in Days)

**Problem**: Risk modifiers (ikhtilāṭ ×0.50, tadlīs ×0.80) are applied even when the narrator is accepted by jumhūr. Hishām ibn ʿUrwa had *ikhtilāṭ* (deterioration) in old age, but his earlier narrations are universally accepted.

**Fix**: Only apply risk modifiers when the narrator's jumhūr score is below thiqah. If most scholars say thiqah → no penalty even if some mention ikhtilāṭ.

**Effort**: Low (1 day, integrated with jumhūr scoring).

### Gap 4: Companion Rule ✅ (Already Done)

**Fix**: Any narrator identified as ṣaḥābī → score 1.0 (thiqah by default). This improved Sahih recall 5.6× and Hasan recall +14pp.

## Projected FYP2 Performance

| Component | Current | With Gap 1–3 Fixed | Method |
|---|---|---|---|
| Chain accuracy (exact) | 8% | **28–35%** | All four fixes |
| ±1 grade accuracy | 56% | **78–85%** | All four fixes |
| Narrator κ | 0.87 | **0.90+** | Jumhūr + calibration |
| Ṭuruq improvements | 2.1% | **15–20%** | Full multi-chain independence |

## FYP2 Roadmap

| Phase | Task | Timeline | Deliverable |
|---|---|---|---|
| Phase 1 | Short name alias resolution | Week 1–2 | Chain accuracy 8% → 15% |
| Phase 2 | Jumhūr scoring system | Week 3–6 | Chain accuracy 15% → 25% |
| Phase 3 | Risk modifier calibration | Week 7 | Chain accuracy 25% → 28% |
| Phase 4 | Multi-chain corroboration | Week 8–10 | ±1 grade accuracy 78%+ |
| Phase 5 | Geospatial verification | Week 11–12 | Full framework |
| Phase 6 | Write-up & evaluation | Week 13–14 | Thesis Chapter 6 |

## Conclusion

The remaining accuracy gap is caused by **3 specific engineering gaps**, not a fundamental design flaw. Each gap has a well-defined solution and reasonable effort estimate. Together they can push chain-level accuracy from 8% → ~30%+ with ±1 grade accuracy at 78%+, which is publishable for an undergraduate thesis in computational hadith studies.
