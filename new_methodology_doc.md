# New Methodology: Jumhūr Sentiment-Based Narrator Scoring

## The Problem with the Current Approach

The current AraBERT 5-class classifier reads an entire biographical text and produces a single prediction. The model weighs all sentences equally. This violates the classical *Jarḥ wa Taʿdīl* principle where critics' statements are counted individually, and the majority opinion (*jumhūr*) determines the final ruling.

A single harsh critic (e.g., "كذاب يضع الحديث") can override ten scholars saying "ثقة" because the classifier treats the entire text as one bag of words.

## The New Approach: Sentence-Level Sentiment Aggregation

Replace the single-pass classification with a **two-stage pipeline**:

```
Bio text → Split by critic → Classify each sentence → Count praise vs criticism → Final grade
```

### Stage 1: Critic Statement Extraction

Split the raw biographical text into individual critic statements using "قال" patterns:

**Input:**
```
وَقَال أَحْمَد بْن عَبد اللَّهِ العجلي: ثقة.
وَقَال أبو الفتح الأزدي: متروك.
وَقَال ابن حجر: صدوق.
```

**Output sentences:**
```python
[
    {"critic": "أحمد بن عبد الله العجلي", "statement": "ثقة"},
    {"critic": "أبو الفتح الأزدي", "statement": "متروك"},
    {"critic": "ابن حجر", "statement": "صدوق"},
]
```

Split criteria:
- **قال X:** (he said) — identifies a new critic statement
- **وَقَال X:** (and he said) — continuation or new critic
- **عن X:** (from) — attribution
- Parenthetical citations like (١) are excluded

### Stage 2: Sentence-Level Classification

Each extracted statement is classified independently by the AraBERT model (the same 5-class model already trained):

| Statement | Predicted Class | Numerical Score |
|---|---|---|
| "ثقة" | thiqah | +1.0 |
| "متروك" | matruk | -1.0 |
| "صدوق" | saduq | +0.5 |

Mapping:
- **thiqah** → +1.0 (strong praise)
- **saduq** → +0.5 (moderate praise)
- **maqbul** → 0.0 (neutral)
- **daif** → -0.5 (moderate criticism)
- **matruk** → -1.0 (strong criticism)

### Stage 3: Jumhūr Aggregation

The final narrator score is computed by majority vote:

```python
def jumhur_score(statements: list[dict]) -> float:
    """
    Given a list of critic statements, return the consensus score.
    
    Rules:
    1. Count praise statements (+1 each) and criticism statements (-1 each).
    2. If all available critics agree → score = that class.
    3. If mixed → take the majority.
    4. If no statements available → default to maqbul (0.40).
    """
    net = sum(statement["score"] for statement in statements)
    n = len(statements)
    
    if n == 0:
        return 0.40  # Default maqbul
    
    average = net / n
    
    if average >= 0.5:
        return 0.85  # thiqah
    elif average >= 0.1:
        return 0.70  # saduq
    elif average >= -0.1:
        return 0.40  # maqbul
    elif average >= -0.5:
        return 0.15  # daif
    else:
        return 0.05  # matruk
```

### Stage 4: Integration with Existing Chain Pipeline

The narrator scores produced by jumhūr aggregation replace the current classifier outputs. Everything else stays the same:

```
Jumhūr scores → Alias index mapping → Chain scoring (weakest-link) → Risk modifiers → Final grade
```

## Key Improvements Over Current Method

| Aspect | Current (FYP1) | **New (Sentiment Jumhūr)** |
|---|---|---|
| Text processing | Whole bio → one prediction | **Each critic → one prediction** |
| Handling mixed opinions | Single critic can override | **Majority vote wins** |
| Hisham ibn Urwah | Matruk (0.05) | **Thiqah (0.85)** — 8 praises > 1 criticism |
| بن جريح | Matruk | **Saduq** — majority say صدوق |
| Missing bio text | GNN fallback | Same — no change needed |
| Interpretability | Black box | **Explainable**: "3 critics said thiqah, 1 said daif" |
| Alignment with rijāl science | Partial | **Exact match**: classical jumhūr principle |

## What This Changes in the Thesis

### Title
> *A Multi-Modal Framework for Quantifying Narrator Reliability in **Ḥadīth Narrator Criticism** Using **Semantic Sentiment Analysis** and Graph Neural Networks*

"Semantic Sentiment Analysis" becomes accurate — we're measuring sentiment (praise/criticism) of individual critic statements and aggregating them.

### Chapter 3: Methodology

**§3.4.1 Continuous Reliability Scoring** (renamed)
Change from regression description to:
- Grade term mapping (GRADE_SCORE_MAP)
- Sentence-level classification
- Jumhūr (majority) aggregation

**§3.4.2 Chain-Level Scoring**
Keep weakest-link as is. Add note that narrator scores now come from sentiment aggregation rather than single-pass classification.

### Chapter 4: Experimental Design

Add new experiment:
**Experiment 3a: Sentence-Level Sentiment Extraction**
- Split biographical text by critic markers
- Classify each statement individually
- Evaluate agreement with ground-truth grade
- Compare against single-pass classification

### Chapter 5: Implementation (FYP2)

The majority of this becomes the core FYP2 deliverable:
- 5.3.1: Critic statement extraction (قال patterns)
- 5.3.2: Sentence classification results
- 5.3.3: Jumhūr aggregation vs single-pass comparison
- 5.3.4: Explainability: case studies of mixed-opinion narrators

## Implementation Plan

| Step | Task | Est. Time | Status |
|---|---|---|---|
| 1 | Extract critic-statements from bio text using قال patterns | 1 day | ⏳ To do |
| 2 | Classify each statement with existing AraBERT model | 1 day | ⏳ To do |
| 3 | Implement jumhūr vote aggregation | 0.5 day | ⏳ To do |
| 4 | Compare against current single-pass results | 0.5 day | ⏳ To do |
| 5 | Integrate into chain scoring pipeline | 1 day | ⏳ To do |
| **Total** | | **~4 days** | |

## Comparison with Current FYP1 Pipeline

```
                    FYP1 (current)                    FYP2 (proposed)
                    ==============                    ================

Bio text ──────┐                              Bio text ──────┐
               │                                              │
               ▼                                              ▼
       AraBERT 5-class                            Split by "قال" markers
       (single pass)                                     │
               │                                          ▼
               ▼                                  Classify each sentence
       One prediction ────┐                     (AraBERT per sentence)
                           │                              │
                           ▼                              ▼
                   GRADE_SCORE_MAP                  Jumhūr vote count
                   (fallback)                              │
                           │                              ▼
                           ▼                          Majority score
                   GNN fallback                            │
                   (no bio text)                           │
                           │                              ▼
                           ▼                      GRADE_SCORE_MAP
                   Final narrator score            (fallback if no text)
                                                          │
                                                          ▼
                                                  GNN fallback
                                                  (no bio text)
                                                          │
                                                          ▼
                                                  Final narrator score
```

## Key Files (to be created)
- `scripts/sentiment_jumhur.py` — Main implementation  
- `scripts/jumhur_vs_classifier.py` — Comparison experiment
- `findings/22_jumhur_sentiment.md` — Results

## References for Thesis

- **Classical**: Ibn Ḥajar al-ʿAsqalānī, *Taqrīb al-Tahḏīb* — documents the jumhūr principle
- **Modern**: Ghriss (2025), ICML — "Crowdsourced Information Authentication: A Graph-based Model from the Science of Hadith" — formalizes majority aggregation
- **NLP**: CAMeL Tools (Obeid et al., 2020) — Arabic sentence segmentation
