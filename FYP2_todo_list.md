Assuming your objectives stay fixed, I would **not touch the title or objectives unless your supervisor explicitly agrees**.

Instead, I'd focus on maximizing marks in the areas the examiners already identified.

# Priority 1 (Highest Impact)

## Fix the Research Design chapter

This directly addresses Lai's comment.

Right now, I suspect your methodology reads like:

> Here's the pipeline → here's the experiments.

FYP2 should read:

> Research Question → Hypothesis → Experiment → Metric → Result → Conclusion.

Create a table like:

| Research Question                                                  | Experiment                      | Metric   |
| ------------------------------------------------------------------ | ------------------------------- | -------- |
| Does semantic information improve narrator reliability estimation? | Graph-only vs Graph+BERT        | Macro-F1 |
| Does graph structure contribute meaningful information?            | Semantic-only vs Graph-only     | Macro-F1 |
| Does multimodal integration outperform individual modalities?      | Semantic vs Graph vs Multimodal | Macro-F1 |

This one change alone will make the thesis feel much more academic.

---

# Priority 2

## Lock your evaluation framework NOW

Tan's comment is easy to fix.

Before coding anything else, decide:

### Primary Metric

* Macro-F1

### Secondary Metrics

* Accuracy
* Cohen's Kappa
* Spearman Correlation
* MAE

And then never change them again.

One mistake many FYP students make is:

> Run experiment → metric bad → change metric.

Don't do that.

Write them into Chapter 3 and keep them fixed.

---

# Priority 3

## Literature comparison table

This is practically free marks.

Create a table like:

| Study          | Method         | Task                       | Dataset           | Metric   | Result |
| -------------- | -------------- | -------------------------- | ----------------- | -------- | ------ |
| Mahmoud et al. | NarratorsKG    | Narrator Disambiguation    | AR-Sanad          | Accuracy | X      |
| Narrator2Vec   | Embeddings     | Narrator Representation    | Hadith corpus     | F1       | Y      |
| Your work      | Multimodal GNN | Reliability Quantification | Sanadset + Taqrib | Macro-F1 | TBD    |

Then in Chapter 5:

> Our system achieved X compared to Y reported by Z.

Even if you don't beat them.

Lai specifically asked for this.

---

# Priority 4

## Fix narrator matching

If I were betting money on where your biggest performance gain comes from, it's here.

This statistic scares me:

> 3 exact matches, 1099 fuzzy matches. 

That means your entire pipeline depends heavily on fuzzy matching.

I would spend serious time improving:

* kunya normalization
* ibn/bin normalization
* nisba handling
* teacher/student context
* tabaqa information

A 5% improvement here may help more than changing the GNN architecture.

---

# Priority 5

## Build a proper ablation study

You already have the foundations.

Run:

### Experiment A

Graph-only

### Experiment B

Semantic-only

### Experiment C

Graph + Semantic

### Experiment D

Graph + Semantic + Temporal

### Experiment E

Graph + Semantic + Temporal + Geographic

Then show:

| Model      | Macro-F1 |
| ---------- | -------- |
| Graph      | 0.xx     |
| Semantic   | 0.xx     |
| Multimodal | 0.xx     |

Examiners love ablation studies because they answer:

> Which component actually matters?

---

# Priority 6

## Revisit Objective 3 interpretation

Even if you don't change the objective itself.

You can still adjust how you write about it.

If Objective 3 currently evaluates against Hadith gradings:

don't present it as:

> "The system grades Hadith."

Present it as:

> "Narrator reliability estimates were evaluated through their ability to produce chain-level assessments consistent with established Hadith classifications."

Subtle difference.

Same objective.

Stronger narrative.

Addresses Lai's concern.

---

# Priority 7

## Improve visualizations

Tan practically told you what he wants.

Keep only:

### Figure 1

System Architecture

### Figure 2

Knowledge Graph Example

### Figure 3

Narrator Reliability Distribution

### Figure 4

Model Comparison Bar Chart

### Figure 5

Confusion Matrix

### Figure 6

Example Chain Assessment

Every figure should answer:

> Why does this figure exist?

If it doesn't answer a question, remove it.

---

# Priority 8

## Strengthen the Discussion chapter

This is where many FYPs lose marks.

Don't just write:

> Graph+BERT got 0.25.

Write:

> The results suggest that semantic biographical information contains stronger reliability signals than graph topology alone, indicating that narrator trustworthiness is primarily encoded in scholarly assessments rather than transmission-network position.

That's the kind of interpretation examiners want.

---

# Priority 9

## Prepare for Viva Questions Now

I can already predict some.

### Why GNN?

### Why not just BERT?

### Why Macro-F1?

### Why use Taqrib?

### Why not train directly on Hadith grades?

### Why use continuous scores?

### Why not use a Transformer on the graph?

### Why is graph-only performance weak?

### Why is semantic information useful?

Start writing answers now.

---

# Priority 10

## Don't add new major components

This is the trap.

You are already tempted by:

* Graph Attention Networks
* Better Arabic LLMs
* Multi-sanad reasoning
* Matn analysis
* Geospatial verification
* Temporal verification

My advice:

### Freeze the architecture by Week 3–4.

After that:

* run experiments
* write results
* write discussion

Not new features.

---

If I were planning your semester, I'd roughly allocate:

**September**

* Research design fixes
* Evaluation framework
* Literature comparison
* Narrator matching improvements

**October**

* Final model training
* Ablation studies

**November**

* Analysis
* Visualizations
* Chapter 5 & 6

**December**

* Thesis polishing
* Viva preparation

The biggest thing I'd tell September-2026 Alan is:

> You're no longer in the "build more AI" phase.
>
> You're in the "prove something scientifically" phase.

That mindset shift is probably worth more marks than any new model architecture.
