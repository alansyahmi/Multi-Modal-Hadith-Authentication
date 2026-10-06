# FYP2 Examiner-Style Critical Audit

**Project:** A Multi-Modal Framework for Quantifying Narrator Reliability in ʿIlm ar-Rijal Using Semantic Sentiment Analysis and Graph Neural Networks
**Audit date:** 2026-10-05 (Week 3)
**Evidence base:** the plan documents in `FYP2/`, the FYP1 report (`fyp_project_1_full_chapters_v6.pdf`), the current code, and two read-only diagnostic scripts run on the frozen cohort and graph (`scratch/audit_data.py`, `scratch/audit_models.py`). Every number below comes from those runs unless it is marked **UNKNOWN**.

> [!CAUTION]
> **Correction to earlier claims.** The PO1/PO2 reports and changelog Entries 014–015 say that PO2 "decisively rejected H₀", that the fusion gain was "statistically significant", and that "graph topology is essential". The diagnostics in §0 show that these claims **are not supported**. No significance test was run. The +10.48-point gain came from one favourable seed. Removing the graph edges does not lower performance. Those statements should be withdrawn, not defended.

---

## 0. Diagnostic Evidence Summary (what the data actually shows)

| # | Check | Result | Severity |
|---|---|---|---|
| D1 | Two label sources mixed in one target | `taqrib_classical` (553) + `hadith_viz_consensus` (660). Where both exist (607 nodes), the FYP1 Taqrīb column agrees with Hadith-Viz only **46.1%** of the time | P0 |
| D2 | Where the minority classes come from | **All 9 `matruk`** and **12/17 `daif`** labels come from the FYP1 Taqrīb column. In that column, 6 of 8 "matrūk" nodes that overlap with Hadith-Viz are graded *thiqah* by Hadith-Viz | P0 |
| D3 | Spot-check of the Taqrīb column | Abū Bakr b. Abī Shayba → "مقبول"; Ḥammād b. Zayd → "صدوق"; ʿAmr b. ʿAlī → "صدوق"; Muḥammad b. Juḥāda → "صدوق". In the actual *Taqrīb*, all four are *thiqah* (ثقة حافظ / ثقة ثبت). The column inherits FYP1's fuzzy-match errors | P0 |
| D4 | Same person split across nodes | 546 graded rows share 247 Hadith-Viz identities; **125 identities span more than one split**; **60/182 test nodes** are the same person as a training node | P0 |
| D5 | Identical text in train and test | **41/182 test texts are byte-identical** to a training text (same person, same Hadith-Viz biography, same label) | P0 |
| D6 | Text-less narrators | **42.5%** of "texts" are a name-plus-ṭabaqa template (no biography). Rate by class: matrūk 89%, ḍaʿīf 71%, maqbūl 70%, ṣadūq 71%, thiqah 23%. Median text length: thiqah 1,608 chars, every other class 40–46 | P0 |
| D7 | What masked TF-IDF actually learns | Top features for classes 0 and 3 are template tokens (`مات سنة`, `من الحادي`, `التاسع`). For class 4 they are critic-name tokens (`وأبو حاتم`, `والنسائي`). The model learns "has a documented biography", not appraisal semantics | P0 |
| D8 | Masking residue | Evaluative words that survive masking and track the label: `ثبت`, `حافظ`, `متقن`, `إمام`, `لين`, `واه`, `صالح`, `يخطئ`, `صحبة` | P1 |
| D9 | Ibn Ḥajar's verdict in the input | `shamela_rijal2.json` holds a dedicated `ibn_hajar_grade` field, *and* Ibn Ḥajar's sentence appears as a `jarh_tadil` block (937 narrators). That block is fed into the text input, so the label source is inside the features (only partly masked) | P0 |
| D10 | Label homophily on the graph | Same-label edge rate is 0.505, against 0.470 expected by chance. The isnād graph is **almost non-homophilous** for reliability | P1 |
| D11 | Edge direction and self-loops | Only 64 of 3,541 edges are reciprocal. Edges are directed, and `GCNConv` aggregates only in one direction. There are 30 self-loops (a narrator transmitting from himself), which point to identity conflation | P1 |
| D12 | Trivial baselines (test set) | Majority macro-F1 = 0.149; stratified-random macro-F1 = 0.200 [0.160, 0.279]. **Graph-only GCN = 0.187 ± 0.035, which is chance level** | P0 |
| D13 | PO2 over 10 seeds (same split) | Semantic-only 0.384 ± 0.027; Fusion-attention 0.443 ± 0.038; **MLP on [text ‖ 8 graph features], no message passing: 0.460 ± 0.043**; **Fusion with edges removed: 0.464 ± 0.050**; Fusion with graph features shuffled: 0.426 ± 0.013 | P0 |
| D14 | Seed sensitivity | Per-arm SD is 0.03–0.05 macro-F1, which is **larger than the 3-point decision threshold**. The reported Arm B score (0.334) sits at the low tail of its own distribution (0.319–0.417) | P0 |
| D15 | Duplicate-free test subset (122 novel identities) | Every model drops by 2.5–4 points. Fusion-attention 0.418 vs. text + features MLP 0.417 vs. edges removed 0.444 | P0 |
| D16 | Node-feature artefact | `has_death_info` by class: matrūk 0.00, ḍaʿīf 0.18, maqbūl 0.17, ṣadūq 0.78, thiqah 0.63. This reflects which dictionary entry a node was matched to, not transmission topology | P1 |

**What the current data honestly supports:**
1. Text (mostly *whether a biography exists*) beats chance.
2. Adding 8 node-level features to text gives a small gain of roughly 2–7 points, depending on the control, and that gain is within about 1–2 SD.
3. **GNN message passing adds nothing measurable** on this graph.
4. Part of the "fusion" gain is architectural: the fusion model still beats the text-only MLP when the graph features are shuffled.

This is a legitimate, reportable finding. It is not the finding currently written in `po2_ablation_report.md`.

---

## 1. Research Problem and Novelty

| Question | Assessment |
|---|---|
| Is the problem clearly defined? | **Partly.** "Quantify narrator reliability" is never pinned to a single operational target. The label is a mix of Ibn Ḥajar's Taqrīb grade and an undefined Hadith-Viz "consensus" tier. "Quantify" implies a continuous score, but the system is a 5-class classifier with hand-set score mappings (`CLASS_TO_SCORE = 0.05/0.15/0.40/0.70/0.85`). |
| Is there a real gap? | **Yes, a narrow one.** Prior work does isnād disambiguation (NarratorsKG, Narrator2Vec) or ḥadīth-level grading. Predicting a *narrator* grade from (a) critics' appraisals and (b) transmission position, under leakage control, is not well covered. |
| Is the novelty defensible? | **Only if it is stated modestly.** "First multimodal GNN framework" will not survive D13. "A controlled study of whether transmission-network information adds predictive value over critic text for narrator grading" is defensible, *whatever the result*. |
| Claims stronger than the method supports | (1) "Graph topology is essential" (contradicted by D13). (2) "Statistically significant" (no test was run). (3) "Multi-critic consensus" ground truth: **UNKNOWN** whether Hadith-Viz `grade_tier` is a consensus or simply a mapping of `ibn_hajar_grade`. Resolve by cross-tabulating `grade_tier` against `ibn_hajar_grade` in `shamela_rijal2.json`. If they agree about 100%, "consensus" is a mislabel. (4) "Semantic sentiment analysis" (see §5). |
| Is "multimodal" correct? | **Weak.** Text and graph are two *views* of the same entity. "Multi-view" or "text-attributed graph" is the precise term. "Multimodal" is used this way in some graph-learning papers, so it can be kept if Ch. 1 defines it. The real problem: three of the 8 "graph" features (`has_death_info`, `has_tabaqa`, `tabaqa_num`) come from the **biographical dictionary**, not the graph. The "graph modality" is therefore contaminated with dictionary metadata. **P1** |
| Contribution type | Currently: **implementation plus dataset engineering** (entity linking to Hadith-Viz, a frozen cohort). After the P0 fixes, the realistic contribution is an **empirical one**: a controlled ablation with an honest answer. That is enough for an undergraduate FYP. |

---

## 2. RQ → Hypothesis → Experiment → Metric → Conclusion Chain

| Link | Problem | Priority |
|---|---|---|
| RQ wording drifts across documents | The action plan RQ ("how well can NLP… and does network information improve") differs from the design-table RQs, the title, and the design-table header ("Automated Narrator Reliability Assessment… Semantic NLP and KG Fusion"). Pick one wording and use it everywhere. | P2 |
| H₀/H₁ say "statistically significant", but the decision rule is a single point estimate (≥ 5 or ≥ 3 points on one test run) | This is internally inconsistent, and D14 shows a single run cannot meet the rule reliably. **Fix:** decision rule = mean paired gain ≥ threshold **and** a 95% CI of the paired difference that excludes 0, across repeated grouped CV folds or seeds. | **P0** |
| PO1 IV lists "AraBERT zero-shot", which was never run; controlled variables cite `narrator_encyclopedia.json` (3,797 entries), but PO1 used Hadith-Viz text | The design table no longer matches the experiment. Remove the zero-shot arm and correct the data source. | P1 |
| PO2 design table still says 1,987 labelled / 299 test, and that `gnn_model.py` is an autoencoder | Stale. Current: 1,213 / 182, supervised. | P1 |
| Does the PO1 conclusion follow? | **No.** PO1 compares models on inputs that are 42.5% name templates (D6). Both models largely learn documentation density (D7). The comparison measures "who exploits the template best", not semantic vs. lexical representation. | **P0** |
| Is Graph / Semantic / Fusion sufficient for PO2? | **No.** It cannot separate the three things that change when you go from Semantic to Fusion: **(i)** extra node features, **(ii)** message passing over edges, **(iii)** a different architecture/capacity. D13 shows (ii) contributes nothing and (iii) contributes part of the gain. The minimum extra controls are a **non-GNN arm** (same features, no edges) and a **feature-group split** (topology vs. dictionary metadata). | **P0** |
| PO3 chain validation | The ḥadīth grades in reference collections were produced *by scholars using narrator grades*, so agreement is partly built in. **UNKNOWN** whether SanadSet carries chain-level grades for your chains, and at what coverage. Count chains whose narrators are all in the cohort *and* that have a reference grade before committing. A better-defined PO3: compare chain scores computed from **predicted** narrator grades against chain scores computed from **true** narrator grades (oracle). That measures error propagation, which is non-circular and cheap. | P1 |

**Better hypothesis for PO2 (it materially improves validity):**
> H₁(PO2): Adding transmission-network information (node topology features and/or message passing) to critic-text features increases grouped-CV Macro-F1 by ≥ 3 points over text alone, with a 95% CI of the paired difference excluding 0. H₁a tests features only (MLP). H₁b tests message passing (GNN vs. MLP on identical inputs).

This keeps your RQ and objectives. It only makes the test able to tell "graph information" apart from "GNN".

---

## 3. Dataset and Ground Truth

### 3.1 Problems (with evidence)
1. **Label definition (P0).** Two sources disagree 54% of the time (D1). The minority classes come almost entirely from the less reliable one (D2–D3). The current 5-class target is therefore partly a measure of FYP1 matching noise.
2. **Identity duplication (P0).** Entity resolution matched surface forms to Hadith-Viz identities but **did not merge nodes**. "إبراهيم بن سعد" appears as two nodes, one in train and one in val. This is train/test contamination (D4–D5), and it also fragments the graph.
3. **Single-token names (P0).** Your own audit gave single-token curated matches **36% precision**. Yet 200 single-token nodes are in the labelled cohort (33 in test), e.g. `سفيان` → ṣadūq, `الحسن` → ṣadūq, `ثابت` → ṣadūq. These are almost certainly mislabelled (al-Thawrī / Ibn ʿUyayna, al-Ḥasan al-Baṣrī, and Thābit al-Bunānī are all *thiqah*).
4. **Companions (P1).** 148 ṣaḥāba are merged into class 4. Companions are not assessed by *jarḥ wa taʿdīl* (their *ʿadāla* is a doctrinal premise), so they are not part of the construct. They are also trivially identifiable (chain end, `صحبة` in the text, ṭabaqa 1), which inflates class-4 F1. Exclude them from the labelled set. They can stay as unlabelled graph nodes.
5. **Text availability is confounded with class (P0).** See D6. A text-based classifier cannot be evaluated fairly on narrators without text.
6. **Class imbalance at test time (P0).** 2 matrūk and 2 ḍaʿīf test nodes. One prediction changes a class F1 by about 0.33–0.5, which moves macro-F1 by about 0.07–0.10. The PO2 "matrūk 0.083 → 0.333" gain is literally **one narrator**, and that narrator's label is probably wrong (D2).

### 3.2 Leakage: how to detect and prevent it
**Leakage channels present today:**
- (a) Ibn Ḥajar's own verdict sentence is in the input text (D9).
- (b) Masking misses evaluative synonyms (D8).
- (c) The same person with the same text appears in train and test (D5).
- (d) Proxy leakage from dictionary metadata (`has_death_info`, D16).
- (e) Text presence as a proxy for class (D6).

**Recommended fix (P0): reformulate the text task so masking is unnecessary.**
- **Label:** Ibn Ḥajar's *Taqrīb* verdict, taken from Hadith-Viz `ibn_hajar_grade` via the resolved identity. Drop the FYP1 Taqrīb column entirely.
- **Text input:** only the statements of *earlier* critics (Ibn Maʿīn, Aḥmad, Abū Ḥātim, al-Nasāʾī, Ibn Ḥibbān, …). **Exclude** every `ابن حجر` block, and also the `ibn_hajar_grade`, `dhahabi_grade`, `grade_tier` and `grades_by_source` fields.
- **Task:** *"Predict a later synthesising critic's verdict from earlier critics' appraisals."* This is a non-circular, historically meaningful task (modelling how Ibn Ḥajar weighs conflicting *jarḥ* and *taʿdīl*). The evaluative words become **legitimate input** rather than leakage, which also makes the "sentiment" part of your title true (§5).
- **Inclusion criterion:** narrators with a valid `ibn_hajar_grade` **and** at least one earlier-critic statement. **UNKNOWN cohort size.** Resolve by counting those narrators among the resolved identities before committing. If any class has fewer than about 30 examples, see §7 on class granularity.

**Leakage tests to run and report (each about one line of code):**
1. Grep the model inputs for `ابن حجر`, `تقريب`, and the label string itself. The expected count is 0.
2. Grouped split: assert that no `hv_shamela_id` appears in more than one fold.
3. Exact and near-duplicate text check across folds (hash and TF-IDF cosine > 0.95). The expected count is 0.
4. "Name-only" probe: a classifier on the narrator name plus ṭabaqa string alone. Its score is the floor any text model must clearly beat.
5. Inspect the top-weighted TF-IDF features per class. They should be appraisal language, not template tokens (compare D7).

---

## 4. Knowledge Graph Methodology

| Aspect | Finding | Priority |
|---|---|---|
| Nodes | Surface-form nodes, not resolved persons (D4). Collapse nodes by resolved identity, then recompute the features. | **P0** |
| Edges | Directed, with 64 reciprocal edges. Semantics (source = teacher or student?) are **UNKNOWN** from the files I read; check `graph_edges.csv` / the Stage 2 builder and state it in Ch. 4. `GCNConv` on a directed `edge_index` passes messages one way only. Use `to_undirected` or justify the direction. | P1 |
| Relation types | One edge type. Transmission formula (`حدثنا`/`عن`) is encoded as an edge attribute but **not used** by `GCNConv` (it ignores `edge_attr`). Either drop the claim or note the limitation. | P2 |
| Self-loops | 30 self-loops, probably conflation (e.g. "عن أبيه" resolved to the same name). Remove them and report the count. | P1 |
| Components / sparsity | 11 components (largest 2,135). Median test degree 2. The graph is built from a small chain sample (FYP1: about 1,030 aligned chains; exact count **UNKNOWN**), so degree features largely measure *sampling frequency in that subset*, not real transmission prominence. | P1 (state as a limitation) |
| Does structure carry signal? | Label homophily 0.505 vs. 0.470 at chance (D10). Graph-only is at chance (D12). Removing edges does not hurt fusion (D13). **On current evidence, the GNN is not justified.** That is a valid finding *if you report it as one*. | **P0** (reporting) |
| "GNN for the sake of GNN"? | Yes, in the current claims. Keep the GNN as **one arm of the test**, not as an assumed component. If GNN ≤ MLP on identical inputs, the thesis concludes that message passing over isnād edges does not add value for this target, and explains why (low homophily: reliable and weak narrators transmit to each other). | — |

---

## 5. Semantic / NLP Methodology

1. **"Semantic sentiment analysis" is challengeable (P0, terminology).** Sentiment analysis means detecting an author's *affective* polarity or opinion. *Jarḥ wa taʿdīl* statements are *technical evaluative judgments* about a transmitter's integrity (*ʿadāla*) and precision (*ḍabṭ*), with a graded, partly conventional vocabulary. FYP1 even defines the component as "semantic sentiment **quantisation** that maps qualitative terms (Thiqa, Ṣadūq, Kaḏḏāb) to continuous weights", which is a lexicon lookup of the label itself.
   - Under the current FYP2 masking design, the system **removes** exactly those terms, so it performs *no* sentiment analysis at all. Title and method contradict each other.
   - **Defensible formulation (no title change needed):** define it in Ch. 1/3 as *"classification of the evaluative polarity (jarḥ vs. taʿdīl) and intensity of critics' appraisal statements"*, i.e. appraisal/stance analysis. The §3.2 reformulation makes this literally what the model does.
2. **AraBERTv2 preprocessing (P1).** `aubmindlab/bert-base-arabertv2` was pretrained on Farasa-segmented text and expects `ArabertPreprocessor`. `run_po1_benchmark.py` passes raw text straight to the tokenizer. That is a tokenisation mismatch. Either apply the preprocessor or use a model that does not need it.
3. **Domain mismatch (P2).** AraBERT was pretrained on Modern Standard Arabic (news/web). CAMeLBERT-CA (pretrained on classical Arabic) is a one-line swap and directly answers the examiner question "why a modern model for classical text?". Run it only if time permits. Otherwise state the mismatch as a limitation.
4. **Truncation (P2).** `max_length=256` tokens while thiqah texts have a median of 1,608 characters. Long biographies are cut. Fine for an FYP, but say so.
5. **Fine-tuned embeddings reused in PO2 (P1).** The PO2 semantic features are [CLS] vectors from the PO1 model fine-tuned on the training labels. Train-node vectors are therefore over-fit, so they look more separable than test vectors, and the checkpoint was selected on validation. **Simplest fix:** use *frozen* pretrained embeddings (mean-pooled) for PO2. That makes cross-validation cheap and removes the coupling. If fine-tuning is kept, it must be redone inside each fold.
6. **NER (UNKNOWN).** FYP1 Chapter 3 describes an NER model. Its role in the FYP2 pipeline and its evaluation are **UNKNOWN**. If entity resolution now uses normalised-name matching against Hadith-Viz, either evaluate NER or stop claiming it.

---

## 6. Multimodal Fusion

| Question | Answer |
|---|---|
| What are the modalities? | Text = critic statements (after the §3.2 fix). "Graph" = currently 5 topology features + 3 dictionary-metadata features + edges. **The metadata features must be moved out** of the graph modality, or reported as a third group. |
| Fusion stage | `ReliabilityFusionGNN` is intermediate fusion: GCN(8 features) ‖ projected text → classifier. |
| Can one modality dominate? | Yes. 768-d text vs. 8-d graph. D13 shows the fused result equals the "MLP on concatenation" result, so the text features carry it. |
| Principled reason for attention gating? | None demonstrated. Attention ≈ concat within noise (seed SD ≈ 0.04). |
| Simplest defensible fusion | **Early fusion:** standardise both blocks and concatenate them into one classifier (logistic regression or a small MLP). Optionally add **late fusion** (average the class probabilities of the two unimodal models) as a robustness check. Use the GNN version only if it beats the MLP on identical inputs. Drop attention gating from the main claims (keep it as optional). |

---

## 7. Model and Evaluation

| Item | Assessment / Recommendation | Priority |
|---|---|---|
| Split strategy | A single 70/15/15 split with n_test = 182 cannot support a 3-point decision rule (D14). **Use repeated stratified *grouped* k-fold** (group = resolved identity; 5 folds × 3–5 repeats). Graph and MLP models train in seconds, so this is cheap. | **P0** |
| Seeds | Report mean ± SD over runs. Never report a single run. | **P0** |
| Significance | Paired comparison per fold. Report the mean difference with a bootstrap 95% CI (or Wilcoxon signed-rank across folds). This is what establishes "the gain is not noise". Nothing more elaborate is needed. | **P0** |
| Class granularity | With 9 matrūk / 17 ḍaʿīf *total* (and probably mislabelled), 5-class macro-F1 is dominated by about 4 test examples. **Recommend: primary metric = Macro-F1 on 3 ordinal tiers** (thiqah / ṣadūq / below-ṣadūq = maqbūl + ḍaʿīf + matrūk), with 5-class reported as secondary. This is a substantive methodological reason to change the metric, not a cosmetic one. Lock it *before* re-running. | **P0** |
| Macro-F1 | Correct primary metric: it weights each tier equally, which is what "detect weak narrators" requires. | adequate |
| Quadratic κ | Keep. It establishes whether errors fall on *adjacent* grades, which matters for an ordinal target. | adequate |
| Per-class P/R/F1 + confusion matrix | Keep. They show *where* gains come from (e.g. only thiqah vs. ṣadūq). | adequate |
| Spearman on mapped scores | Redundant with quadratic κ, because the mapping is arbitrary. Drop it or keep it as minor. | P3 |
| Accuracy | Report it only next to the majority baseline (59.3%). On its own it misleads (graph-only accuracy is 31%, below majority). | P2 |
| Hyperparameters | Fixed defaults (hidden 64, dropout 0.3, lr 0.005) are acceptable **if identical across arms** and stated. Do not tune on test. Optional: a small grid on inner validation, applied equally to all arms. | P2 |
| Class weights | Inverse-frequency weights are fine. State it. | adequate |
| Reproducibility | Seeds are set, but CUDA scatter ops in PyG are non-deterministic. Your two "identical" PO2 runs gave Arm B = 0.391 and then 0.334. Report variance rather than chasing bit-exactness. Pin versions in `requirements.txt` (the `huggingface-hub` monkey-patch is itself a reproducibility risk; pin a compatible version instead). | P1 |

**What is excessive for an FYP:** AUC-OvR, calibration curves (`train_calibrated.py` isotonic), multiple significance-correction schemes, link-prediction AUC, Pearson r on mapped scores.

---

## 8. Baselines and Ablations: Smallest Convincing Set

| Candidate | Rank | What it establishes |
|---|---|---|
| Majority + stratified-random | **CRITICAL** | The chance floor. Graph-only currently does not clear it (D12). Costs nothing. |
| Name/template-only probe | **CRITICAL** | Shows the text model uses appraisal content, not metadata or documentation density (D6–D7). |
| TF-IDF + logistic regression (text) | **CRITICAL** | The lexical baseline; it already exists. |
| AraBERT (frozen embeddings + same classifier head) | **CRITICAL** | The semantic arm. Same head as TF-IDF, so representation is the only variable. |
| Graph features only, MLP/LR (no edges) | **CRITICAL** | The non-GNN graph baseline. Separates "graph information" from "GNN". |
| Graph-only GCN | **CRITICAL** | Tests message passing on its own. |
| Text + graph features, MLP/LR (early fusion, no edges) | **CRITICAL** | Answers the central RQ (H₁a). |
| Text + graph, GCN fusion | **CRITICAL** | Answers H₁b. Compare it to the row above *on identical inputs*. |
| Feature-group split (topology 5 vs. dictionary metadata 3) | **RECOMMENDED** | Shows whether any "graph" gain is really dictionary metadata (D16). It is a feature-subset rerun, so cheap. |
| Shuffled-features / edges-removed controls | **RECOMMENDED** | Cheap and decisive falsification controls (already demonstrated in D13). |
| Attention vs. concat fusion | OPTIONAL | No evidence attention helps. Keep concat as primary. |
| Fine-tuned AraBERT vs. frozen | OPTIONAL | Only if time permits, and only fine-tuned inside each fold. |
| CAMeLBERT-CA swap | OPTIONAL | Addresses domain mismatch. |
| GAT, GraphSAGE, other GNNs | UNNECESSARY | They do not address the RQ. Low homophily will not be fixed by attention. |
| AraBERT zero-shot | UNNECESSARY | There is no zero-shot formulation for a 5-way grade without prompting. Remove it from the design table. |
| Late fusion | OPTIONAL | Robustness check only. |

---

## 9. Threats to Validity

| Type | Threat | Feasible mitigation |
|---|---|---|
| Construct | The label mixes two sources. Companions are included. "Sentiment" is undefined. "Consensus" is unverified | Single label (Ibn Ḥajar via Hadith-Viz); exclude companions; operational definitions in Ch. 1; verify `grade_tier` provenance |
| Construct | The "graph modality" includes dictionary metadata | Split the feature groups and report them separately |
| Internal | Train/test identity contamination (D4–D5) | Merge nodes by identity; use grouped CV |
| Internal | The label source is in the input (D9) and masking is incomplete (D8) | §3.2 reformulation plus the grep / probe tests |
| Internal | Architecture confounded with modality (D13) | Same classifier head across arms; MLP-vs-GNN on identical inputs |
| Data | Label noise from FYP1 fuzzy matching (D3); single-token mismatches | Drop the FYP1 Taqrīb column; exclude single-token names; hand-check a random sample of 50 labels and report the error rate |
| Data | Text availability is confounded with class (D6) | Inclusion criterion: at least one earlier-critic statement |
| Evaluation | n_test = 182, tiny minority classes, seed SD > threshold | Repeated grouped CV; 3-tier primary metric; paired CI |
| External | Cohort is narrators who are both in a small chain sample and in a 3,047-narrator curated set (well-documented, mostly reliable) | State it plainly: results apply to well-documented Kutub-sitta narrators, not obscure transmitters |
| Reproducibility | Non-deterministic CUDA ops, a dependency monkey-patch, cached embeddings from a selected checkpoint | Report mean ± SD; pin versions; derive frozen embeddings deterministically |

---

## 10. Scope Creep: Cut, Postpone or Demote

| Item | Verdict | Reason |
|---|---|---|
| PO3 with four aggregators | **Reduce** | Keep Min (weakest link, which is the doctrinally motivated rule) plus the predicted-vs-oracle comparison. Mean, Product and GeoMean add nothing to the RQ. |
| Attention-gated fusion as the headline | **Demote** to optional | No demonstrated benefit. |
| GAT backbone | **Remove** from the claims | Not needed. |
| Link-prediction diagnostic in `trainer.py` | **Remove** from reporting | Irrelevant to the RQ. |
| `anomaly_score` / `consistency_score` columns | **Remove** or leave unreported | They are not evaluated, so claiming them invites questions. |
| Isotonic calibration (`train_calibrated.py`) | **Postpone** to future work | It answers a question you did not ask. |
| FYP1 edge risk modifiers (tadlīs × 0.80), multiplicative chain scores | **Future work** | Hand-set weights cannot be validated. |
| Geographic maps and timeline visualisations (FYP1 fig9/fig10) | **Do not extend** | Descriptive only. |
| Using the 248K Hadith-Viz index beyond entity linking | **Do not** | Matching only. |
| NER model training | **Do not extend**; clarify whether it is still used | It is not on the RQ path. |
| A "live system demonstration" web app | **Minimal**: a CLI or notebook that scores one narrator | The viva needs evidence, not UI. |
| Rebuilding the graph from all 650K SanadSet chains | **Optional P2, risky** | It would make degree features meaningful, but it is a large job. Mention it as a limitation or future work unless time is clearly available. |

---

## 11. Viva Examiner Questions (15)

| # | Question | Why it is asked / weakness exposed | A strong answer establishes |
|---|---|---|---|
| 1 | "Your label is Ibn Ḥajar's grade, and your input is critics' text. Is his own verdict in that text?" | D9: circularity | A grep showing 0 occurrences; the input is restricted to earlier critics; why this is a non-circular synthesis task |
| 2 | "If you mask the grade words, what 'sentiment' is left? If you don't, isn't it leakage?" | The central title/method contradiction | The §3.2 reformulation: earlier critics' appraisals are legitimate input for predicting a later synthesis |
| 3 | "Define 'sentiment' in your title." | Terminology | An operational definition: evaluative polarity and intensity of jarḥ/taʿdīl appraisals; how it differs from affective sentiment |
| 4 | "Is the same narrator in both train and test?" | D4/D5 | Nodes merged by resolved identity; grouped CV; zero shared identities (show the assertion) |
| 5 | "Does your GNN beat an MLP given the same features with no edges?" | D13: GNN justification | The paired result, whichever way it goes, with a CI. If not, explain low homophily (D10) |
| 6 | "What is your graph-only model's score against random guessing?" | D12 | Graph-only vs. the stratified-random CI. Honest interpretation |
| 7 | "Your fusion gain: is it the graph, or the bigger model?" | Architecture confound | The shuffled-features control and same-head comparisons |
| 8 | "How many matrūk narrators are in your test set? How were they labelled?" | D2/D3/minority instability | 3-tier primary metric; the label source; a hand-verified label sample |
| 9 | "Run it with another seed. Does your conclusion hold?" | D14 | Mean ± SD over repeated grouped CV; the decision rule uses the CI |
| 10 | "Why AraBERTv2, a modern-Arabic model, for 3rd–9th-century biographical Arabic? Did you apply its preprocessing?" | Domain mismatch, missing Farasa preprocessing | Preprocessor applied (or CAMeLBERT-CA compared); the mismatch acknowledged |
| 11 | "Why are ṣaḥāba in your thiqah class?" | Construct validity | They are excluded, with the doctrinal reason |
| 12 | "`has_death_info` is 0.00 for matrūk and 0.78 for ṣadūq. Is that topology?" | D16: metadata masquerading as graph | Feature groups separated; their contributions reported separately |
| 13 | "Your graph comes from about 1,000 chains. What does 'degree' measure?" | Sampling artefact | Acknowledged as sampling frequency; a limitation; the scope of the claim restricted |
| 14 | "Ḥadīth grades were derived from narrator grades. Isn't PO3 circular?" | PO3 validity | PO3 reframed as error propagation (predicted vs. oracle narrator grades); coverage reported |
| 15 | "What does your system do that a scholar looking up the Taqrīb cannot?" | Practical value and claim inflation | The modest framing: a controlled study of how far text and network signals reproduce a classical synthesis; decision-support only, with no authentication claim |

---

## 12. Final Assessment

### A. CRITICAL ISSUES (P0: fix before the Progress Report)
1. **Withdraw the PO1/PO2 conclusions** in `po1_benchmark_report.md`, `po2_ablation_report.md` and Entries 014–015 (single run, no significance test, contradicted by D13).
2. **Single, clean label:** Ibn Ḥajar's grade via Hadith-Viz `ibn_hajar_grade`. Drop the FYP1 Taqrīb column. Verify what `grade_tier` actually is.
3. **Remove Ibn Ḥajar's own statements and all grade fields from the text input.** Switch to the earlier-critics → later-verdict formulation.
4. **Merge nodes by resolved identity** and use **grouped splits**. Zero shared identities or texts across folds.
5. **Exclude** single-token and ambiguous names, companions, and narrators with no critic text from the labelled set.
6. **Repeated grouped CV with a paired CI decision rule.** Lock a **3-tier primary Macro-F1** (5-class as secondary) before re-running.
7. **Add the non-GNN controls** (graph features via MLP; text + graph via MLP) and the chance baselines.
8. **Define "sentiment" and "multimodal" operationally** in Ch. 1/3.

### B. HIGH-VALUE REFINEMENTS (P1)
- Split graph features into topology vs. dictionary metadata.
- Undirect edges (or justify the direction); remove the 30 self-loops.
- Use frozen AraBERT embeddings for PO2 (or fine-tune within folds); apply `ArabertPreprocessor`.
- Reframe PO3 as predicted-vs-oracle error propagation with Min aggregation; check coverage first.
- Hand-verify a random sample of 50 labels and report the error rate.
- Update `research_design_table.md` (stale counts, the unrun zero-shot arm, wrong data source); pin dependency versions.

### C. OPTIONAL (P2–P3)
- CAMeLBERT-CA comparison; late-fusion robustness check; attention vs. concat; inner-CV hyperparameter grid; a note on truncation; recomputing degree features on a larger chain set; harmonising the title wording across documents.

### D. DO NOT DO
- More GNN variants (GAT, SAGE, transformers-on-graphs).
- More datasets.
- Calibration or uncertainty work.
- Link prediction.
- A web app.
- Extending the masking keyword list by hand (whack-a-mole; replaced by the reformulation).
- Tuning anything on the test set.
- Claiming ḥadīth authentication.

### E. MINIMUM DEFENSIBLE EXPERIMENT SET
All runs use the same cleaned cohort, 5-fold × 3+ repeats grouped stratified CV, the 3-tier Macro-F1 primary metric, and paired 95% CIs:
1. Majority and stratified-random baselines.
2. Name/template-only probe.
3. Text: TF-IDF + LR **vs.** frozen AraBERT + LR (PO1).
4. Graph features only, LR/MLP **vs.** graph-only GCN.
5. Text + graph features, MLP (early fusion), **vs.** text-only (PO2 H₁a).
6. Text + graph, GCN fusion, **vs.** row 5 on identical inputs (PO2 H₁b).
7. (Recommended) Row 5 with topology-only vs. metadata-only feature groups.
8. PO3: Min-aggregated chain score from predicted vs. oracle narrator grades, with coverage reported.

That is about 10 model configurations. Everything except AraBERT feature extraction (done once) trains in seconds.

### F. RECOMMENDED FINAL PIPELINE
```
SanadSet chains ──► name normalisation ──► entity resolution to Hadith-Viz identity
                                            (exclude single-token/ambiguous; MERGE nodes by identity)
                                                     │
          ┌──────────────────────────────────────────┴───────────────────────────┐
          ▼                                                                      ▼
 Isnād graph over resolved persons                            Hadith-Viz record per person
 (undirected, self-loops removed)                              • LABEL  = ibn_hajar_grade → 3 tiers (5 secondary)
 • topology features (degree, flow, position)                  • TEXT   = earlier critics' statements only
 • [separate] dictionary metadata (ṭabaqa, death)                         (no Ibn Ḥajar blocks, no grade fields)
          │                                                      • exclude companions & text-less narrators
          │                                                                      │
          │                                         frozen AraBERT (with preprocessor) / TF-IDF
          └──────────────────────┬───────────────────────────────────────────────┘
                                 ▼
          Arms: chance · text-only · graph-feat MLP · graph GCN · text+graph MLP · text+graph GCN
                                 ▼
          Repeated grouped stratified CV → Macro-F1 (3-tier), quadratic κ, per-class F1
          → paired differences with 95% CI → H₁a / H₁b decisions
                                 ▼
          PO3: chain Min-score from predicted vs. oracle narrator grades (+ coverage)
```

### G. TOP 5 RISKS
1. **Circularity / label leakage**: Ibn Ḥajar's verdict inside the input text (D9), plus masking residue (D8).
2. **Train/test identity contamination**: 60/182 test nodes are training narrators (D4–D5).
3. **Overclaimed PO2 result**: the GNN adds nothing over an MLP, and the gain is within seed noise (D13–D14).
4. **Label noise in exactly the classes that matter**: matrūk and ḍaʿīf come from the 46%-agreement FYP1 column (D1–D3).
5. **Text model measures documentation density, not appraisal semantics**: 42.5% template inputs; thiqah texts are about 35× longer (D6–D7).

---

### Open UNKNOWNs and how to resolve them
| UNKNOWN | Resolving evidence |
|---|---|
| Is Hadith-Viz `grade_tier` a consensus, or derived from `ibn_hajar_grade`? | Cross-tabulate the two fields in `shamela_rijal2.json` |
| Cohort size after the §3.2 inclusion criteria | Count resolved identities with `ibn_hajar_grade` and at least one non-Ibn-Ḥajar critic statement, per class |
| Edge direction semantics | Read the Stage 2 edge builder / `graph_edges.csv` header and document it |
| Exact number of chains the graph was built from | Stage 1/2 logs or `graph_metadata.json` |
| Whether SanadSet provides chain-level reference grades for PO3, and at what coverage | Count chains with a grade and full cohort coverage |
| Whether NER is still in the FYP2 pipeline and evaluated | Check `run_stage*.py` imports; report or drop the claim |
