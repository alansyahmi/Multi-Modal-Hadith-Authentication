"""Generate Chapter 4 .docx following the writing style guide."""
import sys
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import re

doc = Document()

# ── Style setup ──
style = doc.styles["Normal"]
font = style.font
font.name = "Times New Roman"
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.5

# Heading styles
for level in range(1, 4):
    hs = doc.styles[f"Heading {level}"]
    hs.font.name = "Times New Roman"
    hs.font.color.rgb = RGBColor(0, 0, 0)
    hs.paragraph_format.space_before = Pt(12)
    hs.paragraph_format.space_after = Pt(6)

def add_paragraph(text, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    return p

def add_list_item(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(1.27 + level * 1.27)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    return p

def add_bold_text(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    return p

def add_table_header(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    return p

def add_code(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(10)
    return p

# ══════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("CHAPTER 4: EXPERIMENTAL DESIGN")
run.bold = True
run.font.size = Pt(16)
run.font.name = "Times New Roman"

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    "A Multi-Modal Framework for Quantifying Narrator Reliability "
    "in ʿIlm ar-Rijal Using Semantic Sentiment Analysis and "
    "Graph Neural Networks"
)
run.font.size = Pt(11)
run.font.name = "Times New Roman"
run.italic = True

doc.add_page_break()

# ══════════════════════════════════════════════════════════
# 4.1 INTRODUCTION
# ══════════════════════════════════════════════════════════

doc.add_heading("4.1 Introduction", level=1)

add_paragraph(
    "This chapter defines the experiments designed to test each of the "
    "three research objectives stated in Section 1.4. The structure follows "
    "the pipeline stages: data preparation, entity extraction, graph construction, "
    "GNN training, and evaluation. Every experiment states a clear question, "
    "describes the setup, lists the key parameters, and specifies the expected outputs."
)

add_paragraph(
    "Seven experiments are designed. Each experiment targets a specific component "
    "of the multi-modal framework. The experiments follow the natural pipeline order: "
    "data acquisition and cleaning (Experiments 1), narrator classification "
    "(Experiment 2), knowledge graph construction (Experiment 3), chain scoring "
    "methodology (Experiment 4), hybrid evaluation against canonical grades "
    "(Experiment 5), multi-chain corroboration (Experiment 6), and theoretical "
    "ceiling analysis (Experiment 7)."
)

add_paragraph(
    "The experimental environment, hardware configuration, and software stack "
    "are described first to establish reproducibility."
)

# ══════════════════════════════════════════════════════════
# 4.2 EXPERIMENTAL ENVIRONMENT
# ══════════════════════════════════════════════════════════

doc.add_heading("4.2 Experimental Environment", level=1)

add_paragraph(
    "All experiments run on a local development workstation. The hardware "
    "configuration includes an AMD Ryzen 5 5600X processor (6 cores, 12 threads), "
    "32 GB of DDR4 RAM, and an NVIDIA GeForce RTX 2070 Max-Q GPU with 8.6 GB VRAM."
)

add_paragraph(
    "The software stack is built on Python 3.13 with the following libraries:"
)

add_list_item("PyTorch 2.12.1 (CUDA 12.6) for all neural network operations.")
add_list_item("HuggingFace Transformers 5.12.1 for AraBERT v2 model loading.")
add_list_item("PyTorch Geometric for graph neural network operations.")
add_list_item("CAMeL Tools for Arabic-specific natural language processing.")
add_list_item("scikit-learn for evaluation metrics and TF-IDF vectorization.")
add_list_item("NetworkX and pandas for graph construction and data management.")

# ══════════════════════════════════════════════════════════
# 4.3 EXPERIMENT 1: ENCYCLOPEDIA CONSTRUCTION
# ══════════════════════════════════════════════════════════

doc.add_heading("4.3 Experiment 1: Narrator Encyclopedia Construction", level=1)

doc.add_heading("4.3.1 Objective", level=2)
add_paragraph(
    "The first experiment constructs a comprehensive narrator encyclopedia from two "
    "classical Arabic biographical sources. The goal is to extract structured "
    "narrator entries with grade terms, teacher-student relationships, and name variants."
)

doc.add_heading("4.3.2 Data Sources", level=2)
add_paragraph(
    "Two primary sources are used for encyclopedia construction:"
)
add_list_item(
    "Tahdhib al-Kamal fi Asma al-Rijal by al-Mizzi. This is a 35-volume biographical "
    "encyclopedia containing detailed entries for thousands of narrators. Each entry "
    "includes the narrator's full name, teachers, students, scholarly evaluations, "
    "and death information."
)
add_list_item(
    "Taqrib al-Tahdhib by Ibn Hajar al-Asqalani. This is a compressed single-volume "
    "abridgment containing concise entries with grade terms, tabaqa classification, "
    "and book codes indicating which canonical collections include each narrator."
)

doc.add_heading("4.3.3 Extraction Pipeline", level=2)
add_paragraph(
    "The extraction pipeline processes both sources through a shared architecture "
    "with source-specific parsers:"
)

add_bold_text("Volume Parsing and Entry Segmentation")
add_paragraph(
    "Raw text volumes are segmented into individual narrator entries using line-based "
    "regular expression detection. The entry header pattern matches lines starting with "
    "Arabic-Indic or Western digits followed by a dash separator. The parser "
    "(`narrator_parser.py`) identifies candidate boundaries, then extracts structured "
    "fields using targeted sub-parsers for teachers, students, grade terms, and death "
    "information."
)

add_bold_text("Taqrib al-Tahdhib Parsing")
add_paragraph(
    "The Taqrib parser (`taqrib_parser.py`) uses a purpose-built extraction routine. "
    "Taqrib entries follow a compressed single-line format where grade, tabaqa, death, "
    "and book codes are embedded within the same line. The parser extracts each field "
    "using prioritized term matching: composite grade terms are checked before "
    "single-word terms to avoid partial matches."
)

add_bold_text("Alias Index Construction")
add_paragraph(
    "An alias index is built from the extracted entries by generating multiple name "
    "variants per entry. Variants include the full normalized name, the first word (ism), "
    "the last word (usually nisba), extracted kunya patterns (Abu X), and father-son "
    "name chains. The alias index (`alias_index.json`) maps each variant to one or more "
    "entry IDs."
)

add_bold_text("Family Graph Construction")
add_paragraph(
    "A family relationship graph is constructed by extracting father-son patterns "
    "from entry names. The pattern \"Fulan bin Fulan\" identifies a father-son "
    "relationship. Father names that do not have their own encyclopedia entries are "
    "registered as virtual narrators, with their reliability scores estimated from "
    "the average scores of their known sons."
)

doc.add_heading("4.3.4 Key Parameters", level=2)
add_list_item("Volume range: 1 to 35 (Tahdhib) plus 1 (Taqrib).")
add_list_item("Confidence labels: High, Medium, and Low based on metadata completeness.")
add_list_item("Entry text cap: 3,000 characters per entry.")
add_list_item("Grade term priority: longest match first for composite terms.")

doc.add_heading("4.3.5 Expected Output", level=2)
add_paragraph(
    "The pipeline produces a unified narrator encyclopedia in JSON format containing "
    "structured entries with fields for entry ID, name, full text, grade terms, "
    "teachers, students, death information, and confidence score. Additional outputs "
    "include the alias index, the family graph, and per-volume extraction logs."
)

# ══════════════════════════════════════════════════════════
# 4.4 EXPERIMENT 2: 5-CLASS NARRATOR CLASSIFICATION
# ══════════════════════════════════════════════════════════

doc.add_heading("4.4 Experiment 2: Five-Class Narrator Classification", level=1)

doc.add_heading("4.4.1 Objective", level=2)
add_paragraph(
    "The second experiment trains a transformer-based classifier to place narrators "
    "into five discrete reliability categories. This replaces the regression-based "
    "approach that collapsed predictions toward the mean."
)

doc.add_heading("4.4.2 Experimental Question", level=2)
add_paragraph(
    "Can a fine-tuned AraBERT v2 model, trained with ordinal-weighted cross-entropy "
    "loss, correctly classify narrators into the five Ibn Hajar reliability tiers?"
)

doc.add_heading("4.4.3 Model Architecture", level=2)
add_paragraph(
    "The model uses AraBERT v2 (`aubmindlab/bert-base-arabertv2`) as the base "
    "encoder with a 5-class classification head. The five output classes correspond "
    "to the Ibn Hajar grading system: thiqah (4), saduq (3), maqbul (2), daif (1), "
    "and matruk (0)."
)

add_paragraph(
    "Class weights are computed using inverse frequency to compensate for the "
    "imbalanced training distribution. The weight for each class is calculated as "
    "the ratio of the maximum class count to that class's count."
)

doc.add_heading("4.4.4 Training Configuration", level=2)
add_list_item("Base model: aubmindlab/bert-base-arabertv2 (135 million parameters).")
add_list_item("Maximum sequence length: 256 tokens.")
add_list_item("Batch size: 16.")
add_list_item("Epochs: 10 (best validation F1 at epoch 7).")
add_list_item("Optimizer: AdamW with learning rate 2e-5.")
add_list_item("Loss function: CrossEntropyLoss with inverse-frequency weights.")
add_list_item("Hardware: RTX 2070 8 GB.")
add_list_item("Training time: approximately 22 minutes.")

doc.add_heading("4.4.5 Data Preparation", level=2)
add_paragraph(
    "Training data is drawn from the narrator encyclopedia constructed in Experiment 1. "
    "Entries with extracted grade terms are mapped to class labels via the "
    "GRADE_SCORE_MAP lookup table. Entries are split into 70% training, 15% validation, "
    "and 15% test sets using stratified sampling to preserve grade distribution."
)

doc.add_heading("4.4.6 Expected Output", level=2)
add_paragraph(
    "The experiment produces a trained classification model, a test set evaluation "
    "report with per-class precision, recall, and F1 scores, and a confusion matrix "
    "showing inter-class errors. The classifier is then applied to score all narrators "
    "in the knowledge graph, producing a narrator score file for downstream chain "
    "evaluation."
)

# ══════════════════════════════════════════════════════════
# 4.5 EXPERIMENT 3: KNOWLEDGE GRAPH AND GNN
# ══════════════════════════════════════════════════════════

doc.add_heading("4.5 Experiment 3: Full-Scale Knowledge Graph and GNN Training", level=1)

doc.add_heading("4.5.1 Objective", level=2)
add_paragraph(
    "The third experiment constructs a knowledge graph from all available Sanadset chains "
    "and trains a graph neural network to predict narrator reliability from structural "
    "features."
)

doc.add_heading("4.5.2 Graph Construction", level=2)
add_paragraph(
    "The knowledge graph is built from the full Sanadset 650K dataset. Each unique "
    "narrator name in the Sanad column becomes a node. Directed edges represent "
    "teacher-student transmission relationships: if narrator A appears immediately "
    "before narrator B in a chain, a directed edge is created from A to B. Edge features "
    "encode the position within the chain, the transmission frequency, and the default "
    "transmission method."
)

add_paragraph(
    "Node features include log-transformed appearance count, in-degree centrality, "
    "out-degree centrality, average chain position, the Ibn Hajar grade score (where "
    "available), and a binary geographic region flag."
)

doc.add_heading("4.5.3 GNN Architecture", level=2)
add_paragraph(
    "The GNN uses a Graph Convolutional Network (GCN) with three layers. Each layer "
    "consists of a GCNConv operation followed by batch normalization, ReLU activation, "
    "and dropout. The output head is a two-layer MLP that produces a scalar reliability "
    "score in the range [0.0, 1.0]."
)

add_paragraph(
    "Training uses a semi-supervised split: 70% of nodes are used for training, 15% "
    "for validation, and 15% for testing. The target label is the Ibn Hajar grade score "
    "from column 0 of the node features. The grade score is masked for validation and "
    "test nodes during training, forcing the model to infer reliability from structural "
    "features alone."
)

doc.add_heading("4.5.4 Hyperparameters", level=2)
add_list_item("Hidden channels: 64.")
add_list_item("Number of layers: 3.")
add_list_item("Dropout rate: 0.3.")
add_list_item("Learning rate: 0.001.")
add_list_item("Weight decay: 5e-4.")
add_list_item("Maximum epochs: 200 (early stopping at patience 30).")

doc.add_heading("4.5.5 Expected Output", level=2)
add_paragraph(
    "The experiment produces a trained GNN model, per-node reliability predictions "
    "for all narrators in the knowledge graph, and evaluation metrics including MAE, "
    "R-squared, and binary accuracy against the masked ground-truth grades."
)

# ══════════════════════════════════════════════════════════
# 4.6 EXPERIMENT 4: CHAIN SCORING METHODOLOGY
# ══════════════════════════════════════════════════════════

doc.add_heading("4.6 Experiment 4: Chain Scoring Methodology", level=1)

doc.add_heading("4.6.1 Objective", level=2)
add_paragraph(
    "The fourth experiment compares different chain aggregation methods and identifies "
    "the optimal approach for translating individual narrator scores into a single "
    "chain-level grade."
)

doc.add_heading("4.6.2 Aggregation Methods", level=2)
add_paragraph(
    "Four aggregation variants are tested against the Kutub as-Sitta evaluation set:"
)

add_bold_text("Method 1: Weakest-Link (Minimum)")
add_paragraph(
    "The chain score equals the minimum narrator score in the chain. This reflects "
    "the classical hadith principle that a chain is only as strong as its weakest "
    "narrator."
)

add_bold_text("Method 2: Average of Bottom Two (Avg-Bottom-2)")
add_paragraph(
    "The chain score equals the average of the two lowest narrator scores. This softens "
    "the strictest interpretation of weakest-link by preventing a single aberrant low "
    "score from determining the entire grade."
)

add_bold_text("Method 3: Weakest-Link with Outlier Skip")
add_paragraph(
    "If the minimum score is below 0.35 and the second minimum exceeds it by more than "
    "0.15, the second minimum is used instead. This treats extreme outliers as potential "
    "data errors."
)

add_bold_text("Method 4: Risk-Modified Scoring")
add_paragraph(
    "Each narrator score is multiplied by a modifier factor before aggregation. "
    "Modifiers are extracted from the biographical text and include tadlis (0.80), "
    "ikhtilat (0.50), kadhib (0.00), and jahala (0.30)."
)

doc.add_heading("4.6.3 Grade Thresholds", level=2)
add_paragraph(
    "Chain scores are mapped to grades using calibrated thresholds optimized against "
    "the Kutub as-Sitta evaluation set. The optimal thresholds are determined through "
    "grid search with 0.025 increments. Default thresholds are established at 0.45 for "
    "Sahih, 0.425 for Hasan, and 0.15 for Daif."
)

doc.add_heading("4.6.4 Expected Output", level=2)
add_paragraph(
    "The experiment yields a comparison table showing accuracy, Sahih recall, and Daif "
    "recall for each aggregation method across all 1,022 evaluation hadiths. The optimal "
    "method is selected for the final pipeline."
)

# ══════════════════════════════════════════════════════════
# 4.7 EXPERIMENT 5: HYBRID SCORING VS CANONICAL GRADES
# ══════════════════════════════════════════════════════════

doc.add_heading("4.7 Experiment 5: Hybrid Scoring Against Canonical Grades", level=1)

doc.add_heading("4.7.1 Objective", level=2)
add_paragraph(
    "The fifth experiment evaluates the complete framework against the ground-truth "
    "grades of the Kutub as-Sitta. Framework predictions are compared against canonical "
    "scholarly classifications to measure accuracy."
)

doc.add_heading("4.7.2 Evaluation Dataset", level=2)
add_paragraph(
    "The evaluation set consists of 1,200 hadiths drawn from the six canonical "
    "collections through stratified sampling (200 hadiths per book). The grades are "
    "normalized from Arabic classifications (Sahih, Hasan, Daif, Mawdu) to a uniform "
    "four-class system. After alignment to the Sanadset through TF-IDF matching, 1,022 "
    "evaluation hadiths are available for scoring."
)

doc.add_heading("4.7.3 Scoring Pipeline", level=2)
add_paragraph(
    "The scoring pipeline processes each evaluation hadith through the following steps:"
)
add_list_item("Parse the sanad string to extract individual narrator names.")
add_list_item("Look up each narrator in the hybrid score table. The hybrid score uses GRADE_SCORE_MAP for the 10,468 known entries, the trained classifier for 251,458 others, and the GNN mean as final fallback.")
add_list_item("Apply risk modifiers from the biographical text. If a narrator entry contains tadlis, ikhtilat, or kadhib indicators, the base score is multiplied by the corresponding modifier.")
add_list_item("Compute the chain score using the selected aggregation method. The avg-bottom-2 method averages the two lowest scores in the chain.")
add_list_item("Map the chain score to a grade using calibrated thresholds. Sahih requires a score above 0.45, Hasan above 0.425, Daif above 0.15, and Matruk for all others.")
add_list_item("Compare the predicted grade against the recorded canonical grade for that hadith.")

doc.add_heading("4.7.4 Evaluation Metrics", level=2)
add_list_item("Overall accuracy: the proportion of hadiths where the predicted grade matches the canonical grade exactly.")
add_list_item("Sahih recall: the proportion of Sahih-graded hadiths correctly identified as Sahih.")
add_list_item("Daif recall: the proportion of Daif-graded hadiths correctly identified as Daif.")
add_list_item("Per-book accuracy: accuracy computed separately for each of the six canonical collections.")
add_list_item("Relaxed accuracy: accuracy allowing predictions to be off by at most one grade level.")

doc.add_heading("4.7.5 Expected Output", level=2)
add_paragraph(
    "The experiment produces a confusion matrix comparing predicted grades against "
    "canonical grades, per-book accuracy breakdowns, and overall accuracy and F1 scores."
)

# ══════════════════════════════════════════════════════════
# 4.8 EXPERIMENT 6: TURUQ AND FAMILY GRAPH CORROBORATION
# ══════════════════════════════════════════════════════════

doc.add_heading("4.8 Experiment 6: Turuq and Family Graph Corroboration", level=1)

doc.add_heading("4.8.1 Objective", level=2)
add_paragraph(
    "The sixth experiment tests two corroboration mechanisms for improving chain "
    "grading. Multi-chain (turuq) corroboration uses the principle that a matn appearing "
    "in multiple independent chains should be graded by its best chain, not its weakest. "
    "Family graph resolution handles relational references such as abih (his father) "
    "by substituting the father's reliability score."
)

doc.add_heading("4.8.2 Multi-Chain Corroboration", level=2)
add_paragraph(
    "The system builds a hash-based index of all matn texts in the 650K Sanadset. "
    "For each evaluation hadith, the system searches the index for other occurrences "
    "of the same matn with different chains. If found, each chain is scored independently. "
    "The final grade uses the highest-scoring chain, provided the chains are sufficiently "
    "independent (sharing no more than 50% of narrators by Jaccard similarity)."
)

doc.add_heading("4.8.3 Family Graph Resolution", level=2)
add_paragraph(
    "The family graph constructed in Experiment 1 is used to resolve references to "
    "his father (abih) in sanad chains. When a chain position contains abih, the system "
    "looks up the previous narrator, finds their father through the father-son mapping, "
    "and substitutes the father's reliability score. If the father is a virtual narrator "
    "(no encyclopedia entry), the score is estimated from the average of his known sons."
)

doc.add_heading("4.8.4 Geospatial Verification", level=2)
add_paragraph(
    "The geospatial verification module extracts narrator locations from nisba patterns "
    "embedded in biographical names. A city coordinate database maps 30 historical Islamic "
    "cities to latitude-longitude pairs. Nisba patterns such as al-Kufi (from Kufa), "
    "al-Basri (from Basra), and al-Baghdadi (from Baghdad) are mapped to their "
    "respective cities."
)

add_paragraph(
    "For each consecutive narrator pair in a chain, the system computes the haversine "
    "distance between their known locations. If the distance exceeds 3,000 kilometers, "
    "a geographic penalty of 0.85 is applied to the chain score. The 3,000 kilometer "
    "threshold represents the maximum plausible travel distance for early Islamic "
    "scholars given contemporary transportation constraints."
)

add_paragraph(
    "The geographic verification serves two purposes. It flags implausible teaching "
    "relationships where a narrator from Khurasan could not have reasonably studied "
    "under a teacher in Egypt without documented travel. It also validates the geographic "
    "coherence of canonical collections, confirming that their compilers already performed "
    "this verification step."
)

doc.add_heading("4.8.5 650K Dataset Analysis", level=2)
add_paragraph(
    "The experiment also runs three descriptive analyses on the complete 650K chain "
    "dataset. Per-book reliability ranking computes the mean chain score for each book "
    "in the Sanadset. Weakest narrator frequency counts how often each narrator appears "
    "as the lowest-scoring link. Chain length analysis examines the correlation between "
    "chain length and average reliability."
)

doc.add_heading("4.8.6 Expected Output", level=2)
add_paragraph(
    "The experiment produces a list of hadiths whose grades improved through turuq "
    "corroboration, the resolution rate for abih references, per-book reliability "
    "rankings, the top weakest narrators by frequency, the chain length versus "
    "reliability correlation, and a count of geographic violations found across the "
    "full dataset with distance distributions."
)

# ══════════════════════════════════════════════════════════
# 4.9 EXPERIMENT 7: THEORETICAL CEILING AND GAP ANALYSIS
# ══════════════════════════════════════════════════════════

doc.add_heading("4.9 Experiment 7: Theoretical Ceiling and Gap Analysis", level=1)

doc.add_heading("4.9.1 Objective", level=2)
add_paragraph(
    "The seventh experiment establishes the theoretical maximum accuracy for "
    "chain-only analysis and measures how much of the remaining error is attributable "
    "to matn-level weakness (outside the project scope). This defines the performance "
    "ceiling and guides FYP2 priorities."
)

doc.add_heading("4.9.2 Methodology", level=2)
add_paragraph(
    "The evaluation set's Daif-graded hadiths are separated into two categories. "
    "Daif hadiths with weak chains (where the chain score falls below 0.45) are "
    "considered within scope for chain analysis. Daif hadiths with strong chains "
    "(where the chain score equals or exceeds 0.45 despite the canonical Daif grade) "
    "are classified as matn-weak. These hadiths have reliable narrators in their chains "
    "but are graded Daif due to textual defects, hidden anomalies, or contradictions "
    "with more authentic narrations."
)

add_paragraph(
    "The theoretical maximum accuracy for chain-only analysis is calculated as:"
)

add_paragraph(
    "Accuracy_max = (N_sahih + N_hasan + N_daif_chain) / Total"
)

add_paragraph(
    "where N_sahih is the number of Sahih-graded hadiths, N_hasan is the number of "
    "Hasan-graded hadiths, and N_daif_chain is the number of Daif-graded hadiths "
    "whose chain scores are below 0.45."
)

doc.add_heading("4.9.3 Key Parameters", level=2)
add_list_item("Strong chain threshold: score of 0.45 or higher for avg-bottom-2.")
add_list_item("Weak chain threshold: score below 0.45.")
add_list_item("Out of scope: Daif hadiths with strong chains (matn weakness per Section 1.5.4).")

doc.add_heading("4.9.4 Expected Output", level=2)
add_paragraph(
    "The experiment produces a ceiling analysis showing the theoretical maximum "
    "accuracy for chain-only analysis, the proportion of Daif hadiths attributable to "
    "matn weakness, and the framework's efficiency as a percentage of the theoretical "
    "maximum. A table summarizes which error categories are addressable in FYP2 versus "
    "those that remain out of scope."
)

# ══════════════════════════════════════════════════════════
# 4.10 SUMMARY
# ══════════════════════════════════════════════════════════

doc.add_heading("4.10 Summary", level=1)

add_paragraph(
    "Seven experiments were designed to test the three research objectives. The "
    "experiments follow the pipeline's natural data flow: data preparation and "
    "encyclopedia construction (Experiment 1), narrator-level classification "
    "(Experiment 2), knowledge graph and GNN training (Experiment 3), chain aggregation "
    "methodology (Experiment 4), full pipeline evaluation against canonical grades "
    "(Experiment 5), multi-chain and family corroboration (Experiment 6), and "
    "theoretical ceiling analysis (Experiment 7)."
)

add_paragraph(
    "Experiment 1 establishes the foundational data layer. Experiments 2 and 3 "
    "build the narrator scoring components using textual and structural modalities. "
    "Experiments 4 and 5 test different chain aggregation approaches and evaluate the "
    "complete scoring pipeline against established scholarly grades. Experiment 6 tests "
    "two refinement mechanisms, and Experiment 7 establishes the theoretical limits "
    "of the chain-only approach."
)

add_paragraph(
    "Each experiment specifies a clear question, defined parameters, measurable outputs, "
    "and the connection to the specific research objective it addresses. The next chapter "
    "presents the execution of these experiments and the preliminary results."
)

# ── Save ──
output_path = r"c:\Projects\FYP Project\Report Drafts\chapter4_experimental_design_v2.docx"
doc.save(output_path)
print(f"Saved: {output_path}")
