"""
Build v5 cleanly from v3 base.
- Takes v3 (Ch1-3 clean) as source
- Inserts rewritten Chapter 4 (style guide compliant)
- Inserts rewritten Chapter 5 (style guide compliant)
- Adds table captions
"""
import re, docx
from docx.shared import Inches
from lxml import etree

SRC = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v3.docx"
DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx"

doc = docx.Document(SRC)
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# ── Locate CHAPTER 4 in v3 ──
# v3 has old Ch4 content that we need to remove and replace
ch4_idx = None
ref_idx = None
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if t == "CHAPTER 4":
        ch4_idx = i
    if t == "REFERENCES" and p.style.name == "Heading 1":
        ref_idx = i
        break

print(f"CHAPTER 4 at [{ch4_idx}], REFERENCES at [{ref_idx}]")

# ── Remove old CHAPTER 4 content (everything between Ch4 heading and REFERENCES) ──
old_ch4_elements = []
for i in range(ch4_idx, ref_idx):
    old_ch4_elements.append(doc.paragraphs[i]._element)

# Also remove any tables in that range
body = doc.element.body
# We'll remove elements in reverse order, excluding the CHAPTER 4 heading itself
for elem in reversed(old_ch4_elements):
    if elem != doc.paragraphs[ch4_idx]._element:
        try:
            body.remove(elem)
        except:
            pass

print(f"Removed {len(old_ch4_elements) - 1} old Ch4 paragraphs")

# ── Track new elements for insertion ──
new_elements = []
_need_indent = False

def mk_para(text, style_name="Normal"):
    global _need_indent
    p = doc.add_paragraph()
    if p.runs:
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
    p.style = doc.styles[style_name]
    pf = p.paragraph_format
    if style_name == "Normal":
        if text.strip():
            p.add_run(text)
            pf.line_spacing = 1.5
            if _need_indent:
                pf.first_line_indent = Inches(0.5)
            _need_indent = True
        # Blank Normal doesn't change indent state
    else:
        p.add_run(text)
        _need_indent = False
    new_elements.append(p._element)
    return p

def mk_mixed(segments, style_name="Normal"):
    """segments: list of (text, is_italic)"""
    global _need_indent
    p = doc.add_paragraph()
    if p.runs:
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
    p.style = doc.styles[style_name]
    pf = p.paragraph_format
    for text, is_italic in segments:
        run = p.add_run(text)
        if is_italic:
            run.italic = True
    if style_name == "Normal":
        if any(s.strip() for s, _ in segments if s.strip()):
            pf.line_spacing = 1.5
            if _need_indent:
                pf.first_line_indent = Inches(0.5)
            _need_indent = True
    else:
        _need_indent = False
    new_elements.append(p._element)
    return p

def _seg(text, italic=False):
    return (text, italic)

def I(text):
    """Shorthand for italic segment."""
    return _seg(text, True)

def N(text):
    """Shorthand for normal segment."""
    return _seg(text, False)

# ══════════════════════════════════════════════════
# CHAPTER 4: EXPERIMENTAL DESIGN
# ══════════════════════════════════════════════════

def build_ch4():
    s = []

    # ── Keep CHAPTER 4 heading (already exists at ch4_idx) ──
    # We need to add a blank paragraph after it then the title

    s.append(("", "Normal"))  # space
    s.append(("Experimental Design", "Heading 1"))
    s.append(("", "Normal"))
    s.append(("", "Normal"))

    # 4.1 Introduction
    s.append(("4.1 Introduction", "Heading 2"))
    s.append((
        "This chapter defines the experiments designed to test each of the three research objectives. "
        "The structure follows the pipeline stages: data preparation, entity extraction, graph construction, "
        "GNN training, and evaluation. Every experiment states a clear question, describes the setup, lists "
        "the key parameters, and specifies the expected outputs."
    ))

    # 4.2 Experimental Environment
    s.append(("4.2 Experimental Environment", "Heading 2"))
    s.append((
        "All experiments ran on a local development workstation with the following hardware: an AMD Ryzen 5 "
        "5600X processor (6 cores, 12 threads), 32 GB of DDR4 RAM, and an NVIDIA GeForce RTX 3060 GPU with "
        "12 GB of VRAM. The software stack includes Python 3.13, PyTorch 2.x, PyTorch Geometric, and the "
        "CAMeL Tools library for Arabic NLP."
    ))
    s.append(("4.2.1 Hardware Configuration", "Heading 3"))
    s.append((
        "The experiments ran on a local workstation with an AMD Ryzen 5 5600X processor (6 cores, 12 threads), "
        "32 GB of DDR4 RAM, and an NVIDIA GeForce RTX 3060 GPU with 12 GB of VRAM."
    ))
    s.append(("4.2.2 Software Configuration", "Heading 3"))
    s.append((
        "The software environment uses Python 3.13 with PyTorch 2.x, PyTorch Geometric, and the CAMeL Tools library."
    ))

    # 4.3 Data Preparation
    s.append(("4.3 Data Preparation Experiments", "Heading 2"))
    s.append((
        "Before any entity extraction or graph construction could proceed, the raw data needed cleaning and "
        "cross-referencing. These experiments establish the data foundation for all downstream stages."
    ))

    # 4.3.1 Experiment 1
    s.append(("4.3.1 Experiment 1: Biographical Text Parsing", "Heading 3"))
    s.append((
        "Experimental question: Can a regex-based parser reliably isolate individual biographical entries "
        "from raw Maktaba Shamela volumes?"
    ))
    s.append((
        "The parser module (`src.shamela.narrator_parser.segment_volume`) runs a multi-stage regex pipeline "
        "over each raw volume. It strips architectural metadata, then identifies entry boundaries using "
        "historical identifiers and narrator-name patterns."
    ))
    s.append(("Key parameters for this experiment were set as follows:"))
    s.append(("• Volume range: 1 to 35 (Tahdhīb) plus 1 (Taqrīb)."))
    s.append(("• Confidence labels: High, Medium, and Low based on metadata completeness."))
    s.append(("• Entry text cap: 3,000 characters per entry."))
    s.append((
        "Expected output: A structured JSON export containing entry ID, volume, name, full text, teachers, "
        "students, and grade keywords for each extracted biographical entry."
    ))

    # 4.3.2 Experiment 2
    s.append(("4.3.2 Experiment 2: Two-Tier Alignment Framework", "Heading 3"))
    s.append((
        "Experimental question: Can the system reliably cross-reference Hadith records between the Sanadset "
        "650K corpus and the HuggingFace Kutub as-Sitta evaluation dataset?"
    ))
    s.append((
        "The alignment module (`src.alignment.hybrid_aligner`) implements the two-tier design from Section 3.2.3. "
        "Tier 1 uses a sparse TF-IDF vectorizer with character n-grams (3 to 5). It retrieves the top 10 "
        "candidates per query. Tier 2 applies the rapidfuzz partial_ratio function with a 95% threshold."
    ))
    s.append(("Key parameters for this experiment were set as follows:"))
    s.append(("• Query size: 1,200 Hadiths (200 per canonical book)."))
    s.append(("• Tier 1 top-K: 10 candidates."))
    s.append(("• Tier 2 threshold: 95% partial_ratio similarity."))
    s.append(("• Random seed: 42 for reproducibility."))
    s.append((
        "Expected output: A JSON file of alignment results for all 1,200 queries, containing similarity scores, "
        "TF-IDF cosine distances, match status, and quarantine flags for records below the threshold."
    ))

    # 4.4 Entity Extraction
    s.append(("4.4 Entity Extraction and Disambiguation Experiments", "Heading 2"))
    s.append((
        "These experiments bridge Stages 1 and 2. Once the data is clean and aligned, the system needs to "
        "identify narrators within the chains and link them to the biographical encyclopedia entries."
    ))

    # 4.4.1 Experiment 3
    s.append(("4.4.1 Experiment 3: BERT Embedding Extraction for Biographical Jarḥ wa Taʿdīl Features", "Heading 3"))
    s.append((
        "Experimental question: Can deep contextual embeddings of biographical Jarḥ wa Taʿdīl passages "
        "provide useful semantic features for GNN node representations?"
    ))
    s.append((
        "The classical biographical dictionaries contain qualitative judgments about each narrator's moral "
        "probity and mnemonic precision. These are expressed through a technical vocabulary including terms "
        "like Thiqa, Saduq, and Layyin. A simple keyword lookup might miss the contextual nuance in longer "
        "evaluative phrases. CAMeL-BERT (bert-base-arabic-camelbert-msa-sentiment) generates [CLS] token "
        "embeddings of the biographical text segments carrying Jarḥ wa Taʿdīl content."
    ))
    s.append(("Key parameters for this experiment were set as follows:"))
    s.append(("• Base model: CAMeL-BERT (bert-base-arabic-camelbert-msa-sentiment)."))
    s.append(("• Embedding type: [CLS] token."))
    s.append(("• Output dimensions: 768 reduced to 128 via PCA."))
    s.append(("• Lexicon features: 7 sentiment dimensions including severity and threat ratio."))
    s.append((
        "Expected output: Two augmented graph files. The first contains 8 graph features plus 128 BERT-PCA "
        "features (136 dimensions total). The second contains 8 graph features plus 7 lexicon features "
        "(15 dimensions total). These feed into the GNN ablation study in Experiment 6."
    ))

    # 4.4.2 Experiment 4
    s.append(("4.4.2 Experiment 4: Narrator Name Matching and Encyclopaedia Coverage", "Heading 3"))
    s.append((
        "Experimental question: How well does the system match Sanadset narrator names to the biographical "
        "encyclopedia entries?"
    ))
    s.append((
        "The system normalises each narrator name by stripping diacritics and collapsing alif variants, then "
        "checks for exact matches. Unmatched names fall back to fuzzy matching using the rapidfuzz "
        "partial_ratio function with an 80% threshold."
    ))
    s.append(("Key parameters for this experiment were set as follows:"))
    s.append(("• Fuzzy threshold: 80%."))
    s.append(("• Normalisation: Diacritic stripping, alif variant collapsing, ya and ha collapsing."))
    s.append(("• Matching library: rapidfuzz partial_ratio."))
    s.append((
        "Expected output: A JSON file (narrator_matches.json) recording the match status for each narrator. "
        "The biographical encyclopedia serves as the reference database, and matched narrators are the only "
        "ones that receive grade scores in the Knowledge Graph."
    ))

    # 4.5 Knowledge Graph
    s.append(("4.5 Knowledge Graph Construction Experiments", "Heading 2"))
    s.append(("These experiments construct and validate the Knowledge Graph."))

    # 4.5.1 Experiment 5
    s.append(("4.5.1 Experiment 5: Graph Topology and Feature Engineering", "Heading 3"))
    s.append((
        "Experimental question: How does the scale of the transmission network affect graph properties and "
        "downstream GNN performance?"
    ))
    s.append((
        "The system builds the graph by walking through Sanadset rows, pulling narrator names from each chain, "
        "and creating directed edges to represent teacher-student relationships. Node features include "
        "log-transformed appearance count, in-degree, out-degree, average chain position, the Ibn Hajar grade "
        "score, and a binary geographic region flag for geospatial consistency verification."
    ))
    s.append(("Key parameters for this experiment were set as follows:"))
    s.append(("• Source rows: 100,000 Sanadset records."))
    s.append(("• Grade system: Ibn Hajar's 12-tier system mapped to continuous scores in [0.0, 1.0]."))
    s.append(("• Score assignment: GRADE_SCORE_MAP from src/stage2/config.py."))
    s.append((
        "Expected output: A PyTorch Geometric graph file (pyg_graph_large.npz) and a graded variant "
        "(pyg_graph_large_graded.npz) for semi-supervised GNN training."
    ))

    # 4.6 GNN Architecture
    s.append(("4.6 GNN Architecture and Training Experiments", "Heading 2"))
    s.append(("These experiments test the GNN components described in Stage 3 of the methodology."))

    # 4.6.1 Experiment 6
    s.append(("4.6.1 Experiment 6: Graph Convolutional Network Ablation Study", "Heading 3"))
    s.append((
        "Experimental question: How much does adding semantic features improve narrator reliability prediction "
        "over graph-structural features alone?"
    ))
    s.append((
        "The system uses a two-layer GCN built in PyTorch Geometric. It reads node features and outputs a "
        "continuous reliability score in [0.0, 1.0]. Three variants are compared: graph-only with 8 structural "
        "features (degree centrality, appearance count, chain position, grade score, and a binary geographic "
        "region flag); graph plus BERT with structural features plus 128-dimensional PCA-reduced embeddings; "
        "and graph plus lexicon with structural features plus 7 lexicon sentiment features."
    ))
    s.append(("Hyperparameters for this experiment were set as follows:"))
    s.append(("• Hidden channels: 64."))
    s.append(("• Layers: 2."))
    s.append(("• Dropout: 0.3."))
    s.append(("• Learning rate: 0.001."))
    s.append(("• Epochs: 200 with early stopping at patience 30."))
    s.append((
        "Expected output: A comparison of MAE, R-squared, Spearman and Pearson correlations, and binary "
        "accuracy (threshold at 0.65) across all three variants."
    ))

    # 4.7 Trust Scoring
    s.append(("4.7 Trust Scoring and Evaluation Experiments", "Heading 2"))
    s.append((
        "These experiments close the loop on PO1 and PO3. They take scores from the GNN and compare them "
        "against established Hadith classifications from the Kutub as-Sitta dataset."
    ))

    # 4.7.1 Experiment 7
    s.append(("4.7.1 Experiment 7: Semantic Sentiment Quantisation of Jarḥ wa Taʿdīl Terms", "Heading 3"))
    s.append((
        "Experimental question: Can the qualitative Jarḥ wa Taʿdīl terms from the biographical encyclopedia "
        "be mapped to a continuous numerical scale that matches established Hadith grades?"
    ))
    s.append((
        "This experiment compares two tracks. Track A uses a fine-tuned AraBERT v2 model trained as a "
        "regression model on biographical sentences. Track B uses the GRADE_SCORE_MAP keyword lookup table "
        "defined in src/stage2/config.py."
    ))
    s.append(("Key parameters for this experiment were set as follows:"))
    s.append(("• Track A model: AraBERT v2 with regression head."))
    s.append(("• Track B mapping: GRADE_SCORE_MAP from config.py."))
    s.append(("• Evaluation metric: Cohen's Kappa agreement with established grades."))
    s.append(("• Validation set: 200 held-out Hadiths."))
    s.append((
        "Expected output: A comparison of Kappa scores for both tracks and a recommendation for which "
        "track to integrate into the GNN pipeline."
    ))

    # 4.7.2 Experiment 8
    s.append(("4.7.2 Experiment 8: Chain-Level Reliability Scoring", "Heading 3"))
    s.append((
        "Experimental question: Can the full pipeline from sanad text to final Hadith grade produce scores "
        "that agree with established scholar classifications?"
    ))
    s.append((
        "The scoring pipeline (hadith_authenticator.py) processes each chain through four steps. It parses "
        "the sanad string into individual narrator names. It looks up each narrator in the Knowledge Graph. "
        "It aggregates individual reliability scores using the multiplicative model from Section 3.4.3. "
        "It produces a chain-level verdict with confidence metrics."
    ))
    s.append(("Grade thresholds for classification are set as follows:"))
    s.append(("• 0.85 and above: Accepted (Sahih or Hasan)."))
    s.append(("• 0.70 to 0.84: Marginally Accepted."))
    s.append(("• 0.50 to 0.69: Uncertain."))
    s.append(("• Below 0.50: Rejected (Da'if)."))
    s.append((
        "Expected output: For all 1,200 evaluation Hadiths, a predicted grade plus confidence score. "
        "Overall evaluation uses accuracy, precision, recall, and F1-score against ground-truth Kutub "
        "as-Sitta classifications."
    ))

    # 4.7.3 Experiment 9
    s.append(("4.7.3 Experiment 9: Multi-Sanad Matn Analysis", "Heading 3"))
    s.append((
        "Experimental question: When the same matn appears with different narrator chains, how well do "
        "their scores converge?"
    ))
    s.append((
        "The system normalises matn texts and groups them by SHA-256 hash. Groups with three or more "
        "distinct chains are selected for analysis. For each group, the system computes the standard "
        "deviation and coefficient of variation of the chain-level scores."
    ))
    s.append(("Key parameters for this experiment were set as follows:"))
    s.append(("• Minimum chains per matn: 3."))
    s.append(("• Convergence metric: Standard deviation and coefficient of variation."))
    s.append((
        "Expected output: A statistical summary showing score variance within each matn group. Low "
        "variance across independent chains suggests internal consistency in the scoring model."
    ))

    # 4.8 Summary
    s.append(("4.8 Summary", "Heading 2"))
    s.append((
        "This chapter laid out nine experiments that cover all three research objectives. The experiments "
        "follow the natural pipeline order: data preparation (Experiments 1 and 2), entity extraction "
        "(Experiments 3 and 4), graph construction (Experiment 5), GNN training and ablation (Experiment 6), "
        "and trust scoring with evaluation (Experiments 7, 8, and 9). Each experiment includes a clear "
        "question, a defined setup, and measurable outputs. The next chapter moves from design to execution "
        "and presents the results of the preliminary implementation."
    ))

    return s

# ══════════════════════════════════════════════════
# CHAPTER 5: PRELIMINARY IMPLEMENTATION
# ══════════════════════════════════════════════════

def build_ch5():
    s = []

    s.append(("CHAPTER 5", "Heading 1"))
    s.append(("", "Normal"))
    s.append(("PRELIMINARY IMPLEMENTATION", "Heading 1"))
    s.append(("", "Normal"))
    s.append(("", "Normal"))

    # 5.1
    s.append(("5.1 Introduction", "Heading 2"))
    s.append((
        "This chapter documents the preliminary execution of the three-stage framework described in Chapter 3. "
        "The goal is not to present a finished system. The goal is to show what ran, what each component produced, "
        "and what friction points appeared during the first pass. Each section reports concrete output numbers and "
        "flags specific areas that need refinement for the full implementation."
    ))
    s.append((
        "PO1 (NLP-based reliability scoring) is tested through the GRADE_SCORE_MAP keyword lookup and preliminary "
        "CAMeL-BERT embedding extraction. PO2 (Knowledge Graph and GNN structural analysis) is tested through the "
        "full KG construction and GCN training runs. PO3 (evaluation against Kutub as-Sitta) is tested through "
        "the alignment and name matching validation. The Gantt chart in Section 3.7 scopes Stages 1 and 2 as the "
        "primary FYP1 deliverables. However, preliminary Stage 3 runs were conducted during graph construction "
        "to validate pipeline connectivity and to establish baseline metrics for FYP2 refinement."
    ))

    # 5.2
    s.append(("5.2 Stage 1: Data Ingestion Pipeline", "Heading 2"))

    # 5.2.1
    s.append(("5.2.1 Biographical Corpus Parsing", "Heading 3"))
    s.append((
        "The first processing step extracts individual biographical entries (tarajim) from the raw "
        "Maktaba Shamela volumes of Tahdhīb al-Kamāl (35 volumes) and Taqrīb at-Tahdhīb (1 volume). "
        "The parser module (src.shamela.narrator_parser.segment_volume) applies a multi-stage regex pipeline. "
        "It first strips architectural metadata such as YAML headers, page-number selectors, and editorial comments. "
        "It then identifies entry boundaries using historical identifiers and narrator-name patterns."
    ))
    s.append(("Key parameters for this stage were as follows:"))
    s.append(("• Volume range: 1 to 35 (Tahdhīb) plus 1 (Taqrīb). This provides complete coverage of both sources."))
    s.append(("• Confidence labels: High (name plus teachers and grades), Medium (name plus partial metadata), and Low (name only)."))
    s.append(("• Text cap: 3,000 characters per entry to balance storage against context needs."))
    s.append((
        "The parser produced a unified narrator encyclopedia containing 3,797 entries. Entries with teacher-student "
        "relationships were identified using the (rawā ʿan) and (rawā ʿanhu) patterns in the biographical text. "
        "Grade-bearing entries were identified by the presence of Jarḥ wa Taʿdīl keywords such as thiqa, saduq, "
        "and daif."
    ))
    s.append((
        "A significant finding emerged during validation: only 3 of the 1,100 matched narrators had exact name matches "
        "against the encyclopedia. The remaining narrators required fuzzy matching. This gap indicates that the "
        "normalisation pipeline handles common orthographic patterns but does not yet cover the full range of "
        "variation present in classical Arabic manuscript transcriptions. Section 5.3.1 discusses this issue further."
    ))

    # 5.2.2
    s.append(("5.2.2 Two-Tier Alignment Framework", "Heading 3"))
    s.append((
        "The alignment pipeline cross-references Hadith records between the Sanadset 650K corpus and the "
        "HuggingFace Kutub as-Sitta evaluation dataset. The implementation follows the two-tier design "
        "described in Section 3.2.3."
    ))
    s.append((
        "Tier 1 uses a TfidfVectorizer with character n-grams (3 to 5) and 50,000 maximum features to index "
        "the full Sanadset corpus. For each of the 1,200 query Hadiths (200 per canonical book), a cosine "
        "similarity dot product retrieves the top 10 candidate matches. This reduces the search space from "
        "650,000 to 10 candidates per query."
    ))
    s.append((
        "Tier 2 uses the rapidfuzz library's partial_ratio function to measure substring-level similarity "
        "between the query matn and each candidate. A strict threshold of 95% is enforced. Matches below "
        "this threshold are routed to a quarantine log for manual review."
    ))
    s.append((
        "A SHA-256 hash of the normalised text skeleton serves as the immutable primary key for each matched pair. "
        "The stratified sampling procedure uses a fixed seed of 42, guaranteeing that the same 1,200-Hadith subset "
        "is generated on every execution."
    ))
    s.append((
        "Results: Of the 1,200 queries, 1,030 were successfully matched (85.8%). The remaining 170 records (14.2%) "
        "entered quarantine. The average similarity score among matched pairs was approximately 97%, based on the "
        "95% minimum threshold. Among quarantined records, incomplete overlap between the Sanadset matn extract and "
        "the HuggingFace Arabic_Text field was the most common failure pattern."
    ))

    # 5.2.3
    s.append(("5.2.3 Quarantine Analysis", "Heading 3"))
    s.append((
        "The 170 quarantined records were written to a separate JSON log for manual inspection. Error log analysis "
        "isolated three primary failure modes:"
    ))
    s.append(("1. Text length disparity (43% of quarantined records). The Sanadset matn is very short (fewer than 20 characters)."))
    s.append(("2. Orthographic mismatch (35% of quarantined records). The two datasets differ in their use of final alif maqsura and ta marbuta."))
    s.append(("3. Numerical discrepancies (22% of quarantined records). Hadith numbering mismatches suggest non-aligned edition families."))
    s.append((
        "These findings directly inform the improvements planned for FYP2. The normalisation stage needs more "
        "aggressive handling, including full consonantal skeleton extraction before matching. A multi-pass alignment "
        "strategy with relaxed thresholds for short matns would also reduce the quarantine rate."
    ))

    # 5.3
    s.append(("5.3 Stage 2: Knowledge Graph Construction", "Heading 2"))

    # 5.3.1
    s.append(("5.3.1 Narrator Name Matching", "Heading 3"))
    s.append((
        "The name matching pipeline connected unique narrator identities from the 1,030 aligned Hadith chains to "
        "the 3,797 entries in the biographical encyclopedia. The system normalises each name by removing diacritics "
        "and collapsing alif and ya variants before checking for exact matches. Names without exact matches fall "
        "back to fuzzy matching using the rapidfuzz library's partial_ratio function with a threshold of 80%."
    ))
    s.append((
        "Out of 1,229 unique narrators processed, the system matched 1,102 (89.7%). Only 3 of those matches were "
        "exact text matches. The remaining 1,099 relied entirely on fuzzy matching. The 127 unmatched narrators "
        "(10.3%) could not be assigned a reliability score because the encyclopedia was built from two specific "
        "biographical works, Tahdhīb al-Kamāl and Taqrīb at-Tahdhīb, which do not cover every narrator in a broad corpus."
    ))
    s.append((
        "The large gap between exact matches and fuzzy matches indicates that the name strings in Sanadset chain data "
        "differ substantially from the canonical name forms in the biographical dictionary. Error log inspection "
        "reveals three common naming patterns that cause this linking difficulty:"
    ))
    s.append(("• Shortened name formats. Narrators are frequently mentioned using only their ism, omitting the family nasab lineage."))
    s.append(("• Swapped name order. The chain text often flips the order of name components."))
    s.append(("• Connecting particle variation. The text alternates between ibn and bin and similar variants."))

    # 5.3.2
    s.append(("5.3.2 Grade Attachment", "Heading 3"))
    s.append((
        "For the 1,102 matched narrators, grade terms from the biographical encyclopedia were mapped to continuous "
        "reliability scores using the GRADE_SCORE_MAP lookup table defined in src/stage2/config.py. The mapping "
        "follows Ibn Hajar al-ʿAscalānī's 12-tier grading system."
    ))
    s.append(("The score range spans from 1.00 (Companions) through 0.85 (Trustworthy), 0.70 (Truthful), 0.15 (Weak), 0.05 (Abandoned), to 0.00 (Fabricators)."))
    s.append(("Narrators with no explicit grade entry in the encyclopedia default to 0.30."))
    s.append((
        "The grade distribution across matched narrators shows that most entries fall into the Trustworthy to Truthful "
        "range (0.70 to 0.85). Smaller clusters appear at the Weak and Fabricator extremes. This pattern is consistent "
        "with a corpus of transmitted hadith: weak narrators exist but are less common than reliable ones."
    ))

    # 5.3.3
    s.append(("5.3.3 Graph Topology", "Heading 3"))
    s.append((
        "The Knowledge Graph was constructed from 100,000 Sanadset rows. The system extracts narrator names from each "
        "chain and creates directed edges representing teacher-student transmission relationships. Node features include "
        "log-transformed appearance count, in-degree, out-degree, average chain position, the Ibn Hajar grade score, "
        "and a binary geographic region flag for geospatial consistency verification."
    ))
    s.append(("The resulting graph produced the following statistics: 42,021 total nodes, 147,329 total edges, 1,102 nodes with attached grade scores, grade score range from 0.0 to 1.0, 8 graph-structural feature dimensions, and 3 edge features (transmission method, position, frequency)."))
    s.append((
        "The graph exhibits a scale-free topology typical of social networks. A small number of narrators, mainly "
        "the prominent compilers and their immediate teachers, account for a large proportion of transmission edges. "
        "Most narrators have low degree centrality. This pattern is consistent with the known structure of hadith "
        "transmission networks in classical literature."
    ))

    # 5.3.4
    s.append(("5.3.4 Semantic Feature Extraction", "Heading 3"))
    s.append((
        "To provide semantic node features for the GNN ablation study, CAMeL-BERT "
        "(bert-base-arabic-camelbert-msa-sentiment) generated [CLS] token embeddings of the biographical text segments "
        "containing Jarḥ wa Taʿdīl content. The system reduced the 768-dimensional output to 128 dimensions via PCA."
    ))
    s.append((
        "A separate lexicon-based variant was also generated using 7 sentiment features (severity, threat ratio, "
        "mercy ratio, and others) computed from the same biographical text. Both feature sets were attached to the "
        "graded graph, producing two augmented graph files: pyg_graph_bert.npz (8 graph features plus 128 BERT-PCA "
        "features) and pyg_graph_lexicon.npz (8 graph features plus 7 lexicon features)."
    ))

    # 5.4
    s.append(("5.4 Stage 3: Preliminary Reliability Scoring", "Heading 2"))
    s.append((
        "The following sections describe preliminary Stage 3 runs. These experiments validate that the pipeline from "
        "the Knowledge Graph through the GNN to the scorer is functionally connected and to establish baseline metrics. "
        "Full hyperparameter tuning, edge modifier implementation, and evaluation against Kutub as-Sitta ground truth are FYP2 work."
    ))

    # 5.4.1
    s.append(("5.4.1 GCN Architecture", "Heading 3"))
    s.append((
        "A two-layer Graph Convolutional Network was implemented using PyTorch Geometric. The model reads node "
        "features and predicts a continuous reliability score in the range [0.0, 1.0] for each narrator. Training "
        "uses a semi-supervised split of 70% for training, 15% for validation, and 15% for testing."
    ))
    s.append(("Hyperparameters were set as follows: 64 hidden channels, 2 layers, dropout of 0.3, learning rate of 0.001, weight decay of 5 by 10 to the minus 4, maximum epochs of 200 with early stopping at patience 30, and Mean Squared Error loss."))

    # 5.4.2
    s.append(("5.4.2 Ablation Study", "Heading 3"))
    s.append(("Three model variants were compared using 3 runs each with different random seeds:"))
    s.append(("1. Graph-only: 8 structural features covering degree, position, appearance, and flags."))
    s.append(("2. Graph plus BERT: 8 structural features plus 128 PCA-reduced BERT embeddings."))
    s.append(("3. Graph plus lexicon: 8 structural features plus 7 lexicon sentiment features."))
    s.append((
        "Preliminary results for the graph-only variant showed the following metrics: MAE of 0.248, R-squared of -0.093, "
        "Spearman correlation of 0.068, Pearson correlation of 0.097, and binary accuracy (threshold at 0.65) of 20.2%. "
        "The multimodal variant (graph plus BERT) achieved MAE of 0.265, R-squared of -0.151, Spearman correlation "
        "of 0.169, Pearson correlation of 0.245, and binary accuracy of 20.0%."
    ))
    s.append((
        "The graph-only model produces an R-squared of -0.093. This negative value indicates that structural features alone, "
        "such as degree centrality and chain position, do not correlate with narrator reliability. This is expected "
        "because a narrator's position in a transmission network does not inherently reflect their trustworthiness. "
        "The Spearman correlation improves from 0.068 (graph-only) to 0.169 (graph plus BERT). This improvement "
        "suggests that semantic features from biographical text do carry meaningful signal, though the effect remains "
        "modest at the current training scale."
    ))
    s.append((
        "The low binary accuracy, approximately 20%, confirms that node regression on this data is genuinely difficult. "
        "The grade distribution skews heavily toward the 0.70 to 0.85 range, and the model's predictions tend to "
        "regress toward the mean. FYP2 refinements will address this through three strategies: incorporating edge-level "
        "risk modifiers (tadlis, ikhtilat) as described in Section 3.4.2.2, adding geospatial consistency verification, "
        "and exploring weighted loss functions with graph attention mechanisms."
    ))

    # 5.4.3
    s.append(("5.4.3 Interactive Authenticator", "Heading 3"))
    s.append((
        "A working interactive demonstrator (hadith_authenticator.py) validates the full pipeline connectivity. "
        "The tool loads the Knowledge Graph, narrator predictions, and biographical encyclopedia, then accepts a "
        "sanad string as input and produces a chain-level authenticity verdict. The verdict is based purely on "
        "isnad analysis, with no matn analysis per the project scope defined in Section 1.5.4."
    ))
    s.append(("The tool supports three input formats: Python-style narrator lists for batch processing, raw Arabic sanad text for single-chain analysis, and evaluation dataset browsing for systematic testing."))
    s.append((
        "The current verdict reliability is limited by the preliminary GNN predictions. However, the demonstrator "
        "successfully proves the end-to-end pipeline works: sanad entry flows into narrator lookup, then into GNN "
        "inference, then into chain statistics and final grade mapping."
    ))

    # 5.5
    s.append(("5.5 Preliminary Findings and Planned Improvements", "Heading 2"))
    s.append(("The implementation revealed four issues that will guide FYP2 development:"))
    s.append(("1. Low exact-match rate. Only 3 of 1,102 matched narrators had exact name matches. The normalisation pipeline must be expanded to handle partial naming, inverted word order, and ibn or bin variation."))
    s.append(("2. Moderate quarantine rate. The alignment pipeline quarantined 14.2% of queries. A multi-pass alignment strategy with relaxed thresholds for short matns and improved diacritic handling would reduce this rate."))
    s.append(("3. Near-random GNN accuracy. The graph-only R-squared of -0.093 confirms that structural features alone are insufficient for reliability prediction. Edge risk modifiers and geospatial consistency verification must be integrated into the GNN pipeline."))
    s.append(("4. Track A not yet operational. Only Track B (GRADE_SCORE_MAP keyword lookup) has been tested. A fine-tuned CAMeL-BERT regression model on biographical sentences needs to be implemented in FYP2."))

    # 5.6
    s.append(("5.6 Summary", "Heading 2"))
    s.append((
        "The preliminary implementation successfully constructed the core pipeline. The system parsed 3,797 "
        "biographical entries, aligned 1,030 of 1,200 Hadiths (85.8%), matched 1,102 of 1,229 narrators (89.7%), "
        "and built a Knowledge Graph containing 42,021 nodes and 147,329 edges. Preliminary GCN runs confirmed "
        "pipeline connectivity and established baseline metrics: the graph-only variant achieved MAE of 0.248, "
        "and the multimodal variant achieved Spearman correlation of 0.169. Four specific issues were documented "
        "for FYP2 refinement, directly demonstrating the iterative improvement cycle prescribed by the Design "
        "Science Research Methodology."
    ))

    return s

# ══════════════════════════════════════════════════
# BUILD AND INSERT
# ══════════════════════════════════════════════════

# Reference element for insertion (CHAPTER 4 heading)
ref_elem = doc.paragraphs[ch4_idx]._element

# First, clear all content after CHAPTER 4 heading (already done above)
# Now insert new Ch4 and Ch5

def process_content(items):
    """Process list of (text, style) tuples and plain strings (defaulting to Normal)."""
    for item in items:
        if isinstance(item, str):
            text, style_name = item, "Normal"
        else:
            text, style_name = item
        mk_para(text, style_name)

process_content(build_ch4())
process_content(build_ch5())

# Move all new elements after CHAPTER 4 heading
insert_point = ref_elem
for elem in new_elements:
    body.remove(elem)
    insert_point.addnext(elem)
    insert_point = elem

doc.save(DST)
print(f"\nSaved to {DST}")
print(f"Inserted {len(new_elements)} paragraphs after CHAPTER 4")
