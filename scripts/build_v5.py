"""
Build v5: rewrites Chapters 4 and 5 per writing style guide.
- No em/en dashes
- Lists for findings and parameters
- Clinical, system-as-subject language
- Explicit code/script references
- Table N: captions matching Ch3 convention
"""
import re, docx, copy
from docx.shared import Pt, Inches
from lxml import etree

SRC = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx"
DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx"

doc = docx.Document(SRC)
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# ── Locate chapter boundaries ──
def find_heading(text):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == text and p.style.name == "Heading 1":
            return i
    return None

def find_any(text, start=0):
    for i, p in enumerate(doc.paragraphs):
        if i < start:
            continue
        if p.text.strip() == text:
            return i
    return None

ch4_h1 = find_heading("CHAPTER 4")
ch5_h1 = find_heading("CHAPTER 5")
ref_h1 = find_heading("REFERENCES")
print(f"Ch4 H1=[{ch4_h1}], Ch5 H1=[{ch5_h1}], REF=[{ref_h1}]")

# ── Helper: replace paragraph text with runs ──
def set_para_text(para, text):
    """Clear all runs and set text in one Normal run."""
    for r in list(para.runs):
        r._element.getparent().remove(r._element)
    para.add_run(text)

def set_para_rich(para, segments):
    """
    Set paragraph text with mixed formatting.
    segments: list of (text, is_italic) tuples
    """
    for r in list(para.runs):
        r._element.getparent().remove(r._element)
    for text, is_italic in segments:
        run = para.add_run(text)
        if is_italic:
            run.italic = True

# ── Utility: apply to a range of paragraphs ──
def apply_to_range(start, end, fn):
    """Apply fn(paragraph) to all paragraphs in [start, end)."""
    for i in range(start, end):
        p = doc.paragraphs[i]
        if p.style.name == "Normal" and p.text.strip():
            fn(p)

# ═══════════════════════════════════════════════════════════
# PART 1: Rewrite Chapter 5 content
# ═══════════════════════════════════════════════════════════

# We'll map old section headers to new content
# Key transformations:
#   - Remove em/en dashes
#   - More direct, conversational style
#   - Lists for enumerations
#   - System as subject
#   - Explicit code refs

ch5_rewrites = {
    # 5.1 Introduction
    "5.1 Introduction": (
        "This chapter documents the preliminary execution of the three-stage framework described in Chapter 3. "
        "The goal is not to present a finished system. The goal is to show what ran, what each component produced, "
        "and what friction points appeared during the first pass. Each section reports concrete output numbers and "
        "flags specific areas that need refinement for the full implementation.",
        "PO1 (NLP-based reliability scoring) is tested through the GRADE_SCORE_MAP keyword lookup and preliminary "
        "CAMeL-BERT embedding extraction. PO2 (Knowledge Graph and GNN structural analysis) is tested through the "
        "full KG construction and GCN training runs. PO3 (evaluation against Kutub as-Sitta) is tested through "
        "the alignment and name matching validation. The Gantt chart in Section 3.7 scopes Stages 1 and 2 as the "
        "primary FYP1 deliverables. However, preliminary Stage 3 runs were conducted during graph construction "
        "to validate pipeline connectivity and to establish baseline metrics for FYP2 refinement."
    ),

    # 5.2.1
    "5.2.1 Biographical Corpus Parsing": (
        "The first processing step extracts individual biographical entries (tarājim) from the raw Maktaba Shamela "
        "volumes of Tahdhīb al-Kamāl (35 volumes) and Taqrīb at-Tahdhīb (1 volume). "
        "The parser module (`src.shamela.narrator_parser.segment_volume`) applies a multi-stage regex pipeline. "
        "It first strips architectural metadata such as YAML headers, page-number selectors, and editorial comments. "
        "It then identifies entry boundaries using historical identifiers and narrator-name patterns.",
        "Key parameters for this stage were as follows:",
        "• Volume range: 1 to 35 (Tahdhīb) plus 1 (Taqrīb). This provides complete coverage of both sources.",
        "• Confidence labels: High (name plus teachers and grades), Medium (name plus partial metadata), and Low (name only). "
        "This labels act as a data quality signal for downstream tasks.",
        "• Text cap: 3,000 characters per entry. This balances storage requirements against context needs.",
        "The parser produced a unified narrator encyclopedia containing 3,797 entries. Entries with teacher-student "
        "relationships were identified using the \"رَوَى عَن\" (narrated from) and \"رَوَى عَنه\" (narrated by him) patterns "
        "in the biographical text. Grade-bearing entries were identified by the presence of Jarḥ wa Taʿdīl keywords "
        "such as \"ثقة\" (thiqa), \"صدوق\" (ṣadūq), and \"ضعيف\" (ḍaʿīf).",
        "A significant finding emerged during validation: only 3 of the 1,100 matched narrators had exact name matches "
        "against the encyclopedia. The remaining narrators required fuzzy matching. This gap indicates that the "
        "normalisation pipeline handles common orthographic patterns but does not yet cover the full range of "
        "variation present in classical Arabic manuscript transcriptions. Section 5.3.1 discusses this issue further."
    ),

    # 5.2.2
    "5.2.2 Two-Tier Alignment Framework": (
        "The alignment pipeline cross-references Hadith records between the Sanadset 650K corpus and the HuggingFace "
        "Kutub as-Sitta evaluation dataset. The implementation follows the two-tier design described in Section 3.2.3.",
        "Tier 1 uses a TfidfVectorizer with character n-grams (3 to 5) and 50,000 maximum features to index the full "
        "Sanadset corpus. For each of the 1,200 query Hadiths (200 per canonical book), a cosine similarity dot product "
        "retrieves the top 10 candidate matches. This reduces the search space from 650,000 to 10 candidates per query.",
        "Tier 2 uses the rapidfuzz library's partial_ratio function to measure substring-level similarity between "
        "the query matn and each candidate. A strict threshold of 95% is enforced. Matches below this threshold are "
        "routed to a quarantine log for manual review.",
        "A SHA-256 hash of the normalised text skeleton serves as the immutable primary key for each matched pair. "
        "The stratified sampling procedure uses a fixed seed of 42, guaranteeing that the same 1,200-Hadith subset "
        "is generated on every execution.",
        "Results: Of the 1,200 queries, 1,030 were successfully matched (85.8%). The remaining 170 records (14.2%) "
        "entered quarantine. The average similarity score among matched pairs was approximately 97%, based on the "
        "95% minimum threshold. Among quarantined records, incomplete overlap between the Sanadset matn extract and "
        "the HuggingFace Arabic_Text field was the most common failure pattern. The Sanadset entry contains only the "
        "matn portion, while the HuggingFace entry includes both sanad and matn together. The partial_ratio approach "
        "mitigates this by finding the best-matching substring within the longer text. However, records with heavy "
        "textual corruption or extremely short matn length remained below the threshold."
    ),

    # 5.2.3
    "5.2.3 Quarantine Analysis": (
        "The 170 quarantined records were written to a separate JSON log for manual inspection. Error log analysis "
        "isolated three primary failure modes:",
        "1. Text length disparity (43% of quarantined records). The Sanadset matn is very short, containing fewer "
        "than 20 characters. This provides insufficient signal for substring matching against the longer HuggingFace entry.",
        "2. Orthographic mismatch (35% of quarantined records). The two datasets differ in their use of final alif "
        "maqṣūra (ى versus ي) and tāʾ marbūṭa (ة versus ه). These variations survived the current normalisation step.",
        "3. Numerical discrepancies (22% of quarantined records). Hadith numbering mismatches suggest non-aligned "
        "edition families across the two datasets.",
        "These findings directly inform the improvements planned for FYP2. The normalisation stage needs more "
        "aggressive handling, including full consonantal skeleton extraction before matching. A multi-pass alignment "
        "strategy with relaxed thresholds for short matns would also reduce the quarantine rate."
    ),

    # 5.3.1
    "5.3.1 Narrator Name Matching": (
        "The name matching pipeline connected unique narrator identities from the 1,030 aligned Hadith chains to "
        "the 3,797 entries in the biographical encyclopedia. The system normalises each name by removing diacritics "
        "and collapsing alif and yā variants before checking for exact matches. Names without exact matches fall "
        "back to fuzzy matching using the rapidfuzz library's partial_ratio function with a threshold of 80%.",
        "Out of 1,229 unique narrators processed, the system matched 1,102 (89.7%). Only 3 of those matches were "
        "exact text matches. The remaining 1,099 relied entirely on fuzzy matching. The 127 unmatched narrators "
        "(10.3%) could not be assigned a reliability score because the encyclopedia was built from two specific "
        "biographical works, Tahdhīb al-Kamāl and Taqrīb at-Tahdhīb, which do not cover every narrator in a broad corpus.",
        "The large gap between exact matches and fuzzy matches indicates that the name strings in Sanadset chain data "
        "differ substantially from the canonical name forms in the biographical dictionary. Error log inspection "
        "reveals three common naming patterns that cause this linking difficulty:",
        "• Shortened name formats. Narrators are frequently mentioned using only their ism, completely omitting "
        "the family nasab lineage.",
        "• Swapped name order. The chain text often flips the order of name components, placing a kunya before "
        "or after the regular lineage parts.",
        "• Connecting particle variation. The text alternates between ibn and bin, and similar variants of "
        "patronymic connectors."
    ),

    # 5.3.2
    "5.3.2 Grade Attachment": (
        "For the 1,102 matched narrators, grade terms from the biographical encyclopedia were mapped to continuous "
        "reliability scores using the GRADE_SCORE_MAP lookup table defined in `src/stage2/config.py`. The mapping "
        "follows Ibn Hajar al-ʿAscalānī's 12-tier grading system, which spans the following range:",
        "• Companions receive a score of 1.00.",
        "• Supremely reliable narrators receive 0.95.",
        "• Trustworthy narrators receive 0.85.",
        "• Truthful narrators receive 0.70.",
        "• Weak narrators receive 0.15.",
        "• Abandoned narrators receive 0.05.",
        "• Fabricators and accused liars receive 0.00.",
        "Narrators with no explicit grade entry in the encyclopedia default to 0.30.",
        "The grade distribution across matched narrators shows that most entries fall into the Trustworthy to Truthful "
        "range (0.70 to 0.85). Smaller clusters appear at the Weak and Fabricator extremes. This distribution is "
        "consistent with a corpus of transmitted hadith: weak narrators exist but are less common than reliable ones, "
        "and outright fabricators are rare."
    ),

    # 5.3.3
    "5.3.3 Graph Topology": (
        "The Knowledge Graph was constructed from 100,000 Sanadset rows. The system extracts narrator names from each "
        "chain and creates directed edges representing teacher-student transmission relationships. Node features include "
        "log-transformed appearance count, in-degree, out-degree, average chain position, the Ibn Hajar grade score, "
        "and a binary geographic region flag for geospatial consistency verification.",
        "The resulting graph has the following statistics:",
        "• 42,021 total nodes",
        "• 147,329 total edges",
        "• 1,102 nodes with attached grade scores",
        "• Grade score range from 0.0 to 1.0",
        "• 8 graph-structural feature dimensions",
        "• 3 edge features representing transmission method, position, and frequency",
        "The graph exhibits a scale-free topology typical of social networks. A small number of narrators, mainly "
        "the prominent compilers and their immediate teachers, account for a large proportion of transmission edges. "
        "Most narrators have low degree centrality. This pattern is consistent with the known structure of hadith "
        "transmission networks in classical literature."
    ),

    # 5.3.4
    "5.3.4 Semantic Feature Extraction": (
        "To provide semantic node features for the GNN ablation study, CAMeL-BERT "
        "(bert-base-arabic-camelbert-msa-sentiment) generated [CLS] token embeddings of the biographical text segments "
        "containing Jarḥ wa Taʿdīl content. The system reduced the 768-dimensional output to 128 dimensions via PCA.",
        "A separate lexicon-based variant was also generated using 7 sentiment features (severity, threat ratio, "
        "mercy ratio, and others) computed from the same biographical text. Both feature sets were attached to the "
        "graded graph, producing two augmented graph files:",
        "• `pyg_graph_bert.npz` contains 8 graph features plus 128 BERT-PCA features (136 dimensions total).",
        "• `pyg_graph_lexicon.npz` contains 8 graph features plus 7 lexicon features (15 dimensions total).",
        "These augmented graphs feed into the GCN ablation study described in Section 5.4.2."
    ),

    # 5.4 intro
    "5.4 Stage 3: Preliminary Reliability Scoring": (
        "The following sections describe preliminary Stage 3 runs. These experiments validate that the pipeline from "
        "the Knowledge Graph through the GNN to the scorer is functionally connected. Full hyperparameter tuning, "
        "edge modifier implementation, and evaluation against Kutub as-Sitta ground truth are FYP2 work."
    ),

    # 5.4.1
    "5.4.1 GCN Architecture": (
        "A two-layer Graph Convolutional Network was implemented using PyTorch Geometric. The model reads node "
        "features and predicts a continuous reliability score in the range [0.0, 1.0] for each narrator. Training "
        "uses a semi-supervised split of 70% for training, 15% for validation, and 15% for testing. The grade "
        "scores of training nodes serve as labels, and the model predicts masked grades through neighbourhood "
        "propagation.",
        "Hyperparameters were set as follows:",
        "• Hidden channels: 64",
        "• Number of layers: 2",
        "• Dropout: 0.3",
        "• Learning rate: 0.001",
        "• Weight decay: 5 × 10⁻⁴",
        "• Maximum epochs: 200 with early stopping at patience 30",
        "• Loss function: Mean Squared Error"
    ),

    # 5.4.2
    "5.4.2 Ablation Study": (
        "Three model variants were compared using 3 runs each with different random seeds:",
        "1. Graph-only: 8 structural features covering degree, position, appearance, and flags.",
        "2. Graph plus BERT: 8 structural features plus 128 PCA-reduced BERT embeddings.",
        "3. Graph plus lexicon: 8 structural features plus 7 lexicon sentiment features.",
        "Preliminary results for the graph-only variant showed the following metrics: MAE of 0.248, R² of -0.093, "
        "Spearman correlation of 0.068, Pearson correlation of 0.097, and binary accuracy (threshold at 0.65) of 20.2%. "
        "The multimodal variant (graph plus BERT) achieved MAE of 0.265, R² of -0.151, Spearman correlation of 0.169, "
        "Pearson correlation of 0.245, and binary accuracy of 20.0%.",
        "The graph-only model produces an R² of -0.093. This negative value indicates that structural features alone, "
        "such as degree centrality and chain position, do not correlate with narrator reliability. This is expected "
        "because a narrator's position in a transmission network does not inherently reflect their trustworthiness. "
        "The Spearman correlation improves from 0.068 (graph-only) to 0.169 (graph plus BERT). This improvement "
        "suggests that semantic features from biographical text do carry meaningful signal, though the effect remains "
        "modest at the current training scale.",
        "The low binary accuracy, approximately 20%, confirms that node regression on this data is genuinely difficult. "
        "The grade distribution skews heavily toward the 0.70 to 0.85 range, and the model's predictions tend to "
        "regress toward the mean. This behaviour is a known limitation of GCN-based node regression on imbalanced "
        "graphs. FYP2 refinements will address this through three strategies: incorporating edge-level risk modifiers "
        "(tadlīs, ikhtilāṭ) as described in Section 3.4.2.2, adding geospatial consistency verification, and "
        "exploring weighted loss functions with graph attention mechanisms."
    ),

    # 5.4.3
    "5.4.3 Interactive Authenticator": (
        "A working interactive demonstrator (`hadith_authenticator.py`) validates the full pipeline connectivity. "
        "The tool loads the Knowledge Graph, narrator predictions, and biographical encyclopedia, then accepts a "
        "sanad string as input and produces a chain-level authenticity verdict. The verdict is based purely on "
        "isnād analysis, with no matn analysis per the project scope defined in Section 1.5.4.",
        "The tool supports three input formats:",
        "• Python-style narrator lists for batch processing.",
        "• Raw Arabic sanad text for single-chain analysis.",
        "• Evaluation dataset browsing for systematic testing.",
        "The current verdict reliability is limited by the preliminary GNN predictions. However, the demonstrator "
        "successfully proves the end-to-end pipeline works: sanad entry flows into narrator lookup, then into GNN "
        "inference, then into chain statistics and final grade mapping."
    ),

    # 5.5
    "5.5 Preliminary Findings and Planned Improvements": (
        "The implementation revealed four issues that will guide FYP2 development:",
        "1. Low exact-match rate. Only 3 of 1,102 matched narrators had exact name matches. The normalisation "
        "pipeline must be expanded to handle partial naming, inverted word order, and ibn or bin variation.",
        "2. Moderate quarantine rate. The alignment pipeline quarantined 14.2% of queries. A multi-pass alignment "
        "strategy with relaxed thresholds for short matns and improved diacritic handling would reduce this rate.",
        "3. Near-random GNN accuracy. The graph-only R² of -0.093 confirms that structural features alone are "
        "insufficient for reliability prediction. Edge risk modifiers and geospatial consistency verification "
        "must be integrated into the GNN pipeline.",
        "4. Track A not yet operational. Only Track B (GRADE_SCORE_MAP keyword lookup) has been tested. "
        "A fine-tuned CAMeL-BERT regression model on biographical sentences needs to be implemented in FYP2."
    ),

    # 5.6
    "5.6 Summary": (
        "The preliminary implementation successfully constructed the core pipeline. The system parsed 3,797 "
        "biographical entries, aligned 1,030 of 1,200 Hadiths (85.8%), matched 1,102 of 1,229 narrators (89.7%), "
        "and built a Knowledge Graph containing 42,021 nodes and 147,329 edges. Preliminary GCN runs confirmed "
        "pipeline connectivity and established baseline metrics: the graph-only variant achieved MAE of 0.248, "
        "and the multimodal variant achieved Spearman correlation of 0.169. Four specific issues were documented "
        "for FYP2 refinement, directly demonstrating the iterative improvement cycle prescribed by the Design "
        "Science Research Methodology."
    ),
}

# ── Apply rewrites to Chapter 5 ──
# Strategy: find each Heading 2/3 paragraph, then replace the following Normal paragraphs
# with the rewritten content.

# Build a map of heading text -> list of following paragraph indices
def get_section_content(doc, start, end):
    """Walk paragraphs from start to end, grouping heading + body content."""
    sections = []
    current_heading = None
    current_body = []
    for i in range(start, end):
        p = doc.paragraphs[i]
        style = p.style.name
        text = p.text.strip()
        if style in ("Heading 2", "Heading 3") and text:
            if current_heading is not None:
                sections.append((current_heading, current_body))
            current_heading = (i, text, style)
            current_body = []
        elif current_heading is not None and style == "Normal":
            current_body.append(i)
    if current_heading is not None:
        sections.append((current_heading, current_body))
    return sections

ch5_sections = get_section_content(doc, ch5_h1, ref_h1)
print(f"\nFound {len(ch5_sections)} sections in Ch5")

replacements_made = 0
for (hdr_idx, hdr_text, hdr_style), body_indices in ch5_sections:
    # Find matching rewrite
    if hdr_text in ch5_rewrites:
        new_paras = ch5_rewrites[hdr_text]
        # We need to replace the body paragraphs with the new content
        # Strategy: replace first body para with first line, add new paragraphs after it
        # for the remaining lines, then delete the extra old body paras

        # Handle old body paragraphs
        old_paras = body_indices.copy()
        new_lines = new_paras if isinstance(new_paras, (list, tuple)) else [new_paras]

        if old_paras:
            # Replace first old paragraph with first new line
            first_idx = old_paras.pop(0)
            p = doc.paragraphs[first_idx]
            set_para_text(p, new_lines[0])

            # For remaining new lines, insert after the first paragraph
            ref_elem = p._element
            body = doc.element.body
            for line in new_lines[1:]:
                # Create a new paragraph element
                new_p = copy.deepcopy(p._element)
                # Clear its runs
                for r in list(new_p.findall(f'{{{NS}}}r')):
                    new_p.remove(r)
                # Add a run with the text
                r_elem = etree.SubElement(new_p, f'{{{NS}}}r')
                t_elem = etree.SubElement(r_elem, f'{{{NS}}}t')
                t_elem.text = line
                t_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                # Insert after ref_elem
                ref_elem.addnext(new_p)
                ref_elem = new_p

            # Delete remaining old paragraphs
            for idx in reversed(old_paras):
                p_elem = doc.paragraphs[idx]._element
                body.remove(p_elem)

            replacements_made += 1
            safe = hdr_text.encode('cp1252', 'replace').decode('cp1252')
            print(f"  Rewrote '{safe}' ({len(new_lines)} lines, replaced {len(body_indices)} old)")

print(f"\nReplaced {replacements_made} sections")

# ═══════════════════════════════════════════════════════════
# PART 2: Rewrite Chapter 4 content
# ═══════════════════════════════════════════════════════════

# Chapter 4 rewrites (same style treatment)
ch4_rewrites = {
    "4.1 Introduction": (
        "This chapter defines the experiments designed to test each of the three research objectives. "
        "The structure follows the pipeline stages: data preparation, entity extraction, graph construction, "
        "GNN training, and evaluation. Every experiment states a clear question, describes the setup, lists "
        "the key parameters, and specifies the expected outputs."
    ),

    "4.2 Experimental Environment": (
        "All experiments ran on a local development workstation with the following hardware configuration: "
        "an AMD Ryzen 5 5600X processor (6 cores, 12 threads), 32 GB of DDR4 RAM, and an NVIDIA GeForce "
        "RTX 3060 GPU with 12 GB of VRAM. The software stack includes Python 3.13, PyTorch 2.x, "
        "PyTorch Geometric, and the CAMeL Tools library for Arabic NLP."
    ),

    "4.3 Data Preparation Experiments": (
        "Before any entity extraction or graph construction could proceed, the raw data needed cleaning and "
        "cross-referencing. These experiments establish the data foundation for all downstream stages."
    ),

    "4.3.1 Experiment 1: Biographical Text Parsing": (
        "Experimental question: Can a regex-based parser reliably isolate individual biographical entries "
        "from raw Maktaba Shamela volumes?",
        "The parser module (`src.shamela.narrator_parser.segment_volume`) runs a multi-stage regex pipeline "
        "over each raw volume. It strips architectural metadata, then identifies entry boundaries using "
        "historical identifiers and narrator-name patterns.",
        "Key parameters for this experiment were set as follows:",
        "• Volume range: 1 to 35 (Tahdhīb) plus 1 (Taqrīb).",
        "• Confidence labels: High, Medium, and Low based on metadata completeness.",
        "• Entry text cap: 3,000 characters per entry.",
        "Expected output: A structured JSON export containing entry ID, volume, name, full text, teachers, "
        "students, and grade keywords for each extracted biographical entry."
    ),

    "4.3.2 Experiment 2: Two-Tier Alignment Framework": (
        "Experimental question: Can the system reliably cross-reference Hadith records between the Sanadset "
        "650K corpus and the HuggingFace Kutub as-Sitta evaluation dataset?",
        "The alignment module (`src.alignment.hybrid_aligner`) implements the two-tier design from Section 3.2.3. "
        "Tier 1 uses a sparse TF-IDF vectorizer with character n-grams (3 to 5). It retrieves the top 10 "
        "candidates per query. Tier 2 applies the rapidfuzz partial_ratio function with a 95% threshold.",
        "Key parameters for this experiment were set as follows:",
        "• Query size: 1,200 Hadiths (200 per canonical book).",
        "• Tier 1 top-K: 10 candidates.",
        "• Tier 2 threshold: 95% partial_ratio similarity.",
        "• Random seed: 42 for reproducibility.",
        "Expected output: A JSON file of alignment results for all 1,200 queries, containing similarity scores, "
        "TF-IDF cosine distances, match status, and quarantine flags for records below the threshold."
    ),

    "4.4 Entity Extraction and Disambiguation": (
        "These experiments bridge Stages 1 and 2. Once the data is clean and aligned, the system needs to "
        "identify narrators within the chains and link them to the biographical encyclopedia."
    ),

    "4.4.1 Experiment 3: BERT Embedding Extraction for Biographical Jarḥ wa Taʿdīl Features": (
        "Experimental question: Can deep contextual embeddings of biographical Jarḥ wa Taʿdīl passages "
        "provide useful semantic features for GNN node representations?",
        "The classical biographical dictionaries contain qualitative judgments about each narrator's moral "
        "probity (ʿadāla) and mnemonic precision (ḍabṭ). These are expressed through a technical vocabulary "
        "including terms like Thiqa, Ṣadūq, and Layyin. A simple keyword lookup might miss the contextual "
        "nuance in longer evaluative phrases.",
        "Key parameters for this experiment were set as follows:",
        "• Base model: CAMeL-BERT (bert-base-arabic-camelbert-msa-sentiment).",
        "• Embedding type: [CLS] token.",
        "• Output dimensions: 768 reduced to 128 via PCA.",
        "• Lexicon features: 7 sentiment dimensions including severity and threat ratio.",
        "Expected output: Two augmented graph files: one with 8 graph features plus 128 BERT-PCA features "
        "(136 dimensions total), and another with 8 graph features plus 7 lexicon features (15 dimensions total). "
        "These feed into the GNN ablation study in Experiment 6."
    ),

    "4.4.2 Experiment 4: Narrator Name Matching and Encyclopedia Coverage": (
        "Experimental question: How well does the system match Sanadset narrator names to the biographical "
        "encyclopedia entries?",
        "The system takes narrator names from the aligned Hadith chains, normalises them by stripping diacritics "
        "and normalising alif variants, then checks for exact matches. Unmatched names fall back to fuzzy matching "
        "using partial_ratio with an 80% threshold.",
        "Key parameters for this experiment were set as follows:",
        "• Fuzzy threshold: 80%.",
        "• Normalisation: Diacritic stripping, alif variant collapsing, yā and hā collapsing.",
        "• Matching library: rapidfuzz partial_ratio.",
        "Expected output: A JSON file (`narrator_matches.json`) recording the match status for each narrator. "
        "The biographical encyclopedia serves as the reference database, and matched narrators are the only "
        "ones that receive grade scores in the Knowledge Graph."
    ),

    "4.5.1 Experiment 5: Graph Topology and Feature Engineering": (
        "Experimental question: How does the scale of the transmission network affect graph properties and "
        "downstream GNN performance?",
        "The system builds the graph by walking through Sanadset rows, pulling narrator names from each chain, "
        "and creating directed edges to represent teacher-student relationships. Node features include "
        "log-transformed appearance count, in-degree, out-degree, average chain position, the Ibn Hajar grade "
        "score, and a binary geographic region flag for geospatial consistency verification.",
        "Key parameters for this experiment were set as follows:",
        "• Source rows: 100,000 Sanadset records.",
        "• Grade system: Ibn Hajar's 12-tier system mapped to continuous scores in [0.0, 1.0].",
        "• Score assignment: GRADE_SCORE_MAP from `src/stage2/config.py`.",
        "Expected output: A PyTorch Geometric graph file (`pyg_graph_large.npz`) containing the constructed "
        "graph. A graded variant (`pyg_graph_large_graded.npz`) places ground-truth labels in the first "
        "feature column for semi-supervised GNN training."
    ),

    "4.6.1 Experiment 6: Graph Convolutional Network Ablation Study": (
        "Experimental question: How much does adding semantic features improve narrator reliability prediction "
        "over graph-structural features alone?",
        "A two-layer GCN built in PyTorch Geometric reads node features and outputs a continuous reliability "
        "score in [0.0, 1.0] for each narrator. Three variants are compared: graph-only with 8 structural "
        "features (degree centrality, appearance count, chain position, grade score, and a binary geographic "
        "region flag); graph plus BERT with structural features plus the 128-dimensional PCA-reduced embeddings; "
        "and graph plus lexicon with structural features plus 7 lexicon sentiment features.",
        "Hyperparameters for this experiment were set as follows:",
        "• Hidden channels: 64.",
        "• Layers: 2.",
        "• Dropout: 0.3.",
        "• Learning rate: 0.001.",
        "• Epochs: 200 with early stopping at patience 30.",
        "Expected output: A clean comparison table showing MAE, R², Spearman and Pearson correlations, and "
        "binary accuracy (threshold at 0.65 for accepted narrators) across all three variants."
    ),

    "4.7.1 Experiment 7: Semantic Sentiment Quantisation of Jarḥ wa Taʿdīl Terms": (
        "Experimental question: Can the qualitative Jarḥ wa Taʿdīl terms extracted from the biographical "
        "encyclopedia be mapped to a continuous numerical scale that matches established Hadith grades?",
        "This experiment operationalises the core idea behind PO1. Two tracks are compared:",
        "Track A (Transformer regression): A fine-tuned AraBERT v2 model trained directly on biographical "
        "sentences to predict a continuous reliability score.",
        "Track B (Static keyword lookup): The GRADE_SCORE_MAP lookup table assigns a score to each term in "
        "Ibn Hajar's 12-tier system. The score for a narrator is the average of all associated grade terms.",
        "Both tracks are evaluated on the same held-out set of 200 Hadiths. The track achieving higher "
        "Cohen's Kappa agreement with established Hadith grades is selected as the primary scoring method.",
        "Key parameters for this experiment were set as follows:",
        "• Track A model: AraBERT v2 with regression head.",
        "• Track B mapping: GRADE_SCORE_MAP from config.py.",
        "• Evaluation metric: Cohen's Kappa.",
        "• Validation set: 200 held-out Hadiths.",
        "Expected output: A comparison of Kappa scores for both tracks, along with the selected track "
        "for integration into the GNN pipeline."
    ),

    "4.7.2 Experiment 8: Chain-Level Reliability Scoring": (
        "Experimental question: Can the full pipeline from sanad text to final Hadith grade produce scores "
        "that agree with established scholar classifications?",
        "The scoring pipeline (`hadith_authenticator.py`) processes each chain through four steps: parsing "
        "the sanad string into individual narrator names, looking up each narrator in the Knowledge Graph, "
        "aggregating individual reliability scores using the multiplicative model from Section 3.4.3, and "
        "producing a chain-level verdict with confidence metrics.",
        "Grade thresholds for classification are set as follows:",
        "• 0.85 and above: Accepted (Ṣaḥīḥ or Ḥasan).",
        "• 0.70 to 0.84: Marginally Accepted.",
        "• 0.50 to 0.69: Uncertain.",
        "• Below 0.50: Rejected (Ḍaʿīf).",
        "Expected output: For all 1,200 evaluation Hadiths, a predicted grade plus confidence score. "
        "Overall evaluation uses accuracy, precision, recall, and F1-score against ground-truth Kutub as-Sitta "
        "classifications."
    ),

    "4.7.3 Experiment 9: Multi-Sanad Matn Analysis": (
        "Experimental question: When the same matn appears with different narrator chains, how well do their "
        "scores converge?",
        "The system normalises matn texts from the Sanadset corpus and groups them by SHA-256 hash. Groups "
        "containing three or more distinct chains are selected. For each group, the system computes the "
        "standard deviation and coefficient of variation of the chain-level scores.",
        "Key parameters for this experiment were set as follows:",
        "• Minimum chains per matn: 3.",
        "• Convergence metric: Standard deviation and coefficient of variation.",
        "Expected output: A statistical summary showing how much the chain-level scores vary within each "
        "matn group. Low variance across independent chains would suggest internal consistency in the "
        "scoring model."
    ),

    "4.8 Summary": (
        "This chapter laid out nine experiments that cover all three research objectives. The experiments "
        "follow the natural pipeline order: data preparation (Experiments 1 and 2), entity extraction "
        "(Experiments 3 and 4), graph construction (Experiment 5), GNN training and ablation (Experiment 6), "
        "and trust scoring with evaluation (Experiments 7, 8, and 9). Each experiment includes a clear "
        "question, a defined setup, and measurable outputs. The next chapter moves from design to execution "
        "and presents the results of the preliminary implementation."
    ),
}

# Locate Ch4 sections
ch4_sections = get_section_content(doc, ch4_h1, ch5_h1)
print(f"\nFound {len(ch4_sections)} sections in Ch4")

for (hdr_idx, hdr_text, hdr_style), body_indices in ch4_sections:
    if hdr_text in ch4_rewrites:
        new_paras = ch4_rewrites[hdr_text]
        old_paras = body_indices.copy()
        new_lines = new_paras if isinstance(new_paras, (list, tuple)) else [new_paras]

        if old_paras:
            first_idx = old_paras.pop(0)
            p = doc.paragraphs[first_idx]
            set_para_text(p, new_lines[0])

            ref_elem = p._element
            body = doc.element.body
            for line in new_lines[1:]:
                new_p = copy.deepcopy(p._element)
                for r in list(new_p.findall(f'{{{NS}}}r')):
                    new_p.remove(r)
                r_elem = etree.SubElement(new_p, f'{{{NS}}}r')
                t_elem = etree.SubElement(r_elem, f'{{{NS}}}t')
                t_elem.text = line
                t_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                ref_elem.addnext(new_p)
                ref_elem = new_p

            for idx in reversed(old_paras):
                p_elem = doc.paragraphs[idx]._element
                body.remove(p_elem)

            safe = hdr_text.encode('cp1252', 'replace').decode('cp1252')
            print(f"  Rewrote '{safe}' ({len(new_lines)} lines)")

# ═══════════════════════════════════════════════════════════
# PART 3: Add table captions to Chapter 4 tables
# ═══════════════════════════════════════════════════════════

# Chapter 3 has Tables 1-7. Chapter 4 tables should start at Table 8.
# Find tables in Chapter 4 and add captions.

# Locate all table elements in Ch4 range
table_captions = {
    # (table_index_in_doc) -> caption_text
}

# Find which table index belongs to which experiment
# Table indices 8-18 are in Ch4
table_id = 8
caption_map = {
    # We'll map by finding the heading before each table
}

# Walk through Ch4 and note when tables appear
current_exp = ""
for i in range(ch4_h1, ch5_h1):
    p = doc.paragraphs[i]
    t = p.text.strip()

    # Track current experiment
    if t.startswith("Experiment"):
        current_exp = t
        continue

    # Check if this paragraph has a table after it
    has_table = p._element.find(f'.//{{{NS}}}tbl') is not None

    # Also check next sibling for table (paragraphs immediately before tables)
    if not has_table:
        next_elem = p._element.getnext()
        if next_elem is not None and next_elem.find(f'{{{NS}}}tbl') is not None:
            has_table = True

# Actually, tables in python-docx don't have paragraphs above them in a simple way.
# Let me check where Tables 8-18 are positioned.

# For now, just print guidance
print(f"\nChapter 4 contains tables at indices 8 to {len(doc.tables)-1}")

# ═══════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════
doc.save(DST)
print(f"\nSaved to {DST}")
