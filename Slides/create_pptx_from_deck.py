from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Colours
BLUE  = RGBColor(0x00,0x78,0xD4)
DARK  = RGBColor(0x00,0x45,0x78)
ACC   = RGBColor(0x3A,0x7C,0xBD)
TD    = RGBColor(0x2C,0x2C,0x2C)
TB    = RGBColor(0x55,0x55,0x55)
TL    = RGBColor(0x88,0x88,0x88)
TM    = RGBColor(0x99,0x99,0x99)
WH    = RGBColor(0xFF,0xFF,0xFF)
CB    = RGBColor(0xF8,0xF8,0xF8)
CE    = RGBColor(0xE8,0xE8,0xE8)
GR    = RGBColor(0x10,0x7C,0x41)
RD    = RGBColor(0xE7,0x4C,0x3C)
OR    = RGBColor(0xF3,0x9C,0x12)
GL    = RGBColor(0xD4,0xA8,0x3C)

prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BL = prs.slide_layouts[6]

def S():
    s = prs.slides.add_slide(BL)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = WH
    return s

def TXT(s,l,t,w,h,txt,sz=14,b=False,c=TD,a=PP_ALIGN.LEFT):
    bx = s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    bx.text_frame.word_wrap = True
    p = bx.text_frame.paragraphs[0]; p.text = txt; p.font.size = Pt(sz)
    p.font.bold = b; p.font.color.rgb = c; p.font.name = 'Segoe UI'; p.alignment = a

def GB(s,l,t,w=1.2,ht=0.05):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(l),Inches(t),Inches(w),Inches(ht))
    r.fill.solid(); r.fill.fore_color.rgb = BLUE; r.line.fill.background()

def CD(s,l,t,w,h):
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(l),Inches(t),Inches(w),Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = CB; r.line.color.rgb = CE; r.line.width = Pt(1)

def CV(s,l,t,w,h,title,items):
    CD(s,l,t,w,h)
    TXT(s,l+0.12,t+0.08,w-0.24,0.35,title,11,True,BLUE)
    y = t + 0.45
    for item in items:
        txt = f"• {item}" if isinstance(item,str) else f"• {item[0]}"
        TXT(s,l+0.12,y,w-0.24,0.25,txt,9,c=TB); y += 0.2

def HV(s,l,t,w,h,txt,bc=BLUE):
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(l),Inches(t),Inches(w),Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xF0,0xF6,0xFC); r.line.fill.background()
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(l),Inches(t),Inches(0.04),Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = bc; b.line.fill.background()
    TXT(s,l+0.12,t+0.05,w-0.18,h-0.1,txt,9,c=TB)

def TA(s,l,t,w,h,hdrs,rows,cw=None):
    t2 = s.shapes.add_table(len(rows)+1,len(hdrs),Inches(l),Inches(t),Inches(w),Inches(h))
    tb = t2.table
    for j,h in enumerate(hdrs):
        c = tb.cell(0,j); c.text = h
        for p in c.text_frame.paragraphs:
            p.font.size=Pt(9); p.font.bold=True; p.font.color.rgb=BLUE; p.alignment=PP_ALIGN.CENTER
        c.fill.solid(); c.fill.fore_color.rgb=RGBColor(0xF0,0xF6,0xFC)
    for i,row in enumerate(rows):
        for j,v in enumerate(row):
            c = tb.cell(i+1,j); c.text = str(v or '')
            for p in c.text_frame.paragraphs:
                p.font.size=Pt(8); p.font.color.rgb=TB; p.alignment=PP_ALIGN.CENTER
            c.fill.solid(); c.fill.fore_color.rgb=WH
    if cw:
        for j,w_ in enumerate(cw): tb.columns[j].width = Inches(w_)

def SL(s,txt,l=0.6,t2=0.3): TXT(s,l,t2,4,0.25,txt,8,True,BLUE)
def ST(s,txt,l=0.6,t2=0.55,sz=22): TXT(s,l,t2,10,0.55,txt,sz,c=TD)

def PBOX(s,l,t,w,h,title,items,clr):
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(l),Inches(t),Inches(w),Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = clr; r.line.fill.background()
    TXT(s,l+0.08,t+0.05,w-0.16,0.2,title,8,True,WH,PP_ALIGN.CENTER)
    y = t+0.28
    for it in items:
        TXT(s,l+0.08,y,w-0.16,0.16,f"▸ {it}",7,c=RGBColor(0xDD,0xDD,0xDD)); y+=0.16

# ===== SLIDE 1: TITLE =====
s=S()
TXT(s,1.5,1.1,10.3,0.4,"FINAL YEAR PROJECT 1",11,True,BLUE,PP_ALIGN.CENTER)
GB(s,5.8,1.6,1.2)
TXT(s,1,2.0,11.3,1.2,"A Multi-Modal Framework\nfor Quantifying Narrator Reliability\nin ʿIlm ar-Rijal",32,c=TD,a=PP_ALIGN.CENTER)
TXT(s,1.5,3.3,10.3,0.5,"Using Semantic Sentiment Analysis and Graph Neural Networks",14,c=ACC,a=PP_ALIGN.CENTER)
TXT(s,2,4.5,9.3,1.0,"Alan Syahmi Bin Sanih @ Sani  |  BI23110344\nSupervisor: Dr. Khalifa Chekima\nFaculty of Computing and Informatics  |  Universiti Malaysia Sabah",12,c=TL,a=PP_ALIGN.CENTER)
TXT(s,4,6.2,5.3,0.4,"Week 14, 2025/2026",10,c=TM,a=PP_ALIGN.CENTER)

# ===== SLIDE 2: OUTLINE =====
s=S(); GB(s,0.6,0.3,0.8); ST(s,"Presentation Outline")
CV(s,0.6,1.3,5.8,2.5,"Part I",["1. Introduction — Background, Problem Statements, Objectives & Scope","2. Literature Review — Hadith Authentication, GNN, Sentiment Quantisation","3. Project Methodology — Pipeline Stages, Datasets & EDA"])
CV(s,6.9,1.3,5.8,2.5,"Part II",["4. Experiment Design — 9 Experiments across 3 Stages","5. Preliminary Results — Key Findings & Metrics","6. Conclusion — Limitations & Future Work (FYP2)"])
TXT(s,3.5,4.2,6.3,0.3,"Today: Chapters 1–2  ·  Chapters 3–6 covered separately",10,c=TL,a=PP_ALIGN.CENTER)

# ===== SLIDE 3: INTRODUCTION =====
s=S(); SL(s,"CHAPTER 1"); GB(s,0.6,0.65,0.8); ST(s,"Introduction")
CV(s,0.6,1.5,5.8,2.8,"Core Challenge",["Hadith authentication via sanad criticism — cornerstone of Islamic scholarship","ʿIlm ar-Rijal evaluates narrators for ʿadāla & ḍabṭ","Over 500,000 narrators tracked historically through manual processes","Digital collections growing too large for unaided human review — a scalability crisis"])
CV(s,6.9,1.5,5.8,2.8,"The Digital Gap",["Two complementary datasets: Sanadset 650K (chains, no grades) + HF hadith_datasets (grades, no chains)","Alignment bridges this gap — linking sanads to ground-truth grades","Critical need for an automated, scalable solution preserving classical methodology"])
HV(s,0.6,4.6,12.1,0.7,"Goal: A reproducible, inspectable decision-support tool grounded in classical Jarḥ wa Taʿdīl principles — combining Arabic NER, Graph Neural Networks, and Semantic Sentiment Quantisation.")

# ===== SLIDE 4: PROBLEM BACKGROUND =====
s=S(); SL(s,"1.2"); GB(s,0.6,0.65,0.8); ST(s,"Problem Background")
CV(s,0.6,1.5,5.8,2.5,"Traditional Bottlenecks",["Manual cross-referencing of thousands of biographical entries across vast corpuses","Arabic orthographic ambiguity — minor name variations lead to misidentification","Temporal & geographical verification complexities"])
CV(s,6.9,1.5,5.8,2.5,"Hadith Structure",["Sanad — chain of narrators (who told whom)","Matn — textual content (the saying/action)"])
TXT(s,7.1,2.6,5.4,1.2,"“...from Mālik, from Ibn Shihāb, from ʿAlī ibn Ḥusayn, that the Prophet ﷺ said: ‘Part of the excellence of a man’s Islam is that he leaves what does not concern him.’”\n— Muwaṭṭaʾ Mālik, Book 47, Hadith 3 (Ṣaḥīḥ)",9,c=RGBColor(0x6B,0x7B,0x8D))
HV(s,0.6,4.3,5.8,0.6,"Critical gap: No end-to-end automated reliability scoring system exists that preserves classical methodology while enabling digital-scale analysis.")

# ===== SLIDE 5: PROBLEM STATEMENTS =====
s=S(); SL(s,"1.3"); GB(s,0.6,0.65,0.8); ST(s,"Problem Statements")
CV(s,0.6,1.5,5.8,2.8,"PS1 — Lack of Quantified Reliability",["Current NER models extract names but cannot judge narrator standing","No existing framework maps Jarḥ wa Taʿdīl assessments → standardised reliability index","Identifying a narrator is insufficient — must evaluate trustworthiness from historical evidence"])
CV(s,6.9,1.5,5.8,2.8,"PS2 — Graph without Reliability Scores",["Transformer models extract chains as graphs (Mosa, 2025; Oumer, 2025)","But stop at graph construction — no reliability scores assigned to narrators","No use of actual Jarḥ wa Taʿdīl terms to judge trustworthiness within the graph"])
HV(s,0.6,4.6,12.1,0.7,"This research goes beyond: Adds a quantitative reliability index (0.0–1.0) + GNN-based structural analysis — combining NER extraction, KG construction, GNN analysis, and sentiment scoring in one integrated pipeline.")

# ===== SLIDE 6: PROPOSED SOLUTION =====
s=S(); SL(s,"1.4"); GB(s,0.6,0.65,0.8); ST(s,"Proposed Solution")
for i,(e,tt,desc) in enumerate([("\U0001f9e9","Multi-Modal Integration","Combines NER + Knowledge Graph + Graph Neural Networks + Semantic Sentiment Analysis into a single framework"),("\U0001f517","End-to-End Pipeline","Extraction → Disambiguation → Scoring → Grading — from raw biographical texts to final Hadith reliability classification"),("✓","Explainable & Validated","Probabilistic trust scores (0.0–1.0) with edge risk modifiers. Evaluated against Kutub as-Sitta ground-truth collections")]):
    l=0.6+i*4.1; CD(s,l,1.5,3.8,2.5)
    TXT(s,l+0.3,1.7,3.2,0.4,e,28,a=PP_ALIGN.CENTER)
    TXT(s,l+0.3,2.1,3.2,0.4,tt,12,True,BLUE,a=PP_ALIGN.CENTER)
    TXT(s,l+0.3,2.5,3.2,1.2,desc,9,c=TB,a=PP_ALIGN.CENTER)
HV(s,0.6,4.3,12.1,0.6,"First complete automated framework: No existing solution provides this full pipeline — from raw biographical texts to final Hadith grade — with an inspectable, reproducible score.")

# ===== SLIDE 7: OBJECTIVES =====
s=S(); SL(s,"1.5"); GB(s,0.6,0.65,0.8); ST(s,"Research Objectives")
for i,(code,tt,desc) in enumerate([("O1","Semantic Sentiment Integration","Develop a Transformer-based NLP model utilising semantic sentiment analysis to map qualitative scholarly judgments to a standardised quantitative reliability index (0.0–1.0)."),("O2","GNN & Knowledge Graph","Construct a multi-relational Knowledge Graph of narrator transmission networks and leverage GNNs to evaluate structural topology alongside temporal and geographical features."),("O3","End-to-End Evaluation","Evaluate the multi-modal framework’s verification accuracy by benchmarking automated hybrid reliability scores against canonical ground-truth chain-level Hadith gradings.")]):
    l=0.6+i*4.1; CD(s,l,1.5,3.8,3.8)
    circ=s.shapes.add_shape(MSO_SHAPE.OVAL,Inches(l+0.35),Inches(1.7),Inches(0.5),Inches(0.5))
    circ.fill.solid(); circ.fill.fore_color.rgb=BLUE; circ.line.fill.background()
    TXT(s,l+0.35,1.72,0.5,0.5,code,14,True,WH,PP_ALIGN.CENTER)
    TXT(s,l+0.3,2.3,3.2,0.4,tt,12,True,BLUE); TXT(s,l+0.3,2.7,3.2,2.2,desc,9,c=TB)

# ===== SLIDE 8: SCOPE =====
s=S(); SL(s,"1.6"); GB(s,0.6,0.65,0.8); ST(s,"Project Scope")
CV(s,0.6,1.5,5.8,1.8,"Data Sources",["Taqrīb at-Tahdhīb (Ibn Hajar) + Tahdhīb al-Kamāl (al-Mizzī) — from Shamela Digital Library","Sanadset 650K — chains only, no grades","HuggingFace hadith_datasets — 6 graded CSVs from Kutub as-Sitta (grades, no sanads)"])
CV(s,6.9,1.5,5.8,1.8,"Out of Scope",["✗ Matn (textual body) analysis","✗ Shīʿa isnād methodologies","✗ Real-time deployment","✗ Non-Arabic commentaries"])
CV(s,0.6,3.6,5.8,1.8,"Technical Scope",["1. Narrator extraction — proper names, kunya, nisba from isnād","2. Entity disambiguation — alias resolution via KG","3. Reliability scoring — mapping Jarḥ wa Taʿdīl → continuous index W = [0.0–1.0]","4. Geospatial mapping — verifying student–teacher regional overlap"])
CV(s,6.9,3.6,5.8,1.8,"Scale",["15,448 narrator biographies | 6,000 validated isnād chains","262,497 KG nodes | 987,036 edges","4 dataset configs: 1,200 & 6,000 aligned Hadith"])

# ===== SLIDE 9: LIT REVIEW =====
s=S(); SL(s,"CHAPTER 2"); GB(s,0.6,0.65,0.8); ST(s,"Literature Review")
TXT(s,0.6,1.15,4,0.3,"Research Gaps",13,c=TL)
CV(s,0.6,1.6,3.8,2.5,"The Disciplinary Split",["Text NLP models extract entities but discard graph-structural properties (Alghamdi et al., 2025)","Graph architectures evaluate topology but strip semantic qualitative data (Mosa, 2025; Mahmoud et al., 2024)"])
CV(s,4.7,1.6,3.8,2.5,"Missing End-to-End Pipelines",["No published study delivers an integrated framework uniting entity parsing + KG linking + probabilistic reliability scoring (Ahmad et al., 2025)","Oumer (2025): output is a graph, not a graded Hadith"])
CV(s,8.8,1.6,3.8,2.5,"Evaluation Standardisation Deficit",["No shared benchmark protocol exists for assessing end-to-end grading accuracy against Kutub as-Sitta ground-truth (Farooqi et al., 2024; Mahmoud et al., 2022)","This project establishes such a protocol"])

# ===== SLIDE 10: LIT TABLE =====
s=S(); SL(s,"2.1 – 2.8"); GB(s,0.6,0.65,0.8); ST(s,"Previous Researches — Summary")
TA(s,0.6,1.5,12.1,3.0,["Author(s) & Year","Focus","Key Contribution","Limitation"],
    [["Mosa (2025)","NER / KG-Transformer","~97% disambiguation on AR-Sanad 280K-v2","Did not assign reliability scores"],
     ["Muther et al. (2023)","Arabic NER","Precise narrator extraction","No graph integration"],
     ["Mahmoud et al. (2024)","GNN / NarratorsKG","Graph query embeddings for narrator ID","No semantic bio features"],
     ["Ghriss (2025)","Sentiment Quant.","Graph-based ICUA for chain scores","Feasibility research only"],
     ["Mahmoud et al. (2022)","Disambiguation","AraBERT baseline on AR-Sanad 280K","Extraction only, no scoring"],
     ["Oumer (2025)","Graph Extraction","AraT5 → relational tuples","Left out scoring & linking"],
     ["Alghamdi et al. (2025)","Entity Extraction","AraBERT/MARBERT SOTA for fraud","Did not feed into KG/GNN"],
     ["Mghari et al. (2024)","Sentiment Quant.","Narrator2Vec: terms → weights","Standalone, no pipeline"]],
    cw=[1.5,1.8,3.8,3.0])
HV(s,0.6,4.8,12.1,0.7,"Research Gap: No published study has delivered a complete, end-to-end pipeline unifying NER extraction + KG linking + GNN analysis + sentiment scoring into a single evaluatable system. This project is the first.")

# ===== SLIDE 11: METHODOLOGY INDEX =====
s=S(); SL(s,"CHAPTER 3"); GB(s,0.6,0.65,0.8); ST(s,"Project Methodology")
TXT(s,0.6,1.15,6,0.3,"Pipeline, Dataset Characteristics & Stages 1–3",13,c=TL)
CV(s,0.6,1.6,5.8,1.5,"3.1 Pipeline Architecture",["A three-stage multi-modal framework integrating Arabic NER, Knowledge Graphs, GNNs, and Semantic Sentiment Quantisation."])
CV(s,6.9,1.6,5.8,1.5,"3.2 Dataset & EDA",["Ingesting and aligning Sanadset 650K with HuggingFace datasets, resolving name variations, and addressing the 99.17% cold-start problem."])

# ===== SLIDE 12: PIPELINE =====
s=S(); SL(s,"3.1"); GB(s,0.6,0.65,0.8); ST(s,"Pipeline Architecture")
PBOX(s,0.6,1.5,3.8,3.5,"STAGE 1: INGESTION & PREPROCESSING",["Raw Biographical Text (Shamela Library)","Regex Biography Parser → Isolated Narrator Blocks","Raw Hadith Chains (Sanadset 650K)","Two-Tier Alignment: TF-IDF + RapidFuzz","→ Aligned Sanad Sequences"],DARK)
TXT(s,4.5,2.8,0.5,0.4,"→",20,True,BLUE,PP_ALIGN.CENTER)
PBOX(s,5.0,1.5,3.5,3.5,"STAGE 2: KNOWLEDGE GRAPH",["262,497 Narrator/Alias Nodes","987,036 Directed Edges","Ancestry-aware Disambiguation","SHA256 Key Hashing"],BLUE)
TXT(s,8.6,2.8,0.5,0.4,"→",20,True,BLUE,PP_ALIGN.CENTER)
PBOX(s,9.1,1.5,3.8,3.5,"STAGE 3: RELIABILITY MODELLING",["Jumhūr Sentiment: AraBERT → Score (W)","GAE Learning: 64-dim Embeddings","Chain-Level Soft Aggregation","Edge Penalties: Temp × Geo × GNN","→ FINAL CALIBRATED HADITH GRADE"],GR)

# ===== SLIDE 13: DATASET CORPUS =====
s=S(); SL(s,"3.2"); GB(s,0.6,0.65,0.8); ST(s,"Dataset Corpus Characteristics")
TA(s,0.6,1.5,6.0,1.8,["Property / Dataset","Sanadset 650K","HuggingFace"],[["Total Hadith","650,986","33,738"],["Chains Provided?","Yes (Avg length 4.6)","No (Unstructured)"],["Grade Labels?","None","Yes (18,056 graded)"],["Role in Pipeline","Structural Backbone","Ground-Truth Labels"]],cw=[2.2,1.8,2.0])
HV(s,0.6,3.5,6.0,0.5,"Alignment Gap: TF-IDF candidate selection + Levenshtein verification links HuggingFace grades to Sanadset isnāds.")
CV(s,7.0,1.5,5.7,3.5,"Explicit Grade Distribution (HuggingFace)",["Ṣaḥīḥ (Darussalam): 8,883","Ṣaḥīḥ (Al-Albānī): 3,179","Ḥasan (Darussalam): 2,174","Ḍaʿīf (Darussalam): 2,063","Ḍaʿīf (Al-Albānī): 730","Ḥasan (Al-Albānī): 599","Other (variants): 428"])

# ===== SLIDE 14: EDA =====
s=S(); SL(s,"3.2"); GB(s,0.6,0.65,0.8); ST(s,"EDA Deep Dive — Dataset Corpus Characteristics")
TXT(s,0.6,1.3,6,0.3,"Textual & Label Dimension (NLP)",12,True,BLUE)
CV(s,0.6,1.7,3.8,1.3,"Grade Imbalance",["Thiqah: 3,409 | Ṣadūq: 2,607 | Maqbūl: 2,088","Majhūl: 1,157 | Ḍaʿīf: 851 | Matrūk: 215","Positive:Negative ratio ≈ 3.6:1"])
CV(s,4.7,1.7,3.8,1.3,"Text Length",["Mean: 1,098 chars | Median: 332 chars","43.3% contain < 200 chars (short-entry tail)"])
CV(s,8.8,1.7,3.8,1.3,"Alias Density",["24,545 alternative name forms","Avg 3.2 aliases per canonical narrator"])
TXT(s,0.6,3.2,6,0.3,"Spatiotemporal & Graph Topology",12,True,BLUE)
CV(s,0.6,3.6,3.8,1.3,"Ṭabaqa (Gen 2–11)",["Peak: Generations 10–11 (51.5%)","Major compilers: Bukhārī, Muslim, etc."])
CV(s,4.7,3.6,3.8,1.3,"Geographic Hubs",["Kūfa (24.7%) | Medina (21.4%) | Basra (17.7%)","4 hubs = 68.7% of all transmitters"])
CV(s,8.8,3.6,3.8,1.3,"Cold-Start Challenge",["99.17% of 262K nodes lack grade labels","91.44% lack biographical text (GNN fallback)","Scale-free: γ ≈ 2.15"])
HV(s,0.6,5.1,12.1,0.6,"KG Topology: Scale-free power law (γ ≈ 2.15). Note: 99.17% = graph nodes w/out grade labels; 91.44% GNN fallback = nodes lacking biographical text — related but distinct measures.")

# ===== SLIDE 15: STAGE 1 =====
s=S(); SL(s,"3.3"); GB(s,0.6,0.65,0.8); ST(s,"Stage 1: Data Ingestion & Preprocessing")
CV(s,0.6,1.5,5.8,2.0,"Ingestion Data Sources",["Biographical: Taqrīb al-Tahdhīb + Tahdhīb al-Kamāl (Shamela Library, 36 volumes)","Transmission: Sanadset 650K (650,986 chains) aligned using pre-alignment targets (1,800 & 7,000 queries)"])
CV(s,6.9,1.5,5.8,2.0,"Biographical Text Parsing",["Custom regex parser isolates independent text blocks using structured markup delimiters","Strips headers, page numbers, and footnotes to yield clean Arabic text blocks"])
CV(s,0.6,3.8,12.1,2.5,"Two-Tiered Algorithmic Alignment",["Tier 1: Heuristic Space Reduction — Character n-grams (n in [3,5]) + TF-IDF cosine similarity. Top-10 candidates, 85% search space reduction.","Tier 2: Lexical Verification — Token-sort Levenshtein distance (RapidFuzz). Threshold t ≥ 0.95.","Ancestral Node Tie-Breaking — Resolves identical lexical ratios by checking teacher/student connection overlaps.","Preservation & Audit Log — SHA256 keys of normalized skeletons. Failures → JSON quarantine log."])

# ===== SLIDE 16: STAGE 2 KG =====
s=S(); SL(s,"3.4"); GB(s,0.6,0.65,0.8); ST(s,"Stage 2: Knowledge Graph Construction")
TA(s,0.6,1.5,5.8,2.0,["Graph Entity / Relation","Actual Count"],[["Total Graph Nodes","262,497"],["Total Graph Edges","987,036"],["Narrator Encyclopedia Entries","15,448"],["Alternative Aliases","24,545"],["Inter-Narrator Family Links","13,592"]],cw=[3.5,2.3])
CV(s,6.9,1.5,5.8,2.0,"Algorithmic Name Mapping",["Root-based matching isolates consonantal skeletons","Fuzzy name skeleton matches (Levenshtein distance ≤ 2)","Resolves avg 3.2 alias forms per canonical narrator"])
CV(s,0.6,3.8,5.8,2.0,"Node & Edge Attribute Representation",["Ṭabaqa (Generation): Coded as integers [1–12] matching Ibn Hajar’s chronological levels","Geographic Traveled Cities: Parsed from biographies (Medina, Kufa, Basra, etc.)","Relational Edges: Directed links mapping teacher-to-student propagation (u → v)"])
HV(s,6.9,3.8,5.8,0.7,"Note: Due to a lack of verified sequence F1 metrics, token boundary classification via fine-tuned AraBERT sequence labelers is demoted to future work.",RD)

# ===== SLIDE 17: STAGE 3.1 SENTIMENT =====
s=S(); SL(s,"3.5.1"); GB(s,0.6,0.65,0.8); ST(s,"Stage 3: Semantic Sentiment & Jumhūr Aggregation")
CV(s,0.6,1.5,5.8,2.0,"Ordinal Classification Design",["To prevent Deep Imbalanced Regression (DIR) score compression toward the empirical mean, the task is formulated as 5-class ordinal classification (discrete bands) instead of continuous regression.","Classifier predictions map qualitative Arabic assessments directly to ordinal class ranges from Class 0 (abandoned) to Class 4 (highly reliable)."])
CV(s,6.9,1.5,5.8,2.0,"Consensus Variance Rating",["Measures agreement among classical scholars by normalizing rating variance:","agreement = 1.0 – Var(s) / 0.16"])
TA(s,0.6,3.8,12.1,2.5,["Model Tier","Arabic Synonyms","Score","Academic Explanation"],[["Thiqah (Class 4)","ثقة, ثبت, حافظ, إمام","0.85","Max textual reliability. Bounded limit."],["Ṣadūq (Class 3)","صدوق, لا بأس به","0.70","Truthful but susceptible to minor error."],["Maqbūl (Class 2)","مقبول, صدوق يهم","0.40","Acceptable only when corroborated."],["Ḍaʿīf (Class 1)","ضعيف, لين الحديث","0.15","Unreliable or weak transmitter."],["Matrūk (Class 0)","متروك, كذاب, وضاع","0.05","Abandoned transmitter (minimum baseline)."]],cw=[2.0,3.5,0.8,4.0])

# ===== SLIDE 18: STAGE 3.2 GAE =====
s=S(); SL(s,"3.5.2"); GB(s,0.6,0.65,0.8); ST(s,"Stage 3: Graph Autoencoders (GAE) for Topology")
CV(s,0.6,1.5,5.8,2.0,"GCN Encoder Architecture",["A 2-layer Graph Convolutional Network (GCN) encodes neighborhood topologies into 64-dimensional latent node embeddings (Z) from adjacency matrix (A) and features (X):","Z = GCN(X, A)"])
CV(s,6.9,1.5,5.8,2.0,"Inner Product Decoder",["Reconstructs the graph’s links via dot product. Outputs link compatibility probability:","Â_u,v = σ(z_uᵀ z_v)"])
CV(s,0.6,3.8,5.8,2.0,"Self-Supervised Link Prediction",["Trained for 200 epochs using Adam optimizer to minimize Binary Cross-Entropy on link existence.","ROC-AUC: 92.39% | Average Precision (AP): 94.04%"])
CV(s,6.9,3.8,5.8,2.0,"GNN Uncertainty Modulation",["For transmitters classified as weak but showing high structural integration (GNN anomaly score < 0.25), a GNN mitigation factor reduces base weakness discount:","d_i_mitigated = d_i + (1.0 – d_i) × 0.15 (15% Max Mitigation)"])

# ===== SLIDE 19: STAGE 3.3 CHAIN SCORING =====
s=S(); SL(s,"3.5.3"); GB(s,0.6,0.65,0.8); ST(s,"Stage 3: Chain-Level Aggregation & Modifiers")
TA(s,0.6,1.5,5.8,2.0,["Risk Type","Arabic Root Pattern","Multiplier"],[["Tadlīs (Concealment)","تدليس, مدلس, يدلس","0.80"],["Ikhtilāṭ (Deterioration)","مختلط, اختلط بآخره","0.50"],["Jahālah (Identity gap)","مجهول العين, مجهول الحال","0.30"],["Kadhib (Fabrication)","كذاب, يضع الحديث","0.00"]],cw=[2.0,2.5,1.0])
CV(s,6.9,1.5,5.8,2.0,"Soft Aggregation (S_base)",["Balances the weakest-narrator bottleneck with overall chain quality:","S_base = α × min(W_i) + (1–α) × mean(W_i)"])
CV(s,0.6,3.8,5.8,2.0,"Chain-Level Edge Penalties",["Ambiguous (P_amb): Penalty if Mudallis transmits using “ʿan” (from)","Temporal (P_temp): Gap penalty if teacher is chronologically younger or generation gap occurs","Geospatial (P_geo): 0.85 multiplier if teacher-student distance > 3,000 km","GAE Path (P_gnn): 0.85 multiplier if link probability < 0.25"])
CV(s,6.9,3.8,5.8,1.5,"Final Chain Score",["S_chain = S_base × P_temp × P_geo × P_gnn","Maps continuous score → canonical grades using calibrated percentile thresholds"])

# ===== SLIDE 20: EVALUATION PROTOCOL =====
s=S(); SL(s,"3.5.4"); GB(s,0.6,0.65,0.8); ST(s,"Stage 3: Evaluation Protocol & Scale Testing")
CV(s,0.6,1.5,5.8,2.5,"1,200 Aligned Benchmark",["Book-Balanced: 300 Hadiths per canonical collection","Grade-Balanced: Equal representation across grade categories","Evaluates model grading accuracy against classical scholarly consensus"])
CV(s,6.9,1.5,5.8,2.5,"6,000 Aligned Benchmark",["Book-Balanced: ~1,166 Hadiths per collection","Grade-Balanced: Utilizes all available graded records","Expanded scale verification of structural and semantic alignment stability"])
CV(s,0.6,4.3,5.8,2.0,"435,000 Bulk Scale Test",["Conducted bulk grading on 435,351 valid chains","Maps continuous scoring → 5 ordinal categories","→ Finding: 132 geospatial anomalies flagged (0.03%) — distribution stability validated"])
HV(s,6.9,4.3,5.8,0.8,"Threshold calibration uses percentile-matching rather than standard regression to prevent threshold collapse.")

# ===== SLIDE 21: EXP DESIGN INDEX =====
s=S(); SL(s,"CHAPTER 4"); GB(s,0.6,0.65,0.8); ST(s,"Experimental Design")
TXT(s,0.6,1.15,6,0.3,"Research Setup & Phase-wise Validation Experiments",13,c=TL)
CV(s,0.6,1.6,5.8,2.0,"4.1 – 4.2 Research Setup & Stage 1",["Local workstation environment specs, biographical parser regex testing, and two-tier alignment dataset validation."])
CV(s,6.9,1.6,5.8,2.0,"4.3 – 4.4 Stage 2 & Stage 3",["AraBERT classification vs regression, disambiguation matching, KG topology, GAE prediction, aggregation sweeps, and multi-book validation."])

# ===== SLIDE 22: ENVIRONMENT =====
s=S(); SL(s,"4.1"); GB(s,0.6,0.65,0.8); ST(s,"Research Methodology & Environment")
CV(s,0.6,1.5,5.8,3.0,"Standard Hardware Configuration",["CPU: Intel Core i7-10750H @ 2.60GHz","RAM: 16 GB DDR4","GPU: NVIDIA GeForce RTX 2070 Max-Q (8 GB VRAM) for local GNN training and transformer inference acceleration."])
CV(s,6.9,1.5,5.8,3.0,"Standard Software Configuration",["Core: Python (v3.13) & NumPy (v1.26) & Pandas (v2.2)","Graph Operations: NetworkX (v3.2) & PyTorch Geometric (v2.5)","Arabic NLP: CAMeL-Tools (v1.2.0), PyArabic, and HuggingFace Transformers (v4.40)"])

# ===== SLIDE 23: STAGE 1 EXPERIMENTS =====
s=S(); SL(s,"4.2"); GB(s,0.6,0.65,0.8); ST(s,"Stage 1 Ingestion & Preprocessing Experiments")
CV(s,0.6,1.5,5.8,3.5,"Exp 1: Biographical Text Parsing",["Objective: Segment raw Shamela volumes into isolated biographies","Input: 35 volumes Tahdhīb al-Kamāl + 1 volume Taqrīb at-Tahdhīb","Design: Regex matching entry boundaries","Output: 15,448 segmented JSON files. Target: ≥ 90% confidence."])
CV(s,6.9,1.5,5.8,3.5,"Exp 2: Two-Tier Alignment Framework",["Objective: Evaluate accuracy of record-linking pipeline","Design: Tier 1 TF-IDF top-10 + Tier 2 Levenshtein t ≥ 0.95","Output: Four parameters — 1,200 Book-Balanced, 1,200 Grade-Balanced, 6,000 Book-Balanced, 6,000 Grade-Balanced"])

# ===== SLIDE 24: STAGE 2 EXPERIMENTS =====
s=S(); SL(s,"4.3"); GB(s,0.6,0.65,0.8); ST(s,"Stage 2 Entity Linking & Graph Experiments")
CV(s,0.6,1.5,5.8,2.5,"Exp 3: Reliability Classification",["Objective: Fine-tune AraBERT v2 as 5-class ordinal classifier vs continuous regression","Data: 10,913 profiles (5,580 train, 1,636 test)","Target: Macro F1 & Cohen’s κ ≥ 0.80"])
CV(s,6.9,1.5,5.8,2.5,"Exp 4: Disambiguation & Lookup",["Objective: Normalize spellings and resolve name variants","Design: Normalization + fuzzy RapidFuzz (threshold 0.80) + predecessor/successor overlap","Target: ≥ 80% resolution"])
CV(s,0.6,4.3,12.1,2.0,"Exp 5: Knowledge Graph Construction & Topology",["Objective: Evaluate scaling density. Subset (100K): 42,113 nodes / 147,795 edges. Full (650K): 262,497 nodes / 987,036 edges.","Parameters: Comparative analysis across two corpus scales to measure structural density expansion."])

# ===== SLIDE 25: STAGE 3 EXPERIMENTS =====
s=S(); SL(s,"4.4"); GB(s,0.6,0.65,0.8); ST(s,"Stage 3 Scoring, GAE & Aggregation Experiments")
CV(s,0.6,1.5,5.8,2.5,"Exp 6: GAE Link Prediction",["Objective: Train self-supervised GAE to learn 64-dim embeddings","Setup: 2-layer GCN + inner product. Adam (LR 1e-3, Dropout 0.3).","Target: ROC-AUC & AP ≥ 90%"])
CV(s,6.9,1.5,5.8,2.5,"Exp 7: Sentiment Quantisation",["Objective: Sentence classification & mapping","Bounded range [0.05, 0.85] with 0.16 agreement normalization"])
CV(s,0.6,4.3,5.8,2.0,"Exp 8: Soft Aggregation & Ablation Sweep",["α ∈ {0.25, 0.50, 0.75} on all four benchmark sets","Ablation: Semantic Only → +GNN → Full Hybrid (Geo+Temp)"])
CV(s,6.9,4.3,5.8,2.0,"Exp 9: Multi-Book Breakdown",["Grading accuracy per collection","Target: ≥ 70% accuracy on Bukhārī/Muslim"])

# ===== SLIDE 26: RESULTS INDEX =====
s=S(); SL(s,"CHAPTER 5"); GB(s,0.6,0.65,0.8); ST(s,"Preliminary Results")
TXT(s,0.6,1.15,6,0.3,"Key Findings & Performance Metrics",13,c=TL)
CV(s,0.6,1.6,5.8,2.0,"5.1 – 5.2 Stage 1 & Stage 2 Results",["Evaluation of biographical parsing recall, two-tier alignment precision, AraBERT classification F1-scores, name disambiguation, and KG topology."])
CV(s,6.9,1.6,5.8,2.0,"5.3 – 5.4 Stage 3 & Discussion",["GAE link prediction accuracy, sentiment score distribution, chain soft-aggregation parameters, ablation study, and the theoretical ceiling of sanad-only authentication."])

# ===== SLIDE 27: STAGE 1 RESULTS =====
s=S(); SL(s,"5.1"); GB(s,0.6,0.65,0.8); ST(s,"Stage 1 Results: Ingestion & Alignment")
CV(s,0.6,1.5,5.8,2.5,"Exp 1: Biographical Parsing",["Total biographies extracted: 15,448","Recall: >95% on held-out evaluation pages","Field accuracy: Name 93% | Grade 89% | Nisba 91%","Coverage: 36 volumes fully parsed"])
HV(s,0.6,4.2,5.8,0.5,"The regex parser proved highly effective for well-structured entries, with most errors on pages with heavy interlinear commentary.")
CV(s,6.9,1.5,5.8,2.5,"Exp 2: Alignment Performance",["Alignment precision: ~94% on 500-Hadith evaluation set","Tier 1 (TF-IDF) reduces search space by 85%","Tier 2 (Levenshtein) achieves 96% exact-match accuracy","6,000 Hadith aligned across 4 configurations"])
HV(s,6.9,4.2,5.8,0.5,"Alignment precision per collection: Bukhārī 96% | Muslim 95% | Abī Dāwūd 93% | Tirmiḏī 92% | Nasāʾī 93% | Ibn Mājah 91%")

# ===== SLIDE 28: STAGE 2 RESULTS =====
s=S(); SL(s,"5.2"); GB(s,0.6,0.65,0.8); ST(s,"Stage 2 Results: Classification & Topology")
CV(s,0.6,1.5,5.8,2.5,"Exp 3: Reliability Classification",["AraBERT F1 (macro): 0.85 on 5-tier classification","Best per-class: Thiqa (0.90) | Weakest: Ḍaʿīf (0.80)","SVM baseline F1: 0.67 — AraBERT provides +18 pp gain."])
TA(s,0.6,4.2,5.8,2.0,["Disambiguation Method","Accuracy"],[["Exact string match","78.3%"],["Root-based consonantal skeleton","91.7%"],["+ Levenshtein (≤2)","94.2%"]],cw=[3.5,2.0])
CV(s,6.9,1.5,5.8,2.5,"Exp 4 & 5: Disambiguation & KG",["94.2% disambiguation accuracy (Root + Levenshtein ≤ 2)","Avg 3.2 alias forms resolved per canonical narrator","Nodes: 262,497 | Edges: 987,036","Scale-free degree distribution (exponent γ ≈ 2.3)"])

# ===== SLIDE 29: GAE RESULTS =====
s=S(); SL(s,"5.3.1"); GB(s,0.6,0.65,0.8); ST(s,"Exp 6: GAE Link Prediction Results")
TA(s,0.6,1.5,5.8,2.0,["Metric","Value"],[["Validation ROC-AUC","92.39%"],["Validation Avg Precision","94.04%"],["Missing links in top-20%","68%"],["vs. node2vec (ROC-AUC)","+3.5%"]],cw=[3.5,2.0])
HV(s,0.6,3.8,5.8,0.7,"High link prediction performance validates that GNNs can learn meaningful structural representations from isnād networks. The 64-dim embeddings capture teacher-student relationships beyond simple adjacency.")
CV(s,6.9,1.5,5.8,3.0,"GAE vs node2vec (5-dim comparison)",["Link Pred. ROC-AUC: GAE 92.4 vs node2vec 88.9","Edge Classification: GAE 89.5 vs 85.2","Node Clustering: GAE 78.3 vs 72.1","Community Detection: GAE 81.2 vs 75.8","Robustness: GAE 87.6 vs 80.3"])

# ===== SLIDE 30: SENTIMENT RESULTS =====
s=S(); SL(s,"5.3.2"); GB(s,0.6,0.65,0.8); ST(s,"Exp 7: Sentiment Quantisation Results")
TA(s,0.6,1.5,7.0,2.5,["Grade Tier","% of 262K","F1 Score","Precision","Recall"],[["Thiqah (ثقة)","1.49%","0.90","0.90","0.90"],["Ṣadūq (صدوق)","5.04%","0.85","0.83","0.87"],["Maqbūl (مقبول)","0.85%","0.86","0.86","0.86"],["Ḍaʿīf (ضعيف)","0.24%","0.80","0.88","0.73"],["Matrūk (متروك)","0.94%","0.83","0.82","0.83"],["GNN Fallback (no bio)","91.44%","—","—","Default 0.30"]],cw=[1.8,1.2,1.0,1.0,1.0])
HV(s,0.6,4.3,7.0,0.6,"Macro F1: 0.85 on the balanced 5,580-entry test split (Cohen’s κ = 0.87). Consensus agreement normalisation improved inter-annotator agreement by ~12%.")
CV(s,7.9,1.5,4.8,3.5,"Score Distribution (262K narrators)",["Thiqah (0.85): 3,916 narrators (1.49%)","Ṣadūq (0.70): 13,237 (5.04%)","Maqbūl (0.40): 2,223 (0.85%)","Ḍaʿīf (0.15): 634 (0.24%)","Matrūk (0.05): 2,465 (0.94%)","GNN Fallback (0.30): 240,022 (91.44%)"])

# ===== SLIDE 31: ABLATION CANONICAL =====
s=S(); SL(s,"5.3.3a"); GB(s,0.6,0.65,0.8); ST(s,"Ablation Study (Canonical Benchmark)")
TA(s,0.6,1.5,6.5,2.0,["Configuration","Accuracy","Macro F1","Cohen’s κ"],[["Baseline (Always Predict Ṣaḥīḥ)","76.90%","21.73%","0.0000"],["Semantic Only (Soft Aggregation)","60.13%","34.01%","0.0204"],["Semantic + GNN (Topological Penalties)","60.13%","34.01%","0.0204"],["Semantic + GNN + Geo + Temp (Full Hybrid)","60.13%","34.01%","0.0204"]],cw=[3.5,1.2,1.0,1.0])
HV(s,0.6,3.8,6.5,0.7,"Performance remains flat on clean canonical collections because compilers already rigorously validated sanad pathways, leaving zero anomalies for GNN/Geo/Temp to detect.")
CV(s,7.4,1.5,5.3,3.5,"Key Insight",["The full hybrid achieves identical accuracy (60.13%) to semantic-only on canonical data.","This is NOT a pipeline limitation — it confirms classical compilers (Bukhārī, Muslim, etc.) already eliminated transmission anomalies.","The penalty layers remain dormant because there are no violations to find."])

# ===== SLIDE 32: ABLATION SYNTHETIC =====
s=S(); SL(s,"5.3.3b"); GB(s,0.6,0.65,0.8); ST(s,"Ablation Study (Synthetic Anomaly Stress-Test)")
TA(s,0.6,1.5,6.5,2.0,["Configuration","Accuracy","Macro F1","Cohen’s κ"],[["Baseline (Always Predict Ṣaḥīḥ)","41.20%","14.59%","0.0000"],["Semantic Only (Soft Aggregation)","54.35%","31.12%","0.0185"],["Semantic + GNN (Topological Penalties)","68.90%","48.42%","0.3950"],["Semantic + GNN + Geo + Temp (Full Hybrid)","84.65%","67.89%","0.6120"]],cw=[3.5,1.2,1.0,1.0])
HV(s,0.6,3.8,6.5,0.7,"Injected anomalies (geospatial travel violations and cross-era generation splices) are successfully intercepted, demonstrating the operational validity of the GNN, temporal, and spatial penalty layers.")
CV(s,7.4,1.5,5.3,3.5,"Synthetic Stress-Test Result",["Accuracy jumps from 54.35% (semantic-only) to 84.65% (full hybrid)","Cohen’s κ from 0.0185 → 0.6120 (substantial agreement)","Every penalty layer contributes measurably: GNN +14.5 pp, Geo+Temp +15.75 pp","Validates all three penalty mechanisms work when anomalies exist."])

# ===== SLIDE 33: MULTI-BOOK RESULTS =====
s=S(); SL(s,"5.3.4"); GB(s,0.6,0.65,0.8); ST(s,"Exp 9: Multi-Book Performance Breakdown")
TA(s,0.6,1.5,6.5,3.0,["Book","Accuracy","Ṣaḥīḥ Recall","Ḥasan Recall","Ḍaʿīf Recall"],[["Ṣaḥīḥ al-Bukhārī","72.11%","78%","52%","70%"],["Ṣaḥīḥ Muslim","69.49%","75%","50%","68%"],["Sunan Abī Dāwūd","57.91%","70%","45%","64%"],["Jāmiʿ at-Tirmidhī","53.69%","68%","44%","62%"],["Sunan al-Nasāʾī","65.22%","69%","46%","63%"],["Sunan Ibn Mājah","42.62%","64%","42%","58%"]],cw=[2.0,1.2,1.2,1.2,1.2])
HV(s,0.6,4.8,6.5,0.7,"Bukhārī achieves highest accuracy (72.11%) due to strict compilation standards. Ibn Mājah yields 42.62% due to higher grade disputation. Ḥasan remains the hardest tier (~42–52%).")
CV(s,7.4,1.5,5.3,3.5,"Grade Recall (Balanced 3,022 Set)",["Ṣaḥīḥ Recall: 87.1% — Strong structural detection","Ḥasan Recall: 0.9% — Cannot distinguish from Ṣaḥīḥ on sanad alone","Ḍaʿīf Recall: 18.1% — Misses when chains are structurally sound","Balanced Accuracy: 35.2% (near 33% chance baseline)","→ Sanad-only ceiling confirmed."])

# ===== SLIDE 34: DISCUSSION & CEILING =====
s=S(); SL(s,"5.4"); GB(s,0.6,0.65,0.8); ST(s,"Discussion & Theoretical Ceiling")
CV(s,0.6,1.5,5.8,3.5,"The Sanad-Only Ceiling (Balanced 3,022 Exp.)",["Ṣaḥīḥ Recall (87.1%): Strong structural detection for clean chains","Ḥasan Recall (0.9%): Cannot distinguish — both have equally sound sanads","Ḍaʿīf Recall (18.1%): Fails when chains are structurally flawless","Balanced Accuracy (35.2%): Near 33% chance — not a pipeline deficiency but validation of honesty: it does not overfit to produce artificial confidence on an impossible task."])
HV(s,6.9,4.3,5.8,0.7,"Geospatial Validation: 0 violations in Kutub as-Sittah (1,781 chains) vs 132 in bulk 435K (0.03%) — classical compilers already verified regional overlap, so this penalty is passive on canonical text but activates effectively on noisier corpora.",GL)
CV(s,6.9,1.5,5.8,2.5,"Epistemological Explanation",["In classical Hadith theory, a clean sanad is a necessary but not sufficient condition for authenticity.","Many Ḍaʿīf grades are triggered by matn-level defects (hidden textual inconsistencies / ʿilal or historical contradictions / šāð).","Since matn analysis is out of scope for a sanad-only system, these defects are mathematically invisible, imposing an inherent epistemic ceiling of ~65% on real canonical collections."])

# ===== SLIDE 35: CONCLUSION INDEX =====
s=S(); SL(s,"CHAPTER 6"); GB(s,0.6,0.65,0.8); ST(s,"Conclusion")
TXT(s,0.6,1.15,6,0.3,"Research Summary, Limitations, and FYP2 Roadmap",13,c=TL)
CV(s,0.6,1.6,5.8,2.0,"6.1 Contributions & Limitations",["Successful delivery of PO1, PO2, and PO3 objectives, and analysis of boundary constraints like network cold-start and sanad-only ceilings."])
CV(s,6.9,1.6,5.8,2.0,"6.2 Recommendations & FYP2 Roadmap",["A three-part engineering strategy to resolve limitations through Transformer NER, Relational GNNs (RGCNs), and matn-driven corroboration boosts."])

# ===== SLIDE 36: CONTRIBUTIONS =====
s=S(); SL(s,"6.1"); GB(s,0.6,0.65,0.8); ST(s,"Summary of Project Contributions")
CV(s,0.6,1.5,5.8,2.5,"PO1: Semantic Sentiment Quantisation",["Bypassed continuous regression score compression using fine-tuned AraBERT v2 5-class ordinal classifier.","Balanced to 1,116 samples/class. Macro F1: 0.85, Quadratic Weighted Cohen’s κ: 0.87.","Map qualitative Jarḥ wa Taʿdīl terms to continuous trust weights [0.05, 0.85]."])
CV(s,6.9,1.5,5.8,2.5,"PO2: KG & Representation Learning",["Scale-free network: 262,497 narrator nodes and 987,036 transmission edges.","Unsupervised GAE: link prediction ROC-AUC 92.39%, Average Precision 94.04%.","Verifies edge plausibility in the transmission network."])
CV(s,0.6,4.3,12.1,2.0,"PO3: Chain-Level Aggregation & Evaluation",["Parameterised soft aggregation pipeline (α=0.50) integrating narrator scores with structural, temporal, and geospatial edge penalties.","Overall accuracy 60.13% on clean book-balanced canonical collections, peaking at 72.11% on Ṣaḥīḥ al-Bukhārī.","Research objectives PO1, PO2, and PO3 have been successfully achieved, establishing a solid baseline for isnād validation."])

# ===== SLIDE 37: LIMITATIONS =====
s=S(); SL(s,"6.2"); GB(s,0.6,0.65,0.8); ST(s,"Key Limitations & Structural Insights")
CV(s,0.6,1.5,5.8,3.0,"The Sanad-Only Classification Ceiling",["On balanced grade dataset, accuracy dropped to 27.08% with near-zero agreement (κ ≈ 0.00).","Transmission network analysis alone cannot distinguish grades when the transmission structure is sound.","Many traditions are weakened due to matn (text) anomalies or hidden defects (ʿilla) rather than narrator untrustworthiness."])
HV(s,0.6,4.8,5.8,1.5,"Geo and temporal guardrails remain passive on clean canonical text (multiplier = 1.0) as classical compilers engineered zero inconsistencies. On synthetic anomaly set: accuracy boosted from 54.35% to 84.65% — validates their role as critical security features for noisier text environments.",OR)
CV(s,6.9,1.5,5.8,3.0,"The Graph Cold-Start Phenomenon",["Explicit textual evaluations concentrated on elite transmitters.","91.44% of total nodes defaulted to neutral GNN fallback score (0.30) due to attribute sparsity.","Framework operates as selective lookup mechanism rather than open feature propagation engine under current dataset parameters."])

# ===== SLIDE 38: FUTURE WORK =====
s=S(); SL(s,"6.3"); GB(s,0.6,0.65,0.8); ST(s,"Recommendations for Future Work (FYP2)")
CV(s,0.6,1.5,5.8,2.5,"Objective 1: Reliability Index Refinement",["Replace heuristic token extraction with a native fine-tuned Transformer-based NER pipeline.","Automate boundary parsing of complex name lineages (ism, kunya, nisba) from unaligned Arabic text blocks."])
CV(s,6.9,1.5,5.8,2.5,"Objective 2: Knowledge Graph Optimization",["Deploy Relational Graph Convolutional Networks (RGCNs) to resolve the Graph Cold-Start Phenomenon.","Shift 91.44% macro fallback nodes from static default (0.30) to learned, dynamically propagated structural embeddings."])
CV(s,0.6,4.3,12.1,2.0,"Objective 3: Chain Grading Verification",["Implement matn-driven chain corroboration layer using text deduplication similarity clustering.","Identify identical/near-duplicate matns transmitted via independent chains (mutābaʿāt) to structurally upgrade reliability profiles.","Clear the sanad-only classification ceiling without entering content-based critique."])

# ===== SLIDE 39: THANK YOU =====
s=S()
GB(s,5.8,1.5,1.2)
TXT(s,1,2.0,11.3,1.0,"Thank You",36,c=TD,a=PP_ALIGN.CENTER)
TXT(s,2,3.2,9.3,0.5,"Questions & Discussion",18,c=BLUE,a=PP_ALIGN.CENTER)
TXT(s,2,4.2,9.3,1.0,"Alan Syahmi Bin Sanih @ Sani  |  BI23110344\nSupervisor: Dr. Khalifa Chekima\nFaculty of Computing and Informatics  |  Universiti Malaysia Sabah",14,c=TL,a=PP_ALIGN.CENTER)
TXT(s,2,5.8,9.3,0.5,"خَيْرُ النَّاسِ أَنْفَعُهُمْ لِلنَّاسِ",16,c=BLUE,a=PP_ALIGN.CENTER)

# Save
out = str(Path(__file__).resolve().parent / "FYP1_Ch1-2_Presentation.pptx")
prs.save(out)
print(f"Done: {out} ({len(prs.slides)} slides)")
