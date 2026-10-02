# Chapter 1: Introduction

## 1.1 Overview

The authentication of prophetic traditions (Hadith) constitutes one of the most intricate scholarly endeavours in classical Islamic civilization. Central to this process is *'Ilm ar-Rijal* — the science of evaluating the men and women who form the chains of narration (isnāds) through which a Hadith is transmitted. Within this tradition, the discipline of *Jarh wa Ta'dīl* (literally "criticism and commendation") provides the epistemological framework by which scholars assess the moral probity (*'adālah*) and mnemonic precision (*dabt*) of every individual narrator in a given chain (Kamali, 2022). Over centuries, this practice produced vast historical corpuses of biographical literature — monuments of scholarly precision such as Ibn Hajar al-'Asqalāni's twelve-volume *Tahdhīb al-Tahdhīb* — each representing thousands of qualitative evaluative judgements that collectively determine the trustworthiness of the prophetic record.
In the contemporary era, the digitisation of classical texts through global repositories such as OpenITI and the Shamela Digital Library has made the Hadith corpus accessible at unprecedented scale (El Ganadi et al., 2025; Dallil et al., 2022). Yet, paradoxically, this very accessibility has created a new challenge: the capacity of modern scholarship to rigorously authenticate and verify Hadith at the speed of digital dissemination has not kept pace with the volume of their circulation. Recent scholarship identifies a "validity crisis" where the speed of digital access often outpaces the ability to verify complex chains of transmission (Azizah, 2023; Nur, 2022). The manual cross-referencing of biographical entries across hundreds of classical volumes remains a laborious, expert-dependent process that is inherently resistant to scale, presenting a compounding challenge for the field as a whole (Pavlova, 2025; Boukhaled, 2022).
This project proposes to close this gap by developing a multi-modal, data-driven framework that automates the process of narrator reliability evaluation grounded in the classical principles of *Jarh wa Ta'dīl*. Specifically, the framework integrates three advanced computational paradigms: (1) Transformer-based Arabic Named Entity Recognition (NER) for the precise extraction and disambiguation of narrator identities from classical texts (Alghamdi et al., 2025; Mahmoud et al., 2022); (2) Graph Neural Networks (GNN) for modelling the structural and temporal integrity of isnād transmission chains (Mosa, 2025; Farooqi et al., 2024); and (3) Semantic Sentiment Quantisation to translate qualitative scholarly evaluations into a standardised, continuous reliability index (Mghari et al., 2024; Ahmad et al., 2025). By uniting these techniques, this research aims to produce a reproducible, explainable, and epistemologically grounded system that directly supports the traditional science of *'Ilm ar-Rijal* through the methods of modern Data Science (A'thoina & Hasballah, 2025).



## 1.2 Problem Background
The evaluation of Hadith authenticity through 'Ilm ar-Rijal remains a cornerstone of Islamic scholarship, as the credibility of a prophetic tradition is fundamentally tied to the reliability of its chain of transmission (isnad). This rigorous verification system, described by classical scholars as a "protective shield" for the Sunnah, necessitates that every link in the chain be scrutinised for both moral integrity (‘adalah) and mnemonic precision (dabt) (Kamali, 2022). Traditionally, this process is a manual, labour-intensive endeavour that requires scholars to cross-reference thousands of qualitative biographical entries across vast historical corpuses, such as the 12-volume Tahdhib at-Tahdhib of Ibn Hajar or the foundational Andalusian Maliki assessments in Ibn ‘Abd al-Barr’s At-Tamhid (Al-Asqalani, d. 852 AH; Ibn ‘Abd al-Barr, d. 463 AH).
Feasibility studies confirm that machine learning can assist in automating isnād analysis, provided that the nuances of classical Arabic and the methodological standards of traditional jurisprudence are preserved (Usman, 2025; Mahmoud et al., 2022). The consonantal script and orthographic ambiguity of classical texts make this particularly challenging, as minor variations in name orthography can lead to misidentification of narrators across manuscript traditions (El Ganadi et al., 2025). Consequently, there exists an urgent need for Natural Language Processing (NLP) techniques to both extract and disambiguate narrator identities, and for Graph Neural Networks (GNN) to model the complex relational networks those narrators form (Farooqi et al., 2024; Mahmoud et al., 2022).
Recent work has confirmed the technical viability of this approach: transformer-based models like AraBERT have achieved high accuracy in identifying ambiguous narrators from biographical text, while GNN frameworks can generate node embeddings to capture the relative position and credibility of a narrator within the broader transmission network (Ahmad et al., 2025; Mahmoud et al., 2022). By transitioning to a systematically quantified framework, researchers can achieve higher consistency and speed in evaluation without compromising the epistemological roots of the science (Tarmom, 2022).


## 1.3 Problem Statements
Despite advancements in Arabic NLP, two primary challenges persist in the computational study of Hadith:
1. While current models have improved in Named Entity Recognition (NER) for extracting narrator names and lineages, there is a critical lack of computational frameworks capable of quantifying narrator reliability. Identifying a narrator is insufficient; the framework must be able to "judge" their standing based on historical evidence (Ahmad, 2025).
2. Current models typically treat the text and the transmission network as separate entities. They fail to integrate the semantic sentiment of Jarh wa Ta'dil (scholarly praise or criticism) found in biographical texts with the structural features of the transmission network. Without this multi-modal integration, automated grading remains superficial and fails to capture the latent patterns of teacher-student reliability explored in recent work by Mosa (2025).


## 1.4 Project Objectives
The main objectives of this project are:
1. To develop an NLP classification model that maps qualitative scholarly judgments to a standardised, quantitative reliability index (0.0 - 1.0).
2. To construct a Knowledge Graph representing teacher-student relationships and utilise GNNs to evaluate the structural integrity and temporal consistency of transmission chains.
3. To evaluate the framework's accuracy by comparing automated reliability scores against established scholarly Hadith gradings (Sahih, Hasan, Dha'if).




## 1.5 Project Scope

This project operates within a clearly defined set of boundaries to ensure that the research remains focused, technically feasible, and evaluatable within the constraints of a single academic year.

**Data Sources**
The framework will be trained and tested exclusively on digitised classical Arabic biographical texts available through the OpenITI and Shamela repositories, specifically the *Taqrīb al-Tahdhīb* of Ibn Hajar al-'Asqalāni and the *Tahdhīb al-Kamāl* of al-Mizzī. These texts contain the most comprehensive and widely referenced collections of narrator biographical assessments (*tarājim*) and thus represent the most suitable ground-truth source for *Jarh wa Ta'dīl* labels.

**Scale and Dataset**
The initial experimental scope targets approximately **1,500 narrator biographical entries** for building the reliability model and **1,000 validated Hadith isnāds** drawn from the *Al-Kutub al-Sittah* (The Six Canonical Books) for end-to-end testing. This scale is consistent with comparable feasibility studies in the literature (Mahmoud et al., 2022; Najeeb, 2022) and is sufficiently large to support statistically meaningful evaluation without exceeding available computational resources.

**Technical Scope**
The system will focus on three interconnected tasks: (1) **Narrator extraction** — identifying proper names, kunya (honorifics), and nisba (place-based identifiers) from isnāds; (2) **Entity disambiguation** — resolving alias variations to canonical narrator identities via a Knowledge Graph; and (3) **Reliability scoring** — mapping aggregated *Jarh wa Ta'dīl* labels to a continuous index [0.0 – 1.0]. The primary programming language is Python, utilising Arabic-specific NLP libraries (e.g., CAMeL Tools, PyArabic) and transformer models (e.g., AraBERT, MARBERT).

**Application Domain Limitations**
This research is strictly concerned with the **structural evaluation of isnāds** (Sanad criticism). The analysis of Hadith textual content (Matn criticism) — which requires comparative jurisprudential reasoning across the four Sunni schools — falls outside the scope of this project. Furthermore, any automated reliability score produced by the framework is intended as a **decision-support tool** for scholars, not as a definitive theological ruling or fatwa. The epistemological authority of qualified human *muhaddithun* is expressly preserved.

**Out of Scope**
The following are explicitly excluded from this project:
- Matn (textual body) analysis or content-based Hadith grading.
- Coverage of Shī'a isnād evaluation methodologies.
- Real-time or web-based deployment of the scoring pipeline.
- Languages other than Arabic (e.g., Urdu or Persian Hadith commentaries).

