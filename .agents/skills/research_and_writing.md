# Academic Research & Thesis Drafting

## Overview
Guidelines for assisting the user in writing the FYP thesis and conducting literature reviews, tailored specifically to a Computer Science/Islamic Studies interdisciplinary topic.

## 1. Quality & Tone
- **Academic Tone**: ALWAYS maintain a formal, objective, and academic tone. Avoid narrative styling (e.g., "In this chapter we will see..."). Rely on passive voice or formal active voice (e.g., "This chapter evaluates...").
- **Anti-Hallucination Protocol**: NEVER hallucinate Hadith chains (Sanads), text (Matn), or narrator names. Always rely on the user's provided data from established corpora (e.g., AR-Sanad 280K, Sanadset 650K). If a specific example is needed, ask the user to provide one.

## 2. Literature Review Strategy
- **Thematic Grouping**: Do not list papers chronologically. Group them by themes: 
  - *Theme 1*: Classic Rule-based NER in Arabic.
  - *Theme 2*: Deep Learning and Transformers for Arabic Text.
  - *Theme 3*: Knowledge Graphs in Islamic Studies.
  - *Theme 4*: Automated Trust/Reliability Scoring (The core gap).
- **Emphasizing the Gap**: Frame the literature review around the transition from simple narrator extraction/disambiguation to the ultimate goal of automated Jarh wa Ta'dil scoring and reliability aggregation.

## 3. Citation and Referencing
- Prioritize **IEEE Citation Style** (standard for CS/Tech) unless the user's university formatting template specifies otherwise.
- When summarizing notes from Scispace or Grok, include inline citation placeholders (e.g., `[Author, Year]`) to ensure the user can trace claims back to the source.

## 4. Drafting Structure 
Whenever asked to draft a chapter, align strictly with the user's provided `Appendix E FYP Thesis Template.docx`. Typical structure:
- **Chapter 1: Introduction** (Background, Problem Statement, Objectives, Scope).
- **Chapter 2: Literature Review** (Theoretical background, prior works, research gap).
- **Chapter 3: Methodology** (System architecture, datasets, algorithms, evaluation metrics).
- **Chapter 4: Results & Discussion** (Experimental setup, quantitative results, error analysis).
- **Chapter 5: Conclusion & Future Work**.
