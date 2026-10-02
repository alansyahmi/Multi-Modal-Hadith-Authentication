# Thesis Writing Style Guide: Accessible System Blueprint

This document defines the custom writing style optimized for data science theses. It combines technical precision with plain-English accessibility, structuring the text like a clear system architecture blueprint.

## 1. Core Philosophy

The objective of this style is to strip away unnecessary academic fluff while maintaining strict data science accuracy. The writing assumes an analytical, system-focused perspective (the INTP approach):
* **Objective Objectivity:** The system, the algorithm, or the experiment is always the active subject. Do not use personal pronouns such as "I" or "We".
* **Accessible Formality:** Use the simplest possible words to explain a concept so that it remains accessible to a broader audience. Do not compromise on exact technical terms (such as embeddings, regression, or topology) when they are necessary. Some occasional informalities are allowed but limited to 1 sentence in a paragraph or even once in 2-3 paragraphs. Important to keep human touch.
* **Systemic Troubleshooting:** Treat errors, anomalies, and poor model performance as valuable diagnostic metrics rather than research failures.

## 2. Punctuation Constraints: The Complete Ban on Dashes

Em dashes and en dashes are completely forbidden in this writing style. They cause sudden shifts in sentence structure and disrupt linear text processing. Use these alternatives to maintain sentence flow:
* **Semicolons (;)** to link two closely related independent clauses.
* **Colons (:)** to introduce direct mechanical breakdowns or definitions.
* **Parentheses ()** to isolate secondary technical configurations or metrics.
* **Sentence Splitting:** Divide complex or long ideas into separate, shorter sentences.

### Style Examples

* **Incorrect (Contains Dashes):** The system cleans the text — removing diacritics and collapsing vowels — before checking for exact matches.
* **Correct (Refined Style):** The system cleans the raw text by executing two operations: it removes diacritics and standardizes specific character variants before checking for exact matches.

## 3. The Structural Method: Frequent Use of Lists

Data science papers heavily favor structured presentation over dense paragraph walls. Use bullet points and numbered lists frequently to represent logical components, step-by-step flows, and system parameters:
* **Algorithmic Flow:** Mirror the linear movement of data through a pipeline using chronological steps.
* **Scannability:** Allow evaluation panels to quickly locate variables, baseline setups, and dataset sizes.
* **Component Isolation:** Separate distinct modal layers (such as text processing layers versus graph processing layers) into isolated items.

## 4. Equation Formatting and Definition Layouts

Equations must not pop up abruptly inside running text. Introduce the equation clearly using a colon, center it on its own line, and define all variables using a structured bulleted list.

### Layout Example

The pipeline calculates the final chain score using a multiplicative model:

$$S_{chain} = \prod_{i=1}^{n} w_{node_{i}} \times m_{edge_{i}}$$

The variables within this calculation are defined as follows:
* $S_{chain}$ represents the total reliability score computed for the narration path.
* $w_{node_{i}}$ represents the baseline reliability weight assigned to an individual narrator.
* $m_{edge_{i}}$ represents the specific penalty multiplier applied if text risks are detected.

## 5. Explicit Code and Script Framing

Always mention specific code scripts, modules, or configuration files explicitly by their exact filenames. Use backticks to highlight them as distinct components within the system architecture.

* **Instead of:** The setup file handles the scoring settings.
* **Use:** The global configuration file (`config.py`) establishes the baseline scoring parameters.
* **Instead of:** The main script scores the hadith chains in four parts.
* **Use:** The evaluation module (`hadith_authenticator.py`) processes incoming chains through a four-step pipeline.

## 6. Clinical Neutrality for Anomaly Reporting

When reporting negative results, low accuracy scores, or data friction, use objective and neutral language. Avoid defensive framing; describe problems as clear diagnostic findings.

* **Instead of:** The model struggled terribly because the structural data was not good enough.
* **Use:** The graph model yields an R2 score of -0.093; this negative variance indicates that topological features alone do not correlate with narrator reliability.
* **Instead of:** We had a big problem where a lot of short records messed up the alignment matching.
* **Use:** The pipeline quarantined 14.2% of alignment queries; error logs isolate text length disparities as the primary source of this processing friction.

## 7. Sample Writing

5.3.1 Name Matching and Database Linking Results
The name matching pipeline connected unique narrator identities from the 1,030 aligned Hadith chains to the 3,797 entries in the biographical encyclopedia. The system cleans the raw text by removing accents and standardizing specific Arabic letters before checking for exact matches. If a name does not match exactly, the pipeline uses fuzzy matching software with a safety threshold of 80% or higher.

Out of 1,229 unique narrators processed, the system matched 89.7% of them, which equals 1,102 individuals. The remaining 10.3% could not be matched because they were missing from the specific books used to build the encyclopedia. However, the data logs showed a major issue; only 3 out of the 1,102 matched narrators were found using exact text matching. This means 1,099 narrators had to rely entirely on fuzzy matching to find a link.

This large difference shows that the raw names written inside narration chains look very different from the official names saved in biographical dictionaries. Reviewing the error logs highlights three common naming habits that cause this linking confusion:  
* Shortened Name Formats: Narrators are frequently mentioned using only their first name, completely leaving out their family lineage. 
* Swapped Name Order: The text often flips the order of names, putting titles before or after the regular lineage names. 
* Variations in Connecting Words: The text changes the way it writes connecting particles, switching randomly between different spelling styles like ibn and bin