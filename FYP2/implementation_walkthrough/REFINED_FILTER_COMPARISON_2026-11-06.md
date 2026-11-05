# Refined snippet-filter rerun comparison

Date: 2026-11-06

## Scope

After auditing the initial broad substring filter, the walkthrough copy was refined to match configured author names as whole tokens, use the full al-Dhahabi spelling `الذهبي`, and require nearby citation context for the ambiguous name `العراقي`. The cohort builder was rerun against the same local inputs. This is a post-report code correction in the walkthrough copy; it is not evidence that the submitted v4 implementation used this exact rule. No PO1/PO2/PO3 model experiment was rerun.

## Cohort output comparison

| Check | Frozen pre-rerun output | Refined output | Result |
|---|---:|---:|---|
| Narrator identities | 335 | 335 | Same identities |
| Retained earlier-critic statements | 13,478 | 13,333 | Refined output has 145 fewer |
| Narrator rows with changed statement counts/text | — | 85 | Changes are confined to earlier-critic text/count fields |
| Cohort summary | — | — | Parsed values match |
| Graph node features | 335 × 8 | 335 × 8 | Arrays match |
| Directed edge entries | 354 | 354 | Edge sets match |

The 145 statement difference consists of 128 author-name matches under the refined rule and 17 snippets rejected by the builder's minimum-length condition (at most five characters). The frozen outputs are retained in `output/cleaned_cohort/` with the `pre_rerun_frozen_20261106` suffix. The refined files are the current unsuffixed cleaned-cohort outputs.

## PO1 population scan under the refined helper

On the full Hadith-Viz source, the current helper finds 835 non-Companion records with a valid mapped Ibn Hajar grade. After earlier-critic extraction and the existing text-length eligibility rule, 819 records remain, matching the PO1 script's stated cohort size. Across those eligible records' earlier-critic entries, 30,859 statements exceed five characters; the helper flags 264, leaving 30,595 before each narrator's concatenated-text length check. These are filter counts, not a manually adjudicated count of genuine later-author quotations.

The report's stated count of 58 secondary quote fragments remains unreconciled. The report does not provide enough detail here to establish whether 58 counts all name mentions, only editorial prefixes, a smaller corpus, or another unit. Do not present 264 as equivalent to 58 or claim the refined filter proves zero leakage. The helper is a name-and-context heuristic and cannot detect all indirect citations or paraphrases.

## Provenance and limitations

- The first broad-filter output set was overwritten by this refined run; it was not separately archived. Its code remains available in Git parent commit `c5eaf60`, and its comparison/audit records remain in this walkthrough folder.
- No model results were regenerated. Existing benchmark result files therefore remain saved historical outputs and are not validated against the refined text corpus.
- The submitted report and forensic v4 bundle were not modified.
- This rerun used the current local Python environment. It does not establish the experiment-time environment used for Chapter 5.
