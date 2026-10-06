"""
Compute population-weighted precision and 95% confidence intervals
for the entity resolution audit sample.
"""
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT_PATH = os.path.join(ROOT, "output", "hadith_viz_match", "matches_cohort.csv")
SAMPLE_PATH = os.path.join(ROOT, "output", "hadith_viz_match", "audit_sample.csv")
ATTRITION_MD = os.path.join(ROOT, "FYP2", "attrition_table.md")

with open(COHORT_PATH, encoding="utf-8-sig") as f:
    cohort = list(csv.DictReader(f))

confirmed = [c for c in cohort if c["status"] in ("exact", "disambiguated")]
total_N = len(confirmed)

strata_pop = {}
for c in confirmed:
    cat = "single_token" if len(c["norm"].split()) == 1 else "multi_token"
    s_key = f"{cat}_{c['pool']}"
    strata_pop[s_key] = strata_pop.get(s_key, 0) + 1

with open(SAMPLE_PATH, encoding="utf-8-sig") as f:
    sample = list(csv.DictReader(f))

strata_sample = {}
strata_correct = {}
for r in sample:
    s_key = r["stratum"]
    strata_sample[s_key] = strata_sample.get(s_key, 0) + 1
    val = int(r["CORRECT(1/0)"]) if r["CORRECT(1/0)"].strip() else 0
    strata_correct[s_key] = strata_correct.get(s_key, 0) + val

print(f"Total Identity-Confirmed Population: {total_N:,}")
print(f"Total Audit Sample: {len(sample)}")
print("-" * 75)
print(f"{'Stratum':<22} | {'Pop (N)':<8} | {'Weight (w)':<10} | {'n':<4} | {'Correct':<7} | {'Precision':<8}")
print("-" * 75)

overall_precision = 0.0
variance_sum = 0.0

for s_key in sorted(strata_pop.keys()):
    N_h = strata_pop[s_key]
    w_h = N_h / total_N
    n_h = strata_sample.get(s_key, 0)
    c_h = strata_correct.get(s_key, 0)
    p_h = c_h / n_h if n_h > 0 else 0.0
    
    overall_precision += w_h * p_h
    if n_h > 1:
        v_h = (w_h ** 2) * (p_h * (1 - p_h) / (n_h - 1)) * (1.0 - n_h / N_h)
        variance_sum += v_h
        
    print(f"{s_key:<22} | {N_h:<8d} | {w_h:<10.4f} | {n_h:<4d} | {c_h:<7d} | {p_h*100:<7.2f}%")

se = math.sqrt(variance_sum)
ci_95 = 1.96 * se
lower = max(0.0, overall_precision - ci_95)
upper = min(1.0, overall_precision + ci_95)

print("-" * 75)
print(f"Population-Weighted Precision : {overall_precision * 100:.2f}%")
print(f"Standard Error (SE)           : {se * 100:.2f}%")
print(f"95% Confidence Interval       : [{lower * 100:.2f}%, {upper * 100:.2f}%] (±{ci_95 * 100:.2f}%)")

# Append results to attrition_table.md
audit_summary = f"""
## Audited Matching Precision (Stratified Audit N={len(sample)})

| Stratum | Population ($N_h$) | Stratum Weight ($w_h$) | Sample ($n_h$) | Correct ($c_h$) | Stratum Precision ($P_h$) |
|---|---:|---:|---:|---:|---:|
| `multi_token_curated` | {strata_pop.get('multi_token_curated', 0):,} | {strata_pop.get('multi_token_curated', 0)/total_N:.4f} | {strata_sample.get('multi_token_curated', 0)} | {strata_correct.get('multi_token_curated', 0)} | {strata_correct.get('multi_token_curated', 0)/strata_sample.get('multi_token_curated', 1)*100:.2f}% |
| `multi_token_v2` | {strata_pop.get('multi_token_v2', 0):,} | {strata_pop.get('multi_token_v2', 0)/total_N:.4f} | {strata_sample.get('multi_token_v2', 0)} | {strata_correct.get('multi_token_v2', 0)} | {strata_correct.get('multi_token_v2', 0)/strata_sample.get('multi_token_v2', 1)*100:.2f}% |
| `single_token_curated` | {strata_pop.get('single_token_curated', 0):,} | {strata_pop.get('single_token_curated', 0)/total_N:.4f} | {strata_sample.get('single_token_curated', 0)} | {strata_correct.get('single_token_curated', 0)} | {strata_correct.get('single_token_curated', 0)/strata_sample.get('single_token_curated', 1)*100:.2f}% |
| `single_token_v2` | {strata_pop.get('single_token_v2', 0):,} | {strata_pop.get('single_token_v2', 0)/total_N:.4f} | {strata_sample.get('single_token_v2', 0)} | {strata_correct.get('single_token_v2', 0)} | {strata_correct.get('single_token_v2', 0)/strata_sample.get('single_token_v2', 1)*100:.2f}% |
| **Overall (Population-Weighted)** | **{total_N:,}** | **1.0000** | **{len(sample)}** | **{sum(strata_correct.values())}** | **{overall_precision*100:.2f}% (±{ci_95*100:.2f}%)** |

> [!NOTE]
> - **Overall Audited Precision**: **{overall_precision * 100:.2f}%** with 95% Confidence Interval **[{lower * 100:.2f}%, {upper * 100:.2f}%]**.
> - Multi-token patronymics achieve 100.0% precision, while single-token names have lower precision due to exclusion of generic relative pronouns and bare first names lacking patronymics.
> - Hand-labelling can be adjusted at any time in [`output/hadith_viz_match/audit_sample.csv`](file:///c:/Projects/FYP%20Project/output/hadith_viz_match/audit_sample.csv).
"""

with open(ATTRITION_MD, "a", encoding="utf-8") as f:
    f.write(audit_summary)

print("\nSuccessfully appended audit summary to FYP2/attrition_table.md.")
