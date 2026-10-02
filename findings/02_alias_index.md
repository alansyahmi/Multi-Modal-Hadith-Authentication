# Finding 2: Alias Index & Name Variant Coverage

## Experiment
Build a comprehensive alias index from the narrator encyclopedia, extracting all name variants (ism, kunya, nisba, laqab) to enable matching KG narrator names to encyclopedia entries.

## Approach
For each encyclopedia entry, extract:
1. **Full name** (normalized)
2. **Kunya** (أبو X) from full text
3. **Nisba** (ال...ي) from name
4. **First word** (ism)
5. **Last word** (usually nisba/laqab)
6. **Ibn/bin patterns**

## Results

| Metric | Value |
|---|---|
| Encyclopedia entries processed | 15,448 |
| **Total alias forms extracted** | **98,747** |
| **Unique alias strings** | **24,391** |
| Ambiguous aliases (map to >1 entry) | 4,636 |
| KG coverage (any match) | **84.2%** (221K/262K) |
| KG coverage (has bio text) | **47.4%** (124K/262K) |
| KG coverage (exact match) | 6.0% (15,767/262K) |

## Father Fragment Extension

After adding multi-word father fragments from "بن X بن Y" patterns:

| Extension | Added | Total |
|---|---|---|
| Initial alias forms | — | 24,391 |
| Father fragment aliases | **+148** | **24,545** |

Example: "يزيد بن إبراهيم" now maps to specific encyclopedia entries instead of just the generic "يزيد" (which matches 207 entries).

## KG Coverage Breakdown

| Tier | Count | % |
|---|---|---|
| Exact alias → bio text | 15,767 | 6.0% |
| Fragment match → bio text | 108,544 | 41.4% |
| **Total with bio text** | **124,311** | **47.4%** |
| Fragment match (parsing artifacts) | ~96,773 | ~36.9% |
| Genuinely missing from rijāl works | ~41,413 | ~15.8% |

The ~16% genuinely missing are very minor transmitters not recorded in Tahdhib al-Kamal or Taqrib al-Tahdhib.

## Key Files
- `output/shamela/alias_index.json` — 24,545 entries (name_form → [entry_ids])
- `output/shamela/name_variants.json` — Per-entry detailed variants
- `scripts/build_alias_index.py` — Initial alias builder
- `scripts/extend_alias_with_fathers.py` — Father fragment extension
