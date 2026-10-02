# Finding 3: Family & Teacher-Student Graph

## Experiment
Build a family relationship graph from the encyclopedia to enable resolution of "أبيه" (his father), "أخيه" (his brother) references in sanad chains.

## Approach

### Father-Son Extraction
From "فلان بن فلان" patterns in entry names:

```python
FATHER_SON_RE = re.compile(r"(?:ابن|بن)\s+([؀-ۿ\s]{3,60}?)(?:\s|$|،|\(|\.)")
```

### Teacher-Student Extraction
From "روى عن" and "روى عنه" patterns (already in encyclopedia entries).

### Virtual Narrator Registry
Generic father names (e.g., "يزيد" with no own entry) are assigned a virtual score = average of their known sons' scores.

## Results

| Relationship | Count |
|---|---|
| Father-son pairs (بن patterns) | **13,592** |
| Resolved to actual encyclopedia entries | **768** (5.6%) |
| Virtual narrators (fathers without own entry) | **1,331** |
| Entries with known teachers | 7,319 |
| Entries with known students | 7,224 |
| Nisba groups | 1,172 (806 multi-member) |

### Nisba-Narrowed Matches
Using the son's nisba to disambiguate generic father names:

| Method | Count |
|---|---|
| Father → own entry (exact) | 768 |
| Father + son's nisba → narrowed | 1,167 |
| Virtual narrator (avg of sons) | 1,331 |

### Top Virtual Fathers

| Father | # Sons | Use Case |
|---|---|---|
| يزيد | 198 | Father of many narrators |
| خالد | 165 | Common name |
| يحيى | 146 | Common name |
| صالح | 79 | Distinctive |
| ميمون | 73 | Rare → findable |

## Sanad Format Finding
After checking 5,000 Sanad column entries: **0 out of 5,000** contain "أبيه" or similar relational terms. The Sanad column in the dataset is **pre-resolved** — the extraction pipeline already converted relational references to actual narrator names. This means the family graph helps with **encyclopedia matching** (resolving "يزيد بن إبراهيم" as a specific entry) rather than direct chain resolution.

## Key Files
- `output/family_graph/father_son_resolved.json` — 768 resolved pairs
- `output/family_graph/virtual_narrators.json` — 1,331 virtual narrators
- `output/family_graph/nisba_narrowed_fathers.json` — 1,167 nisba-narrowed
- `scripts/build_family_graph.py` — Builder script
