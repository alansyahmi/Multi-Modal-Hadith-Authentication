# Finding 1: Narrator Encyclopedia Extraction

## Experiment
Extract biographical entries from Tahdhib al-Kamal (35 volumes) and Taqrib al-Tahdhib (1 volume) to build a narrator encyclopedia.

## Initial Problem
The regex-based entry header parser (`ENTRY_HEADER_RE` in `narrator_parser.py`) used a pattern that only captured single-character book codes (e.g. `ت:`) but failed on multi-character codes with spaces and digits (e.g. `م ٤:`, `خت م ٤:`). This caused only ~2,093 headers to match across all 35 volumes when ~10,000+ entries exist.

Additionally, `parse_all_volumes.py` used the same `segment_volume()` function for Taqrib al-Tahdhib, which has a completely different compressed format. Taqrib extracted only 1 entry (as a "no headers found" fallback block).

## Fix Applied

### 1. Tahdhib Entry Header Regex
Changed from:
```python
r"(?:[؀-ۿA-Za-z]{1,8}\s*[:]\s*)?"    # Letters only, 1-8 chars
```
To:
```python
r"(?:[؀-ۿ٠-٩\s]{1,20}:)?"            # Letters + digits + spaces, 1-20 chars
```

This added **444 more matches** (2,093 → 2,537, a +21% improvement). However, the line-based approach was even better:

```python
r"^[\d٠-٩]{1,4}\s*[-–—]?\s*(?:[؀-ۿ٠-٩\s]{1,20}:)?\s*([؀-ۿ\s,]{10,})"
```
→ **7,997 matches** (3.8× improvement over original regex).

### 2. Taqrib Pipeline
Changed from using `segment_volume()` (Tahdhib parser) to `parse_taqrib_volume()` (purpose-built Taqrib parser). This captured **7,556 entries** from the compressed single-line format.

## Results

| Metric | Before | After |
|---|---|---|
| Tahdhib entries | 3,797 | **7,892** |
| Taqrib entries | 1 (broken) | **7,556** |
| **Total encyclopedia** | **3,797** | **15,448** |
| With explicit grades | ~1,559 | **10,920** |
| High confidence | 1,559 | 8,414 |
| Has teachers | — | 7,319 |
| Has students | — | 7,224 |

## Per-Volume Extraction (Tahdhib)

Volume 1: 1,985 entries → Volumes 2-32: ~200-300 each → Volumes 33-35: ~200-280 each

Volume 1 was over-extracted (includes biographical preface material). Volumes 2-32 are under-extracted (line-based pattern still misses some entries embedded in running prose).

## Residual Issues
- The raw 35 volumes of Tahdhib al-Kamal (~42 MB) should contain ~10,000+ entries, but the line-based pattern only captures ~7,892. The remaining entries are embedded in narrative text without clear numbered headers.
- Some entries have fragmentary names due to parsing artifacts.
- The `source` field distinguishes Tahdhib vs Taqrib entries.

## Key Files
- `output/shamela/narrator_encyclopedia.json` — 15,448 entries
- `src/shamela/narrator_parser.py` — Updated regex
- `scripts/parse_all_volumes.py` — Pipeline script
