"""
Narrator Parser — rule-based segmentation of biography text into narrator entries.

Tahdhib al-Kamal is organized alphabetically by narrator name. Each entry
typically begins with a numbered header (the narrator's full name), followed
by biography text, and ends where the next entry begins.

This module uses regex patterns to:
  1. Split raw text into individual narrator entries
  2. Extract structured fields: name, teachers, students, grades, death info
  3. Label each entry with a confidence score

It is intentionally "best-effort" — imperfect segmentation is expected.
Raw .txt output is always preserved for manual verification.
"""

from __future__ import annotations

import re
from typing import Iterator, Optional

from .models import NarratorEntry


# ── regex patterns ─────────────────────────────────────────────

# Diacritic stripper for regex matching (normalizes قَالَ -> قال, عَنْ -> عن, etc.)
_DIACRITICS_RE = re.compile(r'[ً-ٌٍ-َّْٰ]')

def _strip_dia(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) so regex patterns match raw text."""
    return _DIACRITICS_RE.sub('', text)


# Entry header: a line starting with a number followed by an Arabic name.
# Matches patterns like:
#   "١٣٤ - ت: أبان بن إسحاق الأسدي الكوفي النحوي."
#   "١- دفق: أَحْمَد بن إِبْرَاهِيم بن خالد الموصلي..."
#   "٢٨- س: أَحْمَد بن حَمَّاد..."
#   "134 - ت: أبان بن إسحاق..."  (Western digits)
# Book codes (ت, د, خ, م ٤, س, ق) indicate which canonical collections include them.
# Note: book codes may include Arabic-Indic digits (e.g. م ٤) and spaces.
ENTRY_HEADER_RE = re.compile(
    r"^[\d٠-٩]{1,4}"              # Number (Arabic or Western)
    r"\s*[-–—]?\s*"               # Optional dash
    r"(?:[؀-ۿ٠-٩\s]{1,20}:)?"    # Optional book codes + colon
    r"\s*([؀-ۿ\s,]{10,})",       # Name capture (rest of line)
    re.MULTILINE,
)

# Letter-section headers to skip (not narrator entries)
SECTION_HEADER_RE = re.compile(
    r'(?:^|\n)\s*'
    r'(?:باب|حرف|فصل|القسم)\s+'
    r'[؀-ۿ\s]{3,40}'
    r'(?:\n|\.|،)',
    re.MULTILINE,
)

# ── structured field extraction patterns (DIACRITIC-FREE) ──────
# All patterns use bare Arabic letters only — raw text has mixed/missing diacritics.

# Teachers: "روى عن" / "شيوخه" / "أخذ عن" / "سمع من"
TEACHER_RE = re.compile(
    r'(?:روى عن(?:هما|ه)?|شيوخه|مشايخه|اخذ عن|سمع من)'
    r'\s*[:،]?\s*'
    r'(.+?)'
    r'(?:\.\s*(?:روى عنه|روى عن|تلاميذ|طلبته|الاخذون|وقال|قال|وفاة|توفي|$))',
    re.DOTALL,
)

# Students: "روى عنه" / "تلاميذه" / "طلبة" / "الآخذون عنه"
STUDENT_RE = re.compile(
    r'(?:روى عنه(?:ما)?|تلاميذه|طلبته|الاخذون عنه)'
    r'\s*[:،]?\s*'
    r'(.+?)'
    r'(?:\.\s*(?:قال|وقال|وفاة|توفي|روى|شيوخ|$))',
    re.DOTALL,
)

# Grade evaluations: critic names + evaluation terms
GRADE_TERMS = (
    # Positive (ta'dil)
    r'ثِقَة|ثِقَات|ثَبْت|ثَبَت|حَافِظ|حُجَّة|إِمَام|مَأْمُون|مُتْقِن|ضَابِط|'
    r'صَدُوق|صَدُوقٌ|لَا بَأْسَ بِه|لَيْسَ بِهِ بَأْس|مَحَلُّهُ الصِّدْق|مَقْبُول|'
    # Negative (jarh)
    r'لَيِّن|لَيِّنُ الحَدِيث|لَيْسَ بِالْقَوِيّ|فِيهِ ضَعْف|'
    r'ضَعِيف|ضَعِيفٌ|ضَعِيفُ الحَدِيث|مُنْكَرُ الحَدِيث|مَتْرُوك|مَتْرُوكُ الحَدِيث|'
    r'كَذَّاب|وَضَّاع|دَجَّال|سَاقِط|مَجْهُول|مُخْتَلَط|سَيِّئُ الحِفْظِ'
)

GRADE_RE = re.compile(
    r'(?:قَالَ|وَقَالَ|قِيلَ|عَن)\s*'
    r'[^:.]*?[:]\s*'
    r'(' + GRADE_TERMS + r')',
    re.DOTALL,
)

# Catch standalone grades: "وهو ثقة", "كان ثقة"
STANDALONE_GRADE_RE = re.compile(
    r'(?:هُوَ|كَانَ|وَكَانَ|فَهُوَ|وَهُوَ)\s*(' + GRADE_TERMS + r')',
    re.DOTALL,
)

# Catch bare grade terms: "ثقة." or "ثقة (١)"
BARE_GRADE_RE = re.compile(
    r'(?:^|\n)\s*(' + GRADE_TERMS + r')[\s.،(]',
    re.MULTILINE,
)

# Footnote entries: pronunciation guides, manuscript references
FOOTNOTE_HEADER_RE = re.compile(
    r'^(?:بفتح|بكسر|بضم|بسكون|بتشديد|يعني)',
    re.MULTILINE,
)

# Death info: "وفاته" / "توفي" / "توفي سنة" / "مات"
DEATH_RE = re.compile(
    r'(?:وَفَاتُهُ|تُوُفِّيَ?|توفي|مَاتَ)\s*'
    r'(?:فِي|سَنَةَ)?\s*'
    r'(\d{1,4})\s*'
    r'(?:هـ|لِلْهِجْرَة)?'
    r'(?:[^.]*?'
    r'(?:رَمَضَان|شَوَّال|ذُو\s*\w+|مُحَرَّم|صَفَر|رَبِيع|جُمَادَى|رَجَب|شَعْبَان))?'
    r'[^.]*\.?',
    re.DOTALL,
)

# Minimum entry length in characters (shorter = likely fragment, not a real entry)
MIN_ENTRY_LENGTH = 80

# Maximum fraction of a page that can be a single entry header line
# (names shouldn't exceed ~200 chars)
MAX_NAME_LENGTH = 200


# ── segmentation ───────────────────────────────────────────────

def segment_volume(
    raw_text: str,
    volume: int,
    start_entry_id: int = 1,
) -> tuple[list[NarratorEntry], int]:
    """
    Split a single volume's raw text into narrator entries.

    Returns (entries, next_entry_id).
    """
    # 1. Find all candidate entry boundaries (numbered headers)
    candidates: list[tuple[int, int, str]] = []  # (start, end, name)

    for match in ENTRY_HEADER_RE.finditer(raw_text):
        name = match.group(1).strip()
        if len(name) > MAX_NAME_LENGTH:
            continue
        candidates.append((match.start(), match.end(), name))

    if not candidates:
        # No numbered entries found — treat the whole volume as one block
        if len(raw_text.strip()) > MIN_ENTRY_LENGTH:
            return (
                [
                    NarratorEntry(
                        entry_id=start_entry_id,
                        volume=volume,
                        page=0,
                        name="[Unknown — no entry headers found]",
                        full_text=raw_text.strip(),
                        confidence="low",
                    )
                ],
                start_entry_id + 1,
            )
        return [], start_entry_id

    # 2. Build entries from boundaries
    entries: list[NarratorEntry] = []

    for i, (start, end, name) in enumerate(candidates):
        # Text spans from end of this header to start of next header
        text_start = end
        text_end = candidates[i + 1][0] if i + 1 < len(candidates) else len(raw_text)

        full_text = raw_text[text_start:text_end].strip()

        # Skip section headers that aren't real narrator entries
        if _is_section_header(name, full_text):
            continue

        # Skip pronunciation guide footnotes
        if FOOTNOTE_HEADER_RE.search(name):
            continue

        # Skip fragments too short to be real entries
        if len(full_text) < MIN_ENTRY_LENGTH and len(name) < 20:
            continue

        # Extract structured fields (diacritic stripping happens in helpers)
        teachers = _extract_list(TEACHER_RE, full_text)
        students = _extract_list(STUDENT_RE, full_text)
        grades = _extract_list(GRADE_RE, full_text)
        # Also catch standalone/bare grade terms (strip diacritics for matching)
        stripped = _strip_dia(full_text)
        for pat in (STANDALONE_GRADE_RE, BARE_GRADE_RE):
            stripped_pat = re.compile(_strip_dia(pat.pattern), pat.flags)
            for m in stripped_pat.finditer(stripped):
                g = m.group(1).strip()
                if g and g not in grades:
                    grades.append(g)
        death = _extract_death(DEATH_RE, full_text)

        confidence = _compute_confidence(name, full_text, teachers, students, grades)

        entries.append(
            NarratorEntry(
                entry_id=start_entry_id + len(entries),
                volume=volume,
                page=0,
                name=name,
                full_text=full_text,
                teachers=tuple(teachers),
                students=tuple(students),
                grades=tuple(grades),
                death_info=death,
                confidence=confidence,
            )
        )

    return entries, start_entry_id + len(entries)


def parse_all_volumes(
    volumes: dict[int, str],
) -> list[NarratorEntry]:
    """
    Parse all volumes and return a unified list of narrator entries.

    Args:
        volumes: {volume_number: raw_text}
    """
    all_entries: list[NarratorEntry] = []
    next_id = 1

    for vol_num in sorted(volumes):
        entries, next_id = segment_volume(
            volumes[vol_num], vol_num, start_entry_id=next_id
        )
        all_entries.extend(entries)

    return all_entries


# ── helpers ────────────────────────────────────────────────────

def _is_section_header(name: str, body_text: str) -> bool:
    """Check if a candidate is a letter-section header, not a narrator entry."""
    # Section headers contain words like "باب", "حرف"
    section_keywords = ("باب", "حرف", "فصل", "القسم")
    if any(kw in name for kw in section_keywords):
        return True
    # Section headers have very short body text (just a page break)
    if len(body_text.strip()) < 30:
        return True
    return False


def _extract_list(pattern: re.Pattern, text: str) -> list[str]:
    """Extract items from pattern matches, splitting on common delimiters.

    Strips diacritics from text before matching, since raw Shamela text
    has inconsistent tashkeel that won't match diacritic-heavy regex patterns.
    """
    stripped_text = _strip_dia(text)
    # Build a diacritic-free version of the pattern
    stripped_pattern = re.compile(_strip_dia(pattern.pattern), pattern.flags)
    match = stripped_pattern.search(stripped_text)
    if not match:
        return []

    raw = match.group(1).strip()
    if not raw or len(raw) > 600:
        return [raw[:600] + "…"]

    # Split on Arabic name separators: "و" (and), commas, semicolons
    parts = re.split(r'\s+و\s+|[،;؛]\s*', raw)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]


def _extract_death(pattern: re.Pattern, text: str) -> str:
    """Extract death information — strips diacritics before matching."""
    stripped_text = _strip_dia(text)
    stripped_pattern = re.compile(_strip_dia(pattern.pattern), pattern.flags)
    match = stripped_pattern.search(stripped_text)
    if match:
        return match.group(0).strip()
    return ""


def _compute_confidence(
    name: str,
    text: str,
    teachers: list[str],
    students: list[str],
    grades: list[str],
) -> str:
    """
    Assign a confidence score to the entry segmentation.

    "high"   — Name looks like a valid Arabic full name, text is substantial
    "medium" — One of the criteria is borderline
    "low"    — Multiple criteria are weak
    """
    score = 0

    # Name quality: should have multiple Arabic words, reasonable length
    name_parts = name.split()
    if len(name_parts) >= 2 and 15 <= len(name) <= 150:
        score += 1
    if re.search(r'ابن|بنت|أبو|أم|بن', name):
        score += 1  # Has nasab (patronymic lineage)

    # Text length: substantial biographies
    if len(text) > 200:
        score += 1
    if len(text) > 500:
        score += 1

    # Structured data found
    if teachers:
        score += 1
    if students:
        score += 1
    if grades:
        score += 1

    if score >= 5:
        return "high"
    elif score >= 3:
        return "medium"
    return "low"
