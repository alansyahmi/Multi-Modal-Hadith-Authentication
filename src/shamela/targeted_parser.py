"""
Targeted Page-Level Narrator Parser.

Parses individual biography pages from Tahdhib al-Kamal / Taqrib al-Tahdhib
at the page level by locating a specific narrator's entry and extracting
structured fields (name, teachers, students, grades, death_info).

Unlike narrator_parser.py which segments entire volumes, this module
extracts ONE narrator per call — the one whose page we already know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from rapidfuzz.fuzz import partial_ratio


# ── Entry boundary patterns ─────────────────────────────────────

# Numbered entry header formats found in Tahdhib al-Kamal:
#   "١- دفق: أحمد بن إبراهيم..."  (Arabic digit, dash, Arabic codes, colon)
#   "١٣٤ - ت: أبان بن إسحاق..."  (Arabic digits, dash, space, code, colon)
#   "134 - ت: أبان بن إسحاق..."  (Western digits variant)
# Letter codes (رموز الرواة) are Arabic abbreviations for source books
ENTRY_START_RE = re.compile(
    r"(?:^|\n)\s*"
    r"[٠-٩\d]{1,4}"                          # Arabic or Western digits
    r"\s*[-–—.]?\s*"                           # Optional dash separator
    r"(?:[؀-ۿA-Za-z]{1,8}\s*[:]\s*)?"         # Optional Arabic/Latin code + colon
    r"([؀-ۿ\s,]{10,200})"                       # Arabic name (10-200 chars)
    r"(?:\.|\n|$)",                              # End marker
    re.MULTILINE,
)

# Section/chapter headers to skip
SECTION_RE = re.compile(
    r"(?:باب|حرف|فصل|القسم|المجلد|من اسمه|من اسم)",
    re.MULTILINE,
)


# ── Structured field extractors ─────────────────────────────────

TEACHERS_RE = re.compile(
    r"(?:رَوَى عَنْ(?:هُما|ه)?|شُيُوخُهُ|مَشَايِخُهُ|أَخَذَ عَنْ(?:هُ)?|سَمِعَ مِن)\s*[:،]?\s*"
    r"(.+?)"
    r"(?:\.\s*(?:رَوَى عَنْهُ|تَلَامِيذ|طَلَبَتُه|الآخِذُون|وَقَال|قَالَ|وَفَاة|تُوُفِّي|رَوَى|$))",
    re.DOTALL,
)

STUDENTS_RE = re.compile(
    r"(?:رَوَى عَنْهُ(?:ما)?|تَلَامِيذُهُ|طَلَبَتُهُ|الآخِذُونَ عَنْهُ|رَوَى عَنه)\s*[:،]?\s*"
    r"(.+?)"
    r"(?:\.\s*(?:قَالَ|وَقَال|وَفَاة|تُوُفِّي|رَوَى|شُيُوخ|$))",
    re.DOTALL,
)

GRADE_TERMS = (
    r"ثِقَة|ثِقَات|ثَبْت|ثَبَت|حَافِظ|حُجَّة|إِمَام|مَأْمُون|مُتْقِن|ضَابِط|"
    r"صَدُوق|صَدُوقٌ|لَا بَأْسَ بِه|مَحَلُّهُ الصِّدْق|مَقْبُول|لَيِّن|لَيِّنُ الحَدِيث|"
    r"ضَعِيف|ضَعِيفٌ|ضَعِيفُ الحَدِيث|مُنْكَرُ الحَدِيث|مَتْرُوك|مَتْرُوكُ الحَدِيث|"
    r"كَذَّاب|وَضَّاع|دَجَّال|سَاقِط|مَجْهُول|مُخْتَلَط|سَيِّئُ الحِفْظِ"
)

GRADE_RE = re.compile(
    r"(?:قَالَ|وَقَالَ|قِيلَ)\s*"
    r"(?:أَبُو\s+)?(?:الإِمَامُ\s+)?"
    r"(?:حَاتِم|أَحْمَد|البُخَارِيّ|النَّسَائِيّ|يَحْيَى|ابن\s+\w+|الدَّارَقُطْنِيّ|مُسْلِم|أَبُو\s+\w+|العِجْلِيّ|مَعِين|المَدِينِيّ)?"
    r"[^.]*?"
    r"(" + GRADE_TERMS + r")",
    re.DOTALL | re.IGNORECASE,
)

# Alternative: standalone grade statements like "ثقة" at end of bio
STANDALONE_GRADE_RE = re.compile(
    r"(?:هُوَ|كَانَ|وَكَانَ)\s*(?:" + GRADE_TERMS + r")",
    re.DOTALL | re.IGNORECASE,
)

DEATH_RE = re.compile(
    r"(?:وَفَاتُهُ|تُوُفِّيَ|مَاتَ)\s*"
    r"(?:فِي|سَنَةَ)?\s*"
    r"(\d{1,4})\s*"
    r"(?:هـ|لِلْهِجْرَة)?"
    r"(?:[^.]*?(?:رَمَضَان|شَوَّال|ذُو\s*\w+|مُحَرَّم|صَفَر|رَبِيع|جُمَادَى|رَجَب|شَعْبَان))?"
    r"[^.]*\.?",
    re.DOTALL,
)


@dataclass
class BioEntry:
    """A single narrator biography entry."""
    narrator_name: str
    page_number: int
    full_text: str = ""
    canonical_name: str = ""
    teachers: list[str] = field(default_factory=list)
    students: list[str] = field(default_factory=list)
    grades: list[str] = field(default_factory=list)
    death_info: str = ""
    source_file: str = ""
    confidence: str = "low"  # high, medium, low


def _normalize_arabic(text: str) -> str:
    """Light normalization for matching: strip diacritics, normalize alif/ya."""
    text = re.sub(r"[ً-ٰٟ]", "", text)  # Tashkeel
    text = re.sub(r"[أإآ]", "ا", text)                 # Alif
    text = re.sub(r"ى", "ي", text)                     # Alef maksura
    text = re.sub(r"ة", "ه", text)                     # Ta marbuta
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_page_text(page_path: Path) -> str:
    """Load page text from HTML or TXT file."""
    if page_path.suffix.lower() == ".html":
        html = page_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        # Remove script/style tags
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)
    else:
        return page_path.read_text(encoding="utf-8")


def find_narrator_entry(page_text: str, target_name: str) -> Optional[str]:
    """
    Find the biography text segment for a specific narrator on a page.

    Uses fuzzy matching (partial_ratio) to match short Sanadset names
    against the full canonical names used in entry headers. Returns the
    best-matching entry above a similarity threshold.
    """
    from rapidfuzz.fuzz import partial_ratio

    boundaries = list(ENTRY_START_RE.finditer(page_text))
    if not boundaries:
        return None

    target_norm = _normalize_arabic(target_name)

    best_score = 0.0
    best_boundary: tuple[int, int] | None = None

    for i, match in enumerate(boundaries):
        name = match.group(1).strip()

        # Skip section headers
        if SECTION_RE.search(name):
            continue

        name_norm = _normalize_arabic(name)

        # Score using partial_ratio (handles substring / partial name matching)
        try:
            score = partial_ratio(target_norm, name_norm)
        except Exception:
            score = 0.0

        # Boost score if the entry body text also references the target name
        body_start = match.end()
        body_end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(page_text)
        body = page_text[body_start:body_end]
        if _normalize_arabic(target_name) in _normalize_arabic(body):
            score = max(score, 85.0)  # Strong signal this is the right entry

        if score > best_score:
            best_score = score
            best_boundary = (i, match.start())

    # Require at least 60% similarity
    if best_boundary is None or best_score < 60:
        # Fallback: search entire page for the name
        if target_norm in _normalize_arabic(page_text):
            return page_text.strip()
        return None

    i, start = best_boundary
    end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(page_text)
    return page_text[start:end].strip()


def _extract_teachers(text: str) -> list[str]:
    """Extract teacher list from biography text."""
    m = TEACHERS_RE.search(text)
    if not m:
        return []
    raw = m.group(1).strip()
    if len(raw) > 600:
        raw = raw[:600] + "…"
    # Split on Arabic name separators
    parts = re.split(r"\s+و\s+|[،;؛]\s*", raw)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]


def _extract_students(text: str) -> list[str]:
    """Extract student list from biography text."""
    m = STUDENTS_RE.search(text)
    if not m:
        return []
    raw = m.group(1).strip()
    if len(raw) > 600:
        raw = raw[:600] + "…"
    parts = re.split(r"\s+و\s+|[،;؛]\s*", raw)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]


def _extract_grades(text: str) -> list[str]:
    """Extract critic reliability grades from biography text."""
    grades = []
    for m in GRADE_RE.finditer(text):
        g = m.group(1).strip()
        if g and g not in grades:
            grades.append(g)
    # Also check standalone
    for m in STANDALONE_GRADE_RE.finditer(text):
        g = m.group(1).strip() if m.lastindex else m.group(0).strip()
        if g and g not in grades:
            grades.append(g)
    return grades


def _extract_death(text: str) -> str:
    """Extract death information from biography text."""
    m = DEATH_RE.search(text)
    return m.group(0).strip() if m else ""


def _compute_confidence(name: str, text: str, teachers: list, students: list, grades: list) -> str:
    """Assign confidence score to extracted entry."""
    score = 0
    name_parts = name.split()
    if len(name_parts) >= 2 and 15 <= len(name) <= 150:
        score += 1
    if any(kw in name for kw in ("ابن", "بنت", "أبو", "أم", "بن")):
        score += 1
    if len(text) > 200:
        score += 1
    if len(text) > 500:
        score += 1
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


def parse_narrator_page(
    page_path: Path,
    target_name: str,
    page_number: int,
) -> Optional[BioEntry]:
    """
    Parse a single page to extract the biography entry for a target narrator.

    Args:
        page_path: Path to .html or .txt page file
        target_name: Narrator name to find on the page
        page_number: Shamela page number

    Returns:
        BioEntry if found, None otherwise.
    """
    try:
        page_text = load_page_text(page_path)
    except Exception:
        return None

    entry_text = find_narrator_entry(page_text, target_name)
    if not entry_text:
        return None

    teachers = _extract_teachers(entry_text)
    students = _extract_students(entry_text)
    grades = _extract_grades(entry_text)
    death = _extract_death(entry_text)

    # The canonical name from the entry header
    canonical = target_name
    header_match = ENTRY_START_RE.search(entry_text)
    if header_match:
        canonical = header_match.group(1).strip()

    confidence = _compute_confidence(canonical, entry_text, teachers, students, grades)

    return BioEntry(
        narrator_name=target_name,
        page_number=page_number,
        full_text=entry_text,
        canonical_name=canonical,
        teachers=teachers,
        students=students,
        grades=grades,
        death_info=death,
        source_file=str(page_path),
        confidence=confidence,
    )
