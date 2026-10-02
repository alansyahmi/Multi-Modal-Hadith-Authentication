"""
Taqrib al-Tahdhib specific parser.

Taqrib uses a compressed single-line format:
  NUMBER - FULL_NAME GRADE TABAQA [مات سنة DEATH] BOOK_CODES

Example:
  ١- أحمد ابن إبراهيم ابن خالد الموصلي أبو علي نزيل بغداد
     صدوق من العاشرة مات سنة ست وثلاثين د فق

Unlike Tahdhib al-Kamal, the grade is IMPLICIT — just a word in the line,
not a critic statement. Every entry has exactly one grade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Grade terms in priority order (longer matches first to avoid partial grabs)
GRADE_TERMS_PRIORITY = [
    # Composite grades (check these FIRST)
    "صدوق سيء الحفظ",
    "صدوق يهم",
    "ثقة حافظ",
    "ثقة ثبت",
    "ثقة متقن",
    "لين الحديث",
    "ضعيف الحديث",
    "منكر الحديث",
    "متروك الحديث",
    "سيء الحفظ",
    "ليس بالقوي",
    "لا بأس به",
    "ليس به بأس",
    "محله الصدق",
    "مجهول الحال",
    "مجهول العين",
    # Single-word grades
    "ثقة",
    "صدوق",
    "مقبول",
    "ضعيف",
    "مجهول",
    "متروك",
    "كذاب",
    "وضاع",
    "ساقط",
    "لين",
    "مستور",
    "حافظ",
    "إمام",
    "ثبت",
    "متقن",
    "مأمون",
    "دجال",
    "مختلط",
]

# Build regex: longest-first
_GRADE_PATTERN = "|".join(GRADE_TERMS_PRIORITY)
GRADE_LINE_RE = re.compile(
    r"(?:^|\s)(" + _GRADE_PATTERN + r")(?:\s|$|[.،(])",
    re.MULTILINE,
)

# Tabaqa: "من العاشرة" / "من الثالثة" etc.
TABAQA_RE = re.compile(r"من (?:ال)?(?:[أاولثانثربعخمسسدسعبعتاسعحادي]{3,}|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|الحادية عشرة|الثانية عشرة)")

# Death: "مات سنة X" — different from Tahdhib!
DEATH_TAQRIB_RE = re.compile(
    r"مات سنة\s*"
    r"([^.]{3,60}?)(?:\s[دتسقخمبفلعيصرك]|\s*$)",
    re.DOTALL,
)

# Book codes at end of entry
BOOK_CODES_RE = re.compile(
    r"\s+([دتسقخمبفلعيصرك]{1,30}|ع)\s*$",
    re.MULTILINE,
)

# Entry header: NUMBER - NAME (same as narrator_parser)
ENTRY_START_RE = re.compile(
    r"(?:^|\n)\s*"
    r"[٠-٩\d]{1,4}"                              # Arabic or Western digits
    r"\s*[-–—.]?\s*"                               # Dash
    r"([؀-ۿ\s,]{10,250})"                           # Full name
    r"(?:\.|\n|$)",
    re.MULTILINE,
)


@dataclass
class TaqribEntry:
    """A single narrator entry from Taqrib al-Tahdhib."""
    entry_id: int
    name: str = ""
    grade: str = ""
    tabaqa: str = ""
    death_year_text: str = ""
    book_codes: str = ""
    full_text: str = ""
    confidence: str = "low"


def parse_taqrib_volume(text: str) -> list[TaqribEntry]:
    """Parse all entries from Taqrib raw volume text."""
    boundaries = list(ENTRY_START_RE.finditer(text))
    if not boundaries:
        return []

    entries = []
    for i, match in enumerate(boundaries):
        name = match.group(1).strip()

        # Skip introduction/section headers
        if len(name) < 8 or any(kw in name for kw in ["باب", "حرف", "فصل", "ذكر من", "تقريب", "الطبقة"]):
            continue

        # Text from end of this name to start of next entry
        text_start = match.end()
        text_end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        full_text = text[text_start:text_end].strip()

        # In Taqrib, the grade/tabaqa/death are embedded WITHIN the name line.
        # Search both name and remaining text.
        combined = name + " " + full_text

        # Extract grade — find the best match
        grade = ""
        for gterm in GRADE_TERMS_PRIORITY:
            if gterm in combined:
                grade = gterm
                break

        # Extract tabaqa
        tabaqa = ""
        tm = TABAQA_RE.search(combined)
        if tm:
            tabaqa = tm.group(0)

        # Extract death ("مات سنة X")
        death = ""
        dm = DEATH_TAQRIB_RE.search(combined)
        if dm:
            death = dm.group(0).strip()

        # Extract book codes (at end of the combined text)
        book_codes = ""
        bm = BOOK_CODES_RE.search(combined)
        if bm:
            book_codes = bm.group(0).strip()

        entries.append(TaqribEntry(
            entry_id=len(entries) + 1,
            name=name.strip(),
            grade=grade,
            tabaqa=tabaqa,
            death_year_text=death,
            book_codes=book_codes,
            full_text=combined[:500],
            confidence="high" if grade else "medium",
        ))

    return entries


def _strip_dia(text: str) -> str:
    """Strip diacritics for matching."""
    return re.sub(r'[ً-ٌٍ-َّْٰ]', '', text)
