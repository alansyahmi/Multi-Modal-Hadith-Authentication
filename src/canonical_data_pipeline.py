"""
Canonical Ilm ar-Rijal Data Pipeline & Evaluative Taxonomies.

Single source of truth for:
1. Canonical Ibn Hajar 3-Tier Grade Mapping (with strict precedence for weak/cautionary indicators).
2. Canonical al-Dhahabi 3-Tier Grade Mapping.
3. Strict Pre-Ibn-Hajar Author Filtering (purging all 8th/9th century synthesizing authorities).
"""

import re
from typing import Optional

TIER_TO_INT = {"below_saduq": 0, "saduq": 1, "thiqah": 2}
INT_TO_TIER = {0: "below_saduq", 1: "saduq", 2: "thiqah"}
TIER_DISPLAY_NAMES = {0: "Below Saduq", 1: "Saduq", 2: "Thiqah"}

# Synthesis authorities to exclude from input texts (7th century AH and later):
# Ibn Hajar (852 AH), al-Dhahabi (748 AH), al-Mizzi (742 AH), Mughultay (762 AH),
# al-Iraqi (806 AH), al-Ala'i (761 AH), al-Nawawi (676 AH), Burhan al-Din al-Halabi (841 AH),
# Ibn Abd al-Hadi (744 AH), al-Sakhawi (902 AH), al-Suyuti (911 AH).
EXCLUDE_LATE_SCHOLARS = [
    "ابن حجر", "الذهب", "المزي", "مغلطاي", "العراقي", "العلائي",
    "برهان الدين", "النووي", "ابن عبد الهادي", "السخاوي", "السيوطي"
]


def map_ibn_hajar_tier(grade_str: Optional[str]) -> Optional[str]:
    """
    Maps Ibn Hajar's Taqrīb al-Tahdhīb grade to canonical 3 tiers:
      - 'below_saduq' (0): weak, cautionary, unverified, or disparaged
      - 'saduq' (1): truthful, acceptable with minor memory qualifications
      - 'thiqah' (2): trustworthy, authoritative, established retention
    Strict precedence: weak indicators checked first to avoid negation traps (e.g., 'ليس بثقة').
    """
    if not grade_str:
        return None
    g = str(grade_str).strip()
    if not g or g in ("None", "nan", "null"):
        return None

    # 1. Weak / Cautionary indicators (highest precedence)
    below_indicators = [
        "ضعيف", "متروك", "مجهول", "مقبول", "كذاب", "واه", "ساقط", "لين",
        "تركوه", "ضعفوه", "تالف", "غير ثقة", "ليس بثقة", "ليس بالقوي", "فيه مقال",
        "منكر", "يخطئ كثيرا", "كثير الغلط", "تهموه", "اختلط", "تغير بآخره"
    ]
    if any(k in g for k in below_indicators):
        return "below_saduq"

    # 2. Saduq indicators
    saduq_indicators = [
        "صدوق", "لا بأس به", "خيار", "صالح الحديث", "محله الصدق",
        "حسن الحديث", "روى عنه الناس"
    ]
    if any(k in g for k in saduq_indicators):
        return "saduq"

    # 3. Thiqah indicators
    thiqah_indicators = [
        "ثقة", "حجة", "ثبت", "حافظ", "إمام", "عدل", "متقن", "مأمون", "فقيه"
    ]
    if any(k in g for k in thiqah_indicators):
        return "thiqah"

    return None


def map_dhahabi_tier(grade_str: Optional[str]) -> Optional[str]:
    """
    Maps al-Dhahabi's al-Kāshif / Mīzān grade to canonical 3 tiers.
    """
    if not grade_str:
        return None
    g = str(grade_str).strip()
    if not g or g in ("None", "nan", "null"):
        return None

    below_indicators = [
        "ضعيف", "متروك", "مجهول", "واه", "كذاب", "ساقط", "لين", "ترك",
        "ضعف", "لا يحتج", "فيه نظر", "تكلموا", "ليس بعمدة", "غير ثقة", "منكر"
    ]
    if any(k in g for k in below_indicators):
        return "below_saduq"

    if any(k in g for k in ["صدوق", "لا بأس به", "صالح", "وسط"]):
        return "saduq"

    if any(k in g for k in ["ثقة", "حجة", "ثبت", "حافظ", "إمام", "رأس", "جليل", "كبير", "أحد الأعلام", "فقيه"]):
        return "thiqah"

    return None


def is_earlier_critic(scholar_name: str) -> bool:
    """
    Returns True if the scholar is an earlier classical critic (pre-Ibn Hajar / pre-Mizzi).
    Returns False if the scholar is a later synthesis authority.
    """
    if not scholar_name:
        return False
    sch = scholar_name.strip()
    return not any(late in sch for late in EXCLUDE_LATE_SCHOLARS)


def has_late_compiler_leak(snippet_text: str) -> bool:
    """Return True when text contains a configured later compiler name."""
    if not snippet_text:
        return False
    return any(late in snippet_text for late in EXCLUDE_LATE_SCHOLARS)


def extract_clean_critic_snippets(jarh_tadil_list: list) -> list[str]:
    """
    Extracts text snippets exclusively from earlier classical critics.
    Format: 'قال [Scholar]: [Text]'
    """
    snippets = []
    for item in jarh_tadil_list:
        sch = item.get("scholar", "").strip()
        if not is_earlier_critic(sch):
            continue
        for ev in item.get("evaluations", []):
            txt = ev.get("text", "").strip()
            if txt and len(txt) > 5 and not has_late_compiler_leak(txt):
                snippets.append(f"قال {sch}: {txt}")
    return snippets
