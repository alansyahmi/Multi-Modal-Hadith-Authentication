"""
Reformat Chapter 4 DOCX: italicise non-English terms using Wehr transliteration.
- Reads the original chapter4_experimental_design.docx
- Applies proper Wehr scholarly transliteration to all Arabic/Islamic technical terms
- Italicises only the non-English terms (not the surrounding English text)
- Saves a new copy: chapter4_experimental_design_v4.docx
"""
import re
from docx import Document
from docx.shared import Pt

# ── Wehr Transliteration Map ───────────────────────────────────────────────
# ASCII/plain form → Wehr transliteration (with diacritics)
# Sorted by length descending so longer multi-word phrases match before
# their constituent single words.

WEHR_MAP: dict[str, str] = {
    # ═══ Hadith Structure ═══
    "Hadith": "Ḥadīth",
    "hadith": "ḥadīth",
    "Sunna": "Sunnah",
    "sunna": "sunnah",
    "Sanad": "Sanad",
    "sanad": "sanad",
    "Isnad": "Isnād",
    "isnad": "isnād",
    "Matn": "Matn",
    "matn": "matn",
    "Rijal": "Rijāl",
    "rijal": "rijāl",
    "Muhaddithun": "Muḥaddithūn",
    "muhaddithun": "muḥaddithūn",

    # ═══ Jarh wa Ta'dil (Criticism & Praise) ═══
    "Jarh wa Ta'dil": "Jarḥ wa Taʿdīl",
    "Jarh wa Tadil": "Jarḥ wa Taʿdīl",
    "jarh wa ta'dil": "jarḥ wa taʿdīl",
    "jarh wa tadil": "jarḥ wa taʿdīl",
    "Jarh": "Jarḥ",
    "jarh": "jarḥ",
    "Ta'dil": "Taʿdīl",
    "Ta'dīl": "Taʿdīl",
    "Tadil": "Taʿdīl",
    "ta'dil": "taʿdīl",
    "tadil": "taʿdīl",
    "Ilm ar-Rijal": "ʿIlm ar-Rijāl",
    "Ilm al-Rijal": "ʿIlm al-Rijāl",
    "ilm ar-rijal": "ʿilm ar-rijāl",
    "'Ilm ar-Rijal": "ʿIlm ar-Rijāl",

    # ═══ Hadith Grades ═══
    "Sahih": "Ṣaḥīḥ",
    "sahih": "ṣaḥīḥ",
    "Daif": "Ḍaʿīf",
    "Da'if": "Ḍaʿīf",
    "daif": "ḍaʿīf",
    "da'if": "ḍaʿīf",
    "Hasan": "Ḥasan",
    "hasan": "ḥasan",
    "Mawdu'": "Mawḍūʿ",
    "Mawdu": "Mawḍūʿ",
    "mawdu'": "mawḍūʿ",
    "mawdu": "mawḍūʿ",

    # ═══ Narrator Reliability Terms ═══
    "Thiqa": "Thiqah",
    "thiqah": "thiqah",
    "thiqa": "thiqah",
    "Saduq": "Ṣadūq",
    "saduq": "ṣadūq",
    "Kadhdhab": "Kadhdhāb",
    "Kadhib": "Kadhdhāb",
    "kadhdhab": "kadhdhāb",
    "kadhib": "kadhdhāb",
    "Matruk": "Matrūk",
    "matruk": "matrūk",
    "Maqbul": "Maqbūl",
    "maqbul": "maqbūl",
    "Munkar": "Munkar",
    "munkar": "munkar",
    "Layyin": "Layyin",
    "layyin": "layyin",
    "Majhul": "Majhūl",
    "majhul": "majhūl",
    "Jahala": "Jahālah",
    "jahala": "jahālah",
    "Wadda'": "Waḍḍāʿ",
    "Wadda": "Waḍḍāʿ",
    "wadda'": "waḍḍāʿ",
    "Ma'mun": "Maʾmūn",
    "ma'mun": "maʾmūn",
    "Adbat an-nas": "Aḍbaṭ an-nās",
    "adbat an-nas": "aḍbaṭ an-nās",

    # ═══ Transmission Defects ═══
    "Tadlis": "Tadlīs",
    "tadlis": "tadlīs",
    "Ikhtilat": "Ikhtilāṭ",
    "ikhtilat": "ikhtilāṭ",
    "Inqita'": "Inqiṭāʿ",
    "Inqita": "Inqiṭāʿ",
    "inqita'": "inqiṭāʿ",
    "inqita": "inqiṭāʿ",
    "Shudhudh": "Shudhūdh",
    "shudhudh": "shudhūdh",
    "Shadh": "Shāḏ",
    "shadh": "šāḏ",
    "Illah": "ʿIllah",
    "illah": "ʿillah",
    "'Illah": "ʿIllah",
    "'illah": "ʿillah",

    # ═══ Tabaqat (Generations) ═══
    "Tabaqat": "Ṭabaqāt",
    "tabaqat": "ṭabaqāt",
    "Tabaqa": "Ṭabaqa",
    "tabaqa": "ṭabaqa",
    "Sahaba": "Ṣaḥābah",
    "Sahabah": "Ṣaḥābah",
    "sahaba": "ṣaḥābah",
    "Tabi'un": "Tābiʿūn",
    "Tabiun": "Tābiʿūn",
    "tabi'un": "tābiʿūn",
    "tabiun": "tābiʿūn",
    "Atba' al-Tabi'in": "Atbāʿ al-Tābiʿīn",
    "Atba al-Tabiin": "Atbāʿ al-Tābiʿīn",
    "atba' al-tabi'in": "atbāʿ al-tābiʿīn",
    "Kibar atba' al-atba'": "Kibār atbāʿ al-atbāʿ",
    "kibar atba' al-atba'": "kibār atbāʿ al-atbāʿ",

    # ═══ Turuq / Multi-chain ═══
    "Turuq": "Ṭuruq",
    "turuq": "ṭuruq",

    # ═══ Book Titles ═══
    "Tahdhib al-Kamal": "Tahdhīb al-Kamāl",
    "Tahdhib": "Tahdhīb",
    "Taqrib al-Tahdhib": "Taqrīb al-Tahdhīb",
    "Taqrib": "Taqrīb",
    "Kutub as-Sitta": "Kutub as-Sittah",
    "Kutub al-Sitta": "Kutub al-Sittah",
    "Al-Kutub al-Sittah": "al-Kutub as-Sittah",
    "Sahih al-Bukhari": "Ṣaḥīḥ al-Bukhārī",
    "Sahih Muslim": "Ṣaḥīḥ Muslim",
    "Sunan Abi Dawud": "Sunan Abī Dāwūd",
    "Sunan Abī Dāwūd": "Sunan Abī Dāwūd",
    "Jami' at-Tirmidhi": "Jāmiʿ at-Tirmidhī",
    "Jami` at-Tirmidhi": "Jāmiʿ at-Tirmidhī",
    "Sunan at-Tirmidhi": "Sunan at-Tirmidhī",
    "Sunan al-Nasai": "Sunan al-Nasāʾī",
    "Sunan al-Nasa'i": "Sunan al-Nasāʾī",
    "Sunan Ibn Majah": "Sunan Ibn Mājah",
    "Muwatta": "Muwaṭṭaʾ",
    "Al-Muwatta": "al-Muwaṭṭaʾ",
    "al-Muwatta": "al-Muwaṭṭaʾ",
    "Al-Mu'jam al-Awsat": "al-Muʿjam al-Awsaṭ",

    # ═══ People ═══
    "Mizzi": "Mizzī",
    "al-Mizzi": "al-Mizzī",
    "Ibn Hajar": "Ibn Ḥajar",
    "Ibn Hajar al-Asqalani": "Ibn Ḥajar al-ʿAsqalānī",
    "Ibn Hajar al-'Asqalani": "Ibn Ḥajar al-ʿAsqalānī",
    "Asqalani": "ʿAsqalānī",
    "'Asqalani": "ʿAsqalānī",
    "Dhahabi": "Dhahabī",
    "Bukhari": "Bukhārī",
    "al-Bukhari": "al-Bukhārī",
    "Tirmidhi": "Tirmidhī",
    "at-Tirmidhi": "at-Tirmidhī",
    "Nasai": "Nasāʾī",
    "an-Nasai": "an-Nasāʾī",
    "Dawud": "Dāwūd",
    "Majah": "Mājah",
    "Yahya ibn Ma'in": "Yaḥyā ibn Maʿīn",
    "Ahmad ibn Hanbal": "Aḥmad ibn Ḥanbal",
    "Abu Hurayra": "Abū Hurayrah",
    "Abu Hurayrah": "Abū Hurayrah",
    "Malik ibn Anas": "Mālik ibn Anas",
    "Ibn Shihab": "Ibn Shihāb",

    # ═══ Naming Convention Parts ═══
    "Kunya": "Kunyah",
    "kunya": "kunyah",
    "Nasab": "Nasab",
    "nasab": "nasab",
    "Nisba": "Nisbah",
    "nisba": "nisbah",
    "Laqab": "Laqab",
    "laqab": "laqab",
    "Ism": "Ism",
    "ism": "ism",

    # ═══ Other Technical Terms ═══
    "Fiqh": "Fiqh",
    "fiqh": "fiqh",
    "Aqida": "ʿAqīdah",
    "aqida": "ʿaqīdah",
    "'Aqida": "ʿAqīdah",
    "'aqida": "ʿaqīdah",
    "Dabt": "Ḍabṭ",
    "dabt": "ḍabṭ",
    "Adala": "ʿAdālah",
    "adala": "ʿadālah",
    "'Adala": "ʿAdālah",
    "'adala": "ʿadālah",
    "Hafiz": "Ḥāfiẓ",
    "hafiz": "ḥāfiẓ",
    "Rihlah": "Riḥlah",
    "rihlah": "riḥlah",
    "Tarajim": "Tarājim",
    "tarajim": "tarājim",
    "Dalil": "Dalīl",
    "dalil": "dalīl",
    "Madhhab": "Madhhab",
    "madhhab": "madhhab",
    "Madhabs": "Madhāhib",
    "madhabs": "madhāhib",
    "Kutub": "Kutub",
    "kutub": "kutub",
    "Mudallis": "Mudallis",
    "mudallis": "mudallis",
    "Musnad": "Musnad",
    "musnad": "musnad",
    "An'ana": "ʿAnʿanah",
    "an'ana": "ʿanʿanah",
    "Anana": "ʿAnʿanah",
    "anana": "ʿanʿanah",
    "Asma": "Asmāʾ",
    "asma": "asmāʾ",
    "Hadiths": "Ḥadīths",
    "hadiths": "ḥadīths",
    "Isnad": "Isnād",
    "isnad": "isnād",
    "Isnad's": "Isnād's",
    "Sanadset": "Sanadset",

    # ═══ Geographic ═══
    "Kufa": "Kūfa",
    "Basra": "Baṣra",
}

# ── Terms that are in standard English dictionaries (do NOT italicise) ────
ENGLISH_DICTIONARY_TERMS: set[str] = {
    "Islam", "Islamic", "Muslim", "Muslims", "Quran", "Qur'an", "Qurʾān",
    "Allah", "Muhammad", "Prophet Muhammad", "Arabic", "Sunni", "Shia",
    "Shi'a", "Shīʿa", "Imam", "Caliph", "Jihad", "Sharia", "Shari'a",
    "Mecca", "Medina", "Mosque", "Ramadan", "Hajj", "Sufi", "Sufism",
    "Arab", "Arabs", "Arabia", "Arabian",
}

# ── Patterns for already-properly-transliterated terms ─────────────────────
# These regex patterns match terms that already have Wehr diacritics.
# We still want to italicise them, but we don't need to transliterate them.
WEHR_PATTERN = re.compile(
    r'\b(?:'
    r'[A-Za-z]*[āīūĀĪŪḥḤṣṢḍḌṭṬẓẒʿʾšŠḏḎġĠǧǦḫḪ]'  # at least one Wehr char
    r'[A-Za-zāīūĀĪŪḥḤṣṢḍḌṭṬẓẒʿʾšŠḏḎġĠǧǦḫḪ\-\'’]*'
    r')\b'
)

# ── Build sorted term list ─────────────────────────────────────────────────
SORTED_TERMS = sorted(WEHR_MAP.items(), key=lambda kv: -len(kv[0]))


def apply_wehr_and_italicize(para) -> bool:
    """Process one paragraph: find non-English terms, transliterate to Wehr,
    and italicise only those terms. Returns True if any changes made."""
    text = para.text
    if not text.strip():
        return False

    # ── Step 1: Find all term spans to italicise ────────────────────────
    # Each span is (start, end, wehr_form)
    spans: list[tuple[int, int, str]] = []

    # 1a. Match ASCII terms from WEHR_MAP (word-boundary aware)
    for ascii_term, wehr_form in SORTED_TERMS:
        pattern = re.compile(r'\b' + re.escape(ascii_term) + r'\b')
        for m in pattern.finditer(text):
            # Check it's not an English dictionary word
            if m.group() in ENGLISH_DICTIONARY_TERMS:
                continue
            # Avoid overlap with existing spans
            if not any(s[0] <= m.start() < s[1] or s[0] < m.end() <= s[1]
                       for s in spans):
                spans.append((m.start(), m.end(), wehr_form))

    # 1b. Match already-transliterated Wehr terms
    for m in WEHR_PATTERN.finditer(text):
        matched = m.group()
        # Skip common English words that happen to match
        if matched in ENGLISH_DICTIONARY_TERMS:
            continue
        if not any(s[0] <= m.start() < s[1] or s[0] < m.end() <= s[1]
                   for s in spans):
            spans.append((m.start(), m.end(), matched))  # keep as-is

    if not spans:
        return False

    # ── Step 2: Sort spans by position, merge adjacent/overlapping ──────
    spans.sort(key=lambda s: s[0])
    merged: list[tuple[int, int, str]] = []
    for s in spans:
        if merged and merged[-1][1] >= s[0]:
            # Overlapping — keep the longer span
            prev = merged.pop()
            new_end = max(prev[1], s[1])
            merged.append((prev[0], new_end, prev[2]))  # keep first form
        else:
            merged.append(s)

    # ── Step 3: Rebuild paragraph runs ──────────────────────────────────
    # Clear existing runs
    for run in para.runs:
        run.text = ""

    cursor = 0
    first_run = True
    for start, end, wehr_form in merged:
        # Plain text before this term
        if cursor < start:
            plain_text = text[cursor:start]
            if first_run and para.runs:
                run = para.runs[0]
            else:
                run = para.add_run()
            run.text = plain_text
            run.italic = False
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)
            first_run = False

        # The italicised Wehr term
        run = para.add_run()
        run.text = wehr_form
        run.italic = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        first_run = False
        cursor = end

    # Remaining plain text after last term
    if cursor < len(text):
        run = para.add_run()
        run.text = text[cursor:]
        run.italic = False
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)

    return True


def main() -> None:
    input_path = "Report Drafts/chapter4_experimental_design.docx"
    output_path = "Report Drafts/chapter4_experimental_design_v4.docx"

    print(f"Reading: {input_path}")
    doc = Document(input_path)
    print(f"  {len(doc.paragraphs)} paragraphs")

    count = 0
    for i, para in enumerate(doc.paragraphs):
        try:
            if apply_wehr_and_italicize(para):
                count += 1
        except Exception as e:
            print(f"  Warning: paragraph {i} failed: {e}")

    print(f"Italicised Wehr-transliterated terms in {count} paragraphs")

    doc.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
