"""Diagnose text structure differences between books."""
import sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

# Load Sanadset and check Matn structure
sanad = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\Sanadset 650K Data on Hadith Narrators\Sanadset 650K Data on Hadith Narrators\hadith_samples.csv"
)
print("=== Sanadset Matn (first 10) ===")
for i in range(10):
    m = str(sanad.iloc[i]["Matn"])
    print(f"  [{i}] ({len(m)}c) {m[:150]}")

# Check if SAW appears in Sanadset Matn
saw_count = sum(1 for i in range(len(sanad)) if "صَلَّى" in str(sanad.iloc[i]["Matn"]))
print(f"\nSAW in Sanadset Matn: {saw_count}/{len(sanad)} = {saw_count/len(sanad)*100:.1f}%")

# Check if SAW appears in Sanadset Matn (simplified form)
saw2_count = sum(1 for i in range(len(sanad)) if "صلى" in str(sanad.iloc[i]["Matn"]))
print(f"'صلى' in Sanadset Matn: {saw2_count}/{len(sanad)} = {saw2_count/len(sanad)*100:.1f}%")

# === TIRMIDHI ===
print("\n" + "="*60)
print("TIRMIDHI Structure Analysis")
print("="*60)
hf = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Jami20at-Tirmidhi.csv"
)
print(f"Total hadiths: {len(hf)}")
for i in range(5):
    txt = str(hf.iloc[i]["Arabic_Text"])
    has_abu_isa = "أَبُو عِيسَى" in txt
    print(f"\n  [{i}] ({len(txt)}c) has_abu_isa={has_abu_isa}")
    print(f"    START: {txt[:200]}")
    print(f"    END:   {txt[-200:]}")

# Count editorial commentary prevalence
abu_isa_count = sum(1 for i in range(len(hf)) if "أَبُو عِيسَى" in str(hf.iloc[i]["Arabic_Text"]))
print(f"\n  أَبُو عِيسَى commentary: {abu_isa_count}/{len(hf)} = {abu_isa_count/len(hf)*100:.1f}%")

# === ABU DAWUD ===
print("\n" + "="*60)
print("ABU DAWUD Structure Analysis")
print("="*60)
hf2 = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sunan20Dawud.csv"
)
for i in range(3):
    txt = str(hf2.iloc[i]["Arabic_Text"])
    print(f"\n  [{i}] ({len(txt)}c)")
    print(f"    START: {txt[:200]}")
    print(f"    END:   {txt[-200:]}")

abu_dawud_count = sum(1 for i in range(len(hf2)) if "أَبُو دَاوُدَ" in str(hf2.iloc[i]["Arabic_Text"]))
print(f"\n  أَبُو دَاوُدَ commentary: {abu_dawud_count}/{len(hf2)} = {abu_dawud_count/len(hf2)*100:.1f}%")

# === BUKHARI ===
print("\n" + "="*60)
print("BUKHARI Structure Analysis")
print("="*60)
hf3 = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sahih%20al-Bukhari.csv"
)
for i in range(3):
    txt = str(hf3.iloc[i]["Arabic_Text"])
    print(f"\n  [{i}] ({len(txt)}c)")
    print(f"    START: {txt[:200]}")
    print(f"    END:   {txt[-200:]}")

# Check if Bukhari text starts with standard transmission verbs
for i in range(5):
    txt = str(hf3.iloc[i]["Arabic_Text"])
    print(f"\n[{i}] First 80 chars: {txt[:80]}")

# === MUSLIM ===
print("\n" + "="*60)
print("MUSLIM Structure Analysis")
print("="*60)
hf4 = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sahih%20Muslim.csv"
)
for i in range(3):
    txt = str(hf4.iloc[i]["Arabic_Text"])
    print(f"\n  [{i}] ({len(txt)}c)")
    print(f"    START: {txt[:200]}")
    print(f"    END:   {txt[-200:]}")

# === IBN MAJAH ===
print("\n" + "="*60)
print("IBN MAJAH Structure Analysis")
print("="*60)
hf5 = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sunan20Majah.csv"
)
for i in range(3):
    txt = str(hf5.iloc[i]["Arabic_Text"])
    print(f"\n  [{i}] ({len(txt)}c)")
    print(f"    START: {txt[:200]}")
    print(f"    END:   {txt[-200:]}")

# === NASAI ===
print("\n" + "="*60)
print("NASAI Structure Analysis")
print("="*60)
hf6 = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sunan_an-Nasai.csv"
)
for i in range(3):
    txt = str(hf6.iloc[i]["Arabic_Text"])
    print(f"\n  [{i}] ({len(txt)}c)")
    print(f"    START: {txt[:200]}")
    print(f"    END:   {txt[-200:]}")
