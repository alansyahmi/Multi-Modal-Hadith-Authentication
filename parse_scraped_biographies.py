import sys
import json
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.shamela.narrator_parser import _extract_list, _extract_death, TEACHER_RE, STUDENT_RE, GRADE_RE, DEATH_RE

TARGETS_DIR = Path("output/tahdhib_al_kamal_targets")
OUTPUT_FILE = Path("output/structured_biographies.json")

def main():
    if not TARGETS_DIR.exists():
        print(f"Error: {TARGETS_DIR} directory does not exist. Run the scraper first.")
        sys.exit(1)
        
    txt_files = list(TARGETS_DIR.glob("page_*.txt"))
    print(f"Found {len(txt_files)} biography text files to parse.")
    
    # Load the matched target mappings to associate page numbers back to narrator names
    mapping_path = Path("scratch/matched_targets.json")
    mapping = {}
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            # Invert: page_num -> list of narrator names
            for name, page in json.load(f).items():
                mapping.setdefault(int(page), []).append(name)
                
    structured_data = []
    
    for file_path in txt_files:
        # Extract page number from filename: page_00204.txt -> 204
        try:
            page_num = int(file_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
            
        text = file_path.read_text(encoding="utf-8")
        
        # Extract fields using the project's regex rules
        teachers = _extract_list(TEACHER_RE, text)
        students = _extract_list(STUDENT_RE, text)
        grades = _extract_list(GRADE_RE, text)
        death = _extract_death(DEATH_RE, text)
        
        names = mapping.get(page_num, [f"Page {page_num} Unknown"])
        
        structured_data.append({
            "page_number": page_num,
            "target_names": names,
            "biography_text_preview": text[:200] + "...",
            "teachers": teachers,
            "students": students,
            "grades": grades,
            "death_info": death
        })
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nParsing complete! Saved structured profile database to: {OUTPUT_FILE}")
    print(f"Parsed {len(structured_data)} narrator profiles.")

if __name__ == "__main__":
    main()
