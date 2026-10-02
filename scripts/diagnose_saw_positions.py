"""Diagnose: does SAW formula appear inside quoted speech in low-match books?"""
import sys, re, json
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.alignment.hybrid_aligner import extract_matn_portion, MATN_SPLIT_PATTERNS

eval_df = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\kutub_sitta_evaluation.csv")
with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json") as f:
    results = json.load(f)

def count_saw_positions(text):
    """Find all SAW pattern occurrences and whether each is inside quotes."""
    hits = []
    for pattern, min_len in MATN_SPLIT_PATTERNS[:2]:
        for m in re.finditer(pattern, text):
            # Check if inside Arabic quotation marks
            before = text[:m.start()]
            quote_chars = before.count('"') + before.count('"') + before.count('"') + before.count("'") + before.count("'")
            inside_quotes = quote_chars % 2 == 1

            # Check if inside guillemet-style quotes
            guillemets = before.count("»") - before.count("«")
            inside_guillemets = guillemets > 0

            hits.append({
                "pos": m.start(),
                "pattern": pattern[:20],
                "inside_quotes": inside_quotes or inside_guillemets,
                "context_before": text[max(0,m.start()-30):m.start()][-30:],
                "context_after": text[m.end():m.end()+50][:50],
            })
    return hits

print("=== SAW Position Analysis: Low-Match Books ===")
for book in ["bukhari", "tirmidhi"]:
    book_q = [r for r in results if eval_df.iloc[r["query_idx"]]["book_key"] == book and r.get("quarantine")]
    print(f"\n--- {book.upper()} ({len(book_q)} quarantined) ---")

    saw_inside_quotes = 0
    multi_saw = 0
    total_examined = min(50, len(book_q))

    for d in book_q[:total_examined]:
        row = eval_df.iloc[d["query_idx"]]
        text = str(row["Arabic_Text"])
        hits = count_saw_positions(text)

        if len(hits) > 1:
            multi_saw += 1
        if any(h["inside_quotes"] for h in hits):
            saw_inside_quotes += 1

        # Show first few exemplars
        if d == book_q[0]:
            extracted = extract_matn_portion(text)
            print(f"\n  First quarantined (idx={d['query_idx']}, tfidf={d.get('tfidf_score',0):.3f}):")
            print(f"  SAW hits: {len(hits)}")
            for h in hits:
                print(f"    pos={h['pos']}, in_quotes={h['inside_quotes']}")
                print(f"      before: ...{h['context_before']}")
                print(f"      after:  {h['context_after']}")
            print(f"  Extracted matn ({len(extracted)} chars): {extracted[:150]}")
            print(f"  Full text ({len(text)} chars): {text[:250]}...{text[-100:]}")

    print(f"\n  Sample: {total_examined} entries")
    print(f"  Multi-SAW: {multi_saw}/{total_examined}")
    print(f"  SAW inside quotes: {saw_inside_quotes}/{total_examined}")
