"""Test shamela.ws with cloudscraper."""
import cloudscraper
from pathlib import Path

scraper = cloudscraper.create_scraper()

# Test Taqrib al-Tahdhib
url = "https://shamela.ws/book/1299"
print(f"Testing: {url}")
try:
    r = scraper.get(url, timeout=30)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        text = r.text
        arabic_count = sum(1 for c in text if '؀' <= c <= 'ۿ')
        print(f"  Length: {len(text):,} chars, Arabic: {arabic_count:,}")
        Path("output/shamela_taqrib.html").write_text(text, encoding="utf-8")
        print(f"  Saved to output/shamela_taqrib.html")
    else:
        print(f"  First 300: {r.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# Test Tahdhib al-Kamal
url2 = "https://shamela.ws/book/1222"
print(f"\nTesting: {url2}")
try:
    r2 = scraper.get(url2, timeout=30)
    print(f"  Status: {r2.status_code}")
    if r2.status_code == 200:
        text = r2.text
        arabic_count = sum(1 for c in text if '؀' <= c <= 'ۿ')
        print(f"  Length: {len(text):,} chars, Arabic: {arabic_count:,}")
        Path("output/shamela_tahdhib.html").write_text(text, encoding="utf-8")
        print(f"  Saved to output/shamela_tahdhib.html")
except Exception as e:
    print(f"  Error: {e}")
