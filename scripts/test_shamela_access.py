"""Test shamela.ws accessibility for scraping."""
import requests
from pathlib import Path

# Test with proper headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en;q=0.9",
}

# Test 1: Book info page for Taqrib al-Tahdhib
url = "https://shamela.ws/book/1299"
print(f"Testing: {url}")
try:
    r = requests.get(url, headers=headers, timeout=30)
    print(f"  Status: {r.status_code}")
    print(f"  Content length: {len(r.text):,} chars")
    if r.status_code == 200:
        # Check for Arabic content
        arabic_chars = sum(1 for c in r.text if '؀' <= c <= 'ۿ')
        print(f"  Arabic chars: {arabic_chars:,}")
        # Save sample
        Path("output/shamela_test.html").write_text(r.text, encoding="utf-8")
        print(f"  Saved to output/shamela_test.html")
    else:
        print(f"  Response: {r.text[:500]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 2: Try first page
url2 = "https://shamela.ws/book/1299/1"
print(f"\nTesting: {url2}")
try:
    r2 = requests.get(url2, headers=headers, timeout=30)
    print(f"  Status: {r2.status_code}")
    print(f"  Content length: {len(r2.text):,} chars")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: Try al-maktaba.org as alternative
url3 = "https://al-maktaba.org/book/1299"
print(f"\nTesting (al-maktaba): {url3}")
try:
    r3 = requests.get(url3, headers=headers, timeout=30)
    print(f"  Status: {r3.status_code}")
    print(f"  Content length: {len(r3.text):,} chars")
except Exception as e:
    print(f"  Error: {e}")
