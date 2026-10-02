#!/usr/bin/env python3
"""
Targeted Shamela Scraper — Scrapes only the biographical entries for narrators
present in our aligned evaluation dataset from shamela.ws.

Steps:
1. Extract unique narrator names from the aligned evaluation dataset.
2. Launch headful Playwright with Cloudflare stealth and wait/solve support.
3. Scrape the table of contents (TOC) of Tahdhib al-Kamal (Book 3722) to map
   narrator names to their respective page numbers.
4. Match target narrators with the TOC to find which page numbers to download.
5. Download and parse ONLY those target pages, saving the biographies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import ast
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from src.shamela.web_downloader import is_challenge_page, extract_text_from_html, _try_cdp_connection

BOOK_ID = 3722
BASE_URL = f"https://shamela.ws/book/{BOOK_ID}"
OUTPUT_DIR = Path("output/tahdhib_al_kamal_targets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_target_narrators() -> list[str]:
    """Load the list of unique aligned narrators from the evaluation set."""
    aligned_json = Path("scratch/unique_aligned_narrators.json")
    if aligned_json.exists():
        with open(aligned_json, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # If not found, compute on the fly
    align_path = Path("output/stage1/alignment_results.json")
    sanad_path = Path("output/stage1/sanadset_cleaned.csv")
    if not sanad_path.exists():
        sanad_path = Path("Dataset/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
        
    if not align_path.exists() or not sanad_path.exists():
        print("Error: Stage 1 outputs not found. Run Stage 1 first.")
        sys.exit(1)
        
    print("Extracting target narrators from Stage 1 datasets...")
    with open(align_path, "r", encoding="utf-8") as f:
        alignment_data = json.load(f)
        
    matched_indices = [int(item["corpus_idx"]) for item in alignment_data if item.get("is_match") and item.get("corpus_idx") is not None]
    df_sanad = pd.read_csv(sanad_path)
    
    unique_narrators = set()
    for idx in matched_indices:
        if idx < len(df_sanad):
            sanad_val = df_sanad.iloc[idx]["Sanad"]
            if pd.notna(sanad_val):
                try:
                    narrators = ast.literal_eval(sanad_val)
                    if isinstance(narrators, list):
                        for n in narrators:
                            unique_narrators.add(n.strip())
                except Exception:
                    pass
    
    targets = sorted(list(unique_narrators))
    with open(aligned_json, "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    return targets


def get_html_with_cf_bypass(page, url: str) -> str:
    """Load URL with Playwright, handling navigation timeouts and Cloudflare challenge."""
    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
    except Exception:
        pass
        
    html = page.content()
    if is_challenge_page(html):
        print(f"\n  [CLOUDFLARE DETECTED] Please solve the challenge in the browser window for: {url}")
        start_time = time.time()
        while is_challenge_page(html) and (time.time() - start_time) < 120:
            time.sleep(2)
            try:
                html = page.content()
            except Exception:
                html = ""
        if not is_challenge_page(html):
            print("  [SUCCESS] Challenge solved! Continuing...")
            time.sleep(2)
            try:
                html = page.content()
            except Exception:
                html = ""
        else:
            print("  [TIMEOUT] Challenge not solved.")
            
    return html


def clean_arabic_name(text: str) -> str:
    """Normalise Arabic text by removing diacritics, tags, explanatory words, and kunyas."""
    if not text:
        return ""
    import re
    # Remove XML/HTML tags
    text = re.sub(r"<[^>]*>", "", text)
    # Remove diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Normalise Alifs
    text = re.sub(r"[أإآ]", "ا", text)
    # Normalise Ya/Alef Maksura
    text = re.sub(r"ى", "ي", text)
    # Normalise Kunya cases (أبا / أبي -> أبو)
    text = re.sub(r"\b(ابا|ابي)\b", "ابو", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Targeted Shamela Scraper")
    parser.add_argument("--browser", action="store_true", help="Force Playwright standalone headful browser mode")
    parser.add_argument("--port", type=int, default=9222, help="CDP port for running Chrome (default: 9222)")
    args = parser.parse_args()

    targets = load_target_narrators()
    print(f"Loaded {len(targets):,} target narrators to extract.")
    
    print("\nStarting Playwright browser session...")
    with sync_playwright() as pw:
        # Determine if we should connect to port-debugging Chrome (default) or start fresh Playwright
        use_cdp = not args.browser and _try_cdp_connection(port=args.port)
        
        if use_cdp:
            print(f"Connecting to existing Chrome via CDP on port {args.port}...")
            browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{args.port}")
        else:
            print("Launching Playwright standalone headful browser...")
            browser = pw.chromium.launch(headless=False)
            
        try:
            if use_cdp and browser.contexts:
                context = browser.contexts[0]
            else:
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    locale="ar"
                )
            page = context.new_page()
            
            # Apply stealth if available
            try:
                from playwright_stealth import stealth_sync
                stealth_sync(page)
            except ImportError:
                pass
                
            # ── Step 1: Scrape Table of Contents (TOC) ─────────────────
            print("\n=== STEP 1: SCRAPING TABLE OF CONTENTS ===")
            toc_map = {}  # Narrator name -> page number
            
            # We start with page 1 of the TOC. Shamela TOC might be paginated or single page.
            # For Book 3722, it's typically paginated or structured. We check the index page.
            index_url = BASE_URL
            print(f"Fetching book index from {index_url}...")
            html = get_html_with_cf_bypass(page, index_url)
            
            # Check if there are pagination links for the index
            soup = BeautifulSoup(html, "lxml")
            
            # Collect index links from the current page
            def parse_toc_links(soup_obj):
                found = 0
                for link in soup_obj.find_all("a", href=True):
                    href = link["href"]
                    # URL structure: /book/3722/123
                    if f"/book/{BOOK_ID}/" in href:
                        text = link.get_text(strip=True)
                        if text and not text.isdigit():
                            parts = href.split("?")[0].split("#")[0].strip("/").split("/")
                            if parts:
                                try:
                                    page_num = int(parts[-1])
                                    # Normalise name by stripping whitespace
                                    toc_map[text] = page_num
                                    found += 1
                                except ValueError:
                                    pass
                return found
                
            found_count = parse_toc_links(soup)
            print(f"  Parsed {found_count} narrator entries from index page.")
            
            # Check if index has pagination (e.g. links containing ?page=X)
            toc_pages = set()
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if f"/book/{BOOK_ID}?" in href or f"/book/{BOOK_ID}/?" in href:
                    toc_pages.add(href)
                    
            if toc_pages:
                print(f"  Found {len(toc_pages)} additional index pages. Scraping them...")
                for idx_page in sorted(list(toc_pages)):
                    page_url = f"https://shamela.ws{idx_page}"
                    print(f"  Fetching sub-index: {page_url}")
                    page_html = get_html_with_cf_bypass(page, page_url)
                    sub_soup = BeautifulSoup(page_html, "lxml")
                    sub_found = parse_toc_links(sub_soup)
                    print(f"    Found {sub_found} entries.")
                    time.sleep(1.5)
                    
            print(f"Total entries mapped in TOC: {len(toc_map):,}")
            
            # Save TOC map for reference
            with open("scratch/toc_map_3722.json", "w", encoding="utf-8") as f:
                json.dump(toc_map, f, ensure_ascii=False, indent=2)
                
            # ── Step 2: Match Target Narrators ─────────────────────────
            print("\n=== STEP 2: MATCHING NARRATORS ===")
            matched_targets = {}
            
            # Pre-clean TOC names
            cleaned_toc = {}
            for k, v in toc_map.items():
                cleaned_toc[clean_arabic_name(k)] = (k, v)
                
            for target in targets:
                cleaned_target = clean_arabic_name(target)
                # Ignore trivially short names that match wrong things (like 'أبي')
                if len(cleaned_target) <= 2:
                    continue
                    
                # 1. Try exact cleaned match
                if cleaned_target in cleaned_toc:
                    matched_targets[target] = cleaned_toc[cleaned_target][1]
                    continue
                    
                # 2. Try substring match (e.g. "أحمد بن إبراهيم الدورقي" vs "أحمد بن إبراهيم")
                found = False
                for ck, (raw_k, v) in cleaned_toc.items():
                    if cleaned_target in ck or ck in cleaned_target:
                        matched_targets[target] = v
                        found = True
                        break
                        
                if not found:
                    # 3. Try token subset matching (e.g. if 2 or more main words match)
                    target_words = set(cleaned_target.split())
                    target_words = {w for w in target_words if len(w) > 2 and w not in ("ابن", "بنت", "عبد")}
                    for ck, (raw_k, v) in cleaned_toc.items():
                        toc_words = set(ck.split())
                        toc_words = {w for w in toc_words if len(w) > 2 and w not in ("ابن", "بنت", "عبد")}
                        
                        if len(target_words) >= 2 and (target_words.issubset(toc_words) or toc_words.issubset(target_words)):
                            matched_targets[target] = v
                            found = True
                            break
                            
            # Load manual overrides if they exist
            manual_path = Path("scratch/manual_matches.json")
            if manual_path.exists():
                try:
                    with open(manual_path, "r", encoding="utf-8") as f:
                        manual_matches = json.load(f)
                    print(f"  Loaded {len(manual_matches)} manual match overrides from {manual_path}")
                    # Merge manual overrides (overwriting auto-detected ones if duplicated)
                    for k, v in manual_matches.items():
                        matched_targets[k] = int(v)
                except Exception as e:
                    print(f"  Warning: Failed to load manual overrides: {e}")
                            
            print(f"Matched {len(matched_targets)} / {len(targets)} target narrators to book pages.")
            
            # ── Step 3: Extract matched pages ──────────────────────────
            print("\n=== STEP 3: SCRAPING BIOGRAPHIES ===")
            unique_pages = sorted(list(set(matched_targets.values())))
            print(f"Need to download {len(unique_pages)} unique pages from shamela.ws...")
            
            # Invert map to group narrators by page
            page_to_narrators = {}
            for k, v in matched_targets.items():
                if v not in page_to_narrators:
                    page_to_narrators[v] = []
                page_to_narrators[v].append(k)
                
            success_count = 0
            for i, page_num in enumerate(unique_pages, 1):
                page_url = f"{BASE_URL}/{page_num}"
                print(f"[{i}/{len(unique_pages)}] Fetching page {page_num} for: {page_to_narrators[page_num]}")
                
                page_html = get_html_with_cf_bypass(page, page_url)
                text = extract_text_from_html(page_html)
                
                if text and not is_challenge_page(page_html):
                    # Save page raw html and parsed text
                    page_file = OUTPUT_DIR / f"page_{page_num:05d}.txt"
                    page_file.write_text(text, encoding="utf-8")
                    success_count += 1
                else:
                    print(f"  Failed to fetch page {page_num}")
                    
                time.sleep(2.0)  # Gentle delay
                
            print(f"\nExtraction completed! Successfully saved {success_count} target biography pages.")
        finally:
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
