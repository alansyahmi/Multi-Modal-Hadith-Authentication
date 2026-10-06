"""
Download canonical narrator datasets from Hadith-Viz.

This script fetches the verified scholarly records and narrator cards
from Hadith-Viz into `Dataset/hadith_viz/`.

Artifacts downloaded:
1. `rijal_shamela_cards.json` (~888+ verified narrators with scholarly grades)
2. `shamela_rijal2.json` (Detailed multi-scholar evaluations and biographical excerpts)
3. `rijal_v2_cards.json` (Master 248k narrator index with death years & regions) [Optional/Flagged]

Usage:
    python scripts/download_hadith_viz.py [--all]
"""

import os
import sys
import argparse
import urllib.request
import json
import time

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "hadith_viz")

FILES_TO_DOWNLOAD = {
    "rijal_shamela_cards.json": {
        "url": "https://hadith-viz.com/output/rijal_shamela_cards.json",
        "desc": "Verified narrator cards with grades and canonical death years",
        "essential": True,
    },
    "shamela_rijal2.json": {
        "url": "https://hadith-viz.com/output/shamela_rijal2.json",
        "desc": "Detailed classical multi-scholar evaluations and citations",
        "essential": True,
    },
    "rijal_v2_cards.json": {
        "url": "https://hadith-viz.com/output/rijal_v2_cards.json",
        "desc": "Master 248k narrator index with canonical IDs, death years, and nisbas",
        "essential": False,
    }
}


def download_file(url: str, dest_path: str, desc: str) -> bool:
    print(f"\n[Hadith-Viz] Downloading: {os.path.basename(dest_path)}")
    print(f"  Description: {desc}")
    print(f"  Source URL : {url}")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Hadith-Viz-Academic/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f_out:
            total_bytes = 0
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB chunk
                if not chunk:
                    break
                f_out.write(chunk)
                total_bytes += len(chunk)
                sys.stdout.write(f"\r  Progress: {total_bytes / (1024 * 1024):.2f} MB")
                sys.stdout.flush()
        
        elapsed = time.time() - start_time
        mb = total_bytes / (1024 * 1024)
        print(f"\n  [SUCCESS] {mb:.2f} MB downloaded in {elapsed:.2f}s ({mb/max(elapsed, 0.001):.2f} MB/s)")
        
        # Verify JSON validity
        print("  Verifying JSON integrity...")
        with open(dest_path, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)
            if isinstance(data, list):
                print(f"  Integrity Verified: JSON list with {len(data):,} records.")
            elif isinstance(data, dict):
                print(f"  Integrity Verified: JSON dictionary with {len(data):,} keys.")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Hadith-Viz Canonical Datasets")
    parser.add_argument("--all", action="store_true", help="Download master 248k index (rijal_v2_cards.json) in addition to essential datasets")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Target Directory: {DATASET_DIR}")

    for filename, meta in FILES_TO_DOWNLOAD.items():
        if not meta["essential"] and not args["all"] if isinstance(args, dict) else (not meta["essential"] and not args.all):
            print(f"\n[Skipping Optional] {filename} (Pass `--all` to download the master 248k narrator index)")
            continue

        dest_path = os.path.join(DATASET_DIR, filename)
        if os.path.exists(dest_path) and not args.force:
            print(f"\n[Already Exists] {filename} (Pass `--force` to re-download)")
            continue

        download_file(meta["url"], dest_path, meta["desc"])

    print("\n[Done] All requested Hadith-Viz datasets processed.")


if __name__ == "__main__":
    main()
