#!/usr/bin/env python3
"""
Shamela Book Extractor — Extract books from Al-Maktaba Al-Shamela.

Two extraction paths:
  ISO  : Mount shamela-win-1447.11.iso, read .bok (Access MDB) files via ODBC.
         Fast, offline, no rate limits. Requires Access Database Engine.
  Web  : Page-by-page download from shamela.ws with Cloudflare bypass,
         SQLite checkpointing, and resume support.

Output:
  output/<book_slug>/
    raw_volume_01.txt  .. raw_volume_NN.txt   Per-volume raw Arabic text
    structured.json                            Parsed narrator entries
    extraction_log.json                        Metadata and statistics
    checkpoint.db                              Web download resume state
    pages/                                     Raw HTML pages (web path only)

Usage:
  # Extract Tahdhib al-Kamal (book 3722) — auto-detect method
  python run_shamela_extract.py --book-id 3722

  # Force ISO extraction
  python run_shamela_extract.py --book-id 3722 --method iso

  # Web download with custom delay and resume
  python run_shamela_extract.py --book-id 3722 --method web --delay 2.0 --resume

  # Extract a different book
  python run_shamela_extract.py --book-id 1299 --method web

  # Skip narrator parsing (raw text only)
  python run_shamela_extract.py --book-id 3722 --no-parse

Environment variables:
  SHAMELA_CF_CLEARANCE    Manual Cloudflare clearance cookie (Tier 2 bypass)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    SHAMELA_BOOKS,
    SHAMELA_ISO_PATH,
    SHAMELA_OUTPUT_DIR,
)
from src.shamela import (
    BookMeta,
    CheckpointStore,
    ExtractionConfig,
    download_book,
    extract_from_iso,
    mount_iso,
    parse_all_volumes,
    unmount_iso,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shamela Book Extractor — Extract books from Al-Maktaba Al-Shamela",
    )
    parser.add_argument(
        "--book-id", type=int, required=True,
        help="Shamela book ID (e.g., 3722 for Tahdhib al-Kamal)",
    )
    parser.add_argument(
        "--method", choices=["iso", "web", "auto"], default="auto",
        help="Extraction method. 'auto' tries ISO first, falls back to web. (default: auto)",
    )
    parser.add_argument(
        "--iso", type=Path, default=None,
        help="Path to Shamela ISO file (default: shamela-win-1447.11.iso in project root)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: output/<book_slug>/)",
    )
    parser.add_argument(
        "--start-page", type=int, default=1,
        help="Start page for web extraction (default: 1)",
    )
    parser.add_argument(
        "--end-page", type=int, default=None,
        help="End page for web extraction (default: all pages)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Delay between web requests in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--browser", action="store_true",
        help="Use Playwright browser automation for Cloudflare bypass (Tier 3)",
    )
    parser.add_argument(
        "--no-parse", action="store_true",
        help="Skip narrator entry parsing (output raw text only)",
    )
    return parser.parse_args()


def _get_book_slug(book_id: int) -> str:
    """Get the output directory slug for a book ID."""
    if book_id in SHAMELA_BOOKS:
        return SHAMELA_BOOKS[book_id]["slug"]
    return f"book_{book_id}"


def _get_book_info(book_id: int) -> tuple[str, str]:
    """Get (title, author) for a known book ID, or defaults."""
    if book_id in SHAMELA_BOOKS:
        info = SHAMELA_BOOKS[book_id]
        return info["title"], info["author"]
    return f"Book {book_id}", "Unknown"


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []

    book_id: int = args.book_id

    # ── Determine output directory ─────────────────────────────
    if args.output:
        output_dir = Path(args.output)
    else:
        book_slug = _get_book_slug(book_id)
        output_dir = SHAMELA_OUTPUT_DIR / book_slug

    iso_path = args.iso or SHAMELA_ISO_PATH

    title, author = _get_book_info(book_id)

    print("=" * 60)
    print("SHAMELA BOOK EXTRACTOR")
    print("=" * 60)
    print(f"  Book ID: {book_id}")
    print(f"  Title:   {title}")
    print(f"  Author:  {author}")
    print(f"  Method:  {args.method}")
    print(f"  Output:  {output_dir}")
    print(f"  Parse:   {'yes' if not args.no_parse else 'no (raw text only)'}")
    print()

    # ── Extraction ────────────────────────────────────────────
    volumes: dict[int, str] = {}
    meta: BookMeta = BookMeta(book_id=book_id, title=title, author=author)
    extraction_method = args.method

    # Auto-detect: try ISO first, fall back to web
    if args.method in ("auto", "iso"):
        try:
            if not iso_path.exists():
                raise FileNotFoundError(f"ISO not found: {iso_path}")

            drive = mount_iso(iso_path)
            try:
                # extract_from_iso handles both legacy (.bok) and modern (7z+SQLite)
                volumes, meta = extract_from_iso(drive, book_id)
                extraction_method = "iso"
            finally:
                # Keep ISO mounted for potential re-use
                pass
        except Exception as e:
            msg = f"ISO extraction failed: {e}"
            print(f"  {msg}")
            warnings.append(msg)
            if args.method == "iso":
                print("  FATAL: ISO method was explicitly requested.")
                sys.exit(1)
            print("  Falling back to web download...\n")
            extraction_method = "web"

    # Web fallback
    if extraction_method == "web" or (
        args.method == "auto" and not volumes
    ):
        config = ExtractionConfig(
            book_id=book_id,
            output_dir=output_dir,
            method="web",
            iso_path=iso_path,
            start_page=args.start_page,
            end_page=args.end_page,
            delay_seconds=args.delay,
            resume=args.resume,
            parse_narrators=not args.no_parse,
            use_browser=args.browser,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_db = output_dir / "checkpoint.db"
        checkpoint = CheckpointStore(checkpoint_db)

        try:
            volumes, meta = download_book(config, checkpoint)
        except Exception as e:
            print(f"  Web download failed: {e}")
            sys.exit(1)
        finally:
            checkpoint.close()

    if not volumes:
        print("  No text extracted. Check errors above.")
        sys.exit(1)

    # ── Narrator parsing ──────────────────────────────────────
    entries = []
    if not args.no_parse:
        print(f"\n{'='*60}")
        print("NARRATOR ENTRY PARSING")
        print(f"{'='*60}")
        try:
            entries = parse_all_volumes(volumes)
            print(f"  Parsed {len(entries)} narrator entries")
            high = sum(1 for e in entries if e.confidence == "high")
            med = sum(1 for e in entries if e.confidence == "medium")
            low = sum(1 for e in entries if e.confidence == "low")
            print(f"  Confidence: {high} high, {med} medium, {low} low")
        except Exception as e:
            msg = f"Narrator parsing failed: {e}"
            print(f"  {msg}")
            warnings.append(msg)

    # Update meta with book registry info
    meta = BookMeta(
        book_id=meta.book_id,
        title=title,
        author=author,
        bok_path=meta.bok_path,
        volumes=meta.volumes,
        total_pages=meta.total_pages,
        source_url=meta.source_url,
    )

    # ── Write outputs ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WRITING OUTPUTS")
    print(f"{'='*60}")
    write_outputs(
        volumes=volumes,
        entries=entries,
        output_dir=output_dir,
        book_meta=meta,
        extraction_method=extraction_method,
        warnings=warnings,
        started_at=started_at,
    )

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Book:       {title}")
    print(f"  Author:     {author}")
    print(f"  Method:     {extraction_method}")
    print(f"  Volumes:    {len(volumes)}")
    print(f"  Total text: {sum(len(t) for t in volumes.values()):,} chars")
    print(f"  Entries:    {len(entries)}")
    print(f"  Warnings:   {len(warnings)}")
    print(f"  Output:     {output_dir}")


if __name__ == "__main__":
    main()
