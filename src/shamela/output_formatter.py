"""
Output formatter — writes extracted book text to .txt files and structured JSON.

Produces:
  1. output_dir/raw_volume_01.txt … raw_volume_NN.txt  (per-volume raw Arabic text)
  2. output_dir/structured.json                         (parsed narrator entries)
  3. output_dir/extraction_log.json                     (metadata + stats)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .models import BookMeta, NarratorEntry


def write_outputs(
    volumes: dict[int, str],
    entries: Sequence[NarratorEntry],
    output_dir: Path,
    book_meta: BookMeta,
    *,
    extraction_method: str = "iso",
    warnings: Sequence[str] = (),
    started_at: str = "",
) -> None:
    """
    Write all output files for an extraction run.

    Args:
        volumes: {volume_number: raw_text}
        entries: List of parsed NarratorEntry objects
        output_dir: Target directory (created if needed)
        book_meta: Discovered book metadata
        extraction_method: "iso" or "web"
        warnings: Non-fatal warnings encountered during extraction
        started_at: ISO timestamp of when the run began
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Raw text per volume ────────────────────────────────
    for vol_num in sorted(volumes):
        txt_path = output_dir / f"raw_volume_{vol_num:02d}.txt"
        _write_text_file(txt_path, volumes[vol_num])
        print(f"  Wrote {txt_path.name} ({len(volumes[vol_num]):,} chars)")

    if not volumes:
        print("  WARNING: No volumes to write (empty extraction)")

    # ── 2. Structured JSON ─────────────────────────────────────
    structured_path = output_dir / "structured.json"
    entries_data = [_entry_to_dict(e) for e in entries]
    _write_json_file(structured_path, entries_data)
    print(f"  Wrote {structured_path.name} ({len(entries)} entries)")

    # ── 3. Extraction log ──────────────────────────────────────
    completed_at = datetime.now(timezone.utc).isoformat()
    log_path = output_dir / "extraction_log.json"

    log_data: dict[str, Any] = {
        "book": {
            "id": book_meta.book_id,
            "title": book_meta.title,
            "author": book_meta.author,
            "source_url": book_meta.source_url,
        },
        "extraction": {
            "method": extraction_method,
            "started_at": started_at or completed_at,
            "completed_at": completed_at,
            "volumes_detected": book_meta.volumes,
            "pages_processed": book_meta.total_pages,
            "total_chars": sum(len(t) for t in volumes.values()),
            "narrator_entries_parsed": len(entries),
        },
        "warnings": list(warnings),
    }
    _write_json_file(log_path, log_data)
    print(f"  Wrote {log_path.name}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n  Output directory: {output_dir}")
    print(f"  Volumes:  {len(volumes)}")
    print(f"  Entries:  {len(entries)}")
    print(f"  Warnings: {len(warnings)}")


def _write_text_file(path: Path, text: str) -> None:
    """Write UTF-8 text file with Arabic-friendly encoding."""
    path.write_text(text, encoding="utf-8")


def _write_json_file(path: Path, data: Any) -> None:
    """Write JSON with Arabic text preserved (no ASCII escaping)."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _entry_to_dict(entry: NarratorEntry) -> dict[str, Any]:
    """Convert a NarratorEntry to a JSON-serializable dict."""
    return {
        "entry_id": entry.entry_id,
        "volume": entry.volume,
        "page": entry.page,
        "name": entry.name,
        "full_text": entry.full_text,
        "teachers": list(entry.teachers),
        "students": list(entry.students),
        "grades": list(entry.grades),
        "death_info": entry.death_info,
        "confidence": entry.confidence,
    }
