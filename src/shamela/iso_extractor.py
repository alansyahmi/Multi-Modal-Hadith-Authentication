"""
ISO Extractor — mount shamela-win-14XX.iso and extract book information.

Shamela ISO versions:
  - Legacy (pre-2020): Individual .bok files (Microsoft Access .mdb) per book
    in Books/{id}/{id}.bok. Readable via pyodbc + Access Database Engine.
  - Modern (v1447+, 2024+): Single password-protected 7z archive (shamela.bin)
    containing SQLite metadata (master.db) + Apache Lucene page indexes.

This module handles BOTH formats:
  - Legacy: Full text extraction via ODBC
  - Modern: Metadata extraction via SQLite + Lucene text (with Java/PyLucene
    fallback, or redirection to web downloader for full text)

Provides:
    mount_iso()             — Mount the ISO, return drive letter
    unmount_iso()           — Unmount the ISO
    discover_book()         — Detect format and locate book data
    extract_from_iso()      — Full extraction (multi-format aware)
    get_7z_password()       — Extract the embedded password from shamela.exe
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from .models import BookMeta


# ── 7z archive password (extracted from shamela.exe at runtime) ──

_SHAMELA_7Z_PASSWORD: Optional[str] = None


def get_7z_password(iso_drive: str) -> str:
    """
    Extract the embedded 7z password from shamela.exe on the ISO.

    The password is hardcoded in the launcher executable as a 7za
    command-line argument: -p<password>
    """
    global _SHAMELA_7Z_PASSWORD
    if _SHAMELA_7Z_PASSWORD:
        return _SHAMELA_7Z_PASSWORD

    exe_path = Path(iso_drive) / "shamela.exe"
    if not exe_path.exists():
        raise RuntimeError(f"shamela.exe not found at {exe_path}")

    import re
    raw = exe_path.read_bytes()
    try:
        text = raw.decode("utf-16-le", errors="ignore")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")

    # Pattern: 7za command with -p<password>
    match = re.search(r'-p([a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]{8,64})', text)
    if match:
        _SHAMELA_7Z_PASSWORD = match.group(1)
        return _SHAMELA_7Z_PASSWORD

    raise RuntimeError("Could not extract 7z password from shamela.exe")


# ── format detection ───────────────────────────────────────────

def _detect_format(drive: str) -> str:
    """
    Detect the Shamela ISO format.

    Returns:
        "legacy"  — .bok (Access MDB) files, Books/{id}/{id}.bok
        "modern"  — shamela.bin (7z) + SQLite + Lucene indexes
        "unknown" — unrecognised layout
    """
    bin_path = Path(drive) / "data" / "shamela.bin"
    if bin_path.exists():
        return "modern"

    books_dir = Path(drive) / "Books"
    if books_dir.exists():
        return "legacy"

    return "unknown"


# ── ISO mounting ───────────────────────────────────────────────

def mount_iso(iso_path: Path) -> str:
    """
    Mount the Shamela ISO and return the drive letter (e.g. "D:").

    Uses PowerShell's Mount-DiskImage cmdlet.
    Raises RuntimeError if mounting fails.
    """
    iso_abs = str(iso_path.resolve())

    # Check if already mounted
    result = subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            f'(Get-DiskImage -ImagePath "{iso_abs}" | Get-Volume).DriveLetter'
        ],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode == 0 and result.stdout.strip():
        letter = result.stdout.strip()
        print(f"  ISO already mounted at {letter}:")
        return f"{letter}:"

    # Mount it
    print(f"  Mounting ISO: {iso_path.name} ...")
    result = subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            f'$img = Mount-DiskImage -ImagePath "{iso_abs}" -PassThru; '
            f'$vol = $img | Get-Volume; '
            f'Write-Output $vol.DriveLetter'
        ],
        capture_output=True, text=True, timeout=30,
    )

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"Failed to mount ISO.\n"
            f"  stderr: {result.stderr.strip()}\n"
            f"  Try mounting manually in File Explorer and re-running."
        )

    letter = result.stdout.strip()
    print(f"  Mounted at {letter}:")
    return f"{letter}:"


def unmount_iso(iso_path: Path) -> bool:
    """Unmount the ISO. Returns True on success."""
    iso_abs = str(iso_path.resolve())
    result = subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            f'Dismount-DiskImage -ImagePath "{iso_abs}"'
        ],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode == 0:
        print(f"  Unmounted ISO: {iso_path.name}")
        return True
    print(f"  Warning: Could not unmount ISO: {result.stderr.strip()}")
    return False


@contextmanager
def mounted_iso(iso_path: Path):
    """Context manager: mount ISO, yield drive letter."""
    drive = mount_iso(iso_path)
    try:
        yield drive
    finally:
        pass  # Keep mounted for re-use


# ── modern format: 7z + SQLite ─────────────────────────────────

def _extract_from_7z(
    drive: str,
    archive_path: str,
    output_dir: str,
    password: str,
    verbose: bool = True,
) -> bool:
    """
    Extract file(s) from a password-protected 7z archive.

    Uses the 7za.exe bundled on the ISO (at data/7za.exe or data/7za64.exe).
    Returns True on success.
    """
    # Prefer 64-bit extractor on 64-bit systems
    exe7z = Path(drive) / "data" / ("7za64.exe" if sys.maxsize > 2**32 else "7za.exe")
    if not exe7z.exists():
        exe7z = Path(drive) / "data" / "7za.exe"
    if not exe7z.exists():
        raise RuntimeError(f"7za.exe not found on ISO at {drive}\\data\\")

    archive = Path(drive) / archive_path
    if not archive.exists():
        raise RuntimeError(f"Archive not found: {archive}")

    cmd = [
        str(exe7z),
        "x", str(archive),
        f"-o{output_dir}",
        f"-p{password}",
        "-y",          # Assume Yes on all queries
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if verbose and result.stdout:
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"    {line}")

    return result.returncode == 0


def _read_master_db(db_path: Path) -> sqlite3.Connection:
    """Open the extracted master.db and return a connection."""
    if not db_path.exists():
        raise RuntimeError(f"master.db not found at {db_path}")
    return sqlite3.connect(str(db_path))


def get_book_meta_from_master(
    master_db_path: Path,
    book_id: int,
) -> Optional[BookMeta]:
    """
    Read book metadata from an extracted master.db.

    Returns BookMeta or None if the book ID is not found.
    """
    conn = sqlite3.connect(str(master_db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT b.book_id, b.book_name, a.author_name
               FROM [book] b
               LEFT JOIN [author] a ON b.main_author = a.author_id
               WHERE b.book_id = ?""",
            (book_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return BookMeta(
            book_id=row[0],
            title=row[1] or f"Book {book_id}",
            author=row[2] or "Unknown",
            source_url=f"https://shamela.ws/book/{book_id}",
        )
    finally:
        conn.close()


# ── legacy format: .bok (Access MDB) ───────────────────────────

def discover_book(drive: str, book_id: int) -> Optional[Path]:
    """
    Find book data for a given book ID on the mounted ISO.

    For legacy format: returns path to .bok file.
    For modern format: returns None (text is in Lucene index,
    use extract_from_iso() + web fallback).
    """
    fmt = _detect_format(drive)

    if fmt == "legacy":
        bok = Path(drive) / "Books" / str(book_id) / f"{book_id}.bok"
        if bok.exists():
            return bok
        bok_flat = Path(drive) / "Books" / f"{book_id}.bok"
        if bok_flat.exists():
            return bok_flat
        return None

    # Modern format: no single file per book
    return None


def extract_from_bok(
    bok_path: Path,
    book_id: int,
) -> tuple[dict[int, str], BookMeta]:
    """
    Extract text from a legacy .bok (Access MDB) file via ODBC.

    Only works with the legacy ISO format. For modern Shamela (v1447+),
    use extract_from_iso() which handles both formats.

    Returns (volumes_dict, book_meta).
    """
    import pyodbc

    DRIVERS = [
        "Microsoft Access Driver (*.mdb, *.accdb)",
        "Microsoft Access Driver (*.mdb)",
    ]

    conn = None
    last_error = ""

    for driver in DRIVERS:
        try:
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"DBQ={bok_path};"
                f"ExtendedAnsiSQL=1;"
            )
            conn = pyodbc.connect(conn_str)
            print(f"  Connected via: {driver}")
            break
        except pyodbc.Error as e:
            last_error = str(e)
            continue

    if not conn:
        if "architecture mismatch" in last_error.lower():
            bits = "64" if sys.maxsize > 2**32 else "32"
            raise RuntimeError(
                f"ODBC architecture mismatch ({bits}-bit Python). "
                f"Install the matching Access Database Engine or use --method web."
            )
        raise RuntimeError(
            f"Cannot connect to .bok via ODBC.\n"
            f"  Error: {last_error}\n"
            f"  Install Access Database Engine or use --method web."
        )

    try:
        # Discover schema
        cursor = conn.cursor()
        tables: dict[str, list[str]] = {}
        for row in cursor.tables():
            if row.table_type == "TABLE" and not row.table_name.startswith("MSys"):
                try:
                    cursor.execute(f"SELECT TOP 1 * FROM [{row.table_name}]")
                    tables[row.table_name] = [col[0] for col in cursor.description]
                except Exception:
                    tables[row.table_name] = []

        print(f"  Tables: {list(tables.keys())}")

        # Heuristic: find the text table
        text_table = None
        for name in ("book", "books", "0"):
            if name in tables:
                text_table = name
                break
        if not text_table:
            for tbl, cols in tables.items():
                if any("nass" in c.lower() for c in cols):
                    text_table = tbl
                    break
        if not text_table and tables:
            text_table = max(tables, key=lambda t: len(tables[t]))

        if not text_table:
            raise RuntimeError("No text table found in .bok")

        cols = tables[text_table]
        print(f"  Text table: [{text_table}] — {cols}")

        # Identify key columns
        col_lower = [c.lower() for c in cols]
        text_col = next((c for c in cols if c.lower() == "nass"), cols[1] if len(cols) > 1 else cols[0])
        page_col = next((c for c in cols if c.lower() in ("page", "pg", "id")), "id")
        part_col = next((c for c in cols if c.lower() in ("part", "vol", "juz")), "")

        # Build query
        select = [text_col, page_col]
        if part_col:
            select.append(part_col)
        order = f"[{page_col}]"
        if part_col:
            order = f"[{part_col}], " + order

        query = f"SELECT [{'], ['.join(select)}] FROM [{text_table}] ORDER BY {order}"
        cursor.execute(query)

        # Group by volume
        volumes: dict[int, str] = {}
        for row in cursor:
            text = str(row[0]).strip() if row[0] is not None else ""
            if not text or text == "None":
                continue
            vol = 1
            if part_col and len(row) > 2 and row[2] is not None:
                try:
                    vol = int(row[2])
                except (ValueError, TypeError):
                    vol = 1
            if vol not in volumes:
                volumes[vol] = []
            volumes[vol].append(text)

        joined = {v: "\n\n".join(t) for v, t in sorted(volumes.items())}
        meta = BookMeta(
            book_id=book_id,
            title=f"Book {book_id}",
            author="Unknown",
            bok_path=bok_path,
            volumes=len(joined),
            source_url=f"https://shamela.ws/book/{book_id}",
        )
        print(f"  Extracted {len(joined)} volumes, {sum(len(t) for t in joined.values()):,} chars")
        return joined, meta

    finally:
        conn.close()


# ── unified extraction (both formats) ──────────────────────────

def extract_from_iso(
    drive: str,
    book_id: int,
    *,
    extract_to: Optional[Path] = None,
) -> tuple[dict[int, str], BookMeta]:
    """
    Extract book data from the mounted ISO, handling both formats.

    For legacy format: extracts full text from .bok via ODBC.
    For modern format: extracts metadata from master.db, then raises
    RuntimeError with guidance to use the web downloader for full text
    (since the text lives in Lucene indexes requiring Java/PyLucene).

    Returns (volumes_dict, book_meta).
    """
    fmt = _detect_format(drive)
    print(f"  Detected format: {fmt}")

    if fmt == "legacy":
        bok = discover_book(drive, book_id)
        if not bok:
            raise RuntimeError(
                f"Book {book_id} not found on ISO. Check the book ID."
            )
        return extract_from_bok(bok, book_id)

    elif fmt == "modern":
        # Extract metadata from master.db
        if not extract_to:
            import tempfile
            extract_to = Path(tempfile.mkdtemp(prefix="shamela_"))
        extract_to.mkdir(parents=True, exist_ok=True)

        # Get password
        password = get_7z_password(drive)
        print(f"  Extracting master.db (modern format) ...")

        # Extract master.db only (small, fast)
        success = _extract_from_7z(
            drive,
            archive_path="data/shamela.bin",
            output_dir=str(extract_to),
            password=password,
            verbose=False,
        )
        if not success:
            raise RuntimeError("Failed to extract from shamela.bin")

        master_db = extract_to / "database" / "master.db"
        if not master_db.exists():
            # File might be at root if wildcard extraction happened
            candidates = list(extract_to.glob("**/master.db"))
            if candidates:
                master_db = candidates[0]

        if not master_db.exists():
            raise RuntimeError("master.db not found after extraction")

        meta = get_book_meta_from_master(master_db, book_id)
        if not meta:
            raise RuntimeError(f"Book {book_id} not found in master.db catalogue.")

        # The actual text is in Lucene indexes at database/store/page/
        # These require Java/PyLucene to read, which is impractical on Windows.
        # Redirect to web downloader.
        raise RuntimeError(
            f"Book '{meta.title}' (ID {book_id}) found in catalogue, but the\n"
            f"full text is stored in Apache Lucene indexes (database/store/page/)\n"
            f"which require Java/PyLucene to read.\n\n"
            f"Recommendation: use the web downloader instead:\n"
            f"  python run_shamela_extract.py --book-id {book_id} --method web\n\n"
            f"Or place the extracted Lucene index at:\n"
            f"  {extract_to}/database/store/page/\n"
            f"and process with a Lucene-compatible tool."
        )

    else:
        raise RuntimeError(
            f"Unknown ISO format. Expected Books/ directory (legacy) "
            f"or data/shamela.bin (modern). Got: {drive}"
        )


# ── convenience: extract entire ISO to disk ────────────────────

def extract_full_iso_to(
    drive: str,
    dest_dir: Path,
    *,
    password: Optional[str] = None,
) -> Path:
    """
    Extract the entire shamela.bin archive to disk for offline access.

    This can be useful for bulk processing or if you want to manually
    inspect the Lucene indexes.

    Returns the destination directory.
    """
    if not password:
        password = get_7z_password(drive)

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting full ISO to {dest_dir} ...")
    print(f"  This may take a few minutes (~300 MB uncompressed).")

    success = _extract_from_7z(
        drive,
        archive_path="data/shamela.bin",
        output_dir=str(dest_dir),
        password=password,
        verbose=True,
    )
    if not success:
        raise RuntimeError("Full extraction failed.")

    # Check what we got
    item_count = sum(1 for _ in dest_dir.rglob("*"))
    print(f"  Extracted {item_count} files to {dest_dir}")
    return dest_dir
