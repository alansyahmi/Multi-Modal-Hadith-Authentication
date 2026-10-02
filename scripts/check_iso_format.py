"""Mount the Shamela ISO and detect format + list book availability."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shamela.iso_extractor import mount_iso, _detect_format, get_book_meta_from_master, extract_full_iso_to

ISO_PATH = Path(r"c:\Projects\FYP Project\shamela-win-1447.11.iso")

print("Mounting ISO...")
drive = mount_iso(ISO_PATH)
print(f"Drive: {drive}")

fmt = _detect_format(drive)
print(f"Format: {fmt}")

if fmt == "legacy":
    print("\n=== LEGACY FORMAT ===")
    books_dir = Path(drive) / "Books"
    if books_dir.exists():
        # Check for our books
        for bid in [3722, 1299, 1222]:
            bok_path = books_dir / str(bid) / f"{bid}.bok"
            bok_flat = books_dir / f"{bid}.bok"
            found = bok_path.exists() or bok_flat.exists()
            print(f"  Book {bid}: {'FOUND' if found else 'NOT FOUND'}")

elif fmt == "modern":
    print("\n=== MODERN FORMAT ===")
    # Extract just the master.db to check catalog
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="shamela_check_"))
    print(f"Extracting master.db to {tmp}...")

    try:
        from src.shamela.iso_extractor import get_7z_password, _extract_from_7z
        password = get_7z_password(drive)
        print(f"Password extracted: {password[:4]}...")

        success = _extract_from_7z(
            drive,
            archive_path="data/shamela.bin",
            output_dir=str(tmp),
            password=password,
            verbose=False,
        )
        print(f"7z extraction: {'OK' if success else 'FAILED'}")

        # Find master.db
        candidates = list(tmp.glob("**/master.db"))
        if candidates:
            master_db = candidates[0]
            print(f"master.db: {master_db} ({master_db.stat().st_size:,} bytes)")

            import sqlite3
            conn = sqlite3.connect(str(master_db))
            cur = conn.cursor()

            # Check our target books
            for bid in [3722, 1299, 1222]:
                cur.execute(
                    "SELECT b.book_id, b.book_name, a.author_name FROM [book] b "
                    "LEFT JOIN [author] a ON b.main_author = a.author_id "
                    "WHERE b.book_id = ?", (bid,)
                )
                row = cur.fetchone()
                if row:
                    print(f"\n  Book {bid}: {row[1]}")
                    print(f"    Author: {row[2]}")
                    # Check page count
                    cur.execute(
                        "SELECT COUNT(*) FROM [book_page] WHERE book_id = ?", (bid,)
                    )
                    pc = cur.fetchone()[0]
                    print(f"    Pages: {pc:,}")
            conn.close()

            # Check Lucene index structure
            lucene_dir = tmp / "database" / "store" / "page"
            if lucene_dir.exists():
                lucene_files = list(lucene_dir.glob("*"))
                print(f"\nLucene index: {lucene_dir}")
                print(f"  Files: {len(lucene_files)}")
                for f in lucene_files[:10]:
                    print(f"    {f.name} ({f.stat().st_size:,} bytes)")
        else:
            print("master.db NOT FOUND after extraction")
            print(f"Contents of {tmp}:")
            for item in tmp.rglob("*"):
                if item.is_file():
                    print(f"  {item.relative_to(tmp)}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"\nUnknown format: {fmt}")
    # Show what's on the drive
    for item in Path(drive).iterdir():
        print(f"  {item.name}")
