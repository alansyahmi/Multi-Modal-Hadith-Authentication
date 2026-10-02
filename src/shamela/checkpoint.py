"""
SQLite-backed checkpoint store for resumable book downloads.

Tracks per-page download progress so that interrupted web scrapes can
resume from the last successful page without re-downloading.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pages (
    book_id         INTEGER NOT NULL,
    page            INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    last_attempt_at TEXT,
    downloaded_at   TEXT,
    content_hash    TEXT,
    PRIMARY KEY (book_id, page)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_status ON pages(book_id, status);
"""


class CheckpointStore:
    """Tracks per-page download progress in a SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── connection management ─────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Create schema on first use."""
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.execute(CREATE_INDEX_SQL)
        self.conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── page lifecycle ────────────────────────────────────────

    def mark_pending(self, book_id: int, page: int) -> None:
        """Register a page as pending (upsert)."""
        self.conn.execute(
            """INSERT OR IGNORE INTO pages (book_id, page, status)
               VALUES (?, ?, 'pending')""",
            (book_id, page),
        )
        self.conn.commit()

    def mark_downloaded(self, book_id: int, page: int, content_hash: str = "") -> None:
        """Mark a page as successfully downloaded."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO pages
                   (book_id, page, status, attempts, downloaded_at, content_hash)
               VALUES (?, ?, 'downloaded',
                   COALESCE((SELECT attempts FROM pages WHERE book_id=? AND page=?), 0) + 1,
                   ?, ?)""",
            (book_id, page, book_id, page, now, content_hash),
        )
        self.conn.commit()

    def mark_failed(self, book_id: int, page: int, error: str) -> None:
        """Record a failed download attempt."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO pages
                   (book_id, page, status, attempts, last_error, last_attempt_at)
               VALUES (?, ?, 'failed',
                   COALESCE((SELECT attempts FROM pages WHERE book_id=? AND page=?), 0) + 1,
                   ?, ?)""",
            (book_id, page, book_id, page, error, now),
        )
        self.conn.commit()

    # ── queries ───────────────────────────────────────────────

    def get_next_pending(self, book_id: int) -> Optional[int]:
        """Return the lowest pending page number, or None."""
        row = self.conn.execute(
            "SELECT page FROM pages WHERE book_id=? AND status='pending' ORDER BY page LIMIT 1",
            (book_id,),
        ).fetchone()
        return row[0] if row else None

    def get_progress(self, book_id: int) -> tuple[int, int, int]:
        """Return (done, total, failed) counts for a book."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) FROM pages WHERE book_id=? GROUP BY status",
            (book_id,),
        ).fetchall()
        counts = {row[0]: row[1] for row in rows}
        done = counts.get("downloaded", 0)
        failed = counts.get("failed", 0)
        total = done + failed + counts.get("pending", 0)
        return done, total, failed

    def is_downloaded(self, book_id: int, page: int) -> bool:
        """Check if a specific page was already downloaded."""
        row = self.conn.execute(
            "SELECT status FROM pages WHERE book_id=? AND page=?",
            (book_id, page),
        ).fetchone()
        return row is not None and row[0] == "downloaded"

    def reset_failed(self, book_id: int) -> int:
        """Reset all failed pages to pending. Returns count reset."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM pages WHERE book_id=? AND status='failed'",
            (book_id,),
        ).fetchone()[0]
        self.conn.execute(
            "UPDATE pages SET status='pending', last_error=NULL WHERE book_id=? AND status='failed'",
            (book_id,),
        )
        self.conn.commit()
        return count

    def reset_all(self, book_id: int) -> None:
        """Delete all checkpoint data for a book (fresh start)."""
        self.conn.execute("DELETE FROM pages WHERE book_id=?", (book_id,))
        self.conn.commit()
