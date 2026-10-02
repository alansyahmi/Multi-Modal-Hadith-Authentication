"""
Dataclasses for the Shamela book extraction pipeline.

All dataclasses are frozen (immutable) per project conventions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for a single extraction run."""
    book_id: int
    output_dir: Path
    method: str = "auto"              # "iso" | "web" | "auto"
    iso_path: Optional[Path] = None
    start_page: int = 1
    end_page: Optional[int] = None
    delay_seconds: float = 1.5
    resume: bool = True
    parse_narrators: bool = True
    use_browser: bool = False


@dataclass(frozen=True)
class BookMeta:
    """Metadata about a Shamela book."""
    book_id: int
    title: str                        # Arabic title
    author: str                       # Author name
    bok_path: Optional[Path] = None   # Path to .bok file (ISO path only)
    volumes: int = 1                  # Number of volumes/parts detected
    total_pages: int = 0              # Total pages discovered
    source_url: str = ""              # shamela.ws URL


@dataclass(frozen=True)
class NarratorEntry:
    """A single narrator biography entry extracted from the text."""
    entry_id: int                     # Sequential ID within the book
    volume: int                       # Volume number (1-based)
    page: int                         # Page number
    name: str                         # Extracted narrator full name
    full_text: str                    # Complete biography text
    teachers: tuple[str, ...] = ()    # Names of teachers (ruwa 'an / shuyukh)
    students: tuple[str, ...] = ()    # Names of students (ruwa 'anhu / talamidh)
    grades: tuple[str, ...] = ()      # Jarh wa ta'dil evaluations
    death_info: str = ""              # Death date/location
    confidence: str = "low"           # "high" | "medium" | "low"
