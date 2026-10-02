"""
Shamela book extraction package.

Provides dual-path extraction:
  - ISO  : Mount shamela-win-14XX.iso, read legacy .bok (Access MDB) files
           or extract metadata from modern 7z+SQLite+Lucene format.
  - Web  : Resilient page-by-page download from shamela.ws with Cloudflare
           bypass, SQLite checkpointing, and resume support.

Plus rule-based narrator biography segmentation.
"""

from .checkpoint import CheckpointStore
from .iso_extractor import (
    discover_book,
    extract_from_bok,
    extract_from_iso,
    extract_full_iso_to,
    get_7z_password,
    mount_iso,
    unmount_iso,
)
from .models import BookMeta, ExtractionConfig, NarratorEntry
from .narrator_parser import parse_all_volumes, segment_volume
from .output_formatter import write_outputs
from .web_downloader import download_book

__all__ = [
    # Models
    "ExtractionConfig",
    "BookMeta",
    "NarratorEntry",
    # ISO path
    "mount_iso",
    "unmount_iso",
    "discover_book",
    "extract_from_bok",
    "extract_from_iso",
    "extract_full_iso_to",
    "get_7z_password",
    # Web path
    "download_book",
    "CheckpointStore",
    # Output
    "write_outputs",
    # Narrator parsing
    "parse_all_volumes",
    "segment_volume",
]
