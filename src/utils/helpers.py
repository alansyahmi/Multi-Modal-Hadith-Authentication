"""
Utility helpers: file I/O, timing, dataset stats, and serialisation.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@contextmanager
def timer(label: str = "Timer"):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {elapsed:.2f}s")


def save_json(data: Any, path: Path, **kwargs):
    """Save data as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)
    print(f"  Saved: {path} ({len(json.dumps(data, ensure_ascii=False)):,} chars)")


def save_csv(df: pd.DataFrame, path: Path):
    """Save DataFrame as CSV."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {path} ({len(df):,} rows)")


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataframe_stats(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Any]:
    """Print and return useful DataFrame statistics."""
    stats = {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing": {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()},
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
    }

    print(f"\n=== {name} Stats ===")
    print(f"  Rows:       {stats['rows']:,}")
    print(f"  Columns:    {len(stats['columns'])}")
    print(f"  Memory:     {stats['memory_mb']:.1f} MB")
    if stats["missing"]:
        print(f"  Missing:    {stats['missing']}")
    return stats
