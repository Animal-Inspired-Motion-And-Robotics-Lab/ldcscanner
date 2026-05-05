# ---------------------------------------------------------------------------
# io_utils.py — File I/O, GUI file picker, and label-cache management
# ---------------------------------------------------------------------------

import hashlib
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import CRACK_LABELS


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

def create_run_output_dir(base_dir="outputs"):
    """Create and return a unique timestamped directory for one script run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# CSV file selection
# ---------------------------------------------------------------------------

def select_csv_files_via_gui(initial_dir="."):
    """
    Open a GUI file picker to choose CSV files for analysis.

    Returns:
        List[str]: Selected file paths, or empty list if none selected/canceled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"GUI file picker unavailable ({exc}).")
        return []

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        selected = filedialog.askopenfilenames(
            title="Select CSV files for analysis",
            initialdir=str(Path(initial_dir).resolve()),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    return list(selected)


def build_source_path_lookup(file_paths):
    """Build a lookup from CSV basename to absolute source path."""
    return {Path(p).name: str(Path(p).resolve()) for p in file_paths}


def load_csv_files(file_paths):
    """
    Load multiple CSV files into a dict of {filename: DataFrame}.

    Args:
        file_paths: List of file paths to CSV files

    Returns:
        Dict mapping filename to DataFrame (only successfully loaded files).
    """
    dataframes = {}

    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue

            if not file_path.lower().endswith(".csv"):
                print(f"Warning: File is not a CSV - {file_path}")
                continue

            df = pd.read_csv(file_path)
            filename = Path(file_path).name
            dataframes[filename] = df
            print(f"Successfully loaded: {filename} ({len(df)} rows, {len(df.columns)} columns)")

        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")

    return dataframes


# ---------------------------------------------------------------------------
# Label cache — persist manual crack windows across runs
# ---------------------------------------------------------------------------

def get_file_fingerprint(file_path):
    """Return a lightweight fingerprint dict for cache validation."""
    try:
        stat = Path(file_path).stat()
        return {
            "source_path": str(Path(file_path).resolve()),
            "file_size":   int(stat.st_size),
            "mtime_ns":    int(stat.st_mtime_ns),
        }
    except OSError:
        return None


def get_label_cache_path(cache_dir, source_path):
    """Map a source file path to a stable, human-readable cache filename."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(str(Path(source_path).resolve()).encode()).hexdigest()[:12]
    stem = Path(source_path).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return cache_dir / f"{safe_stem}_{digest}_manual_windows.csv"


def load_cached_windows(cache_path, fingerprint):
    """
    Load cached crack windows if the fingerprint matches the source file.

    Returns a list of window dicts, an empty list if the cache exists but is
    empty, or None if the cache is missing or stale.
    """
    if fingerprint is None or not cache_path.exists():
        return None

    try:
        cached_df = pd.read_csv(cache_path)
    except Exception:
        return None

    required_meta = ["source_path", "file_size", "mtime_ns"]
    if any(col not in cached_df.columns for col in required_meta):
        return None

    if cached_df.empty:
        return []

    first = cached_df.iloc[0]
    if not (
        str(first.get("source_path", "")) == str(fingerprint["source_path"])
        and int(first.get("file_size", -1)) == int(fingerprint["file_size"])
        and int(first.get("mtime_ns",  -1)) == int(fingerprint["mtime_ns"])
    ):
        return None

    required_data = ["manual_label", "window_start_x", "window_end_x"]
    if any(col not in cached_df.columns for col in required_data):
        return None

    if "window_order" in cached_df.columns:
        cached_df = cached_df.sort_values("window_order")

    windows = []
    for _, row in cached_df.iterrows():
        start_x = row.get("window_start_x")
        end_x   = row.get("window_end_x")
        label   = str(row.get("manual_label", "")).strip()
        if pd.isna(start_x) or pd.isna(end_x):
            continue
        if label not in CRACK_LABELS:
            continue
        windows.append({"start_x": float(start_x), "end_x": float(end_x),
                         "label": label, "patch": None, "text": None})
    return windows


def save_cached_windows(cache_path, fingerprint, windows):
    """Persist manual crack windows to CSV for reuse in future runs."""
    if fingerprint is None:
        return

    rows = [
        {
            "source_path":   fingerprint["source_path"],
            "file_size":     fingerprint["file_size"],
            "mtime_ns":      fingerprint["mtime_ns"],
            "window_order":  i,
            "window_start_x": float(w["start_x"]),
            "window_end_x":   float(w["end_x"]),
            "manual_label":   str(w["label"]),
        }
        for i, w in enumerate(windows, start=1)
    ]

    pd.DataFrame(rows).to_csv(cache_path, index=False)
