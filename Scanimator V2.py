import pandas as pd
import os
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from scipy.signal import find_peaks

# Runtime toggles for faster execution.
FAST_MODE = True
FRAME_STEP = 5 if FAST_MODE else 1 #1 for highest quality, 5 for faster runs
SHOW_PLOTS = False if FAST_MODE else True
SAVE_GIFS = True


def create_run_output_dir(base_dir="outputs"):
    """
    Create and return a unique timestamped directory for one script run.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    """
    Build a lookup from CSV basename to absolute source path.
    """
    lookup = {}
    for path in file_paths:
        resolved = str(Path(path).resolve())
        lookup[Path(path).name] = resolved
    return lookup


def get_file_fingerprint(file_path):
    """
    Return a lightweight fingerprint for cache validation.
    """
    try:
        stat = Path(file_path).stat()
        return {
            "source_path": str(Path(file_path).resolve()),
            "file_size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return None


def get_label_cache_path(cache_dir, source_path):
    """
    Map source file path to a stable cache filename.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(str(Path(source_path).resolve()).encode("utf-8")).hexdigest()[:12]
    stem = Path(source_path).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return cache_dir / f"{safe_stem}_{digest}_manual_windows.csv"


def load_cached_windows(cache_path, fingerprint):
    """
    Load cached windows if fingerprint matches the current source file.
    """
    if fingerprint is None or not cache_path.exists():
        return None

    try:
        cached_df = pd.read_csv(cache_path)
    except Exception:
        return None

    required_meta_cols = ["source_path", "file_size", "mtime_ns"]
    if any(col not in cached_df.columns for col in required_meta_cols):
        return None
    if cached_df.empty:
        return []

    first = cached_df.iloc[0]
    path_match = str(first.get("source_path", "")) == str(fingerprint["source_path"])
    size_match = int(first.get("file_size", -1)) == int(fingerprint["file_size"])
    mtime_match = int(first.get("mtime_ns", -1)) == int(fingerprint["mtime_ns"])
    if not (path_match and size_match and mtime_match):
        return None

    label_col = "manual_label"
    start_col = "window_start_x"
    end_col = "window_end_x"
    order_col = "window_order"
    if any(col not in cached_df.columns for col in [label_col, start_col, end_col]):
        return None

    sort_cols = [c for c in [order_col] if c in cached_df.columns]
    if sort_cols:
        cached_df = cached_df.sort_values(sort_cols)

    windows = []
    for _, row in cached_df.iterrows():
        start_x = row.get(start_col)
        end_x = row.get(end_col)
        label = str(row.get(label_col, "")).strip()
        if pd.isna(start_x) or pd.isna(end_x):
            continue
        if label not in ("Crack 1", "Crack 2", "Crack 3"):
            continue
        windows.append(
            {
                "start_x": float(start_x),
                "end_x": float(end_x),
                "label": label,
                "patch": None,
                "text": None,
            }
        )
    return windows


def save_cached_windows(cache_path, fingerprint, windows):
    """
    Persist manual crack windows for reuse in future runs.
    """
    if fingerprint is None:
        return

    rows = []
    for i, window in enumerate(windows, start=1):
        rows.append(
            {
                "source_path": fingerprint["source_path"],
                "file_size": fingerprint["file_size"],
                "mtime_ns": fingerprint["mtime_ns"],
                "window_order": i,
                "window_start_x": float(window["start_x"]),
                "window_end_x": float(window["end_x"]),
                "manual_label": str(window["label"]),
            }
        )

    cache_df = pd.DataFrame(rows)
    cache_df.to_csv(cache_path, index=False)

def load_csv_files(file_paths):
    """
    Load multiple CSV files and create dataframes for each one.
    
    Args:
        file_paths: List of file paths to CSV files
        
    Returns:
        Dictionary with filenames as keys and dataframes as values
    """
    dataframes = {}
    
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue
                
            if not file_path.lower().endswith('.csv'):
                print(f"Warning: File is not a CSV - {file_path}")
                continue
            
            df = pd.read_csv(file_path)
            filename = Path(file_path).name
            dataframes[filename] = df
            print(f"Successfully loaded: {filename} ({len(df)} rows, {len(df.columns)} columns)")
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return dataframes


def resolve_first_existing_column(df, candidate_names):
    """
    Return the first matching column from a candidate list.
    """
    stripped_lookup = {col.strip(): col for col in df.columns}
    for candidate in candidate_names:
        if candidate in df.columns:
            return candidate
        if candidate.strip() in stripped_lookup:
            return stripped_lookup[candidate.strip()]
    return None


def detect_raw_peak_candidates(y_vals, max_peaks=10, min_peak_distance=60):
    """
    Detect candidate peaks directly on raw incoming values.

    Uses a robust prominence threshold, then keeps top peaks by prominence.
    """
    if len(y_vals) < 10:
        return np.array([], dtype=int), np.array([], dtype=float)

    y_vals = np.asarray(y_vals, dtype=float)
    y_median = float(np.median(y_vals))
    y_mad = float(np.median(np.abs(y_vals - y_median)))
    robust_sigma = 1.4826 * y_mad
    if robust_sigma <= 0:
        robust_sigma = float(np.std(y_vals))

    dynamic_prominence = max(0.05, robust_sigma * 1.0)
    peak_indices, peak_props = find_peaks(
        y_vals,
        prominence=dynamic_prominence,
        distance=max(1, int(min_peak_distance)),
    )
    peak_prominences = peak_props.get("prominences", np.array([], dtype=float))

    if len(peak_indices) == 0:
        return peak_indices, peak_prominences

    if len(peak_indices) > max_peaks:
        keep = np.argsort(peak_prominences)[::-1][:max_peaks]
        peak_indices = peak_indices[keep]
        peak_prominences = peak_prominences[keep]

    sort_idx = np.argsort(peak_indices)
    return peak_indices[sort_idx], peak_prominences[sort_idx]


def ask_manual_peak_label(prompt_text):
    """
    Prompt user for a manual crack label.
    """
    label_map = {
        "1": "Crack 1",
        "2": "Crack 2",
        "3": "Crack 3",
        "0": "Not a crack",
        "n": "Not a crack",
        "not": "Not a crack",
        "not a crack": "Not a crack",
        "crack 1": "Crack 1",
        "crack 2": "Crack 2",
        "crack 3": "Crack 3",
    }

    while True:
        response = input(prompt_text).strip().lower()
        if response == "":
            return "Not a crack"
        if response in label_map:
            return label_map[response]
        print("Invalid label. Use 1, 2, 3, or 0 (Not a crack).")

def run_manual_raw_peak_labeling(
    dataframes,
    output_dir,
    source_path_lookup=None,
    cache_dir="window_label_cache",
    max_peaks_per_file=10,
    min_peak_distance=60,
    show_plot=True,
):
    """
    Manual labeling step on raw incoming data before processing/analysis.

    Labels available:
      - Crack 1
      - Crack 2
      - Crack 3
      - Not a crack
    """
    if not dataframes:
        return pd.DataFrame()

    records = []
    x_candidates = ["timestamp", "time", "sample", "index"]
    y_candidates = ["sensor2_smooth_rot", " sensor2_smooth_rot", "sensor2", " sensor2", "sensor1_smooth_rot", " sensor1_smooth_rot", "sensor1", " sensor1"]

    print("\nManual raw window labeling step")
    print("Drag a window, then press 1/2/3 for crack type or 0 for Not a crack. Press Enter when done.")

    for filename, df in dataframes.items():
        if df.empty:
            continue

        source_path = None
        if source_path_lookup is not None:
            source_path = source_path_lookup.get(filename)

        fingerprint = get_file_fingerprint(source_path) if source_path else None
        cache_path = get_label_cache_path(cache_dir, source_path) if source_path else None
        cached_windows = load_cached_windows(cache_path, fingerprint) if cache_path is not None else None

        y_col = resolve_first_existing_column(df, y_candidates)
        if y_col is None:
            print(f"Skipping manual labels for {filename}: raw sensor column not found.")
            continue

        x_col = resolve_first_existing_column(df, x_candidates)

        y_series = pd.to_numeric(df[y_col], errors="coerce")
        if x_col is None:
            x_series = pd.Series(np.arange(len(df), dtype=float), index=df.index)
            x_col_name = "sample_index"
        else:
            x_series = pd.to_numeric(df[x_col], errors="coerce")
            x_col_name = x_col

        valid = x_series.notna() & y_series.notna()
        x_vals = x_series[valid].to_numpy(dtype=float)
        y_vals = y_series[valid].to_numpy(dtype=float)
        raw_row_indices = np.flatnonzero(valid.to_numpy())

        if len(y_vals) < 10:
            print(f"Skipping manual labels for {filename}: insufficient numeric raw samples.")
            continue

        windows = []
        if cached_windows is not None:
            windows = cached_windows
            print(
                f"Loaded cached manual windows for {filename}: {len(windows)} window(s). "
                "Press Enter to accept, or edit them before continuing."
            )

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x_vals, y_vals, color="#1f77b4", linewidth=1.2, label=f"Raw signal ({y_col.strip()})")

        label_color_map = {
            "Crack 1": "#d62728",
            "Crack 2": "#2ca02c",
            "Crack 3": "#ff7f0e",
            "Not a crack": "#7f7f7f",
        }
        label_short_map = {
            "Crack 1": "C1",
            "Crack 2": "C2",
            "Crack 3": "C3",
            "Not a crack": "N",
        }

        active_selection = {"start_x": None, "end_x": None, "patch": None}

        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        y_range = max(y_max - y_min, 1e-9)
        text_y = y_max + (0.04 * y_range)

        status_text = ax.text(
            0.01,
            0.98,
            "Drag a span to create a window. 1/2/3/0 label the selected window. Enter=done",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        ax.set_title(f"Manual Window Labeling: {filename}")
        ax.set_xlabel(x_col_name)
        ax.set_ylabel(y_col.strip())
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        plt.tight_layout()

        # Keep user zoom/pan stable while interactive annotations are updated.
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)

        base_plot_path = Path(output_dir) / f"raw_window_selection_{Path(filename).stem}.png"
        fig.savefig(base_plot_path, dpi=180)
        print(f"Saved base window-selection plot: {base_plot_path}")

        def refresh_window_display():
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            crack_windows = [w for w in windows if w["label"] in ("Crack 1", "Crack 2", "Crack 3")]
            active_text = (
                "none"
                if active_selection["start_x"] is None
                else f"{active_selection['start_x']:.3f} to {active_selection['end_x']:.3f}"
            )
            status_text.set_text(
                f"Active selection: {active_text} | Crack windows: {len(crack_windows)} | "
                "0 clears active span and overlapping annotations | Enter=done"
            )

            crack_idx = 0
            for window in windows:
                if window["label"] in ("Crack 1", "Crack 2", "Crack 3"):
                    crack_idx += 1
                    if window["patch"] is None:
                        window["patch"] = ax.axvspan(
                            float(window["start_x"]),
                            float(window["end_x"]),
                            facecolor=label_color_map[window["label"]],
                            edgecolor="#303030",
                            alpha=0.24,
                            linewidth=1.2,
                            zorder=2,
                        )
                    else:
                        window["patch"].set_visible(True)
                        window["patch"].set_facecolor(label_color_map[window["label"]])
                        window["patch"].set_alpha(0.24)
                        window["patch"].set_linewidth(1.2)

                    center_x = 0.5 * (window["start_x"] + window["end_x"])
                    if window["text"] is None:
                        window["text"] = ax.text(
                            center_x,
                            text_y,
                            "",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="#111111",
                            zorder=4,
                        )
                    window["text"].set_visible(True)
                    window["text"].set_position((center_x, text_y))
                    window["text"].set_text(f"W{crack_idx}:{label_short_map[window['label']]}")
                else:
                    if window["patch"] is not None:
                        window["patch"].set_visible(False)
                    if window["text"] is not None:
                        window["text"].set_visible(False)

            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)
            fig.canvas.draw_idle()

        if show_plot:
            print(
                f"\nInteractive labeling for {filename}: drag a window, then press "
                "1=Crack1, 2=Crack2, 3=Crack3, 0=clear span + overlapping annotations, Enter=finish, Backspace/Delete=remove last committed window."
            )

            x_span = max(np.ptp(x_vals), 1e-12)

            def on_span_select(xmin, xmax):
                start_x = float(min(xmin, xmax))
                end_x = float(max(xmin, xmax))
                if abs(end_x - start_x) <= (0.002 * x_span):
                    return

                if active_selection["patch"] is not None:
                    active_selection["patch"].remove()

                active_selection["patch"] = ax.axvspan(
                    start_x,
                    end_x,
                    facecolor="#b0b0b0",
                    edgecolor="#505050",
                    alpha=0.3,
                    linewidth=1.2,
                    zorder=3,
                )
                active_selection["start_x"] = start_x
                active_selection["end_x"] = end_x
                refresh_window_display()

            span_selector = SpanSelector(
                ax,
                on_span_select,
                "horizontal",
                useblit=True,
                props={"facecolor": "#bbbbbb", "alpha": 0.16},
                interactive=False,
                drag_from_anywhere=True,
            )

            def on_key(event):
                if event.key is None:
                    return
                key = event.key.lower()
                if key in ("1", "2", "3", "0"):
                    if active_selection["start_x"] is None or active_selection["end_x"] is None:
                        print("Drag a window first.")
                        return

                    if key == "1":
                        label = "Crack 1"
                    elif key == "2":
                        label = "Crack 2"
                    elif key == "3":
                        label = "Crack 3"
                    else:
                        label = "Not a crack"

                    start_x = float(active_selection["start_x"])
                    end_x = float(active_selection["end_x"])

                    if key == "0":
                        kept_windows = []
                        for window in windows:
                            w_start = float(window["start_x"])
                            w_end = float(window["end_x"])
                            overlaps = max(w_start, start_x) <= min(w_end, end_x)
                            if overlaps:
                                if window["patch"] is not None:
                                    window["patch"].remove()
                                if window["text"] is not None:
                                    window["text"].remove()
                            else:
                                kept_windows.append(window)
                        windows[:] = kept_windows
                    else:
                        patch = ax.axvspan(
                            start_x,
                            end_x,
                            facecolor=label_color_map[label],
                            edgecolor="#303030",
                            alpha=0.24,
                            linewidth=1.2,
                            zorder=2,
                        )
                        text = ax.text(
                            0.5 * (start_x + end_x),
                            text_y,
                            "",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="#111111",
                            zorder=4,
                        )
                        windows.append(
                            {
                                "start_x": start_x,
                                "end_x": end_x,
                                "label": label,
                                "patch": patch,
                                "text": text,
                            }
                        )

                    if active_selection["patch"] is not None:
                        active_selection["patch"].remove()
                    active_selection["patch"] = None
                    active_selection["start_x"] = None
                    active_selection["end_x"] = None
                    refresh_window_display()
                elif key in ("backspace", "delete"):
                    if not windows:
                        return
                    window = windows.pop()
                    if window["patch"] is not None:
                        window["patch"].remove()
                    if window["text"] is not None:
                        window["text"].remove()
                    refresh_window_display()
                elif key in ("enter", "return"):
                    plt.close(fig)

            key_cid = fig.canvas.mpl_connect("key_press_event", on_key)
            refresh_window_display()
            plt.show()
            fig.canvas.mpl_disconnect(key_cid)
            span_selector.set_active(False)

            if active_selection["patch"] is not None:
                active_selection["patch"].remove()
                active_selection["patch"] = None
                active_selection["start_x"] = None
                active_selection["end_x"] = None
        elif cached_windows is None:
            print(f"\nFile: {filename}")
            print("Non-plot mode: enter windows as start,end,label where label is 1/2/3/0. Label 0 removes overlapping windows. Blank line to finish.")
            while True:
                entry = input("Window (start,end,label): ").strip()
                if not entry:
                    break
                parts = [p.strip() for p in entry.split(",")]
                if len(parts) != 3:
                    print("Invalid format. Use start,end,label")
                    continue
                try:
                    start_x = float(parts[0])
                    end_x = float(parts[1])
                except ValueError:
                    print("Invalid numeric start/end values.")
                    continue

                label_raw = parts[2].lower()
                if label_raw == "1":
                    label = "Crack 1"
                elif label_raw == "2":
                    label = "Crack 2"
                elif label_raw == "3":
                    label = "Crack 3"
                elif label_raw == "0":
                    label = "Not a crack"
                else:
                    print("Invalid label. Use 1, 2, 3, or 0.")
                    continue

                start_x, end_x = sorted((start_x, end_x))
                if label_raw == "0":
                    windows = [
                        w for w in windows
                        if max(float(w["start_x"]), start_x) > min(float(w["end_x"]), end_x)
                    ]
                else:
                    windows.append({
                        "start_x": start_x,
                        "end_x": end_x,
                        "label": label,
                        "patch": None,
                        "text": None,
                    })

        if cache_path is not None and fingerprint is not None:
            save_cached_windows(cache_path, fingerprint, windows)

        labeled_plot_path = Path(output_dir) / f"raw_window_labeled_{Path(filename).stem}.png"
        fig_labeled, ax_labeled = plt.subplots(figsize=(12, 4))
        ax_labeled.plot(x_vals, y_vals, color="#1f77b4", linewidth=1.2, label=f"Raw signal ({y_col.strip()})")

        for i, window in enumerate(windows, start=1):
            ax_labeled.axvspan(
                window["start_x"],
                window["end_x"],
                facecolor=label_color_map[window["label"]],
                edgecolor="#303030",
                alpha=0.22,
                linewidth=1.0,
                zorder=2,
            )
            center_x = 0.5 * (window["start_x"] + window["end_x"])
            ax_labeled.text(
                center_x,
                text_y,
                f"W{i}:{label_short_map[window['label']]}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111111",
                zorder=4,
            )

        ax_labeled.set_title(f"Labeled Raw Windows: {filename}")
        ax_labeled.set_xlabel(x_col_name)
        ax_labeled.set_ylabel(y_col.strip())
        ax_labeled.grid(alpha=0.25)
        y_pad = 0.12 * y_range
        ax_labeled.set_ylim(y_min - (0.03 * y_range), y_max + y_pad)
        ax_labeled.legend(loc="best")
        plt.tight_layout()
        fig_labeled.savefig(labeled_plot_path, dpi=180)
        plt.close(fig_labeled)
        print(f"Saved labeled window plot: {labeled_plot_path}")

        for i, window in enumerate(windows, start=1):
            start_x = float(window["start_x"])
            end_x = float(window["end_x"])
            in_window = (x_vals >= start_x) & (x_vals <= end_x)
            point_count = int(np.count_nonzero(in_window))
            y_mean = float(np.mean(y_vals[in_window])) if point_count > 0 else np.nan

            if point_count > 0:
                clean_indices = np.flatnonzero(in_window)
                start_raw_idx = int(raw_row_indices[int(clean_indices[0])])
                end_raw_idx = int(raw_row_indices[int(clean_indices[-1])])

                local_peak_rel = int(np.argmax(y_vals[clean_indices]))
                peak_clean_idx = int(clean_indices[local_peak_rel])
                peak_raw_idx = int(raw_row_indices[peak_clean_idx])
                peak_raw_x = float(x_vals[peak_clean_idx])
                peak_raw_y = float(y_vals[peak_clean_idx])
            else:
                start_raw_idx = np.nan
                end_raw_idx = np.nan
                peak_raw_idx = np.nan
                peak_raw_x = np.nan
                peak_raw_y = np.nan

            records.append(
                {
                    "file": filename,
                    "raw_y_col": y_col,
                    "raw_x_col": x_col_name,
                    "window_order": i,
                    "window_start_x": start_x,
                    "window_end_x": end_x,
                    "window_start_raw_idx": start_raw_idx,
                    "window_end_raw_idx": end_raw_idx,
                    "window_width": float(end_x - start_x),
                    "window_point_count": point_count,
                    "window_mean_y": y_mean,
                    "window_peak_raw_idx": peak_raw_idx,
                    "window_peak_raw_x": peak_raw_x,
                    "window_peak_raw_y": peak_raw_y,
                    "manual_label": window["label"],
                }
            )

    labels_df = pd.DataFrame(records)
    if labels_df.empty:
        print("No manual peak labels were recorded.")
        return labels_df

    labels_path = Path(output_dir) / "manual_window_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved manual window labels: {labels_path}")

    summary_df = (
        labels_df.groupby(["file", "manual_label"]) ["window_order"]
        .count()
        .rename("count")
        .reset_index()
        .sort_values(["file", "manual_label"])
    )
    summary_path = Path(output_dir) / "manual_window_label_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved manual window label summary: {summary_path}")

    return labels_df


def resolve_column_name(df, target_name):
    """
    Resolve a column name with tolerance for leading/trailing spaces.

    Args:
        df: Source dataframe
        target_name: Desired column name

    Returns:
        The matching real column name in df

    Raises:
        KeyError if no suitable column is found
    """
    if target_name in df.columns:
        return target_name

    stripped_lookup = {col.strip(): col for col in df.columns}
    if target_name.strip() in stripped_lookup:
        return stripped_lookup[target_name.strip()]

    raise KeyError(f"Missing required column '{target_name}'")


def prepare_xy_series(dataframes, x_col, y_col, normalize=False):
    """
    Build cleaned XY arrays from each dataframe.

    Args:
        dataframes: Dict of filename -> dataframe
        x_col: Requested x-axis source column
        y_col: Requested y-axis source column
        normalize: If True, min-max normalize x and y per file

    Returns:
        List of tuples (filename, x_vals, y_vals)
    """
    prepared = []

    for filename, df in dataframes.items():
        try:
            real_x_col = resolve_column_name(df, x_col)
            real_y_col = resolve_column_name(df, y_col)

            x = pd.to_numeric(df[real_x_col], errors="coerce")
            y = pd.to_numeric(df[real_y_col], errors="coerce")
            valid = x.notna() & y.notna()
            x_vals = x[valid].to_numpy()
            y_vals = y[valid].to_numpy()

            if len(x_vals) == 0:
                print(f"Skipping {filename}: no numeric XY pairs after cleaning.")
                continue

            if normalize:
                x_min, x_max = np.min(x_vals), np.max(x_vals)
                y_min, y_max = np.min(y_vals), np.max(y_vals)

                if x_max > x_min:
                    x_vals = (x_vals - x_min) / (x_max - x_min)
                else:
                    x_vals = np.zeros_like(x_vals)

                if y_max > y_min:
                    y_vals = (y_vals - y_min) / (y_max - y_min)
                else:
                    y_vals = np.zeros_like(y_vals)

            prepared.append((filename, x_vals, y_vals))

        except KeyError as err:
            print(f"Skipping {filename}: {err}")

    return prepared


def compute_xy_crack_snr_metrics(
    x_vals,
    y_vals,
    baseline_window=101,
    max_peaks=9,
    min_peak_distance=100,
    sigma_threshold_multiplier=3.0,
    manual_windows=None,
    row_indices=None,
    allow_automatic_fallback=False,
):
    """
    Compute crack-focused SNR.

    In strict ground-truth mode (default), peaks are taken only from manual windows
    and each window contributes its strongest baseline-removed point.
    """
    if len(x_vals) < 20 or len(y_vals) < 20:
        return None

    baseline_window = max(11, baseline_window)
    if baseline_window % 2 == 0:
        baseline_window += 1
    max_window = len(x_vals) if len(x_vals) % 2 == 1 else len(x_vals) - 1
    baseline_window = min(baseline_window, max_window)
    if baseline_window < 3:
        return None

    y_series = pd.Series(y_vals)
    y_base = y_series.rolling(window=baseline_window, center=True, min_periods=1).median().to_numpy()
    y_dev = y_vals - y_base

    # Estimate noise from non-crack regions when ground-truth windows are available.
    noise_source = y_dev
    if manual_windows is not None and len(manual_windows) > 0 and row_indices is not None and len(row_indices) == len(y_dev):
        crack_mask = np.zeros(len(y_dev), dtype=bool)
        row_indices = np.asarray(row_indices, dtype=int)
        for window in manual_windows:
            start_row = window.get("start_row")
            end_row = window.get("end_row")
            if start_row is None or end_row is None:
                continue
            if not np.isfinite(start_row) or not np.isfinite(end_row):
                continue
            lo_row, hi_row = sorted((int(start_row), int(end_row)))
            crack_mask |= (row_indices >= lo_row) & (row_indices <= hi_row)

        non_crack = y_dev[~crack_mask]
        if len(non_crack) >= 10:
            noise_source = non_crack

    noise_floor = float(np.median(noise_source))
    noise_mad = float(np.median(np.abs(noise_source - noise_floor)))
    noise_sigma = 1.4826 * noise_mad
    if noise_sigma <= 0:
        noise_sigma = float(np.std(noise_source))
    if noise_sigma <= 0:
        return None

    peak_labels = []
    per_window_records = []
    peak_height_threshold = noise_floor + (sigma_threshold_multiplier * noise_sigma)

    if manual_windows is not None and len(manual_windows) > 0:
        selected_peak_indices = []
        selected_peak_heights = []

        for window in manual_windows:
            label = str(window["label"])

            in_window_idx = np.array([], dtype=int)
            start_row = window.get("start_row")
            end_row = window.get("end_row")
            if (
                start_row is not None
                and end_row is not None
                and row_indices is not None
                and len(row_indices) == len(y_dev)
                and np.isfinite(start_row)
                and np.isfinite(end_row)
            ):
                lo_row, hi_row = sorted((int(start_row), int(end_row)))
                in_window_idx = np.flatnonzero((row_indices >= lo_row) & (row_indices <= hi_row))
            elif "start_x" in window and "end_x" in window:
                start_x = float(window["start_x"])
                end_x = float(window["end_x"])
                if np.isfinite(start_x) and np.isfinite(end_x):
                    lo, hi = sorted((start_x, end_x))
                    in_window_idx = np.flatnonzero((x_vals >= lo) & (x_vals <= hi))

            if len(in_window_idx) == 0:
                continue

            # Use the strongest baseline-removed point inside each labeled crack window.
            local_rel = int(np.argmax(y_dev[in_window_idx]))
            peak_idx = int(in_window_idx[local_rel])
            peak_signal = float(y_dev[peak_idx] - noise_floor)
            peak_snr_linear = (peak_signal / noise_sigma) if noise_sigma > 0 and peak_signal > 0 else 0.0
            peak_snr_db = float(20 * np.log10(peak_snr_linear)) if peak_snr_linear > 0 else float("-inf")

            start_x_record = float(x_vals[in_window_idx[0]])
            end_x_record = float(x_vals[in_window_idx[-1]])
            peak_x = float(x_vals[peak_idx])
            peak_y = float(y_vals[peak_idx])

            start_row_record = np.nan
            end_row_record = np.nan
            peak_row_record = np.nan
            if row_indices is not None and len(row_indices) == len(y_dev):
                start_row_record = int(row_indices[int(in_window_idx[0])])
                end_row_record = int(row_indices[int(in_window_idx[-1])])
                peak_row_record = int(row_indices[peak_idx])

            selected_peak_indices.append(peak_idx)
            selected_peak_heights.append(float(y_dev[peak_idx]))
            peak_labels.append(label)
            per_window_records.append(
                {
                    "manual_label": label,
                    "window_start_x": start_x_record,
                    "window_end_x": end_x_record,
                    "window_start_raw_idx": start_row_record,
                    "window_end_raw_idx": end_row_record,
                    "peak_index_in_clean_series": int(peak_idx),
                    "peak_raw_idx": peak_row_record,
                    "peak_x": peak_x,
                    "peak_y": peak_y,
                    "noise_floor": float(noise_floor),
                    "noise_sigma": float(noise_sigma),
                    "peak_signal_amplitude": peak_signal,
                    "peak_snr_linear": float(peak_snr_linear),
                    "peak_snr_db": peak_snr_db,
                }
            )

        peak_indices = np.array(selected_peak_indices, dtype=int)
        peak_heights = np.array(selected_peak_heights, dtype=float)
    elif allow_automatic_fallback:
        peak_indices, peak_props = find_peaks(
            y_dev,
            height=peak_height_threshold,
            distance=max(1, min_peak_distance),
        )

        peak_heights = peak_props.get("peak_heights", np.array([]))

        if len(peak_indices) > max_peaks:
            keep = np.argsort(peak_heights)[::-1][:max_peaks]
            peak_indices = peak_indices[keep]
            peak_heights = peak_heights[keep]

        if len(peak_indices) > 0:
            sort_idx = np.argsort(peak_indices)
            peak_indices = peak_indices[sort_idx]
            peak_heights = peak_heights[sort_idx]
    else:
        return None

    if len(peak_heights) == 0:
        return None

    peak_signal = peak_heights - noise_floor
    signal_amplitude = float(np.median(peak_signal))

    snr_linear = signal_amplitude / noise_sigma if noise_sigma > 0 else 0.0
    snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

    peak_snr_db_values = []
    for height in peak_heights:
        peak_signal = float(height - noise_floor)
        if noise_sigma > 0 and peak_signal > 0:
            peak_snr_db_values.append(float(20 * np.log10(peak_signal / noise_sigma)))
        else:
            peak_snr_db_values.append(float("-inf"))

    return {
        "noise_sigma": float(noise_sigma),
        "noise_floor": float(noise_floor),
        "signal_amplitude": float(signal_amplitude),
        "snr_linear": float(snr_linear),
        "snr_db": float(snr_db),
        "peak_count": int(len(peak_indices)),
        "peak_indices": ";".join(str(int(i)) for i in peak_indices),
        "peak_snr_db_values": ";".join(f"{v:.6f}" for v in peak_snr_db_values),
        "peak_labels": ";".join(peak_labels),
        "peak_height_threshold": float(peak_height_threshold),
        "per_window_records": per_window_records,
    }


def analyze_snr(
    dataframes,
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    manual_labels_df=None,
    require_manual_windows=False,
    ground_truth_only=True,
    baseline_window=101,
    max_peaks=9,
    min_peak_distance=100,
    sigma_threshold_multiplier=3.0,
    verbose=True,
):
    """
    Quantify crack-detection SNR per file.

    If manual_labels_df is provided, crack windows are used to choose peaks.
    In ground_truth_only mode, automatic peak finding is disabled.

        Returns a tuple:
            - file-level dataframe sorted by descending SNR (dB), one row per file
            - per-crack dataframe with one row per labeled crack window
    """
    if not dataframes:
        print("No dataframes available for SNR analysis.")
        return pd.DataFrame(), pd.DataFrame()

    if ground_truth_only and (manual_labels_df is None or manual_labels_df.empty):
        print("Ground-truth mode requires manual crack window labels.")
        return pd.DataFrame(), pd.DataFrame()

    records = []
    per_crack_records = []

    for filename, df in dataframes.items():
        if df.empty:
            print(f"Skipping {filename} for SNR: file is empty.")
            continue

        try:
            real_x_col = resolve_column_name(df, x_col)
            real_y_col = resolve_column_name(df, y_col)
        except KeyError as err:
            print(f"Skipping {filename} for SNR: {err}")
            continue

        x = pd.to_numeric(df[real_x_col], errors="coerce")
        y = pd.to_numeric(df[real_y_col], errors="coerce")
        valid = x.notna() & y.notna()
        x_vals = x[valid].to_numpy()
        y_vals = y[valid].to_numpy()
        proc_row_indices = np.flatnonzero(valid.to_numpy())

        file_manual_windows = []
        if manual_labels_df is not None and not manual_labels_df.empty:
            file_windows_df = manual_labels_df[
                (manual_labels_df["file"] == filename)
                & (manual_labels_df["manual_label"].isin(["Crack 1", "Crack 2", "Crack 3"]))
            ].copy()

            sort_cols = [c for c in ["window_order", "manual_label"] if c in file_windows_df.columns]
            if sort_cols:
                file_windows_df = file_windows_df.sort_values(sort_cols)

            for _, win_row in file_windows_df.iterrows():
                start_x = win_row.get("window_start_x")
                end_x = win_row.get("window_end_x")
                start_row = win_row.get("window_start_raw_idx")
                end_row = win_row.get("window_end_raw_idx")

                if pd.isna(start_row) or pd.isna(end_row):
                    if pd.isna(start_x) or pd.isna(end_x):
                        continue
                    file_manual_windows.append(
                        {
                            "start_x": float(start_x),
                            "end_x": float(end_x),
                            "label": str(win_row.get("manual_label", "")),
                        }
                    )
                    continue

                file_manual_windows.append(
                    {
                        "start_row": float(start_row),
                        "end_row": float(end_row),
                        "label": str(win_row.get("manual_label", "")),
                    }
                )

            if require_manual_windows and len(file_manual_windows) == 0:
                print(f"Skipping {filename} for SNR: no crack-labeled windows found.")
                continue

        metrics = compute_xy_crack_snr_metrics(
            x_vals,
            y_vals,
            baseline_window=baseline_window,
            max_peaks=max_peaks,
            min_peak_distance=min_peak_distance,
            sigma_threshold_multiplier=sigma_threshold_multiplier,
            manual_windows=file_manual_windows if len(file_manual_windows) > 0 else None,
            row_indices=proc_row_indices,
            allow_automatic_fallback=not ground_truth_only,
        )
        if metrics is None:
            continue

        records.append(
            {
                "file": filename,
                "x_col": real_x_col,
                "y_col": real_y_col,
                "n_samples": int(len(x_vals)),
                "noise_floor": metrics["noise_floor"],
                "noise_sigma": metrics["noise_sigma"],
                "signal_amplitude": metrics["signal_amplitude"],
                "snr_linear": metrics["snr_linear"],
                "snr_db": metrics["snr_db"],
                "peak_count": metrics["peak_count"],
                "peak_indices": metrics["peak_indices"],
                "peak_snr_db_values": metrics["peak_snr_db_values"],
                "peak_labels": metrics["peak_labels"],
                "peak_height_threshold": metrics["peak_height_threshold"],
            }
        )

        for window_record in metrics.get("per_window_records", []):
            per_crack_records.append(
                {
                    "file": filename,
                    "x_col": real_x_col,
                    "y_col": real_y_col,
                    "manual_label": window_record.get("manual_label"),
                    "window_start_x": window_record.get("window_start_x"),
                    "window_end_x": window_record.get("window_end_x"),
                    "window_start_raw_idx": window_record.get("window_start_raw_idx"),
                    "window_end_raw_idx": window_record.get("window_end_raw_idx"),
                    "peak_index_in_clean_series": window_record.get("peak_index_in_clean_series"),
                    "peak_raw_idx": window_record.get("peak_raw_idx"),
                    "peak_x": window_record.get("peak_x"),
                    "peak_y": window_record.get("peak_y"),
                    "noise_floor": window_record.get("noise_floor"),
                    "noise_sigma": window_record.get("noise_sigma"),
                    "peak_signal_amplitude": window_record.get("peak_signal_amplitude"),
                    "peak_snr_linear": window_record.get("peak_snr_linear"),
                    "peak_snr_db": window_record.get("peak_snr_db"),
                }
            )

    if not records:
        print("No valid sensor signals found for SNR analysis.")
        return pd.DataFrame(), pd.DataFrame()

    snr_df = pd.DataFrame(records).sort_values(by="snr_db", ascending=False).reset_index(drop=True)
    per_crack_df = pd.DataFrame(per_crack_records)
    if not per_crack_df.empty:
        sort_cols = [c for c in ["file", "manual_label", "window_start_raw_idx"] if c in per_crack_df.columns]
        if sort_cols:
            per_crack_df = per_crack_df.sort_values(by=sort_cols).reset_index(drop=True)

    if verbose:
        print("\nCrack-detection SNR by file (highest first):")
        print(snr_df[["file", "snr_db", "snr_linear", "signal_amplitude", "noise_sigma", "peak_count"]].head(50))
        if not per_crack_df.empty:
            print("\nPer-crack window SNR details:")
            print(per_crack_df[["file", "manual_label", "peak_snr_db", "peak_signal_amplitude", "noise_sigma"]].head(50))

    return snr_df, per_crack_df


def tune_snr_parameters_for_peak_targets(
    dataframes,
    x_col,
    y_col,
    target_min_peaks=4,
    target_ideal_peaks=6,
    max_peaks=6,
):
    """
    Find SNR detector settings that push each file toward desired peak counts.

    Primary objective: all files have at least target_min_peaks.
    Secondary objective: average peaks close to target_ideal_peaks.
    """
    candidate_settings = [
        {"baseline_window": 121, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.6},
        {"baseline_window": 101, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.4},
        {"baseline_window": 81, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.2},
        {"baseline_window": 61, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.0},
        {"baseline_window": 51, "min_peak_distance": 100, "sigma_threshold_multiplier": 2.8},
    ]

    best = None
    best_score = None

    for params in candidate_settings:
        snr_df, _ = analyze_snr(
            dataframes,
            x_col=x_col,
            y_col=y_col,
            ground_truth_only=False,
            baseline_window=params["baseline_window"],
            max_peaks=max_peaks,
            min_peak_distance=params["min_peak_distance"],
            sigma_threshold_multiplier=params["sigma_threshold_multiplier"],
            verbose=False,
        )

        if snr_df.empty:
            continue

        peak_counts = snr_df["peak_count"].to_numpy()
        min_count = int(np.min(peak_counts))
        mean_count = float(np.mean(peak_counts))
        below_target_penalty = max(0, target_min_peaks - min_count)
        ideal_distance = abs(mean_count - target_ideal_peaks)
        overflow_penalty = max(0.0, mean_count - max_peaks)
        score = (below_target_penalty * 100.0) + ideal_distance + (overflow_penalty * 5.0)

        if best is None or score < best_score:
            best = {"params": params, "snr_df": snr_df, "min_count": min_count, "mean_count": mean_count}
            best_score = score

        if min_count >= target_min_peaks and mean_count >= (target_ideal_peaks - 1):
            break

    if best is None:
        return None, pd.DataFrame()

    return best, best["snr_df"]


def get_snr_value_db(snr_df, filename):
    """
    Fetch file-level SNR dB for crack detection.
    """
    if snr_df is None or snr_df.empty:
        return None

    matches = snr_df[snr_df["file"] == filename]
    if matches.empty:
        return None
    return float(matches.iloc[0]["snr_db"])


def get_peak_count(snr_df, filename):
    """
    Fetch detected crack peak count for a file.
    """
    if snr_df is None or snr_df.empty:
        return None

    matches = snr_df[snr_df["file"] == filename]
    if matches.empty:
        return None
    return int(matches.iloc[0]["peak_count"])


def get_peak_annotations(snr_df, filename):
    """
    Fetch per-peak annotations as (index, peak_snr_db, peak_label) for a file.
    """
    if snr_df is None or snr_df.empty:
        return []

    matches = snr_df[snr_df["file"] == filename]
    if matches.empty:
        return []

    row = matches.iloc[0]
    idx_str = row.get("peak_indices", "")
    snr_str = row.get("peak_snr_db_values", "")
    label_str = row.get("peak_labels", "")

    if pd.isna(idx_str) or str(idx_str).strip() == "":
        return []

    indices = [int(x) for x in str(idx_str).split(";") if str(x).strip() != ""]
    snr_vals = [float(x) for x in str(snr_str).split(";") if str(x).strip() != ""]
    labels = [str(x).strip() for x in str(label_str).split(";") if str(x).strip() != ""]

    annotations = []
    for i, idx in enumerate(indices):
        peak_snr_db = snr_vals[i] if i < len(snr_vals) else np.nan
        peak_label = labels[i] if i < len(labels) else "Crack"
        annotations.append((idx, peak_snr_db, peak_label))
    return annotations


def fmt_snr_db(value):
    """
    Format SNR value for on-plot labels.
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} dB"


def parse_peak_snr_series(peak_snr_db_values):
    """
    Parse semicolon-separated per-peak SNR dB values into finite floats.
    """
    if pd.isna(peak_snr_db_values):
        return []

    values = []
    for token in str(peak_snr_db_values).split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def parse_peak_label_series(peak_labels):
    """
    Parse semicolon-separated per-peak crack labels.
    """
    if pd.isna(peak_labels):
        return []
    return [token.strip() for token in str(peak_labels).split(";") if token.strip()]


def create_snr_visualizations(snr_df, output_dir=".", show_plot=False, save_plots=True):
    """
    Create static charts that compare SNR differences across sensor runs.

    Outputs:
      - Overview PNG with ranking, delta-to-best, and signal-vs-noise scatter.
      - Peak-level SNR strip plot PNG (if peak SNR values are available).
    """
    if snr_df is None or snr_df.empty:
        print("No SNR data available for visualization.")
        return []

    plot_outputs = []
    snr_plot_df = snr_df.copy().sort_values(by="snr_db", ascending=False).reset_index(drop=True)

    rank = np.arange(1, len(snr_plot_df) + 1)
    best_snr = float(snr_plot_df.loc[0, "snr_db"])
    snr_plot_df["delta_to_best_db"] = best_snr - snr_plot_df["snr_db"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_rank = axes[0, 0]
    ax_delta = axes[0, 1]
    ax_signal_noise = axes[1, 0]
    ax_peaks = axes[1, 1]

    bar_colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(snr_plot_df)))
    ax_rank.barh(snr_plot_df["file"], snr_plot_df["snr_db"], color=bar_colors)
    ax_rank.invert_yaxis()
    ax_rank.set_title("Run Ranking by Crack SNR")
    ax_rank.set_xlabel("SNR (dB)")
    ax_rank.grid(axis="x", alpha=0.25)
    for y_idx, value in enumerate(snr_plot_df["snr_db"]):
        ax_rank.text(value, y_idx, f" {value:.1f}", va="center", ha="left", fontsize=8)

    ax_delta.plot(rank, snr_plot_df["snr_db"], marker="o", linewidth=2, color="#1f77b4", label="SNR (dB)")
    ax_delta.bar(rank, snr_plot_df["delta_to_best_db"], alpha=0.25, color="#ff7f0e", label="Delta to best (dB)")
    ax_delta.set_xticks(rank)
    ax_delta.set_xticklabels([f"#{i}" for i in rank])
    ax_delta.set_title("SNR Spread Across Runs")
    ax_delta.set_xlabel("Rank (best to worst)")
    ax_delta.set_ylabel("dB")
    ax_delta.grid(alpha=0.25)
    ax_delta.legend(loc="best")

    scatter = ax_signal_noise.scatter(
        snr_plot_df["noise_sigma"],
        snr_plot_df["signal_amplitude"],
        c=snr_plot_df["snr_db"],
        cmap="viridis",
        s=70,
        alpha=0.9,
    )
    ax_signal_noise.set_title("Signal vs Noise by Run")
    ax_signal_noise.set_xlabel("Noise sigma")
    ax_signal_noise.set_ylabel("Signal amplitude")
    ax_signal_noise.grid(alpha=0.25)
    for _, row in snr_plot_df.iterrows():
        ax_signal_noise.annotate(row["file"], (row["noise_sigma"], row["signal_amplitude"]), fontsize=7)
    fig.colorbar(scatter, ax=ax_signal_noise, label="SNR (dB)")

    peak_points = []
    for i, (_, row) in enumerate(snr_plot_df.iterrows(), start=1):
        peak_snr_values = parse_peak_snr_series(row.get("peak_snr_db_values", ""))
        peak_labels = parse_peak_label_series(row.get("peak_labels", ""))
        for j, peak_value in enumerate(peak_snr_values):
            peak_label = peak_labels[j] if j < len(peak_labels) else "Unknown"
            peak_points.append((i, peak_value, row["file"], peak_label))

    if peak_points:
        x_vals = np.array([pt[0] for pt in peak_points], dtype=float)
        y_vals = np.array([pt[1] for pt in peak_points], dtype=float)
        labels = [pt[3] for pt in peak_points]
        jitter = np.linspace(-0.12, 0.12, len(x_vals)) if len(x_vals) > 1 else np.array([0.0])

        crack_color_map = {
            "Crack 1": "#d62728",
            "Crack 2": "#2ca02c",
            "Crack 3": "#ff7f0e",
            "Unknown": "#7f7f7f",
        }

        for crack_label in ["Crack 1", "Crack 2", "Crack 3", "Unknown"]:
            mask = np.array([lbl == crack_label for lbl in labels], dtype=bool)
            if not np.any(mask):
                continue
            ax_peaks.scatter(
                x_vals[mask] + jitter[mask],
                y_vals[mask],
                color=crack_color_map[crack_label],
                alpha=0.75,
                s=28,
                label=crack_label,
            )

        for i, file_name in enumerate(snr_plot_df["file"], start=1):
            run_peak_values = [pt[1] for pt in peak_points if pt[2] == file_name]
            if run_peak_values:
                ax_peaks.hlines(np.median(run_peak_values), i - 0.2, i + 0.2, colors="#d62728", linewidth=2)

        ax_peaks.set_xticks(rank)
        ax_peaks.set_xticklabels([f"#{i}" for i in rank])
        ax_peaks.set_title("Per-Peak SNR Distribution by Run")
        ax_peaks.set_xlabel("Run rank")
        ax_peaks.set_ylabel("Peak SNR (dB)")
        ax_peaks.grid(alpha=0.25)
        ax_peaks.legend(loc="best", title="Crack label")
    else:
        ax_peaks.text(0.5, 0.5, "No finite peak SNR values available", ha="center", va="center")
        ax_peaks.set_axis_off()

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_plots:
        overview_path = Path(output_dir) / f"snr_visual_summary_{timestamp}.png"
        fig.savefig(overview_path, dpi=200)
        plot_outputs.append(str(overview_path))
        print(f"Saved SNR visual summary: {overview_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return plot_outputs


def create_overlay_animation(
    dataframes,
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    snr_df=None,
    interval=30,
    tail_length=100,
    frame_step=1,
    output_dir=".",
    save_gif=True,
    show_plot=True,
):
    """
    Animate overlaid XY traces from multiple dataframes.

    Each dataframe is drawn in a unique color and updated frame-by-frame.
    """
    if not dataframes:
        print("No dataframes available for animation.")
        return

    prepared = prepare_xy_series(dataframes, x_col, y_col, normalize=True)
    if not prepared:
        print("No valid files had the required columns for animation.")
        return

    global_x_min = np.inf
    global_x_max = -np.inf
    global_y_min = np.inf
    global_y_max = -np.inf

    for _, x_vals, y_vals in prepared:
        global_x_min = min(global_x_min, np.min(x_vals))
        global_x_max = max(global_x_max, np.max(x_vals))
        global_y_min = min(global_y_min, np.min(y_vals))
        global_y_max = max(global_y_max, np.max(y_vals))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Overlay Animation (Normalized): sensor1_smooth_rot vs sensor2_smooth_rot")
    ax.set_xlabel("R_p (normalized)")
    ax.set_ylabel("Inductance (normalized)")

    # Compute x limits from peak positions; fall back to full data range if no peaks.
    all_peak_x = []
    for fname, xv, _ in prepared:
        for (peak_idx, _, _) in get_peak_annotations(snr_df, fname):
            if peak_idx < len(xv):
                all_peak_x.append(xv[peak_idx])
    if all_peak_x:
        pk_x_min = min(all_peak_x)
        pk_x_max = max(all_peak_x)
        pk_span = pk_x_max - pk_x_min if pk_x_max > pk_x_min else (global_x_max - global_x_min) * 0.1
        x_pad = pk_span * 0.1875
        display_x_min = pk_x_min - x_pad
        display_x_max = pk_x_max + x_pad
    else:
        x_pad = (global_x_max - global_x_min) * 0.0625 if global_x_max > global_x_min else 1
        display_x_min = global_x_min - x_pad
        display_x_max = global_x_max + x_pad
    y_range = (global_y_max - global_y_min) if global_y_max > global_y_min else 1
    y_pad_bottom = y_range * 0.05
    base_y_pad_top = y_range * 0.05
    base_y_span = y_range + y_pad_bottom + base_y_pad_top
    target_y_span = base_y_span * 1.25
    y_pad_top = max(base_y_pad_top, target_y_span - (y_range + y_pad_bottom))
    ax.set_xlim(display_x_min, display_x_max)
    ax.set_ylim(global_y_min - y_pad_bottom, global_y_max + y_pad_top)

    color_map = plt.get_cmap("tab10", len(prepared))
    artists = []
    max_frames = max(len(item[1]) for item in prepared)

    for i, (filename, _, _) in enumerate(prepared):
        base_color = color_map(i)
        _, _, y_vals_for_noise = prepared[i]

        # Visualize per-trace noise floor as median +/- robust sigma in plotted y-space.
        y_floor = float(np.median(y_vals_for_noise))
        y_mad = float(np.median(np.abs(y_vals_for_noise - y_floor)))
        y_sigma = 1.4826 * y_mad
        if y_sigma <= 0:
            y_sigma = float(np.std(y_vals_for_noise))

        noise_bottom = y_floor - y_sigma
        noise_top = y_floor + y_sigma

        ax.axhline(noise_bottom, color=base_color, linestyle="--", linewidth=1.0, alpha=0.45, zorder=1)
        ax.axhline(noise_top, color=base_color, linestyle="--", linewidth=1.0, alpha=0.45, zorder=1)

        snr_db = get_snr_value_db(snr_df, filename)
        peak_count = get_peak_count(snr_df, filename)
        peak_annotations = get_peak_annotations(snr_df, filename)
        label = (
            f"{filename} | "
            f"Crack SNR: {fmt_snr_db(snr_db)} | "
            f"Peaks: {peak_count if peak_count is not None else 'n/a'}"
        )
        tail_segments = LineCollection([], linewidths=2.0, zorder=2)
        ax.add_collection(tail_segments)
        point, = ax.plot([], [], marker="o", ms=5, color=base_color, zorder=3, label=label)
        peak_markers = []
        peak_texts = []
        for _, _, _ in peak_annotations:
            peak_marker, = ax.plot([], [], marker="^", ms=6, color=base_color, zorder=4, alpha=0.9)
            peak_text = ax.text(
                0,
                0,
                "",
                fontsize=8,
                color=base_color,
                ha="left",
                va="bottom",
                zorder=5,
                visible=False,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
            peak_markers.append(peak_marker)
            peak_texts.append(peak_text)

        artists.append((tail_segments, point, base_color, peak_annotations, peak_markers, peak_texts))

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    def init():
        for tail_segments, point, _, _, peak_markers, peak_texts in artists:
            tail_segments.set_segments([])
            tail_segments.set_color([])
            point.set_data([], [])
            for peak_marker, peak_text in zip(peak_markers, peak_texts):
                peak_marker.set_data([], [])
                peak_text.set_visible(False)
        return [
            artist
            for tail_segments, point, _, _, peak_markers, peak_texts in artists
            for artist in (tail_segments, point, *peak_markers, *peak_texts)
        ]

    def update(frame):
        for (_, x_vals, y_vals), (tail_segments, point, base_color, peak_annotations, peak_markers, peak_texts) in zip(prepared, artists):
            last_index = min(frame, len(x_vals) - 1)

            start_idx = max(0, last_index - tail_length + 1)
            tail_x = x_vals[start_idx : last_index + 1]
            tail_y = y_vals[start_idx : last_index + 1]

            if len(tail_x) > 1:
                points = np.column_stack([tail_x, tail_y])
                segments = np.stack([points[:-1], points[1:]], axis=1)

                # Older segments are more transparent; most recent ~100 points stay dark.
                segment_count = len(segments)
                alphas = np.linspace(0.05, 1.0, segment_count)
                colors = np.tile(np.array(base_color), (segment_count, 1))
                colors[:, 3] = alphas

                tail_segments.set_segments(segments)
                tail_segments.set_color(colors)
            else:
                tail_segments.set_segments([])
                tail_segments.set_color([])

            if frame < len(x_vals):
                point.set_data([x_vals[frame]], [y_vals[frame]])
            else:
                point.set_data([x_vals[-1]], [y_vals[-1]])

            for (peak_idx, peak_snr_db, peak_label), peak_marker, peak_text in zip(peak_annotations, peak_markers, peak_texts):
                if peak_idx < len(x_vals) and frame >= peak_idx:
                    peak_x = x_vals[peak_idx]
                    peak_y = y_vals[peak_idx]
                    peak_marker.set_data([peak_x], [peak_y])
                    peak_text.set_position((peak_x, peak_y))
                    peak_text.set_text(f"{peak_label} | {fmt_snr_db(peak_snr_db)}")
                    peak_text.set_visible(True)
                else:
                    peak_marker.set_data([], [])
                    peak_text.set_visible(False)

        return [
            artist
            for tail_segments, point, _, _, peak_markers, peak_texts in artists
            for artist in (tail_segments, point, *peak_markers, *peak_texts)
        ]

    frame_indices = range(0, max_frames, max(1, frame_step))

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_gif:
        gif_path = Path(output_dir) / f"overlay_animation_{timestamp}.gif"
        fps = max(1, int(round(1000 / interval / max(1, frame_step))))
        anim.save(str(gif_path), writer="pillow", fps=fps)
        print(f"Saved animation GIF: {gif_path}")

    # Keep a reference alive for the life of the figure.
    fig._overlay_anim = anim

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def create_stacked_animation(
    dataframes,
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    snr_df=None,
    interval=30,
    tail_length=100,
    frame_step=1,
    output_dir=".",
    save_gif=True,
    show_plot=True,
):
    """
    Animate one trace per vertically stacked subplot using raw (non-normalized) units.
    """
    if not dataframes:
        print("No dataframes available for stacked animation.")
        return

    prepared = prepare_xy_series(dataframes, x_col, y_col, normalize=False)
    if not prepared:
        print("No valid files had the required columns for stacked animation.")
        return

    fig, axes = plt.subplots(
        len(prepared),
        1,
        figsize=(10, max(3, 3 * len(prepared))),
        squeeze=False,
    )
    axes = axes.flatten()
    fig.suptitle("Stacked Trace Animation (Raw Units)")
    subplot_hspace = 0.5

    color_map = plt.get_cmap("tab10", len(prepared))
    artists = []
    max_frames = max(len(item[1]) for item in prepared)

    for i, ((filename, x_vals, y_vals), ax) in enumerate(zip(prepared, axes)):
        base_color = color_map(i)
        snr_db = get_snr_value_db(snr_df, filename)
        peak_count = get_peak_count(snr_df, filename)
        peak_annotations = get_peak_annotations(snr_df, filename)

        x_min, x_max = np.min(x_vals), np.max(x_vals)
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        peak_x_positions = [x_vals[pk_idx] for (pk_idx, _, _) in peak_annotations if pk_idx < len(x_vals)]
        if peak_x_positions:
            pk_min = min(peak_x_positions)
            pk_max = max(peak_x_positions)
            pk_span = pk_max - pk_min if pk_max > pk_min else (x_max - x_min) * 0.1
            x_pad = pk_span * 0.1875
            ax_x_min, ax_x_max = pk_min - x_pad, pk_max + x_pad
        else:
            x_pad = (x_max - x_min) * 0.0625 if x_max > x_min else 1
            ax_x_min, ax_x_max = x_min - x_pad, x_max + x_pad
        y_range = (y_max - y_min) if y_max > y_min else 1
        y_pad_bottom = y_range * 0.05
        base_y_pad_top = y_range * 0.15
        base_y_span = y_range + y_pad_bottom + base_y_pad_top
        target_y_span = base_y_span * 1.25
        y_pad_top = max(base_y_pad_top, target_y_span - (y_range + y_pad_bottom))

        ax.set_xlim(ax_x_min, ax_x_max)
        ax.set_ylim(y_min - y_pad_bottom, y_max + y_pad_top)
        ax.set_title(filename)
        ax.set_xlabel("R_p (ohm)")
        ax.set_ylabel("Inductance (uH)")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.01,
            0.98,
            (
                f"Crack SNR: {fmt_snr_db(snr_db)}\n"
                f"Detected peaks: {peak_count if peak_count is not None else 'n/a'}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            zorder=10,
        )

        tail_segments = LineCollection([], linewidths=2.0, zorder=2)
        ax.add_collection(tail_segments)
        point, = ax.plot([], [], marker="o", ms=5, color=base_color, zorder=3)
        peak_markers = []
        peak_texts = []
        for _, _, _ in peak_annotations:
            peak_marker, = ax.plot([], [], marker="^", ms=6, color=base_color, zorder=4, alpha=0.9)
            peak_text = ax.text(
                0,
                0,
                "",
                fontsize=8,
                color=base_color,
                ha="left",
                va="bottom",
                zorder=5,
                visible=False,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
            peak_markers.append(peak_marker)
            peak_texts.append(peak_text)

        artists.append((tail_segments, point, base_color, peak_annotations, peak_markers, peak_texts))

    def init():
        for tail_segments, point, _, _, peak_markers, peak_texts in artists:
            tail_segments.set_segments([])
            tail_segments.set_color([])
            point.set_data([], [])
            for peak_marker, peak_text in zip(peak_markers, peak_texts):
                peak_marker.set_data([], [])
                peak_text.set_visible(False)
        return [
            artist
            for tail_segments, point, _, _, peak_markers, peak_texts in artists
            for artist in (tail_segments, point, *peak_markers, *peak_texts)
        ]

    def update(frame):
        for (_, x_vals, y_vals), (tail_segments, point, base_color, peak_annotations, peak_markers, peak_texts) in zip(prepared, artists):
            last_index = min(frame, len(x_vals) - 1)

            start_idx = max(0, last_index - tail_length + 1)
            tail_x = x_vals[start_idx : last_index + 1]
            tail_y = y_vals[start_idx : last_index + 1]

            if len(tail_x) > 1:
                points = np.column_stack([tail_x, tail_y])
                segments = np.stack([points[:-1], points[1:]], axis=1)

                segment_count = len(segments)
                alphas = np.linspace(0.05, 1.0, segment_count)
                colors = np.tile(np.array(base_color), (segment_count, 1))
                colors[:, 3] = alphas

                tail_segments.set_segments(segments)
                tail_segments.set_color(colors)
            else:
                tail_segments.set_segments([])
                tail_segments.set_color([])

            if frame < len(x_vals):
                point.set_data([x_vals[frame]], [y_vals[frame]])
            else:
                point.set_data([x_vals[-1]], [y_vals[-1]])

            for (peak_idx, peak_snr_db, peak_label), peak_marker, peak_text in zip(peak_annotations, peak_markers, peak_texts):
                if peak_idx < len(x_vals) and frame >= peak_idx:
                    peak_x = x_vals[peak_idx]
                    peak_y = y_vals[peak_idx]
                    peak_marker.set_data([peak_x], [peak_y])
                    peak_text.set_position((peak_x, peak_y))
                    peak_text.set_text(f"{peak_label} | {fmt_snr_db(peak_snr_db)}")
                    peak_text.set_visible(True)
                else:
                    peak_marker.set_data([], [])
                    peak_text.set_visible(False)

        return [
            artist
            for tail_segments, point, _, _, peak_markers, peak_texts in artists
            for artist in (tail_segments, point, *peak_markers, *peak_texts)
        ]

    frame_indices = range(0, max_frames, max(1, frame_step))

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )

    # Apply layout before saving so spacing changes are reflected in the GIF.
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=subplot_hspace)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_gif:
        gif_path = Path(output_dir) / f"stacked_animation_{timestamp}.gif"
        fps = max(1, int(round(1000 / interval / max(1, frame_step))))
        anim.save(str(gif_path), writer="pillow", fps=fps)
        print(f"Saved stacked animation GIF: {gif_path}")

    fig._stacked_anim = anim

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def create_stacked_baseline_animation(
    dataframes,
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    snr_df=None,
    interval=30,
    tail_length=100,
    frame_step=1,
    output_dir=".",
    save_gif=True,
    show_plot=True,
):
    """
    Animate one trace per stacked subplot after subtracting each file's own baseline.

    For each dataset, x and y are shifted by their respective minima.
    """
    if not dataframes:
        print("No dataframes available for baseline stacked animation.")
        return

    prepared_raw = prepare_xy_series(dataframes, x_col, y_col, normalize=False)
    if not prepared_raw:
        print("No valid files had the required columns for baseline stacked animation.")
        return

    prepared = []
    for filename, x_vals, y_vals in prepared_raw:
        x_shifted = x_vals - np.min(x_vals)
        y_shifted = y_vals - np.min(y_vals)
        prepared.append((filename, x_shifted, y_shifted))

    global_y_min = min(np.min(y_vals) for _, _, y_vals in prepared)
    global_y_max = max(np.max(y_vals) for _, _, y_vals in prepared)
    global_y_pad = (global_y_max - global_y_min) * 0.05 if global_y_max > global_y_min else 1

    fig, axes = plt.subplots(
        len(prepared),
        1,
        figsize=(10, max(3, 3 * len(prepared))),
        squeeze=False,
        sharey=True,
    )
    axes = axes.flatten()
    fig.suptitle("Stacked Trace Animation (Baseline-Shifted)")
    subplot_hspace = 0.5

    color_map = plt.get_cmap("tab10", len(prepared))
    artists = []
    max_frames = max(len(item[1]) for item in prepared)

    for i, ((filename, x_vals, y_vals), ax) in enumerate(zip(prepared, axes)):
        base_color = color_map(i)
        snr_db = get_snr_value_db(snr_df, filename)
        peak_count = get_peak_count(snr_df, filename)
        peak_annotations = get_peak_annotations(snr_df, filename)

        x_min, x_max = np.min(x_vals), np.max(x_vals)
        peak_x_positions = [x_vals[pk_idx] for (pk_idx, _, _) in peak_annotations if pk_idx < len(x_vals)]
        if peak_x_positions:
            pk_min = min(peak_x_positions)
            pk_max = max(peak_x_positions)
            pk_span = pk_max - pk_min if pk_max > pk_min else (x_max - x_min) * 0.1
            x_pad = pk_span * 0.1875
            ax_x_min, ax_x_max = pk_min - x_pad, pk_max + x_pad
        else:
            x_pad = (x_max - x_min) * 0.0625 if x_max > x_min else 1
            ax_x_min, ax_x_max = x_min - x_pad, x_max + x_pad

        ax.set_xlim(ax_x_min, ax_x_max)
        ax.set_ylim(global_y_min - global_y_pad, global_y_max + (global_y_pad * 3.0))
        ax.set_title(f"{filename} (baseline-shifted)")
        ax.set_xlabel("R_p (ohm)")
        ax.set_ylabel("Inductance (uH)")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.01,
            0.98,
            (
                f"Crack SNR: {fmt_snr_db(snr_db)}\n"
                f"Detected peaks: {peak_count if peak_count is not None else 'n/a'}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            zorder=10,
        )

        tail_segments = LineCollection([], linewidths=2.0, zorder=2)
        ax.add_collection(tail_segments)
        point, = ax.plot([], [], marker="o", ms=5, color=base_color, zorder=3)
        peak_markers = []
        peak_texts = []
        for _, _, _ in peak_annotations:
            peak_marker, = ax.plot([], [], marker="^", ms=6, color=base_color, zorder=4, alpha=0.9)
            peak_text = ax.text(
                0,
                0,
                "",
                fontsize=8,
                color=base_color,
                ha="left",
                va="bottom",
                zorder=5,
                visible=False,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
            peak_markers.append(peak_marker)
            peak_texts.append(peak_text)

        artists.append((tail_segments, point, base_color, peak_annotations, peak_markers, peak_texts))

    def init():
        for tail_segments, point, _, _, peak_markers, peak_texts in artists:
            tail_segments.set_segments([])
            tail_segments.set_color([])
            point.set_data([], [])
            for peak_marker, peak_text in zip(peak_markers, peak_texts):
                peak_marker.set_data([], [])
                peak_text.set_visible(False)
        return [
            artist
            for tail_segments, point, _, _, peak_markers, peak_texts in artists
            for artist in (tail_segments, point, *peak_markers, *peak_texts)
        ]

    def update(frame):
        for (_, x_vals, y_vals), (tail_segments, point, base_color, peak_annotations, peak_markers, peak_texts) in zip(prepared, artists):
            last_index = min(frame, len(x_vals) - 1)

            start_idx = max(0, last_index - tail_length + 1)
            tail_x = x_vals[start_idx : last_index + 1]
            tail_y = y_vals[start_idx : last_index + 1]

            if len(tail_x) > 1:
                points = np.column_stack([tail_x, tail_y])
                segments = np.stack([points[:-1], points[1:]], axis=1)

                segment_count = len(segments)
                alphas = np.linspace(0.05, 1.0, segment_count)
                colors = np.tile(np.array(base_color), (segment_count, 1))
                colors[:, 3] = alphas

                tail_segments.set_segments(segments)
                tail_segments.set_color(colors)
            else:
                tail_segments.set_segments([])
                tail_segments.set_color([])

            if frame < len(x_vals):
                point.set_data([x_vals[frame]], [y_vals[frame]])
            else:
                point.set_data([x_vals[-1]], [y_vals[-1]])

            for (peak_idx, peak_snr_db, peak_label), peak_marker, peak_text in zip(peak_annotations, peak_markers, peak_texts):
                if peak_idx < len(x_vals) and frame >= peak_idx:
                    peak_x = x_vals[peak_idx]
                    peak_y = y_vals[peak_idx]
                    peak_marker.set_data([peak_x], [peak_y])
                    peak_text.set_position((peak_x, peak_y))
                    peak_text.set_text(f"{peak_label} | {fmt_snr_db(peak_snr_db)}")
                    peak_text.set_visible(True)
                else:
                    peak_marker.set_data([], [])
                    peak_text.set_visible(False)

        return [
            artist
            for tail_segments, point, _, _, peak_markers, peak_texts in artists
            for artist in (tail_segments, point, *peak_markers, *peak_texts)
        ]

    frame_indices = range(0, max_frames, max(1, frame_step))

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )

    # Apply layout before saving so spacing changes are reflected in the GIF.
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=subplot_hspace)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_gif:
        gif_path = Path(output_dir) / f"stacked_baseline_animation_{timestamp}.gif"
        fps = max(1, int(round(1000 / interval / max(1, frame_step))))
        anim.save(str(gif_path), writer="pillow", fps=fps)
        print(f"Saved baseline-stacked animation GIF: {gif_path}")

    fig._stacked_baseline_anim = anim

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def create_3d_contour_plots(
    dataframes,
    time_col="timestamp",
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    snr_df=None,
    output_dir=".",
    save_plots=True,
    show_plot=True,
):
    """
    For each run, produce a 3D surface / contour plot where:
      x = time (timestamp)
      y = R_p  (sensor1_smooth_rot)
      z = Inductance (sensor2_smooth_rot)

    Because the data is a single 1-D scan (not a 2-D grid), we build a
    thin pseudo-surface by stacking two offset copies of the trace so
    matplotlib's plot_surface / contour machinery has a proper 2-D array
    to work with.  The top face is the actual scan; peak positions are
    marked with scatter points.
    """
    color_map = plt.get_cmap("tab10", max(len(dataframes), 1))

    time_candidates = [time_col, "time", "sample", "index"]
    rp_candidates = [x_col, x_col.strip(), "sensor1", " sensor1"]
    ind_candidates = [y_col, y_col.strip(), "sensor2", " sensor2"]

    for i, (filename, df) in enumerate(dataframes.items()):
        t_col = next((c for c in time_candidates if c in df.columns), None)
        rp_col = next((c for c in rp_candidates if c in df.columns), None)
        ind_col = next((c for c in ind_candidates if c in df.columns), None)

        if rp_col is None or ind_col is None:
            print(f"3D plot: required columns not found in '{filename}', skipping.")
            continue

        rp_vals = df[rp_col].to_numpy(dtype=float)
        ind_vals = df[ind_col].to_numpy(dtype=float)

        if t_col is not None:
            t_raw = df[t_col].to_numpy(dtype=float)
        else:
            t_raw = np.arange(len(rp_vals), dtype=float)

        # Normalise time to [0, 1] for a clean axis.
        t_min, t_max = t_raw.min(), t_raw.max()
        t_vals = (t_raw - t_min) / (t_max - t_min) if t_max > t_min else t_raw

        # Build a 2-row pseudo-surface (row 0 = baseline at ind_min, row 1 = actual trace).
        N = len(t_vals)
        T = np.vstack([t_vals, t_vals])          # shape (2, N)
        R = np.vstack([rp_vals, rp_vals])        # shape (2, N)
        ind_floor = np.full(N, ind_vals.min())
        Z = np.vstack([ind_floor, ind_vals])     # shape (2, N)

        base_color = color_map(i)

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            T, R, Z,
            facecolor=base_color,
            alpha=0.55,
            linewidth=0,
            antialiased=True,
        )

        # Draw the scan line on top for clarity.
        ax.plot(t_vals, rp_vals, ind_vals, color=base_color, linewidth=1.5, zorder=5)

        # Project a contour onto the floor (z = ind_min).
        ax.contourf(
            T, R, Z,
            zdir="z",
            offset=ind_vals.min(),
            levels=15,
            cmap="viridis",
            alpha=0.4,
        )

        # Mark detected peaks.
        peak_annotations = get_peak_annotations(snr_df, filename) if snr_df is not None else []
        if peak_annotations:
            pk_t = np.array([t_vals[pk] for (pk, _, _) in peak_annotations if pk < N])
            pk_r = np.array([rp_vals[pk] for (pk, _, _) in peak_annotations if pk < N])
            pk_z = np.array([ind_vals[pk] for (pk, _, _) in peak_annotations if pk < N])
            pk_labels = [lbl for (pk, _, lbl) in peak_annotations if pk < N]
            pk_snrs = [snr for (pk, snr, _) in peak_annotations if pk < N]
            ax.scatter(pk_t, pk_r, pk_z, color="red", s=60, zorder=6, depthshade=False)
            for tx, rx, zx, lbl, snr_val in zip(pk_t, pk_r, pk_z, pk_labels, pk_snrs):
                ax.text(
                    tx, rx, zx,
                    f"  {lbl}\n  {fmt_snr_db(snr_val)}",
                    fontsize=7,
                    color="red",
                    zorder=7,
                )

        ax.set_xlabel("Time (norm.)")
        ax.set_ylabel("R_p")
        ax.set_zlabel("Inductance")
        ax.set_title(f"3D Contour — {filename}")
        plt.tight_layout()

        if save_plots:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = filename.replace(".csv", "").replace(" ", "_")
            out_path = Path(output_dir) / f"3d_contour_{safe_name}_{ts}.png"
            fig.savefig(str(out_path), dpi=150)
            print(f"Saved 3D contour plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def main():
    print("CSV Loader + Animation + SNR Analysis")
    print("-" * 40)

    file_paths = select_csv_files_via_gui(initial_dir=Path.cwd())
    
    if not file_paths:
        print("No CSV files selected. Exiting.")
        return
    
    # Load the CSV files
    dataframes = load_csv_files(file_paths)
    source_path_lookup = build_source_path_lookup(file_paths)
    
    if dataframes:
        run_output_dir = create_run_output_dir()
        print(f"Run output directory: {run_output_dir}")

        labels_df = run_manual_raw_peak_labeling(
            dataframes,
            output_dir=run_output_dir,
            source_path_lookup=source_path_lookup,
            max_peaks_per_file=10,
            min_peak_distance=60,
            show_plot=True,
        )
        if not labels_df.empty:
            print("Manual labels captured and saved before analysis.")
        else:
            print("No manual crack windows were labeled. Ground-truth analysis requires labels; stopping run.")
            return

        print("\n" + "=" * 40)
        print(f"Loaded {len(dataframes)} file(s) successfully:")
        for filename, df in dataframes.items():
            print(f"\n{filename}:")
            print(df.head())

        snr_df, per_crack_df = analyze_snr(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            manual_labels_df=labels_df,
            require_manual_windows=True,
            ground_truth_only=True,
            baseline_window=101,
            max_peaks=6,
            min_peak_distance=100,
            sigma_threshold_multiplier=3.0,
            verbose=False,
        )

        if snr_df.empty:
            print("No SNR results generated from manual crack windows.")
        else:
            print("\nSNR computed from manually labeled crack windows (one peak per window).")

        if not snr_df.empty:
            print("\nCrack-detection SNR by file (highest first):")
            print(snr_df[["file", "snr_db", "snr_linear", "signal_amplitude", "noise_sigma", "peak_count"]].head(50))
        if not snr_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snr_path = run_output_dir / f"snr_analysis_{timestamp}.csv"
            snr_df.to_csv(snr_path, index=False)
            print(f"Saved SNR analysis CSV: {snr_path}")

            if per_crack_df is not None and not per_crack_df.empty:
                per_crack_path = run_output_dir / f"snr_per_crack_{timestamp}.csv"
                per_crack_df.to_csv(per_crack_path, index=False)
                print(f"Saved per-crack SNR CSV: {per_crack_path}")

            create_snr_visualizations(
                snr_df,
                output_dir=run_output_dir,
                show_plot=SHOW_PLOTS,
                save_plots=True,
            )

        create_overlay_animation(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            snr_df=snr_df,
            frame_step=FRAME_STEP,
            output_dir=run_output_dir,
            save_gif=SAVE_GIFS,
            show_plot=SHOW_PLOTS,
        )

        create_stacked_animation(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            snr_df=snr_df,
            frame_step=FRAME_STEP,
            output_dir=run_output_dir,
            save_gif=SAVE_GIFS,
            show_plot=SHOW_PLOTS,
        )

        create_stacked_baseline_animation(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            snr_df=snr_df,
            frame_step=FRAME_STEP,
            output_dir=run_output_dir,
            save_gif=SAVE_GIFS,
            show_plot=SHOW_PLOTS,
        )

        create_3d_contour_plots(
            dataframes,
            time_col="timestamp",
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            snr_df=snr_df,
            output_dir=run_output_dir,
            save_plots=True,
            show_plot=SHOW_PLOTS,
        )
    else:
        print("No files were successfully loaded.")


if __name__ == "__main__":
    main()
