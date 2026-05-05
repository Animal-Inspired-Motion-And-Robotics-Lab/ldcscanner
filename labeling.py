# ---------------------------------------------------------------------------
# labeling.py — Interactive manual crack-window labeling
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from pathlib import Path

from config import (
    CRACK_LABELS,
    LABEL_COLOR_MAP,
    LABEL_SHORT_MAP,
    LABEL_INPUT_MAP,
    X_CANDIDATES,
    Y_CANDIDATES,
)
from io_utils import (
    get_file_fingerprint,
    get_label_cache_path,
    load_cached_windows,
    save_cached_windows,
)
from signal_processing import resolve_first_existing_column


# ---------------------------------------------------------------------------
# Text-mode helper
# ---------------------------------------------------------------------------

def ask_manual_peak_label(prompt_text):
    """Prompt the user for a crack label via text input."""
    while True:
        response = input(prompt_text).strip().lower()
        if response == "":
            return "Not a crack"
        if response in LABEL_INPUT_MAP:
            return LABEL_INPUT_MAP[response]
        print("Invalid label. Use 1, 2, 3, or 0 (Not a crack).")


# ---------------------------------------------------------------------------
# Main labeling entry point
# ---------------------------------------------------------------------------

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
    Interactive step: user draws crack windows on the raw sensor signal.

    For each file the user drags spans on a matplotlib figure and presses:
      1 / 2 / 3  — label the current span as Crack 1/2/3
      0          — erase all windows overlapping the current span
            x          — trim (delete) datapoints in the current span
      Backspace  — remove the last committed window
      Enter      — finish and move to the next file

    If show_plot=False and no cache exists, falls back to text entry.

    Labeled windows are cached per-file so subsequent runs skip re-labeling
    unless the source CSV changes.

    Returns:
        DataFrame with one row per labeled window across all files.
    """
    if not dataframes:
        return pd.DataFrame()

    records = []
    print("\nManual raw window labeling step")
    print("Drag a window, then press 1/2/3 for crack type, 0 to clear, x to trim. Enter when done.")

    for filename, df in dataframes.items():
        if df.empty:
            continue

        source_path = source_path_lookup.get(filename) if source_path_lookup else None
        fingerprint = get_file_fingerprint(source_path) if source_path else None
        cache_path  = get_label_cache_path(cache_dir, source_path) if source_path else None
        cached      = load_cached_windows(cache_path, fingerprint) if cache_path else None

        y_col = resolve_first_existing_column(df, Y_CANDIDATES)
        if y_col is None:
            print(f"Skipping manual labels for {filename}: raw sensor column not found.")
            continue

        x_col = resolve_first_existing_column(df, X_CANDIDATES)

        working_df = df.copy()

        def get_plot_arrays(df_in):
            y_series_local = pd.to_numeric(df_in[y_col], errors="coerce")
            if x_col is None:
                x_series_local = pd.Series(np.arange(len(df_in), dtype=float), index=df_in.index)
                x_col_name_local = "sample_index"
            else:
                x_series_local = pd.to_numeric(df_in[x_col], errors="coerce")
                x_col_name_local = x_col

            valid_local = x_series_local.notna() & y_series_local.notna()
            x_vals_local = x_series_local[valid_local].to_numpy(dtype=float)
            y_vals_local = y_series_local[valid_local].to_numpy(dtype=float)
            raw_row_indices_local = np.flatnonzero(valid_local.to_numpy())
            return x_vals_local, y_vals_local, raw_row_indices_local, x_col_name_local

        x_vals, y_vals, raw_row_indices, x_col_name = get_plot_arrays(working_df)

        if len(y_vals) < 10:
            print(f"Skipping manual labels for {filename}: insufficient numeric raw samples.")
            continue

        windows = list(cached) if cached is not None else []
        if cached is not None:
            print(
                f"Loaded {len(windows)} cached window(s) for {filename}. "
                "Press Enter to accept, or edit before continuing."
            )

        y_min   = float(np.min(y_vals))
        y_max   = float(np.max(y_vals))
        y_range = max(y_max - y_min, 1e-9)
        text_y  = y_max + (0.04 * y_range)

        # active_sel must be defined before refresh_window_display (captured by the closure).
        active_sel = {"start_x": None, "end_x": None, "patch": None}

        fig, ax = plt.subplots(figsize=(12, 4))
        signal_line, = ax.plot(
            x_vals,
            y_vals,
            color="#1f77b4",
            linewidth=1.2,
            label=f"Raw signal ({y_col.strip()})",
        )

        status_text = ax.text(
            0.01, 0.98,
            "Drag a span to create a window. 1/2/3 label. 0=clear. x=trim. Enter=done",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.set_title(f"Manual Window Labeling: {filename}")
        ax.set_xlabel(x_col_name)
        ax.set_ylabel(y_col.strip())
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        plt.tight_layout()

        # Disable autoscaling AFTER tight_layout so the layout pass can apply
        # the data-driven axis limits before we lock them in place.
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)

        base_plot_path = Path(output_dir) / f"raw_window_selection_{Path(filename).stem}.png"
        fig.savefig(base_plot_path, dpi=180)
        print(f"Saved base window-selection plot: {base_plot_path}")

        # ── helpers shared between interactive and text modes ──────────────

        def refresh_window_display():
            nonlocal x_vals, y_vals, raw_row_indices, y_min, y_max, y_range, text_y
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            if len(x_vals) > 0 and len(y_vals) > 0:
                signal_line.set_data(x_vals, y_vals)
                y_min   = float(np.min(y_vals))
                y_max   = float(np.max(y_vals))
                y_range = max(y_max - y_min, 1e-9)
                text_y  = y_max + (0.04 * y_range)

            crack_windows = [w for w in windows if w["label"] in CRACK_LABELS]
            active_text = (
                "none" if active_sel["start_x"] is None
                else f"{active_sel['start_x']:.3f} to {active_sel['end_x']:.3f}"
            )
            status_text.set_text(
                f"Active selection: {active_text} | Crack windows: {len(crack_windows)} | "
                "0 clears active span + overlapping | x trims selected data | Enter=done"
            )

            crack_idx = 0
            for window in windows:
                if window["label"] in CRACK_LABELS:
                    crack_idx += 1
                    if window["patch"] is None:
                        window["patch"] = ax.axvspan(
                            float(window["start_x"]), float(window["end_x"]),
                            facecolor=LABEL_COLOR_MAP[window["label"]],
                            edgecolor="#303030", alpha=0.24, linewidth=1.2, zorder=2,
                        )
                    else:
                        window["patch"].set_visible(True)
                        window["patch"].set_facecolor(LABEL_COLOR_MAP[window["label"]])
                        window["patch"].set_alpha(0.24)
                        window["patch"].set_linewidth(1.2)

                    center_x = 0.5 * (window["start_x"] + window["end_x"])
                    if window["text"] is None:
                        window["text"] = ax.text(
                            center_x, text_y, "",
                            ha="center", va="bottom", fontsize=8,
                            color="#111111", zorder=4,
                        )
                    window["text"].set_visible(True)
                    window["text"].set_position((center_x, text_y))
                    window["text"].set_text(f"W{crack_idx}:{LABEL_SHORT_MAP[window['label']]}")
                else:
                    if window["patch"] is not None:
                        window["patch"].set_visible(False)
                    if window["text"] is not None:
                        window["text"].set_visible(False)

            if len(x_vals) > 0 and len(y_vals) > 0:
                ax.set_xlim(current_xlim)
                ax.set_ylim(current_ylim)
            fig.canvas.draw_idle()

        # ── interactive (GUI) mode ─────────────────────────────────────────

        if show_plot:
            print(
                f"\nInteractive labeling for {filename}: drag a window, then press "
                "1=Crack1, 2=Crack2, 3=Crack3, 0=clear span+overlapping, x=trim selected data, "
                "Enter=finish, Backspace/Delete=remove last window."
            )

            x_span = max(np.ptp(x_vals), 1e-12)

            def on_span_select(xmin, xmax):
                start_x = float(min(xmin, xmax))
                end_x   = float(max(xmin, xmax))
                if abs(end_x - start_x) <= (0.002 * x_span):
                    return
                if active_sel["patch"] is not None:
                    active_sel["patch"].remove()
                active_sel["patch"] = ax.axvspan(
                    start_x, end_x,
                    facecolor="#b0b0b0", edgecolor="#505050",
                    alpha=0.3, linewidth=1.2, zorder=3,
                )
                active_sel["start_x"] = start_x
                active_sel["end_x"]   = end_x
                refresh_window_display()

            span_selector = SpanSelector(
                ax, on_span_select, "horizontal",
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
                    if active_sel["start_x"] is None:
                        print("Drag a window first.")
                        return

                    start_x = float(active_sel["start_x"])
                    end_x   = float(active_sel["end_x"])
                    label   = LABEL_INPUT_MAP.get(key, "Not a crack")

                    if key == "0":
                        kept = []
                        for w in windows:
                            overlaps = max(float(w["start_x"]), start_x) <= min(float(w["end_x"]), end_x)
                            if overlaps:
                                if w["patch"] is not None:
                                    w["patch"].remove()
                                if w["text"] is not None:
                                    w["text"].remove()
                            else:
                                kept.append(w)
                        windows[:] = kept
                    else:
                        patch = ax.axvspan(
                            start_x, end_x,
                            facecolor=LABEL_COLOR_MAP[label],
                            edgecolor="#303030", alpha=0.24, linewidth=1.2, zorder=2,
                        )
                        text = ax.text(
                            0.5 * (start_x + end_x), text_y, "",
                            ha="center", va="bottom", fontsize=8,
                            color="#111111", zorder=4,
                        )
                        windows.append({"start_x": start_x, "end_x": end_x,
                                        "label": label, "patch": patch, "text": text})

                    if active_sel["patch"] is not None:
                        active_sel["patch"].remove()
                    active_sel["patch"]   = None
                    active_sel["start_x"] = None
                    active_sel["end_x"]   = None
                    refresh_window_display()

                elif key == "x":
                    nonlocal working_df, x_vals, y_vals, raw_row_indices

                    if active_sel["start_x"] is None:
                        print("Drag a window first.")
                        return

                    start_x = float(active_sel["start_x"])
                    end_x   = float(active_sel["end_x"])
                    in_trim = (x_vals >= start_x) & (x_vals <= end_x)
                    trim_count = int(np.count_nonzero(in_trim))
                    if trim_count == 0:
                        print("No points in selected span to trim.")
                        return

                    # Drop trimmed points from the dataframe used by downstream analysis.
                    drop_raw_indices = raw_row_indices[in_trim]
                    working_df = working_df.drop(index=working_df.index[drop_raw_indices]).reset_index(drop=True)
                    dataframes[filename] = working_df

                    # Remove windows that now overlap the trimmed span.
                    kept = []
                    removed_window_count = 0
                    for w in windows:
                        overlaps = max(float(w["start_x"]), start_x) <= min(float(w["end_x"]), end_x)
                        if overlaps:
                            removed_window_count += 1
                            if w["patch"] is not None:
                                w["patch"].remove()
                            if w["text"] is not None:
                                w["text"].remove()
                        else:
                            kept.append(w)
                    windows[:] = kept

                    x_vals, y_vals, raw_row_indices, _ = get_plot_arrays(working_df)
                    if len(y_vals) < 10:
                        print(
                            f"Trimmed {trim_count} point(s); only {len(y_vals)} numeric points remain. "
                            "Finishing this file."
                        )
                        plt.close(fig)
                        return

                    if active_sel["patch"] is not None:
                        active_sel["patch"].remove()
                    active_sel["patch"] = None
                    active_sel["start_x"] = None
                    active_sel["end_x"] = None

                    print(
                        f"Trimmed {trim_count} point(s) from {filename}; "
                        f"removed {removed_window_count} overlapping window(s)."
                    )
                    refresh_window_display()

                elif key in ("backspace", "delete"):
                    if not windows:
                        return
                    w = windows.pop()
                    if w["patch"] is not None:
                        w["patch"].remove()
                    if w["text"] is not None:
                        w["text"].remove()
                    refresh_window_display()

                elif key in ("enter", "return"):
                    plt.close(fig)

            key_cid = fig.canvas.mpl_connect("key_press_event", on_key)
            refresh_window_display()
            plt.show()
            fig.canvas.mpl_disconnect(key_cid)
            span_selector.set_active(False)

            if active_sel["patch"] is not None:
                active_sel["patch"].remove()

        # ── text-input fallback (no display or cached result skips GUI) ────

        elif cached is None:
            print(f"\nFile: {filename}")
            print("Text mode: enter windows as start,end,label (1/2/3/0). Blank line to finish.")
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
                    end_x   = float(parts[1])
                except ValueError:
                    print("Invalid numeric start/end values.")
                    continue

                label_raw = parts[2].lower()
                if label_raw not in LABEL_INPUT_MAP:
                    print("Invalid label. Use 1, 2, 3, or 0.")
                    continue

                label = LABEL_INPUT_MAP[label_raw]
                start_x, end_x = sorted((start_x, end_x))

                if label_raw == "0":
                    windows = [
                        w for w in windows
                        if max(float(w["start_x"]), start_x) > min(float(w["end_x"]), end_x)
                    ]
                else:
                    windows.append({"start_x": start_x, "end_x": end_x,
                                    "label": label, "patch": None, "text": None})

        # ── persist cache and save labeled summary plot ────────────────────

        if cache_path is not None and fingerprint is not None:
            save_cached_windows(cache_path, fingerprint, windows)

        _save_labeled_plot(filename, x_vals, y_vals, x_col_name, y_col,
                           windows, text_y, y_min, y_max, y_range, output_dir)

        records.extend(_collect_window_records(filename, y_col, x_col_name,
                                               windows, x_vals, y_vals, raw_row_indices))

    labels_df = pd.DataFrame(records)
    if labels_df.empty:
        print("No manual peak labels were recorded.")
        return labels_df

    labels_path = Path(output_dir) / "manual_window_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved manual window labels: {labels_path}")

    summary_df = (
        labels_df.groupby(["file", "manual_label"])["window_order"]
        .count()
        .rename("count")
        .reset_index()
        .sort_values(["file", "manual_label"])
    )
    summary_path = Path(output_dir) / "manual_window_label_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved manual window label summary: {summary_path}")

    return labels_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_labeled_plot(filename, x_vals, y_vals, x_col_name, y_col,
                       windows, text_y, y_min, y_max, y_range, output_dir):
    """Save a static PNG showing all labeled windows on the raw signal."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_vals, y_vals, color="#1f77b4", linewidth=1.2,
            label=f"Raw signal ({y_col.strip()})")

    for i, window in enumerate(windows, start=1):
        ax.axvspan(
            window["start_x"], window["end_x"],
            facecolor=LABEL_COLOR_MAP[window["label"]],
            edgecolor="#303030", alpha=0.22, linewidth=1.0, zorder=2,
        )
        ax.text(
            0.5 * (window["start_x"] + window["end_x"]), text_y,
            f"W{i}:{LABEL_SHORT_MAP[window['label']]}",
            ha="center", va="bottom", fontsize=8, color="#111111", zorder=4,
        )

    ax.set_title(f"Labeled Raw Windows: {filename}")
    ax.set_xlabel(x_col_name)
    ax.set_ylabel(y_col.strip())
    ax.grid(alpha=0.25)
    y_pad = 0.12 * y_range
    ax.set_ylim(y_min - (0.03 * y_range), y_max + y_pad)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = Path(output_dir) / f"raw_window_labeled_{Path(filename).stem}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved labeled window plot: {out_path}")


def _collect_window_records(filename, y_col, x_col_name, windows, x_vals, y_vals, raw_row_indices):
    """Build one result record per labeled window."""
    records = []
    for i, window in enumerate(windows, start=1):
        start_x = float(window["start_x"])
        end_x   = float(window["end_x"])
        in_win  = (x_vals >= start_x) & (x_vals <= end_x)
        point_count = int(np.count_nonzero(in_win))

        if point_count > 0:
            clean_indices   = np.flatnonzero(in_win)
            start_raw_idx   = int(raw_row_indices[clean_indices[0]])
            end_raw_idx     = int(raw_row_indices[clean_indices[-1]])
            local_peak_rel  = int(np.argmax(y_vals[clean_indices]))
            peak_clean_idx  = int(clean_indices[local_peak_rel])
            peak_raw_idx    = int(raw_row_indices[peak_clean_idx])
            peak_raw_x      = float(x_vals[peak_clean_idx])
            peak_raw_y      = float(y_vals[peak_clean_idx])
            y_mean          = float(np.mean(y_vals[in_win]))
        else:
            start_raw_idx = end_raw_idx = peak_raw_idx = np.nan
            peak_raw_x = peak_raw_y = y_mean = np.nan

        records.append({
            "file":                  filename,
            "raw_y_col":             y_col,
            "raw_x_col":             x_col_name,
            "window_order":          i,
            "window_start_x":        start_x,
            "window_end_x":          end_x,
            "window_start_raw_idx":  start_raw_idx,
            "window_end_raw_idx":    end_raw_idx,
            "window_width":          float(end_x - start_x),
            "window_point_count":    point_count,
            "window_mean_y":         y_mean,
            "window_peak_raw_idx":   peak_raw_idx,
            "window_peak_raw_x":     peak_raw_x,
            "window_peak_raw_y":     peak_raw_y,
            "manual_label":          window["label"],
        })
    return records
