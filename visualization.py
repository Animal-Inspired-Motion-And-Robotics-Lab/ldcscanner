# ---------------------------------------------------------------------------
# visualization.py — Static plots, animations, and 3D contour figures
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from datetime import datetime
from pathlib import Path

from config import (
    LABEL_COLOR_MAP,
    CRACK_LABELS,
    X_COL,
    Y_COL,
    SNR_DISPLAY_MODE,
    SNR_CHART_GROUP_MODE,
)
from io_utils import (
    get_file_fingerprint,
    get_rotation_cache_path,
    load_cached_rotation,
    save_cached_rotation,
)
from signal_processing import resolve_first_existing_column


# ---------------------------------------------------------------------------
# SNR lookup helpers (read results produced by signal_processing.analyze_snr)
# ---------------------------------------------------------------------------

def get_snr_value_db(snr_df, filename):
    """Return the file-level SNR (dB) for filename, or None."""
    if snr_df is None or snr_df.empty:
        return None
    matches = snr_df[snr_df["file"] == filename]
    return float(matches.iloc[0]["snr_db"]) if not matches.empty else None


def _use_linear_snr_display():
    return str(SNR_DISPLAY_MODE).strip().lower() == "linear"


def _snr_unit_label():
    return "linear" if _use_linear_snr_display() else "dB"


def _use_grouped_snr_chart_display():
    return str(SNR_CHART_GROUP_MODE).strip().lower() == "grouped"


def get_snr_value_display(snr_df, filename):
    """Return file-level SNR in configured display units."""
    if snr_df is None or snr_df.empty:
        return None
    matches = snr_df[snr_df["file"] == filename]
    if matches.empty:
        return None
    row = matches.iloc[0]
    key = "snr_linear" if _use_linear_snr_display() else "snr_db"
    if key not in row or pd.isna(row[key]):
        return None
    return float(row[key])


def get_peak_count(snr_df, filename):
    """Return the detected crack peak count for filename, or None."""
    if snr_df is None or snr_df.empty:
        return None
    matches = snr_df[snr_df["file"] == filename]
    return int(matches.iloc[0]["peak_count"]) if not matches.empty else None


def get_peak_annotations(snr_df, filename):
    """
    Return per-peak annotations for filename as a list of
    (peak_index, peak_snr_value, peak_label) tuples in configured units.
    """
    if snr_df is None or snr_df.empty:
        return []
    matches = snr_df[snr_df["file"] == filename]
    if matches.empty:
        return []

    row       = matches.iloc[0]
    idx_str   = row.get("peak_indices", "")
    snr_str   = row.get("peak_snr_db_values", "")
    label_str = row.get("peak_labels", "")

    if pd.isna(idx_str) or str(idx_str).strip() == "":
        return []

    indices = [int(x)   for x in str(idx_str).split(";")   if x.strip()]
    snr_vals_db = [float(x) for x in str(snr_str).split(";")   if x.strip()]
    if _use_linear_snr_display():
        snr_vals = [float(10 ** (v / 20.0)) for v in snr_vals_db]
    else:
        snr_vals = snr_vals_db
    labels   = [x.strip() for x in str(label_str).split(";") if x.strip()]

    return [
        (idx, snr_vals[i] if i < len(snr_vals) else np.nan,
              labels[i]   if i < len(labels)   else "Crack")
        for i, idx in enumerate(indices)
    ]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_snr_db(value):
    """Format an SNR value for on-plot labels."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} dB"


def fmt_snr_display(value):
    """Format SNR for configured display mode."""
    if value is None or not np.isfinite(value):
        return "n/a"
    if _use_linear_snr_display():
        return f"{value:.3f}"
    return f"{value:.1f} dB"


def parse_peak_snr_series(peak_snr_db_values):
    """Parse a semicolon-separated SNR string into a list of finite floats."""
    if pd.isna(peak_snr_db_values):
        return []
    values = []
    for token in str(peak_snr_db_values).split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            v = float(token)
        except ValueError:
            continue
        if np.isfinite(v):
            if _use_linear_snr_display():
                values.append(float(10 ** (v / 20.0)))
            else:
                values.append(v)
    return values


def parse_peak_label_series(peak_labels):
    """Parse a semicolon-separated label string into a list of stripped strings."""
    if pd.isna(peak_labels):
        return []
    return [t.strip() for t in str(peak_labels).split(";") if t.strip()]


def _extract_sensor_group_from_filename(filename):
    """Map a run filename to sensor group key (e.g., flex_8, flex_12)."""
    stem = Path(str(filename)).stem
    match = re.match(r"^([A-Za-z]+_\d+)", stem)
    if match:
        return match.group(1)
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


def _get_inductor_color_map(filenames):
    """
    Create a color map for filenames grouped by inductor type.
    
    Each inductor type (e.g., flex_4, flex_8, flex_12) gets a base color,
    and files of the same type get different shades of that color.
    
    Returns:
        Dict mapping filename → (R, G, B, A) color tuple.
    """
    # Base colors for each inductor type (RGBA tuples)
    type_base_colors = {
        "flex_4": (0.2, 0.8, 0.2, 1.0),    # green
        "flex_8": (0.2, 0.6, 1.0, 1.0),    # blue
        "flex_12": (1.0, 0.4, 0.4, 1.0),   # red
    }
    
    # Group filenames by inductor type
    type_groups = {}
    for fn in filenames:
        sensor_group = _extract_sensor_group_from_filename(fn)
        if sensor_group not in type_groups:
            type_groups[sensor_group] = []
        type_groups[sensor_group].append(fn)
    
    # Assign colors: for each type, generate shades for each file
    color_map = {}
    for sensor_type, file_list in type_groups.items():
        base_color = type_base_colors.get(sensor_type, (0.5, 0.5, 0.5, 1.0))
        n_files = len(file_list)
        
        for idx, fn in enumerate(file_list):
            # Create a shade by interpolating between the base color and white
            if n_files > 1:
                shade_factor = 0.5 + (idx / (n_files - 1)) * 0.5
            else:
                shade_factor = 0.75
            
            r, g, b, a = base_color
            # Interpolate towards white
            r_shade = r + (1.0 - r) * (1.0 - shade_factor)
            g_shade = g + (1.0 - g) * (1.0 - shade_factor)
            b_shade = b + (1.0 - b) * (1.0 - shade_factor)
            
            color_map[fn] = (r_shade, g_shade, b_shade, a)
    
    return color_map


# ---------------------------------------------------------------------------
# Static SNR summary charts
# ---------------------------------------------------------------------------

def create_snr_visualizations(snr_df, output_dir=".", show_plot=False, save_plots=True, label_color_map=None):
    """
    Produce static comparison charts for crack-detection SNR.

    Outputs (when save_plots=True):
      - Overview PNG: ranking, delta-to-best, signal vs noise scatter,
        and per-peak SNR strip plot.

    Returns:
        List of saved file paths.
    """
    if snr_df is None or snr_df.empty:
        print("No SNR data available for visualization.")
        return []

    effective_label_color_map = label_color_map or LABEL_COLOR_MAP

    plot_outputs = []
    snr_col = "snr_linear" if _use_linear_snr_display() else "snr_db"
    df_runs = snr_df.copy().sort_values(snr_col, ascending=False).reset_index(drop=True)

    use_grouped_display = _use_grouped_snr_chart_display()

    if use_grouped_display:
        # Display summary grouped by sensor type prefix (e.g., flex_8, flex_12).
        df_runs["sensor_group"] = df_runs["file"].map(_extract_sensor_group_from_filename)
        grouped = (
            df_runs.groupby("sensor_group", as_index=False)
            .agg(
                snr_mean=(snr_col, "mean"),
                snr_std=(snr_col, "std"),
                noise_sigma_mean=("noise_sigma", "mean"),
                signal_amplitude_mean=("signal_amplitude", "mean"),
                run_count=("file", "count"),
            )
        )
        grouped["snr_std"] = grouped["snr_std"].fillna(0.0)
        grouped = grouped.sort_values("snr_mean", ascending=False).reset_index(drop=True)
        rank = np.arange(1, len(grouped) + 1)
    else:
        grouped = None
        rank = np.arange(1, len(df_runs) + 1)

    unit_label = _snr_unit_label()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_rank, ax_delta, ax_signal_noise, ax_peaks = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    if use_grouped_display:
        entity_df = grouped.copy()
        entity_col = "sensor_group"
        x_label = "Inductor group rank (best to worst)"
    else:
        entity_df = df_runs[["file", snr_col, "noise_sigma", "signal_amplitude"]].copy()
        entity_df = entity_df.rename(
            columns={
                "file": "entity",
                snr_col: "snr_mean",
                "noise_sigma": "noise_sigma_mean",
                "signal_amplitude": "signal_amplitude_mean",
            }
        )
        entity_df["snr_std"] = 0.0
        entity_df["run_count"] = 1
        entity_col = "entity"
        x_label = "Run rank (best to worst)"

    if entity_col != "entity":
        entity_df = entity_df.rename(columns={entity_col: "entity"})

    crack_records = []
    for _, row in df_runs.iterrows():
        file_name = row.get("file")
        sensor_group = _extract_sensor_group_from_filename(file_name)
        noise_sigma = float(row.get("noise_sigma", np.nan))
        snr_vals = parse_peak_snr_series(row.get("peak_snr_db_values", ""))
        lbl_vals = parse_peak_label_series(row.get("peak_labels", ""))
        for i, snr_val in enumerate(snr_vals):
            crack_label = lbl_vals[i] if i < len(lbl_vals) else "Unknown"
            if crack_label not in CRACK_LABELS:
                continue
            if _use_linear_snr_display():
                snr_linear_val = float(snr_val)
            else:
                snr_linear_val = float(10 ** (float(snr_val) / 20.0))
            signal_amp = snr_linear_val * noise_sigma if np.isfinite(noise_sigma) else np.nan
            crack_records.append(
                {
                    "entity": sensor_group if use_grouped_display else file_name,
                    "crack_label": crack_label,
                    "peak_snr": float(snr_val),
                    "peak_signal_amplitude": float(signal_amp),
                }
            )

    per_crack_df = pd.DataFrame(crack_records)
    if per_crack_df.empty:
        ax_rank.text(0.5, 0.5, "No per-crack SNR data", ha="center", va="center")
        ax_rank.set_axis_off()
        ax_delta.text(0.5, 0.5, "No per-crack signal data", ha="center", va="center")
        ax_delta.set_axis_off()
    else:
        per_crack_summary = (
            per_crack_df.groupby(["entity", "crack_label"], as_index=False)
            .agg(
                snr_mean=("peak_snr", "mean"),
                snr_std=("peak_snr", "std"),
                signal_mean=("peak_signal_amplitude", "mean"),
                signal_std=("peak_signal_amplitude", "std"),
            )
        )
        per_crack_summary["snr_std"] = per_crack_summary["snr_std"].fillna(0.0)
        per_crack_summary["signal_std"] = per_crack_summary["signal_std"].fillna(0.0)

        entity_order = entity_df["entity"].tolist()
        crack_order = list(CRACK_LABELS)
        x = np.arange(len(entity_order), dtype=float)
        width = 0.24
        crack_colors = {
            crack_label: effective_label_color_map.get(crack_label, "#7f7f7f")
            for crack_label in crack_order
        }

        snr_pivot = per_crack_summary.pivot(index="entity", columns="crack_label", values="snr_mean")
        snr_std_pivot = per_crack_summary.pivot(index="entity", columns="crack_label", values="snr_std")
        sig_pivot = per_crack_summary.pivot(index="entity", columns="crack_label", values="signal_mean")
        sig_std_pivot = per_crack_summary.pivot(index="entity", columns="crack_label", values="signal_std")


        for idx, crack_label in enumerate(crack_order):
            offset = (idx - (len(crack_order) - 1) / 2.0) * width
            snr_vals = snr_pivot.reindex(entity_order).get(crack_label, pd.Series(index=entity_order, dtype=float)).fillna(0.0).to_numpy(dtype=float)
            snr_errs = snr_std_pivot.reindex(entity_order).get(crack_label, pd.Series(index=entity_order, dtype=float)).fillna(0.0).to_numpy(dtype=float)
            sig_vals = sig_pivot.reindex(entity_order).get(crack_label, pd.Series(index=entity_order, dtype=float)).fillna(0.0).to_numpy(dtype=float)
            sig_errs = sig_std_pivot.reindex(entity_order).get(crack_label, pd.Series(index=entity_order, dtype=float)).fillna(0.0).to_numpy(dtype=float)

            ax_rank.bar(
                x + offset,
                snr_vals,
                width=width,
                yerr=snr_errs,
                capsize=2,
                color=crack_colors.get(crack_label, "#7f7f7f"),
                alpha=0.85,
                label=crack_label,
            )
            ax_delta.bar(
                x + offset,
                sig_vals,
                width=width,
                yerr=sig_errs,
                capsize=2,
                color=crack_colors.get(crack_label, "#7f7f7f"),
                alpha=0.85,
                label=crack_label,
            )

        # Overlay individual crack amplitudes as semi-transparent small grey dots (upper-right panel)
        # Only if in grouped display mode
        if use_grouped_display:
            # Map entity to x position
            entity_x_map = {entity: i for i, entity in enumerate(entity_order)}
            # For reproducibility, use a fixed random seed
            rng = np.random.default_rng(42)
            for idx, row in per_crack_df.iterrows():
                entity = row["entity"]
                crack_label = row["crack_label"]
                amp = row["peak_signal_amplitude"]
                if entity in entity_x_map:
                    x_pos = entity_x_map[entity]
                    # Offset by crack index for visual separation
                    crack_idx = crack_order.index(crack_label) if crack_label in crack_order else 0
                    offset = (crack_idx - (len(crack_order) - 1) / 2.0) * width
                    # Add small random jitter to x position
                    jitter = rng.uniform(-0.09, 0.09)
                    ax_delta.scatter(
                        x_pos + offset + jitter,
                        amp,
                        color="#888888",
                        alpha=0.35,
                        s=18,
                        zorder=5,
                        linewidths=0.5,
                        edgecolors="none",
                    )

        ax_rank.set_xticks(x)
        ax_rank.set_xticklabels(entity_order, rotation=20, ha="right")
        ax_rank.set_title("Per-Crack SNR by Inductor")
        ax_rank.set_xlabel(x_label)
        ax_rank.set_ylabel(f"SNR ({unit_label})")
        ax_rank.grid(axis="y", alpha=0.25)
        ax_rank.legend(loc="best", title="Crack size")

        ax_delta.set_xticks(x)
        ax_delta.set_xticklabels(entity_order, rotation=20, ha="right")
        ax_delta.set_title("Per-Crack Signal Amplitude by Inductor")
        ax_delta.set_xlabel(x_label)
        ax_delta.set_ylabel("Amplitude")
        ax_delta.grid(axis="y", alpha=0.25)
        ax_delta.legend(loc="best", title="Crack size")

    if use_grouped_display:
        scatter = ax_signal_noise.scatter(
            grouped["noise_sigma_mean"], grouped["signal_amplitude_mean"],
            c=grouped["snr_mean"], cmap="viridis", s=80, alpha=0.9,
        )
        ax_signal_noise.set_title("Mean Signal vs Mean Noise by Sensor Group")
        ax_signal_noise.set_xlabel("Mean noise sigma")
        ax_signal_noise.set_ylabel("Mean signal amplitude")
        ax_signal_noise.grid(alpha=0.25)
        for _, row in grouped.iterrows():
            ax_signal_noise.annotate(
                row["sensor_group"],
                (row["noise_sigma_mean"], row["signal_amplitude_mean"]),
                fontsize=7,
            )
    else:
        scatter = ax_signal_noise.scatter(
            df_runs["noise_sigma"], df_runs["signal_amplitude"],
            c=df_runs[snr_col], cmap="viridis", s=80, alpha=0.9,
        )
        ax_signal_noise.set_title("Signal vs Noise by Run")
        ax_signal_noise.set_xlabel("Noise sigma")
        ax_signal_noise.set_ylabel("Signal amplitude")
        ax_signal_noise.grid(alpha=0.25)
        for _, row in df_runs.iterrows():
            ax_signal_noise.annotate(
                row["file"],
                (row["noise_sigma"], row["signal_amplitude"]),
                fontsize=7,
            )
    fig.colorbar(scatter, ax=ax_signal_noise, label=f"SNR ({unit_label})")

    df = df_runs
    peak_rank = np.arange(1, len(df) + 1)
    peak_points = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        snr_vals  = parse_peak_snr_series(row.get("peak_snr_db_values", ""))
        lbl_vals  = parse_peak_label_series(row.get("peak_labels", ""))
        for j, pv in enumerate(snr_vals):
            peak_points.append((i, pv, row["file"], lbl_vals[j] if j < len(lbl_vals) else "Unknown"))

    if peak_points:
        x_arr = np.array([pt[0] for pt in peak_points], dtype=float)
        y_arr = np.array([pt[1] for pt in peak_points], dtype=float)
        lbls  = [pt[3] for pt in peak_points]
        jitter = np.linspace(-0.12, 0.12, len(x_arr)) if len(x_arr) > 1 else np.array([0.0])

        crack_color_map = {**{k: v for k, v in effective_label_color_map.items()}, "Unknown": "#7f7f7f"}
        for crack_label in [*CRACK_LABELS, "Unknown"]:
            mask = np.array([lbl == crack_label for lbl in lbls], dtype=bool)
            if not np.any(mask):
                continue
            ax_peaks.scatter(x_arr[mask] + jitter[mask], y_arr[mask],
                             color=crack_color_map[crack_label], alpha=0.75, s=28,
                             label=crack_label)

        for i, file_name in enumerate(df["file"], start=1):
            run_vals = [pt[1] for pt in peak_points if pt[2] == file_name]
            if run_vals:
                ax_peaks.hlines(np.median(run_vals), i - 0.2, i + 0.2,
                                colors="#d62728", linewidth=2)

        ax_peaks.set_xticks(peak_rank)
        ax_peaks.set_xticklabels(df["file"].tolist(), rotation=45, ha="right")
        ax_peaks.set_title("Per-Peak SNR Distribution by Run")
        ax_peaks.set_xlabel("Run (rank-ordered)")
        ax_peaks.set_ylabel(f"Peak SNR ({unit_label})")
        if _use_linear_snr_display():
            ax_peaks.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.9)
        ax_peaks.grid(alpha=0.25)
        ax_peaks.legend(loc="best", title="Crack label")
    else:
        ax_peaks.text(0.5, 0.5, "No finite peak SNR values available", ha="center", va="center")
        ax_peaks.set_axis_off()

    plt.tight_layout()

    if save_plots:
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        overview_path = Path(output_dir) / f"snr_visual_summary_{timestamp}.png"
        fig.savefig(overview_path, dpi=200)
        plot_outputs.append(str(overview_path))
        print(f"Saved SNR visual summary: {overview_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return plot_outputs


# ---------------------------------------------------------------------------
# Shared animation helpers
# ---------------------------------------------------------------------------

def _compute_x_limits(x_vals, peak_annotations):
    """Return (x_min, x_max) axis limits centred around detected peaks."""
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    peak_xs = [x_vals[pk] for (pk, _, _) in peak_annotations if pk < len(x_vals)]
    if peak_xs:
        pk_min, pk_max = min(peak_xs), max(peak_xs)
        pk_span = pk_max - pk_min if pk_max > pk_min else (x_max - x_min) * 0.1
        pad = pk_span * 0.1875
        return pk_min - pad, pk_max + pad
    pad = (x_max - x_min) * 0.0625 if x_max > x_min else 1
    return x_min - pad, x_max + pad


def _compute_y_limits(y_min, y_max, top_multiplier=1.25):
    """Return (y_lo, y_hi) with proportional padding above for annotations."""
    y_range       = y_max - y_min if y_max > y_min else 1
    y_pad_bottom  = y_range * 0.05
    base_pad_top  = y_range * 0.15
    base_span     = y_range + y_pad_bottom + base_pad_top
    y_pad_top     = max(base_pad_top, base_span * top_multiplier - (y_range + y_pad_bottom))
    return y_min - y_pad_bottom, y_max + y_pad_top


def _resolve_std_column(df, base_col):
    """Return the std column name corresponding to base_col, or None."""
    candidates = [f"{base_col}_std", f"{base_col.strip()}_std"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    target = f"{base_col.strip()}_std"
    for col in df.columns:
        if str(col).strip() == target:
            return col
    return None


def _prepare_xy_with_std(dataframes, x_col, y_col, normalize=False):
    """
    Extract clean XY arrays plus optional Y std arrays for error-band plotting.

    Returns:
        List of tuples: (filename, x_vals, y_vals, y_std_vals_or_none)
    """
    prepared = []

    for filename, df in dataframes.items():
        x_col_found = resolve_first_existing_column(df, [x_col, x_col.strip()])
        y_col_found = resolve_first_existing_column(df, [y_col, y_col.strip()])

        if x_col_found is None or y_col_found is None:
            print(f"Skipping {filename}: required XY columns not found.")
            continue

        x = pd.to_numeric(df[x_col_found], errors="coerce")
        y = pd.to_numeric(df[y_col_found], errors="coerce")
        valid = x.notna() & y.notna()
        x_vals = x[valid].to_numpy(dtype=float)
        y_vals = y[valid].to_numpy(dtype=float)

        if len(x_vals) == 0:
            print(f"Skipping {filename}: no numeric XY pairs after cleaning.")
            continue

        std_col = _resolve_std_column(df, y_col_found)
        y_std_vals = None
        if std_col is not None:
            y_std_series = pd.to_numeric(df[std_col], errors="coerce").fillna(0.0)
            y_std_vals = np.clip(y_std_series[valid].to_numpy(dtype=float), a_min=0.0, a_max=None)

        if normalize:
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            y_min, y_max = np.min(y_vals), np.max(y_vals)
            x_scale = (x_max - x_min) if x_max > x_min else None
            y_scale = (y_max - y_min) if y_max > y_min else None
            x_vals = (x_vals - x_min) / x_scale if x_scale else np.zeros_like(x_vals)
            y_vals = (y_vals - y_min) / y_scale if y_scale else np.zeros_like(y_vals)
            if y_std_vals is not None:
                y_std_vals = y_std_vals / y_scale if y_scale else np.zeros_like(y_std_vals)

        prepared.append((filename, x_vals, y_vals, y_std_vals))

    return prepared


def _build_tail_update(tail_length):
    """Return an update closure for the fading-tail animation pattern."""
    def update_trace(frame, x_vals, y_vals, tail_segments, point,
                     base_color, peak_annotations, peak_markers, peak_texts):
        last_idx = min(frame, len(x_vals) - 1)
        start_idx = max(0, last_idx - tail_length + 1)
        tail_x = x_vals[start_idx: last_idx + 1]
        tail_y = y_vals[start_idx: last_idx + 1]

        if len(tail_x) > 1:
            pts      = np.column_stack([tail_x, tail_y])
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            n        = len(segments)
            alphas   = np.linspace(0.05, 1.0, n)
            colors   = np.tile(np.array(base_color), (n, 1))
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

        for (pk_idx, pk_snr, pk_label), pk_marker, pk_text in zip(
                peak_annotations, peak_markers, peak_texts):
            if pk_idx < len(x_vals) and frame >= pk_idx:
                px, py = x_vals[pk_idx], y_vals[pk_idx]
                pk_marker.set_data([px], [py])
                pk_text.set_position((px, py))
                pk_text.set_text(f"{pk_label} | {fmt_snr_display(pk_snr)}")
                pk_text.set_visible(True)
            else:
                pk_marker.set_data([], [])
                pk_text.set_visible(False)

    return update_trace


def _add_peak_artists(ax, base_color, peak_annotations):
    """Create and return (peak_markers, peak_texts) for one trace."""
    peak_markers, peak_texts = [], []
    for _ in peak_annotations:
        marker, = ax.plot([], [], marker="^", ms=6, color=base_color, zorder=4, alpha=0.9)
        text = ax.text(
            0, 0, "", fontsize=8, color=base_color, ha="left", va="bottom",
            zorder=5, visible=False,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
        )
        peak_markers.append(marker)
        peak_texts.append(text)
    return peak_markers, peak_texts


def _save_and_close_animation(fig, anim, output_dir, gif_name, interval, frame_step,
                               save_gif, show_plot, attr_name):
    """Save the animation GIF (if requested), then show or close the figure."""
    if save_gif:
        gif_path = Path(output_dir) / gif_name
        fps = max(1, int(round(1000 / interval / max(1, frame_step))))
        anim.save(str(gif_path), writer="pillow", fps=fps)
        print(f"Saved animation GIF: {gif_path}")

    setattr(fig, attr_name, anim)   # keep a reference alive for the figure lifetime

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Overlay animation (all traces on one axes, normalized)
# ---------------------------------------------------------------------------

def create_overlay_animation(
    dataframes,
    x_col="sensor1",
    y_col=" sensor2",
    snr_df=None,
    interval=30,
    tail_length=100,
    frame_step=1,
    output_dir=".",
    save_gif=True,
    show_plot=True,
):
    """
    Animate overlaid, normalized XY traces from all loaded files.

    Each file is drawn in a distinct color with a fading tail.
    Noise-floor bands and detected crack peaks are annotated per trace.
    """
    if not dataframes:
        print("No dataframes available for animation.")
        return

    prepared = _prepare_xy_with_std(dataframes, x_col, y_col, normalize=True)
    if not prepared:
        print("No valid files had the required columns for animation.")
        return

    global_x_min = min(np.min(xv) for _, xv, _, _ in prepared)
    global_x_max = max(np.max(xv) for _, xv, _, _ in prepared)
    global_y_min = min(np.min(yv) for _, _, yv, _ in prepared)
    global_y_max = max(np.max(yv) for _, _, yv, _ in prepared)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Overlay Animation (Normalized): sensor1_smooth_rot vs sensor2_smooth_rot")
    ax.set_xlabel("R_p (normalized)")
    ax.set_ylabel("Inductance (normalized)")

    all_peak_x = [
        xv[pk] for fname, xv, _, _ in prepared
        for (pk, _, _) in get_peak_annotations(snr_df, fname)
        if pk < len(xv)
    ]
    if all_peak_x:
        pk_span = max(all_peak_x) - min(all_peak_x) if max(all_peak_x) > min(all_peak_x) \
                  else (global_x_max - global_x_min) * 0.1
        x_pad = pk_span * 0.1875
        display_x_min = min(all_peak_x) - x_pad
        display_x_max = max(all_peak_x) + x_pad
    else:
        x_pad = (global_x_max - global_x_min) * 0.0625 if global_x_max > global_x_min else 1
        display_x_min = global_x_min - x_pad
        display_x_max = global_x_max + x_pad

    y_range       = global_y_max - global_y_min if global_y_max > global_y_min else 1
    y_pad_bottom  = y_range * 0.05
    base_pad_top  = y_range * 0.05
    y_pad_top     = max(base_pad_top, (y_range + y_pad_bottom + base_pad_top) * 1.25
                        - (y_range + y_pad_bottom))
    ax.set_xlim(display_x_min, display_x_max)
    ax.set_ylim(global_y_min - y_pad_bottom, global_y_max + y_pad_top)

    # Generate colors based on inductor type
    inductor_color_map = _get_inductor_color_map([fname for fname, _, _, _ in prepared])
    update_func = _build_tail_update(tail_length)
    artists     = []
    max_frames  = max(len(xv) for _, xv, _, _ in prepared)

    for i, (filename, xv, yv, y_std) in enumerate(prepared):
        base_color = inductor_color_map.get(filename, plt.get_cmap("tab10")(i % 10))

        if y_std is not None and len(y_std) == len(yv):
            ax.fill_between(
                xv,
                yv - y_std,
                yv + y_std,
                color=base_color,
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )

        y_floor = float(np.median(yv))
        y_sigma = 1.4826 * float(np.median(np.abs(yv - y_floor)))
        if y_sigma <= 0:
            y_sigma = float(np.std(yv))
        ax.axhline(y_floor - y_sigma, color=base_color, linestyle="--", linewidth=1.0, alpha=0.45)
        ax.axhline(y_floor + y_sigma, color=base_color, linestyle="--", linewidth=1.0, alpha=0.45)

        snr_value      = get_snr_value_display(snr_df, filename)
        peak_count     = get_peak_count(snr_df, filename)
        peak_annotations = get_peak_annotations(snr_df, filename)
        label = (f"{filename} | Crack SNR: {fmt_snr_display(snr_value)} | "
                 f"Peaks: {peak_count if peak_count is not None else 'n/a'}")

        tail_segments = LineCollection([], linewidths=2.0, zorder=2)
        ax.add_collection(tail_segments)
        point, = ax.plot([], [], marker="o", ms=5, color=base_color, zorder=3, label=label)
        peak_markers, peak_texts = _add_peak_artists(ax, base_color, peak_annotations)
        artists.append((tail_segments, point, base_color, peak_annotations, peak_markers, peak_texts))

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, fontsize=9)
    ax.grid(True, alpha=0.3)

    def init():
        for ts, pt, _, _, pms, ptxs in artists:
            ts.set_segments([])
            ts.set_color([])
            pt.set_data([], [])
            for pm, ptx in zip(pms, ptxs):
                pm.set_data([], [])
                ptx.set_visible(False)
        return [a for ts, pt, _, _, pms, ptxs in artists
                for a in (ts, pt, *pms, *ptxs)]

    def update(frame):
        for (_, xv, yv, _), (ts, pt, col, pa, pms, ptxs) in zip(prepared, artists):
            update_func(frame, xv, yv, ts, pt, col, pa, pms, ptxs)
        return [a for ts, pt, _, _, pms, ptxs in artists
                for a in (ts, pt, *pms, *ptxs)]

    anim = FuncAnimation(fig, update, frames=range(0, max_frames, max(1, frame_step)),
                         init_func=init, interval=interval, blit=True, repeat=False)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_and_close_animation(fig, anim, output_dir, f"overlay_animation_{timestamp}.gif",
                               interval, frame_step, save_gif, show_plot, "_overlay_anim")


# ---------------------------------------------------------------------------
# Stacked animation (one subplot per file, raw units)
# ---------------------------------------------------------------------------

def create_stacked_animation(
    dataframes,
    x_col="sensor1",
    y_col=" sensor2",
    snr_df=None,
    interval=30,
    tail_length=100,
    frame_step=1,
    output_dir=".",
    save_gif=True,
    show_plot=True,
):
    """Animate one trace per vertically stacked subplot using normalized units."""
    if not dataframes:
        print("No dataframes available for stacked animation.")
        return

    prepared = _prepare_xy_with_std(dataframes, x_col, y_col, normalize=True)
    if not prepared:
        print("No valid files had the required columns for stacked animation.")
        return

    grouped = {}
    for row in prepared:
        fname = row[0]
        grouped.setdefault(_extract_sensor_group_from_filename(fname), []).append(row)
    grouped_items = sorted(grouped.items(), key=lambda kv: kv[0])

    n = len(grouped_items)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(3, 3.4 * n)), squeeze=False)
    axes = axes.flatten()
    fig.suptitle("Stacked Trace Animation (Normalized, Faceted by Inductor)")

    # Generate colors based on inductor type
    inductor_color_map = _get_inductor_color_map([fname for fname, _, _, _ in prepared])
    update_func = _build_tail_update(tail_length)
    artists     = []
    max_frames  = max(len(xv) for _, xv, _, _ in prepared)

    for ax, (inductor_name, rows) in zip(axes, grouped_items):
        group_x_min = min(np.min(xv) for _, xv, _, _ in rows)
        group_x_max = max(np.max(xv) for _, xv, _, _ in rows)
        group_y_min = min(np.min(yv) for _, _, yv, _ in rows)
        group_y_max = max(np.max(yv) for _, _, yv, _ in rows)

        group_peak_x = [
            xv[pk]
            for fname, xv, _, _ in rows
            for (pk, _, _) in get_peak_annotations(snr_df, fname)
            if pk < len(xv)
        ]
        if group_peak_x:
            x_span = max(group_peak_x) - min(group_peak_x)
            x_pad = (x_span if x_span > 0 else (group_x_max - group_x_min)) * 0.1875
            ax_x_min = min(group_peak_x) - x_pad
            ax_x_max = max(group_peak_x) + x_pad
        else:
            x_pad = (group_x_max - group_x_min) * 0.0625 if group_x_max > group_x_min else 1
            ax_x_min = group_x_min - x_pad
            ax_x_max = group_x_max + x_pad

        y_lo, y_hi = _compute_y_limits(group_y_min, group_y_max)
        ax.set_xlim(ax_x_min, ax_x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{inductor_name} ({len(rows)} file{'s' if len(rows) != 1 else ''})")
        ax.set_xlabel("R_p (normalized)")
        ax.set_ylabel("Inductance (normalized)")
        ax.grid(True, alpha=0.3)

        for i, (filename, x_vals, y_vals, y_std) in enumerate(rows):
            base_color       = inductor_color_map.get(filename, plt.get_cmap("tab10")(i % 10))
            peak_annotations = get_peak_annotations(snr_df, filename)
            snr_value        = get_snr_value_display(snr_df, filename)
            peak_count       = get_peak_count(snr_df, filename)

            if y_std is not None and len(y_std) == len(y_vals):
                ax.fill_between(
                    x_vals,
                    y_vals - y_std,
                    y_vals + y_std,
                    color=base_color,
                    alpha=0.14,
                    linewidth=0,
                    zorder=1,
                )

            label = (
                f"{filename} | SNR: {fmt_snr_display(snr_value)} | "
                f"Peaks: {peak_count if peak_count is not None else 'n/a'}"
            )
            tail_segments = LineCollection([], linewidths=2.0, zorder=2)
            ax.add_collection(tail_segments)
            point, = ax.plot([], [], marker="o", ms=5, color=base_color, zorder=3, label=label)
            peak_markers, peak_texts = _add_peak_artists(ax, base_color, peak_annotations)
            artists.append((x_vals, y_vals, tail_segments, point, base_color,
                            peak_annotations, peak_markers, peak_texts))

        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0, fontsize=8)

    def init():
        for _, _, ts, pt, _, _, pms, ptxs in artists:
            ts.set_segments([])
            ts.set_color([])
            pt.set_data([], [])
            for pm, ptx in zip(pms, ptxs):
                pm.set_data([], [])
                ptx.set_visible(False)
        return [a for _, _, ts, pt, _, _, pms, ptxs in artists
                for a in (ts, pt, *pms, *ptxs)]

    def update(frame):
        for xv, yv, ts, pt, col, pa, pms, ptxs in artists:
            update_func(frame, xv, yv, ts, pt, col, pa, pms, ptxs)
        return [a for _, _, ts, pt, _, _, pms, ptxs in artists
                for a in (ts, pt, *pms, *ptxs)]

    anim = FuncAnimation(fig, update, frames=range(0, max_frames, max(1, frame_step)),
                         init_func=init, interval=interval, blit=True, repeat=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_and_close_animation(fig, anim, output_dir, f"stacked_animation_{timestamp}.gif",
                               interval, frame_step, save_gif, show_plot, "_stacked_anim")


# ---------------------------------------------------------------------------
# Stacked baseline animation (one subplot per file, baseline-shifted)
# ---------------------------------------------------------------------------

def create_stacked_baseline_animation(
    dataframes,
    x_col="sensor1",
    y_col=" sensor2",
    snr_df=None,
    interval=30,
    tail_length=100,
    frame_step=1,
    output_dir=".",
    save_gif=True,
    show_plot=True,
):
    """
    Animate one trace per stacked subplot after subtracting each file's own
    baseline (x and y are both shifted by their respective minima).
    """
    if not dataframes:
        print("No dataframes available for baseline stacked animation.")
        return

    prepared_raw = _prepare_xy_with_std(dataframes, x_col, y_col, normalize=False)
    if not prepared_raw:
        print("No valid files had the required columns for baseline stacked animation.")
        return

    prepared = [
        (fname, xv - np.min(xv), yv - np.min(yv), y_std)
        for fname, xv, yv, y_std in prepared_raw
    ]

    global_y_min = min(np.min(yv) for _, _, yv, _ in prepared)
    global_y_max = max(np.max(yv) for _, _, yv, _ in prepared)
    global_y_pad = (global_y_max - global_y_min) * 0.05 if global_y_max > global_y_min else 1

    grouped = {}
    for row in prepared:
        fname = row[0]
        grouped.setdefault(_extract_sensor_group_from_filename(fname), []).append(row)
    grouped_items = sorted(grouped.items(), key=lambda kv: kv[0])

    n = len(grouped_items)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(3, 3.4 * n)),
                             squeeze=False, sharey=True)
    axes = axes.flatten()
    fig.suptitle("Stacked Trace Animation (Baseline-Shifted, Faceted by Inductor)")

    # Generate colors based on inductor type
    inductor_color_map = _get_inductor_color_map([fname for fname, _, _, _ in prepared])
    update_func = _build_tail_update(tail_length)
    artists     = []
    max_frames  = max(len(xv) for _, xv, _, _ in prepared)

    for ax, (inductor_name, rows) in zip(axes, grouped_items):
        group_x_min = min(np.min(xv) for _, xv, _, _ in rows)
        group_x_max = max(np.max(xv) for _, xv, _, _ in rows)
        group_peak_x = [
            xv[pk]
            for fname, xv, _, _ in rows
            for (pk, _, _) in get_peak_annotations(snr_df, fname)
            if pk < len(xv)
        ]
        if group_peak_x:
            x_span = max(group_peak_x) - min(group_peak_x)
            x_pad = (x_span if x_span > 0 else (group_x_max - group_x_min)) * 0.1875
            ax_x_min = min(group_peak_x) - x_pad
            ax_x_max = max(group_peak_x) + x_pad
        else:
            x_pad = (group_x_max - group_x_min) * 0.0625 if group_x_max > group_x_min else 1
            ax_x_min = group_x_min - x_pad
            ax_x_max = group_x_max + x_pad

        ax.set_xlim(ax_x_min, ax_x_max)
        ax.set_ylim(global_y_min - global_y_pad, global_y_max + (global_y_pad * 3.0))
        ax.set_title(f"{inductor_name} ({len(rows)} file{'s' if len(rows) != 1 else ''}, baseline-shifted)")
        ax.set_xlabel("R_p (ohm)")
        ax.set_ylabel("Inductance (uH)")
        ax.grid(True, alpha=0.3)

        for i, (filename, x_vals, y_vals, y_std) in enumerate(rows):
            base_color       = inductor_color_map.get(filename, plt.get_cmap("tab10")(i % 10))
            peak_annotations = get_peak_annotations(snr_df, filename)
            snr_value        = get_snr_value_display(snr_df, filename)
            peak_count       = get_peak_count(snr_df, filename)

            if y_std is not None and len(y_std) == len(y_vals):
                ax.fill_between(
                    x_vals,
                    y_vals - y_std,
                    y_vals + y_std,
                    color=base_color,
                    alpha=0.14,
                    linewidth=0,
                    zorder=1,
                )

            label = (
                f"{filename} | SNR: {fmt_snr_display(snr_value)} | "
                f"Peaks: {peak_count if peak_count is not None else 'n/a'}"
            )
            tail_segments = LineCollection([], linewidths=2.0, zorder=2)
            ax.add_collection(tail_segments)
            point, = ax.plot([], [], marker="o", ms=5, color=base_color, zorder=3, label=label)
            peak_markers, peak_texts = _add_peak_artists(ax, base_color, peak_annotations)
            artists.append((x_vals, y_vals, tail_segments, point, base_color,
                            peak_annotations, peak_markers, peak_texts))

        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0, fontsize=8)

    def init():
        for _, _, ts, pt, _, _, pms, ptxs in artists:
            ts.set_segments([])
            ts.set_color([])
            pt.set_data([], [])
            for pm, ptx in zip(pms, ptxs):
                pm.set_data([], [])
                ptx.set_visible(False)
        return [a for _, _, ts, pt, _, _, pms, ptxs in artists
                for a in (ts, pt, *pms, *ptxs)]

    def update(frame):
        for xv, yv, ts, pt, col, pa, pms, ptxs in artists:
            update_func(frame, xv, yv, ts, pt, col, pa, pms, ptxs)
        return [a for _, _, ts, pt, _, _, pms, ptxs in artists
                for a in (ts, pt, *pms, *ptxs)]

    anim = FuncAnimation(fig, update, frames=range(0, max_frames, max(1, frame_step)),
                         init_func=init, interval=interval, blit=True, repeat=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_and_close_animation(fig, anim, output_dir,
                               f"stacked_baseline_animation_{timestamp}.gif",
                               interval, frame_step, save_gif, show_plot,
                               "_stacked_baseline_anim")


# ---------------------------------------------------------------------------
# 3-D overlay animation (time × R_p × Inductance)
# ---------------------------------------------------------------------------

def _rotate_points_2d(points, degrees):
    """Rotate 2D points by degrees."""
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(points, R.T)


def _rotate_yz_arrays(y_data, z_data, degrees, cy=0.0, cz=0.0, sy=1.0, sz=1.0):
    """
    Rotate (y, z) pairs by angle while preserving axis-scale behaviour.

    Rotation is done in normalized space, then mapped back.
    """
    if len(y_data) == 0:
        return np.asarray(y_data, dtype=float), np.asarray(z_data, dtype=float)

    points = np.column_stack((np.asarray(y_data, dtype=float), np.asarray(z_data, dtype=float)))
    centered = points - np.array([cy, cz], dtype=float)
    normalized = centered / np.array([sy, sz], dtype=float)
    rotated_norm = _rotate_points_2d(normalized, -degrees)
    rotated = rotated_norm * np.array([sy, sz], dtype=float) + np.array([cy, cz], dtype=float)
    return rotated[:, 0], rotated[:, 1]


def _get_yz_rotation_params(y_vals, z_vals):
    """Return center/scale parameters used for stable y/z rotation."""
    if len(y_vals) == 0:
        return 0.0, 0.0, 1.0, 1.0
    cy = float(np.mean(y_vals))
    cz = float(np.mean(z_vals))
    sy = float(np.ptp(y_vals))
    sz = float(np.ptp(z_vals))
    return cy, cz, (sy if sy > 0 else 1.0), (sz if sz > 0 else 1.0)


def create_interactive_raw_3d_transform_preview(
    dataframes,
    time_col="timestamp",
    rp_col="sensor1_smooth",
    ind_col="sensor2_smooth",
    interval=30,
    frame_step=1,
    show_plot=True,
    source_path_lookup=None,
    rotation_cache_dir="rotation_cache",
):
    """
    Interactive real-time 3D preview of one CSV at a time.

    Uses smoothed, non-rotated channels for visualization by default
    (sensor1_smooth / sensor2_smooth). x-axis (time) stays fixed while
    y/z are transformed with a live angle control.

    Per file, pressing the "overwrite rotated data" button writes the
    transformed y/z values into the rotated columns used downstream
    (X_COL / Y_COL). Closing the window without pressing the button keeps
    the existing rotated data from that CSV.
    """
    if not dataframes:
        print("No dataframes available for interactive raw 3D preview.")
        return

    time_candidates = [time_col, "time", "sample", "index"]
    rp_candidates = [rp_col, rp_col.strip(), "sensor1_smooth", " sensor1_smooth", "sensor1"]
    ind_candidates = [ind_col, ind_col.strip(), "sensor2_smooth", " sensor2_smooth", "sensor2"]

    any_valid = False

    for file_idx, (filename, df) in enumerate(dataframes.items(), start=1):
        source_path = source_path_lookup.get(filename) if source_path_lookup else None
        fingerprint = get_file_fingerprint(source_path) if source_path else None
        rotation_cache_path = (
            get_rotation_cache_path(rotation_cache_dir, source_path) if source_path else None
        )
        cached_angle, cached_smooth = (None, None)
        if rotation_cache_path is not None:
            cache_result = load_cached_rotation(rotation_cache_path, fingerprint)
            if isinstance(cache_result, tuple):
                cached_angle, cached_smooth = cache_result
            else:
                cached_angle = cache_result
                cached_smooth = None

        t_col = resolve_first_existing_column(df, time_candidates)
        y_col = resolve_first_existing_column(df, rp_candidates)
        z_col = resolve_first_existing_column(df, ind_candidates)

        if y_col is None or z_col is None:
            print(f"Raw 3D preview: required columns not found in '{filename}', skipping.")
            continue

        t_series = pd.to_numeric(df[t_col], errors="coerce") if t_col else pd.Series(np.arange(len(df), dtype=float), index=df.index)
        y_series = pd.to_numeric(df[y_col], errors="coerce")
        z_series = pd.to_numeric(df[z_col], errors="coerce")
        valid = t_series.notna() & y_series.notna() & z_series.notna()

        if int(np.count_nonzero(valid.to_numpy())) < 3:
            print(f"Raw 3D preview: insufficient numeric samples in '{filename}', skipping.")
            continue

        any_valid = True
        t_vals = t_series[valid].to_numpy(dtype=float)
        y_raw = y_series[valid].to_numpy(dtype=float)
        z_raw = z_series[valid].to_numpy(dtype=float)

        # Smoothing window state
        initial_smooth = int(cached_smooth) if cached_smooth is not None else 0
        smooth_state = {"window": initial_smooth}


        def smooth(arr, window):
            if window <= 0:
                return arr
            window = int(window)
            if window < 1:
                return arr
            # Use pandas rolling mean for better edge handling
            import pandas as pd
            return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()

        y_vals = smooth(y_raw, smooth_state["window"])
        z_vals = smooth(z_raw, smooth_state["window"])
        rot_params = _get_yz_rotation_params(y_vals, z_vals)

        # Determine the rotation angle to apply (cached, or None if not available)
        angle_to_apply = cached_angle if cached_angle is not None else None

        # Only create/overwrite the rotated columns if we have an explicit rotation
        # (either from cache or from user interaction in the preview window).
        # If neither exists, leave the original CSV's rotated columns as-is.
        if angle_to_apply is not None:
            y_rot_applied, z_rot_applied = _rotate_yz_arrays(y_vals, z_vals, angle_to_apply, *rot_params)
            if X_COL not in df.columns:
                df[X_COL] = np.nan
            if Y_COL not in df.columns:
                df[Y_COL] = np.nan
            df.loc[valid, X_COL] = y_rot_applied
            df.loc[valid, Y_COL] = z_rot_applied
            dataframes[filename] = df
            print(
                f"Applied cached rotation for {filename}: {angle_to_apply:.1f}°, smoothing={smooth_state['window']} "
                f"-> '{X_COL}', '{Y_COL}'."
            )

        if not show_plot:
            continue

        fig = plt.figure(figsize=(15, 8))
        grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0])
        ax = fig.add_subplot(grid[0, 0], projection="3d")
        ax_plane = fig.add_subplot(grid[0, 1])
        fig.subplots_adjust(bottom=0.21)

        ax.set_xlabel("Time")
        ax.set_ylabel("R_p (smoothed)")
        ax.set_zlabel("Inductance (smoothed)")
        ax_plane.set_xlabel("R_p (smoothed, transformed)")
        ax_plane.set_ylabel("Inductance (smoothed, transformed)")
        ax_plane.set_title("Top-Down Y/Z Plane")
        ax_plane.grid(alpha=0.25)

        initial_angle = float(cached_angle) if cached_angle is not None else 0.0
        angle_state = {"deg": initial_angle}
        interaction_state = {"angle_changed": False}
        overwrite_state = {"selected": False}
        base_color = plt.get_cmap("tab10", max(len(dataframes), 1))((file_idx - 1) % 10)

        # Ghost trace for reference (smoothed, non-rotated source).
        ax.plot(t_vals, y_vals, z_vals, color=base_color, linewidth=0.9, alpha=0.18)

        y_rot0, z_rot0 = _rotate_yz_arrays(y_vals, z_vals, angle_state["deg"], *rot_params)
        N_s = len(t_vals)
        T_s = np.vstack([t_vals, t_vals])
        Y_s = np.vstack([y_rot0, y_rot0])
        z_floor0 = float(np.min(z_rot0))
        Z_s = np.vstack([np.full(N_s, z_floor0), z_rot0])

        surf = ax.plot_surface(
            T_s,
            Y_s,
            Z_s,
            facecolor=base_color,
            alpha=0.14,
            linewidth=0,
            antialiased=False,
        )
        contour = ax.contourf(
            T_s,
            Y_s,
            Z_s,
            zdir="z",
            offset=z_floor0,
            levels=8,
            colors=[base_color],
            alpha=0.08,
        )

        line, = ax.plot([], [], [], color=base_color, linewidth=2.2, label=filename)
        point, = ax.plot([], [], [], marker="o", ms=5, color=base_color, linestyle="none")
        plane_ghost, = ax_plane.plot(y_vals, z_vals, color=base_color, linewidth=0.9, alpha=0.18)
        plane_line, = ax_plane.plot([], [], color=base_color, linewidth=2.2)
        plane_point, = ax_plane.plot([], [], marker="o", ms=5, color=base_color, linestyle="none")
        y_pad = max(np.ptp(y_rot0) * 0.08, 1e-9)
        z_pad = max(np.ptp(z_rot0) * 0.08, 1e-9)
        ax_plane.set_xlim(np.min(y_rot0) - y_pad, np.max(y_rot0) + y_pad)
        ax_plane.set_ylim(np.min(z_rot0) - z_pad, np.max(z_rot0) + z_pad)
        ax.legend(loc="upper left", fontsize=8)



        # Place the angle slider above, smoothing slider below
        slider_ax = fig.add_axes([0.14, 0.13, 0.58, 0.035])
        angle_slider = Slider(
            slider_ax,
            "Y/Z Transform (deg)",
            -180.0,
            180.0,
            valinit=angle_state["deg"],
            valstep=0.1,
        )

        smooth_ax = fig.add_axes([0.14, 0.08, 0.58, 0.035])
        smooth_slider = Slider(
            smooth_ax,
            "Smoothing (window)",
            0,
            100,
            valinit=smooth_state["window"],
            valstep=1,
        )

        button_ax = fig.add_axes([0.75, 0.07, 0.2, 0.06])
        overwrite_button = Button(button_ax, "overwrite rotated data")

        fig.text(
            0.02,
            0.02,
            "Adjust angle/smoothing + close to apply/save, or click overwrite rotated data.",
            fontsize=9,
            color="#222222",
        )

        def on_angle_change(val):
            angle_state["deg"] = float(val)
            interaction_state["angle_changed"] = True
            # Update smoothed/rotated data
            update(0)

        def on_smooth_change(val):
            smooth_state["window"] = int(val)
            interaction_state["angle_changed"] = True
            # Update smoothed/rotated data
            nonlocal y_vals, z_vals, rot_params
            y_vals = smooth(y_raw, smooth_state["window"])
            z_vals = smooth(z_raw, smooth_state["window"])
            rot_params = _get_yz_rotation_params(y_vals, z_vals)
            update(0)

        def on_overwrite_click(_event):
            overwrite_state["selected"] = True
            plt.close(fig)

        angle_slider.on_changed(on_angle_change)
        smooth_slider.on_changed(on_smooth_change)
        overwrite_button.on_clicked(on_overwrite_click)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            plane_line.set_data([], [])
            plane_point.set_data([], [])
            return []

        def update(frame):
            nonlocal surf, contour

            angle_deg = angle_state["deg"]
            last_idx = min(frame, len(t_vals) - 1)
            if last_idx < 0:
                return []

            cy, cz, sy, sz = rot_params
            y_rot, z_rot = _rotate_yz_arrays(y_vals, z_vals, angle_deg, cy, cz, sy, sz)

            if surf is not None:
                surf.remove()
            if contour is not None:
                if hasattr(contour, "remove"):
                    contour.remove()
                elif hasattr(contour, "collections"):
                    for coll in contour.collections:
                        coll.remove()

            N_r = len(t_vals)
            T_r = np.vstack([t_vals, t_vals])
            Y_r = np.vstack([y_rot, y_rot])
            z_floor = float(np.min(z_rot))
            Z_r = np.vstack([np.full(N_r, z_floor), z_rot])

            surf = ax.plot_surface(
                T_r,
                Y_r,
                Z_r,
                facecolor=base_color,
                alpha=0.14,
                linewidth=0,
                antialiased=False,
            )
            contour = ax.contourf(
                T_r,
                Y_r,
                Z_r,
                zdir="z",
                offset=z_floor,
                levels=8,
                colors=[base_color],
                alpha=0.08,
            )

            t_seg = t_vals[: last_idx + 1]
            y_seg = y_rot[: last_idx + 1]
            z_seg = z_rot[: last_idx + 1]

            line.set_data(t_seg, y_seg)
            line.set_3d_properties(z_seg)
            point.set_data([t_seg[-1]], [y_seg[-1]])
            point.set_3d_properties([z_seg[-1]])
            plane_line.set_data(y_seg, z_seg)
            plane_point.set_data([y_seg[-1]], [z_seg[-1]])

            y_pad_dyn = max(np.ptp(y_rot) * 0.08, 1e-9)
            z_pad_dyn = max(np.ptp(z_rot) * 0.08, 1e-9)
            ax_plane.set_xlim(np.min(y_rot) - y_pad_dyn, np.max(y_rot) + y_pad_dyn)
            ax_plane.set_ylim(np.min(z_rot) - z_pad_dyn, np.max(z_rot) + z_pad_dyn)

            ax.set_title(
                f"{filename} | Smoothed non-rotated source | Angle: {angle_deg:.1f}°"
            )
            return []

        anim = FuncAnimation(
            fig,
            update,
            frames=range(0, len(t_vals), max(1, frame_step)),
            init_func=init,
            interval=interval,
            blit=False,
            repeat=True,
        )

        fig._raw_3d_preview_anim = anim
        plt.show()

        should_commit_rotation = overwrite_state["selected"] or interaction_state["angle_changed"]

        if should_commit_rotation:
            final_angle = float(angle_state["deg"])
            final_smooth = int(smooth_state["window"])
            y_vals_final = smooth(y_raw, final_smooth)
            z_vals_final = smooth(z_raw, final_smooth)
            y_rot_final, z_rot_final = _rotate_yz_arrays(y_vals_final, z_vals_final, final_angle, *_get_yz_rotation_params(y_vals_final, z_vals_final))

            if X_COL not in df.columns:
                df[X_COL] = np.nan
            if Y_COL not in df.columns:
                df[Y_COL] = np.nan

            df.loc[valid, X_COL] = y_rot_final
            df.loc[valid, Y_COL] = z_rot_final
            dataframes[filename] = df
            if rotation_cache_path is not None and fingerprint is not None:
                save_cached_rotation(rotation_cache_path, fingerprint, final_angle, final_smooth)
            commit_reason = "overwrite" if overwrite_state["selected"] else "slider-change"
            print(
                f"Applied rotation for {filename} ({commit_reason}): "
                f"wrote transformed data to '{X_COL}' and '{Y_COL}' at {final_angle:.1f}°, smoothing={final_smooth}."
            )
        else:
            print(f"Kept existing rotated data for {filename}.")

    if not any_valid:
        print("No valid files for interactive raw 3D preview.")


def create_3d_overlay_animation(
    dataframes,
    time_col="timestamp",
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
    Animate overlaid 3-D traces where:
      X-axis = time, normalised per-file to [0, 1]
      Y-axis = R_p  (sensor1_smooth_rot, raw units)
      Z-axis = Inductance (sensor2_smooth_rot, normalised per-file to [0, 1])

    A faint ghost trace is drawn first for context, then a fading-tail
    point sweeps along each file's path.  Detected crack peaks appear as
    triangle markers when the animation reaches their position.

    Note: 3-D animations cannot use matplotlib blitting, so they render
    more slowly than the 2-D animations — use frame_step > 1 if needed.
    """
    if not dataframes:
        print("No dataframes available for 3D overlay animation.")
        return

    time_candidates = [time_col, "time", "sample", "index"]
    rp_candidates   = [x_col, x_col.strip(), "sensor1", " sensor1"]
    ind_candidates  = [y_col, y_col.strip(), "sensor2", " sensor2"]

    # ── load all three channels per file ──────────────────────────────────
    prepared = []
    for filename, df in dataframes.items():
        t_col_found = resolve_first_existing_column(df, time_candidates)
        rp_col      = resolve_first_existing_column(df, rp_candidates)
        ind_col     = resolve_first_existing_column(df, ind_candidates)

        if rp_col is None or ind_col is None:
            print(f"3D overlay animation: required columns not found in '{filename}', skipping.")
            continue

        rp_raw  = df[rp_col].to_numpy(dtype=float)
        ind_raw = df[ind_col].to_numpy(dtype=float)
        t_raw   = (df[t_col_found].to_numpy(dtype=float) if t_col_found
                   else np.arange(len(rp_raw), dtype=float))

        t_min, t_max = t_raw.min(), t_raw.max()
        t_vals = (t_raw - t_min) / (t_max - t_min) if t_max > t_min else t_raw.copy()

        ind_min, ind_max = ind_raw.min(), ind_raw.max()
        ind_vals = (ind_raw - ind_min) / (ind_max - ind_min) if ind_max > ind_min else ind_raw.copy()

        ind_std_col = _resolve_std_column(df, ind_col)
        ind_std_vals = None
        if ind_std_col is not None:
            ind_std_raw = pd.to_numeric(df[ind_std_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ind_scale = (ind_max - ind_min) if ind_max > ind_min else None
            ind_std_vals = (np.clip(ind_std_raw, a_min=0.0, a_max=None) / ind_scale) if ind_scale else np.zeros_like(ind_std_raw)

        prepared.append((filename, t_vals, rp_raw, ind_vals, ind_std_vals))

    if not prepared:
        print("No valid files for 3D overlay animation.")
        return

    # ── build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Time (norm.)")
    ax.set_ylabel("R_p (ohm)")
    ax.set_zlabel("Inductance (norm.)")
    ax.set_title("3D Overlay Animation")

    # Generate colors based on inductor type
    inductor_color_map = _get_inductor_color_map(list(dataframes.keys()))
    artists    = []
    max_frames = max(len(tv) for _, tv, _, _, _ in prepared)

    for i, (filename, t_vals, rp_vals, ind_vals, ind_std_vals) in enumerate(prepared):
        base_color = inductor_color_map.get(filename, plt.get_cmap("tab10")(i % 10))
        N          = len(t_vals)

        peak_annotations = get_peak_annotations(snr_df, filename)
        snr_value        = get_snr_value_display(snr_df, filename)
        peak_count       = get_peak_count(snr_df, filename)

        # Ghost trace for context
        ax.plot(t_vals, rp_vals, ind_vals,
                color=base_color, linewidth=0.8, alpha=0.18, zorder=1)

        if ind_std_vals is not None and len(ind_std_vals) == len(ind_vals):
            z_lower = np.clip(ind_vals - ind_std_vals, 0.0, 1.0)
            z_upper = np.clip(ind_vals + ind_std_vals, 0.0, 1.0)
            ax.plot(t_vals, rp_vals, z_lower, color=base_color, linewidth=0.6, alpha=0.22, zorder=1)
            ax.plot(t_vals, rp_vals, z_upper, color=base_color, linewidth=0.6, alpha=0.22, zorder=1)
            T_std = np.vstack([t_vals, t_vals])
            R_std = np.vstack([rp_vals, rp_vals])
            Z_std = np.vstack([z_lower, z_upper])
            ax.plot_surface(T_std, R_std, Z_std, facecolor=base_color, alpha=0.08,
                    linewidth=0, antialiased=False)

        # Filled curtain under trace + floor projection (matches 3D contour style)
        N_s   = len(t_vals)
        T_s   = np.vstack([t_vals,  t_vals])
        R_s   = np.vstack([rp_vals, rp_vals])
        Z_s   = np.vstack([np.zeros(N_s), ind_vals])   # floor at 0 (normalised)
        ax.plot_surface(T_s, R_s, Z_s, facecolor=base_color, alpha=0.15,
                        linewidth=0, antialiased=False)
        ax.contourf(T_s, R_s, Z_s, zdir="z", offset=0.0,
                    levels=8, cmap="viridis", alpha=0.4)

        # Fading tail (Line3DCollection — same idea as LineCollection in 2-D).
        # Must be initialised with a non-empty segment list; add_collection3d
        # calls auto_scale_xyz(*segments.transpose()) which fails on shape (0,).
        # The dummy segment is invisible (alpha=0) and overwritten on frame 1.
        _dummy = [[[t_vals[0], rp_vals[0], ind_vals[0]],
                   [t_vals[0], rp_vals[0], ind_vals[0]]]]
        tail_col = Line3DCollection(_dummy, linewidths=2.0, zorder=2, alpha=0.0)
        ax.add_collection3d(tail_col)

        # Moving point
        label = (f"{filename} | SNR: {fmt_snr_display(snr_value)} | "
                 f"Peaks: {peak_count if peak_count is not None else 'n/a'}")
        point, = ax.plot([], [], [], marker="o", ms=5,
                         color=base_color, zorder=3, label=label)

        # Peak markers — triangle markers that appear when the trace arrives
        peak_markers = []
        for (pk_idx, _, _) in peak_annotations:
            if pk_idx < N:
                pm, = ax.plot([t_vals[pk_idx]], [rp_vals[pk_idx]], [ind_vals[pk_idx]],
                              marker="^", ms=7, color=base_color, zorder=4,
                              alpha=0.0, linestyle="none")
            else:
                pm, = ax.plot([], [], [], marker="^", ms=7,
                              color=base_color, zorder=4, linestyle="none")
            peak_markers.append(pm)

        # Text annotations that appear when the sweeping point reaches each peak
        peak_texts = []
        for (pk_idx, snr_val, lbl) in peak_annotations:
            if pk_idx < N:
                pt = ax.text(t_vals[pk_idx], rp_vals[pk_idx], ind_vals[pk_idx],
                             f"  {lbl}\n  {fmt_snr_display(snr_val)}",
                             fontsize=7, color=base_color, zorder=5, alpha=0.0)
            else:
                pt = ax.text(0, 0, 0, "", fontsize=7, alpha=0.0)
            peak_texts.append(pt)

        artists.append((tail_col, point, base_color,
                        peak_annotations, peak_markers, peak_texts,
                        t_vals, rp_vals, ind_vals, N))

    ax.legend(loc="upper left", fontsize=7)

    # ── animation callbacks ───────────────────────────────────────────────

    def init():
        for (tail_col, point, _, _, peak_markers, peak_texts,
             t_vals, rp_vals, ind_vals, _) in artists:
            # Reset tail to the invisible dummy; never set segments to []
            # because that triggers the same auto_scale_xyz shape error.
            tail_col.set_segments([[[t_vals[0], rp_vals[0], ind_vals[0]],
                                    [t_vals[0], rp_vals[0], ind_vals[0]]]])
            tail_col.set_alpha(0.0)
            point.set_data([], [])
            point.set_3d_properties([])
            for pm in peak_markers:
                pm.set_alpha(0.0)
            for pt in peak_texts:
                pt.set_alpha(0.0)
        return []

    def update(frame):
        for (tail_col, point, base_color,
             peak_annotations, peak_markers, peak_texts,
             t_vals, rp_vals, ind_vals, N) in artists:

            last_idx  = min(frame, N - 1)
            start_idx = max(0, last_idx - tail_length + 1)

            tt = t_vals[start_idx: last_idx + 1]
            rr = rp_vals[start_idx: last_idx + 1]
            ii = ind_vals[start_idx: last_idx + 1]

            if len(tt) > 1:
                pts      = np.column_stack([tt, rr, ii])
                segments = np.stack([pts[:-1], pts[1:]], axis=1)   # shape (n, 2, 3)
                n        = len(segments)
                alphas   = np.linspace(0.05, 1.0, n)
                colors   = np.tile(np.array(base_color), (n, 1))
                colors[:, 3] = alphas
                tail_col.set_segments(segments)
                tail_col.set_color(colors)
            else:
                tail_col.set_segments([[[t_vals[0], rp_vals[0], ind_vals[0]],
                                        [t_vals[0], rp_vals[0], ind_vals[0]]]])
                tail_col.set_alpha(0.0)

            cur = min(frame, N - 1)
            point.set_data([t_vals[cur]], [rp_vals[cur]])
            point.set_3d_properties([ind_vals[cur]])

            # Reveal peak markers and text labels as the animation sweeps past them
            for (pk_idx, _, _), pm, pt in zip(peak_annotations, peak_markers, peak_texts):
                visible = pk_idx < N and frame >= pk_idx
                pm.set_alpha(0.9 if visible else 0.0)
                pt.set_alpha(0.9 if visible else 0.0)

        return []

    anim = FuncAnimation(
        fig, update,
        frames=range(0, max_frames, max(1, frame_step)),
        init_func=init,
        interval=interval,
        blit=False,   # 3-D axes do not support blitting
        repeat=False,
    )

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_and_close_animation(
        fig, anim, output_dir,
        f"overlay_3d_animation_{timestamp}.gif",
        interval, frame_step, save_gif, show_plot,
        "_overlay_3d_anim",
    )


# ---------------------------------------------------------------------------
# 3-D contour plots
# ---------------------------------------------------------------------------

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
    Produce one 3D surface/contour figure per run where:
      x-axis = time (normalised to [0, 1])
      y-axis = R_p  (sensor1_smooth_rot)
      z-axis = Inductance (sensor2_smooth_rot)

    Because the data is a 1-D scan (not a 2-D grid), a thin pseudo-surface
    is built by stacking two copies of the trace so plot_surface has a
    proper 2-D array to work with.
    """
    # Generate colors based on inductor type
    inductor_color_map = _get_inductor_color_map(list(dataframes.keys()))

    time_candidates = [time_col, "time", "sample", "index"]
    rp_candidates   = [x_col, x_col.strip(), "sensor1", " sensor1"]
    ind_candidates  = [y_col, y_col.strip(), "sensor2", " sensor2"]

    for i, (filename, df) in enumerate(dataframes.items()):
        t_col   = resolve_first_existing_column(df, time_candidates)
        rp_col  = resolve_first_existing_column(df, rp_candidates)
        ind_col = resolve_first_existing_column(df, ind_candidates)

        if rp_col is None or ind_col is None:
            print(f"3D plot: required columns not found in '{filename}', skipping.")
            continue

        rp_vals  = df[rp_col].to_numpy(dtype=float)
        ind_vals = df[ind_col].to_numpy(dtype=float)
        t_raw    = df[t_col].to_numpy(dtype=float) if t_col else np.arange(len(rp_vals), dtype=float)

        t_min, t_max = t_raw.min(), t_raw.max()
        t_vals = (t_raw - t_min) / (t_max - t_min) if t_max > t_min else t_raw

        N = len(t_vals)
        T = np.vstack([t_vals,  t_vals])
        R = np.vstack([rp_vals, rp_vals])
        Z = np.vstack([np.full(N, ind_vals.min()), ind_vals])

        base_color = inductor_color_map.get(filename, plt.get_cmap("tab10")(i % 10))
        fig = plt.figure(figsize=(12, 7))
        ax  = fig.add_subplot(111, projection="3d")

        ax.plot_surface(T, R, Z, facecolor=base_color, alpha=0.55,
                        linewidth=0, antialiased=True)
        ax.plot(t_vals, rp_vals, ind_vals, color=base_color, linewidth=1.5, zorder=5)

        ind_std_col = _resolve_std_column(df, ind_col)
        if ind_std_col is not None:
            ind_std_vals = pd.to_numeric(df[ind_std_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ind_std_vals = np.clip(ind_std_vals, a_min=0.0, a_max=None)
            z_lower = ind_vals - ind_std_vals
            z_upper = ind_vals + ind_std_vals
            T_std = np.vstack([t_vals, t_vals])
            R_std = np.vstack([rp_vals, rp_vals])
            Z_std = np.vstack([z_lower, z_upper])
            ax.plot_surface(T_std, R_std, Z_std, facecolor=base_color, alpha=0.14,
                            linewidth=0, antialiased=False)
            ax.plot(t_vals, rp_vals, z_lower, color=base_color, linewidth=0.7, alpha=0.32, zorder=4)
            ax.plot(t_vals, rp_vals, z_upper, color=base_color, linewidth=0.7, alpha=0.32, zorder=4)

        ax.contourf(T, R, Z, zdir="z", offset=ind_vals.min(),
                    levels=15, cmap="viridis", alpha=0.4)

        peak_annotations = get_peak_annotations(snr_df, filename) if snr_df is not None else []
        if peak_annotations:
            valid_peaks = [(pk, snr, lbl) for pk, snr, lbl in peak_annotations if pk < N]
            if valid_peaks:
                pk_t   = np.array([t_vals[pk]  for pk, _, _  in valid_peaks])
                pk_r   = np.array([rp_vals[pk]  for pk, _, _  in valid_peaks])
                pk_z   = np.array([ind_vals[pk] for pk, _, _  in valid_peaks])
                ax.scatter(pk_t, pk_r, pk_z, color="red", s=60, zorder=6, depthshade=False)
                for tx, rx, zx, (_, snr_val, lbl) in zip(pk_t, pk_r, pk_z, valid_peaks):
                        ax.text(tx, rx, zx, f"  {lbl}\n  {fmt_snr_display(snr_val)}",
                            fontsize=7, color="red", zorder=7)

        ax.set_xlabel("Time (norm.)")
        ax.set_ylabel("R_p (ohm)")
        ax.set_zlabel("Inductance (uH)")
        ax.set_title(f"3D Contour — {filename}")
        plt.tight_layout()

        if save_plots:
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = filename.replace(".csv", "").replace(" ", "_")
            out_path  = Path(output_dir) / f"3d_contour_{safe_name}_{ts}.png"
            fig.savefig(str(out_path), dpi=150)
            print(f"Saved 3D contour plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
