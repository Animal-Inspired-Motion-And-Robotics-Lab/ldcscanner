import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks


# Runtime toggles for faster execution.
FAST_MODE = False
FRAME_STEP = 5 if FAST_MODE else 1
SHOW_PLOTS = False if FAST_MODE else True
SAVE_GIFS = True

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
):
    """
    Compute crack-focused SNR from vertical-direction thresholded peaks.

    A peak is any local maximum in baseline-removed y that exceeds
    noise_floor + sigma_threshold_multiplier * noise_sigma.
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

    # Estimate baseline level and sigma from the full baseline-removed vertical signal.
    noise_floor = float(np.median(y_dev))
    noise_mad = float(np.median(np.abs(y_dev - noise_floor)))
    noise_sigma = 1.4826 * noise_mad
    if noise_sigma <= 0:
        noise_sigma = float(np.std(y_dev))
    if noise_sigma <= 0:
        return None

    peak_height_threshold = noise_floor + (sigma_threshold_multiplier * noise_sigma)
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

    if len(peak_heights) > 0:
        peak_signal = peak_heights - noise_floor
        signal_amplitude = float(np.median(peak_signal))
    else:
        signal_amplitude = float(max(0.0, np.percentile(y_dev, 95) - noise_floor))

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
        "peak_height_threshold": float(peak_height_threshold),
    }


def analyze_snr(
    dataframes,
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    baseline_window=101,
    max_peaks=9,
    min_peak_distance=100,
    sigma_threshold_multiplier=3.0,
    verbose=True,
):
    """
    Quantify crack-detection SNR per file from vertical thresholded peaks.

    Returns a dataframe sorted by descending SNR (dB), one row per file.
    """
    if not dataframes:
        print("No dataframes available for SNR analysis.")
        return pd.DataFrame()

    records = []

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

        metrics = compute_xy_crack_snr_metrics(
            x_vals,
            y_vals,
            baseline_window=baseline_window,
            max_peaks=max_peaks,
            min_peak_distance=min_peak_distance,
            sigma_threshold_multiplier=sigma_threshold_multiplier,
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
                "peak_height_threshold": metrics["peak_height_threshold"],
            }
        )

    if not records:
        print("No valid sensor signals found for SNR analysis.")
        return pd.DataFrame()

    snr_df = pd.DataFrame(records).sort_values(by="snr_db", ascending=False).reset_index(drop=True)

    if verbose:
        print("\nCrack-detection SNR by file (highest first):")
        print(snr_df[["file", "snr_db", "snr_linear", "signal_amplitude", "noise_sigma", "peak_count"]].head(50))

    return snr_df


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
        snr_df = analyze_snr(
            dataframes,
            x_col=x_col,
            y_col=y_col,
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
    Fetch per-peak annotations as (index, peak_snr_db) for a file.
    """
    if snr_df is None or snr_df.empty:
        return []

    matches = snr_df[snr_df["file"] == filename]
    if matches.empty:
        return []

    row = matches.iloc[0]
    idx_str = row.get("peak_indices", "")
    snr_str = row.get("peak_snr_db_values", "")

    if pd.isna(idx_str) or str(idx_str).strip() == "":
        return []

    indices = [int(x) for x in str(idx_str).split(";") if str(x).strip() != ""]
    snr_vals = [float(x) for x in str(snr_str).split(";") if str(x).strip() != ""]

    annotations = []
    for i, idx in enumerate(indices):
        peak_snr_db = snr_vals[i] if i < len(snr_vals) else np.nan
        annotations.append((idx, peak_snr_db))
    return annotations


def fmt_snr_db(value):
    """
    Format SNR value for on-plot labels.
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} dB"


def create_overlay_animation(
    dataframes,
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    snr_df=None,
    interval=30,
    tail_length=100,
    frame_step=1,
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

    x_pad = (global_x_max - global_x_min) * 0.05 if global_x_max > global_x_min else 1
    y_pad = (global_y_max - global_y_min) * 0.05 if global_y_max > global_y_min else 1
    ax.set_xlim(global_x_min - x_pad, global_x_max + x_pad)
    ax.set_ylim(global_y_min - y_pad, global_y_max + y_pad)

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
        for _, _ in peak_annotations:
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

            for (peak_idx, peak_snr_db), peak_marker, peak_text in zip(peak_annotations, peak_markers, peak_texts):
                if peak_idx < len(x_vals) and frame >= peak_idx:
                    peak_x = x_vals[peak_idx]
                    peak_y = y_vals[peak_idx]
                    peak_marker.set_data([peak_x], [peak_y])
                    peak_text.set_position((peak_x, peak_y))
                    peak_text.set_text(f"{fmt_snr_db(peak_snr_db)}")
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
        gif_filename = f"overlay_animation_{timestamp}.gif"
        fps = max(1, int(round(1000 / interval / max(1, frame_step))))
        anim.save(gif_filename, writer="pillow", fps=fps)
        print(f"Saved animation GIF: {gif_filename}")

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
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
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
        for _, _ in peak_annotations:
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

            for (peak_idx, peak_snr_db), peak_marker, peak_text in zip(peak_annotations, peak_markers, peak_texts):
                if peak_idx < len(x_vals) and frame >= peak_idx:
                    peak_x = x_vals[peak_idx]
                    peak_y = y_vals[peak_idx]
                    peak_marker.set_data([peak_x], [peak_y])
                    peak_text.set_position((peak_x, peak_y))
                    peak_text.set_text(f"{fmt_snr_db(peak_snr_db)}")
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
        gif_filename = f"stacked_animation_{timestamp}.gif"
        fps = max(1, int(round(1000 / interval / max(1, frame_step))))
        anim.save(gif_filename, writer="pillow", fps=fps)
        print(f"Saved stacked animation GIF: {gif_filename}")

    fig._stacked_anim = anim

    plt.tight_layout(rect=[0, 0, 1, 0.98])
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

    color_map = plt.get_cmap("tab10", len(prepared))
    artists = []
    max_frames = max(len(item[1]) for item in prepared)

    for i, ((filename, x_vals, y_vals), ax) in enumerate(zip(prepared, axes)):
        base_color = color_map(i)
        snr_db = get_snr_value_db(snr_df, filename)
        peak_count = get_peak_count(snr_df, filename)
        peak_annotations = get_peak_annotations(snr_df, filename)

        x_min, x_max = np.min(x_vals), np.max(x_vals)
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(global_y_min - global_y_pad, global_y_max + global_y_pad)
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
        for _, _ in peak_annotations:
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

            for (peak_idx, peak_snr_db), peak_marker, peak_text in zip(peak_annotations, peak_markers, peak_texts):
                if peak_idx < len(x_vals) and frame >= peak_idx:
                    peak_x = x_vals[peak_idx]
                    peak_y = y_vals[peak_idx]
                    peak_marker.set_data([peak_x], [peak_y])
                    peak_text.set_position((peak_x, peak_y))
                    peak_text.set_text(f"{fmt_snr_db(peak_snr_db)}")
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
        gif_filename = f"stacked_baseline_animation_{timestamp}.gif"
        fps = max(1, int(round(1000 / interval / max(1, frame_step))))
        anim.save(gif_filename, writer="pillow", fps=fps)
        print(f"Saved baseline-stacked animation GIF: {gif_filename}")

    fig._stacked_baseline_anim = anim

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    print("CSV Loader + Animation + SNR Analysis")
    print("-" * 40)
    
    # Requested files to load and animate.
    user_input = "flex_4.csv,flex_8.csv"
    
    if user_input.lower() == 'quit':
        print("Exiting...")
        return
    
    # Parse input and filter empty strings
    file_paths = [path.strip() for path in user_input.split(',') if path.strip()]
    
    if not file_paths:
        print("No files provided.")
        return
    
    # Load the CSV files
    dataframes = load_csv_files(file_paths)
    
    if dataframes:
        print("\n" + "=" * 40)
        print(f"Loaded {len(dataframes)} file(s) successfully:")
        for filename, df in dataframes.items():
            print(f"\n{filename}:")
            print(df.head())

        tuned, snr_df = tune_snr_parameters_for_peak_targets(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            target_min_peaks=4,
            target_ideal_peaks=6,
            max_peaks=6,
        )

        if tuned is not None:
            p = tuned["params"]
            print("\nAuto-tuned crack peak settings:")
            print(
                "baseline_window={bw}, min_peak_distance={mpd}, sigma_threshold_multiplier={sm}".format(
                    bw=p["baseline_window"],
                    mpd=p["min_peak_distance"],
                    sm=p["sigma_threshold_multiplier"],
                )
            )
            print(
                "Peak count summary: min={mn}, mean={avg:.2f}".format(
                    mn=tuned["min_count"],
                    avg=tuned["mean_count"],
                )
            )

        if not snr_df.empty:
            print("\nCrack-detection SNR by file (highest first):")
            print(snr_df[["file", "snr_db", "snr_linear", "signal_amplitude", "noise_sigma", "peak_count"]].head(50))
        if not snr_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snr_filename = f"snr_analysis_{timestamp}.csv"
            snr_df.to_csv(snr_filename, index=False)
            print(f"Saved SNR analysis CSV: {snr_filename}")

        create_overlay_animation(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            snr_df=snr_df,
            frame_step=FRAME_STEP,
            save_gif=SAVE_GIFS,
            show_plot=SHOW_PLOTS,
        )

        create_stacked_animation(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            snr_df=snr_df,
            frame_step=FRAME_STEP,
            save_gif=SAVE_GIFS,
            show_plot=SHOW_PLOTS,
        )

        create_stacked_baseline_animation(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
            snr_df=snr_df,
            frame_step=FRAME_STEP,
            save_gif=SAVE_GIFS,
            show_plot=SHOW_PLOTS,
        )
    else:
        print("No files were successfully loaded.")


if __name__ == "__main__":
    main()
