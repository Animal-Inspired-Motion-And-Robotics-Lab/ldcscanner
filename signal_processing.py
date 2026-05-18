# ---------------------------------------------------------------------------
# signal_processing.py — Column resolution, SNR computation, peak detection
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from natsort import natsorted  # Ensure natural sorting

from config import CRACK_LABELS


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def resolve_column_name(df, target_name, required=True):
    """
    Find a column in df, tolerating leading/trailing whitespace in headers.

    Args:
        df: Source DataFrame.
        target_name: Desired column name (exact or with whitespace).
        required: If True, raise KeyError when not found; if False, return None.
    """
    if target_name in df.columns:
        return target_name

    stripped_lookup = {col.strip(): col for col in df.columns}
    if target_name.strip() in stripped_lookup:
        return stripped_lookup[target_name.strip()]

    if required:
        raise KeyError(f"Missing required column '{target_name}'")
    return None


def resolve_first_existing_column(df, candidate_names):
    """
    Return the first candidate column that exists in df, or None.

    Whitespace-tolerant: each candidate is checked both exactly and stripped.
    """
    for candidate in candidate_names:
        result = resolve_column_name(df, candidate, required=False)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_xy_series(dataframes, x_col, y_col, normalize=False):
    """
    Extract and clean XY arrays from each DataFrame.

    Args:
        dataframes: Dict of {filename: DataFrame}.
        x_col: X-axis column name.
        y_col: Y-axis column name.
        normalize: If True, min-max normalize x and y independently per file.

    Returns:
        List of (filename, x_vals, y_vals) tuples for files that succeeded.
    """
    prepared = []

    for filename, df in dataframes.items():
        try:
            real_x = resolve_column_name(df, x_col)
            real_y = resolve_column_name(df, y_col)

            x = pd.to_numeric(df[real_x], errors="coerce")
            y = pd.to_numeric(df[real_y], errors="coerce")
            valid = x.notna() & y.notna()
            x_vals = x[valid].to_numpy()
            y_vals = y[valid].to_numpy()

            if len(x_vals) == 0:
                print(f"Skipping {filename}: no numeric XY pairs after cleaning.")
                continue

            if normalize:
                x_min, x_max = np.min(x_vals), np.max(x_vals)
                y_min, y_max = np.min(y_vals), np.max(y_vals)
                x_vals = (x_vals - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(x_vals)
                y_vals = (y_vals - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y_vals)

            prepared.append((filename, x_vals, y_vals))

        except KeyError as err:
            print(f"Skipping {filename}: {err}")

    return prepared


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def detect_raw_peak_candidates(y_vals, max_peaks=10, min_peak_distance=60):
    """
    Detect candidate peaks on raw signal values using robust prominence thresholding.

    Returns:
        (peak_indices, peak_prominences) — both sorted by index position.
    """
    if len(y_vals) < 10:
        return np.array([], dtype=int), np.array([], dtype=float)

    y_vals = np.asarray(y_vals, dtype=float)
    y_median = float(np.median(y_vals))
    y_mad = float(np.median(np.abs(y_vals - y_median)))
    robust_sigma = 1.4826 * y_mad
    if robust_sigma <= 0:
        robust_sigma = float(np.std(y_vals))

    peak_indices, peak_props = find_peaks(
        y_vals,
        prominence=max(0.05, robust_sigma * 1.0),
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


# ---------------------------------------------------------------------------
# SNR computation
# ---------------------------------------------------------------------------

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
    Compute crack-focused SNR metrics for a single sensor trace.

    In strict ground-truth mode (default), peaks are drawn only from
    manual_windows — one peak per window (the strongest baseline-removed
    point inside).  Set allow_automatic_fallback=True to use scipy
    find_peaks instead when no manual windows are provided.

    Returns:
        Dict of metrics, or None if computation is not possible.
    """
    if len(x_vals) < 20 or len(y_vals) < 20:
        return None

    y_vals = np.asarray(y_vals, dtype=float)

    peak_labels        = []
    per_window_records = []
    peak_height_threshold = np.nan

    if manual_windows:
        windows_info = []
        crack_mask = np.zeros(len(y_vals), dtype=bool)
        row_indices_arr = np.asarray(row_indices, dtype=int) if row_indices is not None else None

        for window in manual_windows:
            label = str(window["label"])
            in_window_idx = np.array([], dtype=int)

            start_row = window.get("start_row")
            end_row   = window.get("end_row")
            if (
                start_row is not None
                and end_row is not None
                and row_indices_arr is not None
                and len(row_indices_arr) == len(y_vals)
                and np.isfinite(start_row)
                and np.isfinite(end_row)
            ):
                lo, hi = sorted((int(start_row), int(end_row)))
                in_window_idx = np.flatnonzero((row_indices_arr >= lo) & (row_indices_arr <= hi))
            elif "start_x" in window and "end_x" in window:
                start_x, end_x = float(window["start_x"]), float(window["end_x"])
                if np.isfinite(start_x) and np.isfinite(end_x):
                    lo, hi = sorted((start_x, end_x))
                    in_window_idx = np.flatnonzero((x_vals >= lo) & (x_vals <= hi))

            if len(in_window_idx) == 0:
                continue

            start_idx = int(in_window_idx[0])
            end_idx = int(in_window_idx[-1])
            crack_mask[start_idx:end_idx + 1] = True
            windows_info.append({
                "label": label,
                "indices": in_window_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
            })

        if not windows_info:
            return None

        windows_info.sort(key=lambda w: (w["start_idx"], w["end_idx"]))

        non_crack_vals = y_vals[~crack_mask]
        if len(non_crack_vals) < 2:
            return None

        sigma_baseline = float(np.std(non_crack_vals, ddof=0))
        if sigma_baseline <= 0:
            return None

        selected_peak_indices = []
        selected_peak_values = []
        peak_signals = []

        for i, info in enumerate(windows_info):
            label = info["label"]
            in_window_idx = info["indices"]

            peak_idx = int(in_window_idx[int(np.argmax(y_vals[in_window_idx]))])
            a_peak = float(y_vals[peak_idx])

            left_start = 0 if i == 0 else int(windows_info[i - 1]["end_idx"]) + 1
            left_end = int(info["start_idx"]) - 1
            right_start = int(info["end_idx"]) + 1
            right_end = (len(y_vals) - 1) if i == (len(windows_info) - 1) else int(windows_info[i + 1]["start_idx"]) - 1

            left_vals = y_vals[left_start:left_end + 1] if left_end >= left_start else np.array([], dtype=float)
            right_vals = y_vals[right_start:right_end + 1] if right_end >= right_start else np.array([], dtype=float)

            if len(left_vals) == 0 and len(right_vals) == 0:
                continue

            baseline_neighbors = np.concatenate([left_vals, right_vals])
            mu_baseline = float(np.mean(baseline_neighbors))

            peak_signal = float(a_peak - mu_baseline)
            peak_snr_linear = peak_signal / sigma_baseline
            peak_snr_db = float(20 * np.log10(peak_snr_linear)) if peak_snr_linear > 0 else float("-inf")

            start_row_record = end_row_record = peak_row_record = np.nan
            if row_indices_arr is not None and len(row_indices_arr) == len(y_vals):
                start_row_record = int(row_indices_arr[int(in_window_idx[0])])
                end_row_record = int(row_indices_arr[int(in_window_idx[-1])])
                peak_row_record = int(row_indices_arr[peak_idx])

            selected_peak_indices.append(peak_idx)
            selected_peak_values.append(a_peak)
            peak_signals.append(peak_signal)
            peak_labels.append(label)
            per_window_records.append({
                "manual_label":               label,
                "window_start_x":             float(x_vals[in_window_idx[0]]),
                "window_end_x":               float(x_vals[in_window_idx[-1]]),
                "window_start_raw_idx":       start_row_record,
                "window_end_raw_idx":         end_row_record,
                "peak_index_in_clean_series": int(peak_idx),
                "peak_raw_idx":               peak_row_record,
                "peak_x":                     float(x_vals[peak_idx]),
                "peak_y":                     float(y_vals[peak_idx]),
                "noise_floor":                mu_baseline,
                "noise_sigma":                float(sigma_baseline),
                "peak_signal_amplitude":      peak_signal,
                "peak_snr_linear":            float(peak_snr_linear),
                "peak_snr_db":                peak_snr_db,
            })

        peak_indices = np.array(selected_peak_indices, dtype=int)
        peak_values = np.array(selected_peak_values, dtype=float)
        peak_signals_arr = np.array(peak_signals, dtype=float)
        noise_floor = float(np.mean(non_crack_vals))
        noise_sigma = float(sigma_baseline)

    elif allow_automatic_fallback:
        peak_indices, peak_props = find_peaks(
            y_vals,
            height=np.median(y_vals) + (sigma_threshold_multiplier * np.std(y_vals)),
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

    if manual_windows:
        if len(peak_indices) == 0:
            return None

        signal_amplitude = float(np.median(peak_signals_arr))
        snr_linear = signal_amplitude / noise_sigma if noise_sigma > 0 else 0.0
        snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

        peak_snr_db_values = []
        for s in peak_signals_arr:
            if noise_sigma > 0:
                s_lin = float(s / noise_sigma)
                peak_snr_db_values.append(float(20 * np.log10(s_lin)) if s_lin > 0 else float("-inf"))
            else:
                peak_snr_db_values.append(float("-inf"))

    else:
        if len(peak_heights) == 0:
            return None

        noise_floor = float(np.median(y_vals))
        noise_sigma = float(np.std(y_vals))
        if noise_sigma <= 0:
            return None

        signal_amplitude = float(np.median(peak_heights - noise_floor))
        snr_linear = signal_amplitude / noise_sigma if noise_sigma > 0 else 0.0
        snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

        peak_snr_db_values = []
        for height in peak_heights:
            s = float(height - noise_floor)
            if noise_sigma > 0 and s > 0:
                peak_snr_db_values.append(float(20 * np.log10(s / noise_sigma)))
            else:
                peak_snr_db_values.append(float("-inf"))

    if manual_windows and len(peak_signals_arr) == 0:
        return None

    return {
        "noise_sigma":          float(noise_sigma),
        "noise_floor":          float(noise_floor),
        "signal_amplitude":     float(signal_amplitude),
        "snr_linear":           float(snr_linear),
        "snr_db":               float(snr_db),
        "peak_count":           int(len(peak_indices)),
        "peak_indices":         ";".join(str(int(i)) for i in peak_indices),
        "peak_snr_db_values":   ";".join(f"{v:.6f}" for v in peak_snr_db_values),
        "peak_labels":          ";".join(peak_labels),
        "peak_height_threshold": float(peak_height_threshold),
        "per_window_records":   per_window_records,
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
    Quantify crack-detection SNR for every file in dataframes.

    If manual_labels_df is provided, labeled crack windows drive peak
    selection.  In ground_truth_only mode, automatic peak-finding is
    disabled entirely.

    Returns:
        (snr_df, per_crack_df) — file-level and per-crack DataFrames.
    """
    if not dataframes:
        print("No dataframes available for SNR analysis.")
        return pd.DataFrame(), pd.DataFrame()

    if ground_truth_only and (manual_labels_df is None or manual_labels_df.empty):
        print("Ground-truth mode requires manual crack window labels.")
        return pd.DataFrame(), pd.DataFrame()

    records           = []
    per_crack_records = []

    for filename, df in dataframes.items():
        if df.empty:
            print(f"Skipping {filename} for SNR: file is empty.")
            continue

        try:
            real_x = resolve_column_name(df, x_col)
            real_y = resolve_column_name(df, y_col)
        except KeyError as err:
            print(f"Skipping {filename} for SNR: {err}")
            continue

        x = pd.to_numeric(df[real_x], errors="coerce")
        y = pd.to_numeric(df[real_y], errors="coerce")
        valid = x.notna() & y.notna()
        x_vals           = x[valid].to_numpy()
        y_vals           = y[valid].to_numpy()
        proc_row_indices = np.flatnonzero(valid.to_numpy())

        file_manual_windows = []
        if manual_labels_df is not None and not manual_labels_df.empty:
            file_df = manual_labels_df[
                (manual_labels_df["file"] == filename)
                & (manual_labels_df["manual_label"].isin(CRACK_LABELS))
            ].copy()

            sort_cols = [c for c in ["window_order", "manual_label"] if c in file_df.columns]
            if sort_cols:
                file_df = file_df.sort_values(sort_cols)

            for _, win_row in file_df.iterrows():
                start_x   = win_row.get("window_start_x")
                end_x     = win_row.get("window_end_x")
                start_row = win_row.get("window_start_raw_idx")
                end_row   = win_row.get("window_end_raw_idx")
                label     = str(win_row.get("manual_label", ""))

                if pd.isna(start_row) or pd.isna(end_row):
                    if pd.isna(start_x) or pd.isna(end_x):
                        continue
                    file_manual_windows.append(
                        {"start_x": float(start_x), "end_x": float(end_x), "label": label}
                    )
                else:
                    file_manual_windows.append(
                        {"start_row": float(start_row), "end_row": float(end_row), "label": label}
                    )

            if require_manual_windows and not file_manual_windows:
                print(f"Skipping {filename} for SNR: no crack-labeled windows found.")
                continue

        metrics = compute_xy_crack_snr_metrics(
            x_vals, y_vals,
            baseline_window=baseline_window,
            max_peaks=max_peaks,
            min_peak_distance=min_peak_distance,
            sigma_threshold_multiplier=sigma_threshold_multiplier,
            manual_windows=file_manual_windows or None,
            row_indices=proc_row_indices,
            allow_automatic_fallback=not ground_truth_only,
        )
        if metrics is None:
            continue

        records.append({
            "file":                filename,
            "x_col":               real_x,
            "y_col":               real_y,
            "n_samples":           int(len(x_vals)),
            "noise_floor":         metrics["noise_floor"],
            "noise_sigma":         metrics["noise_sigma"],
            "signal_amplitude":    metrics["signal_amplitude"],
            "snr_linear":          metrics["snr_linear"],
            "snr_db":              metrics["snr_db"],
            "peak_count":          metrics["peak_count"],
            "peak_indices":        metrics["peak_indices"],
            "peak_snr_db_values":  metrics["peak_snr_db_values"],
            "peak_labels":         metrics["peak_labels"],
            "peak_height_threshold": metrics["peak_height_threshold"],
        })

        for wr in metrics.get("per_window_records", []):
            per_crack_records.append({
                "file":                        filename,
                "x_col":                       real_x,
                "y_col":                       real_y,
                "manual_label":                wr.get("manual_label"),
                "window_start_x":              wr.get("window_start_x"),
                "window_end_x":                wr.get("window_end_x"),
                "window_start_raw_idx":        wr.get("window_start_raw_idx"),
                "window_end_raw_idx":          wr.get("window_end_raw_idx"),
                "peak_index_in_clean_series":  wr.get("peak_index_in_clean_series"),
                "peak_raw_idx":                wr.get("peak_raw_idx"),
                "peak_x":                      wr.get("peak_x"),
                "peak_y":                      wr.get("peak_y"),
                "noise_floor":                 wr.get("noise_floor"),
                "noise_sigma":                 wr.get("noise_sigma"),
                "peak_signal_amplitude":       wr.get("peak_signal_amplitude"),
                "peak_snr_linear":             wr.get("peak_snr_linear"),
                "peak_snr_db":                 wr.get("peak_snr_db"),
            })

    if not records:
        print("No valid sensor signals found for SNR analysis.")
        return pd.DataFrame(), pd.DataFrame()

    snr_df = pd.DataFrame(records)
    snr_df["file"] = snr_df["file"].astype(str)  # Ensure file column is string
    snr_df = snr_df.loc[natsorted(snr_df.index, key=lambda i: snr_df.loc[i, "file"])]
    snr_df = snr_df.reset_index(drop=True)

    per_crack_df = pd.DataFrame(per_crack_records)
    if not per_crack_df.empty:
        sort_cols = [c for c in ["file", "manual_label", "window_start_raw_idx"]
                     if c in per_crack_df.columns]
        if sort_cols:
            per_crack_df = per_crack_df.sort_values(sort_cols).reset_index(drop=True)

    if verbose:
        print("\nCrack-detection SNR by file (highest first):")
        print(snr_df[["file", "snr_db", "snr_linear", "signal_amplitude",
                       "noise_sigma", "peak_count"]].head(50))
        if not per_crack_df.empty:
            print("\nPer-crack window SNR details:")
            print(per_crack_df[["file", "manual_label", "peak_snr_db",
                                 "peak_signal_amplitude", "noise_sigma"]].head(50))

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
    Grid-search SNR detector settings to hit desired peak counts.

    Primary objective:   every file has at least target_min_peaks.
    Secondary objective: mean peaks close to target_ideal_peaks.

    Returns:
        (best_params_dict, snr_df) or (None, empty DataFrame) if no files loaded.
    """
    candidate_settings = [
        {"baseline_window": 121, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.6},
        {"baseline_window": 101, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.4},
        {"baseline_window":  81, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.2},
        {"baseline_window":  61, "min_peak_distance": 100, "sigma_threshold_multiplier": 3.0},
        {"baseline_window":  51, "min_peak_distance": 100, "sigma_threshold_multiplier": 2.8},
    ]

    best = None
    best_score = None

    for params in candidate_settings:
        snr_df, _ = analyze_snr(
            dataframes, x_col=x_col, y_col=y_col,
            ground_truth_only=False,
            baseline_window=params["baseline_window"],
            max_peaks=max_peaks,
            min_peak_distance=params["min_peak_distance"],
            sigma_threshold_multiplier=params["sigma_threshold_multiplier"],
            verbose=False,
        )
        if snr_df.empty:
            continue

        peak_counts          = snr_df["peak_count"].to_numpy()
        min_count            = int(np.min(peak_counts))
        mean_count           = float(np.mean(peak_counts))
        below_target_penalty = max(0, target_min_peaks - min_count)
        ideal_distance       = abs(mean_count - target_ideal_peaks)
        overflow_penalty     = max(0.0, mean_count - max_peaks)
        score = (below_target_penalty * 100.0) + ideal_distance + (overflow_penalty * 5.0)

        if best is None or score < best_score:
            best = {"params": params, "snr_df": snr_df,
                    "min_count": min_count, "mean_count": mean_count}
            best_score = score

        if min_count >= target_min_peaks and mean_count >= (target_ideal_peaks - 1):
            break

    if best is None:
        return None, pd.DataFrame()
    return best, best["snr_df"]
