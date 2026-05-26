# ---------------------------------------------------------------------------
# labeling.py — Compatibility stubs (manual crack labeling removed)
# ---------------------------------------------------------------------------

import pandas as pd


def ask_manual_peak_label(prompt_text):
    """Compatibility helper retained for callers; always returns empty label."""
    _ = prompt_text
    return ""


def run_manual_raw_peak_labeling(
    dataframes,
    output_dir,
    source_path_lookup=None,
    cache_dir="window_label_cache",
    max_peaks_per_file=10,
    min_peak_distance=60,
    show_plot=True,
):
    """Manual crack-window labeling has been removed from this project."""
    _ = (
        dataframes,
        output_dir,
        source_path_lookup,
        cache_dir,
        max_peaks_per_file,
        min_peak_distance,
        show_plot,
    )
    print("Manual crack labeling is disabled.")
    return pd.DataFrame()
