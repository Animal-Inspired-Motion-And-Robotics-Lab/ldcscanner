"""
Scanimator V4 — LDC sensor signal analysis pipeline
===================================================

Pipeline overview
-----------------
1. GUI file picker  →  select one or more CSV scan files
2. SNR analysis     →  signal-to-noise ratio per file
4. Visualizations   →  static SNR charts + three animated GIF types + 3D contours

Outputs land in  outputs/run_<timestamp>/  and are never overwritten.

Configuration
-------------
Edit config.py to change:
  - FAST_MODE / FRAME_STEP / SHOW_PLOTS / SAVE_GIFS
  - X_COL / Y_COL / TIME_COL  (CSV column names)
"""

from datetime import datetime
from pathlib import Path

from config import (
    FRAME_STEP,
    SHOW_PLOTS,
    SAVE_GIFS,
    X_COL,
    Y_COL,
    TIME_COL,
)
from io_utils import (
    select_csv_files_via_gui,
    load_csv_files,
    build_source_path_lookup,
    create_run_output_dir,
)
from signal_processing import analyze_snr
from visualization import (
    create_snr_visualizations,
    create_overlay_animation,
    create_stacked_animation,
    create_stacked_baseline_animation,
    create_interactive_raw_3d_transform_preview,
    create_3d_overlay_animation,
    create_3d_contour_plots,
)

def main():
    print("CSV Loader + Animation + SNR Analysis")
    print("-" * 40)

    file_paths = select_csv_files_via_gui(initial_dir=Path.cwd())
    if not file_paths:
        print("No CSV files selected. Exiting.")
        return

    dataframes = load_csv_files(file_paths)
    source_path_lookup = build_source_path_lookup(file_paths)

    if not dataframes:
        print("No files were successfully loaded.")
        return

    run_output_dir = create_run_output_dir()
    print(f"Run output directory: {run_output_dir}")

    # ── Step 0: interactive raw 3D preview with y/z transform control ────
    create_interactive_raw_3d_transform_preview(
        dataframes,
        time_col=TIME_COL,
        rp_col=X_COL,
        ind_col=Y_COL,
        frame_step=FRAME_STEP,
        show_plot=SHOW_PLOTS,
        source_path_lookup=source_path_lookup,
    )

    # ── Step 1: SNR analysis ──────────────────────────────────────────────
    print("\n" + "=" * 40)
    print(f"Loaded {len(dataframes)} file(s) successfully:")
    for filename, df in dataframes.items():
        print(f"\n{filename}:")
        print(df.head())

    snr_df, _ = analyze_snr(
        dataframes,
        x_col=X_COL,
        y_col=Y_COL,
        baseline_window=101,
        max_peaks=6,
        min_peak_distance=100,
        sigma_threshold_multiplier=3.0,
        verbose=False,
    )

    if snr_df.empty:
        print("No SNR results generated.")
    else:
        print("\nSNR by file (highest first):")
        print(snr_df[["file", "snr_db", "snr_linear", "signal_amplitude",
                       "noise_sigma", "peak_count"]].head(50))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snr_path = run_output_dir / f"snr_analysis_{timestamp}.csv"
        snr_df.to_csv(snr_path, index=False)
        print(f"Saved SNR analysis CSV: {snr_path}")

        create_snr_visualizations(
            snr_df,
            output_dir=run_output_dir,
            show_plot=SHOW_PLOTS,
            save_plots=True,
        )

    # ── Step 2: animations and 3D plots ───────────────────────────────────
    # Always use only the raw columns from CSV for all analysis/animation input.
    anim_kwargs = dict(
        x_col=X_COL, y_col=Y_COL, snr_df=snr_df,
        frame_step=FRAME_STEP, output_dir=run_output_dir,
        save_gif=SAVE_GIFS, show_plot=SHOW_PLOTS,
    )

    create_overlay_animation(dataframes, **anim_kwargs)
    create_stacked_animation(dataframes, **anim_kwargs)
    create_stacked_baseline_animation(dataframes, **anim_kwargs)

    create_3d_overlay_animation(
        dataframes,
        time_col=TIME_COL,
        x_col=X_COL,
        y_col=Y_COL,
        snr_df=snr_df,
        frame_step=FRAME_STEP,
        output_dir=run_output_dir,
        save_gif=SAVE_GIFS,
        show_plot=SHOW_PLOTS,
    )

    create_3d_contour_plots(
        dataframes,
        time_col=TIME_COL,
        x_col=X_COL,
        y_col=Y_COL,
        snr_df=snr_df,
        output_dir=run_output_dir,
        save_plots=True,
        show_plot=SHOW_PLOTS,
    )

if __name__ == "__main__":
    main()
