import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

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


def create_overlay_animation(
    dataframes,
    x_col="sensor1_smooth_rot",
    y_col=" sensor2_smooth_rot",
    interval=30,
    tail_length=100,
):
    """
    Animate overlaid XY traces from multiple dataframes.

    Each dataframe is drawn in a unique color and updated frame-by-frame.
    """
    if not dataframes:
        print("No dataframes available for animation.")
        return

    prepared = []
    global_x_min = np.inf
    global_x_max = -np.inf
    global_y_min = np.inf
    global_y_max = -np.inf

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

            prepared.append((filename, x_vals, y_vals))
            global_x_min = min(global_x_min, np.min(x_vals))
            global_x_max = max(global_x_max, np.max(x_vals))
            global_y_min = min(global_y_min, np.min(y_vals))
            global_y_max = max(global_y_max, np.max(y_vals))

        except KeyError as err:
            print(f"Skipping {filename}: {err}")

    if not prepared:
        print("No valid files had the required columns for animation.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Overlay Animation: sensor1_smooth_rot vs sensor2_smooth_rot")
    ax.set_xlabel("sensor1_smooth_rot")
    ax.set_ylabel("sensor2_smooth_rot")

    x_pad = (global_x_max - global_x_min) * 0.05 if global_x_max > global_x_min else 1
    y_pad = (global_y_max - global_y_min) * 0.05 if global_y_max > global_y_min else 1
    ax.set_xlim(global_x_min - x_pad, global_x_max + x_pad)
    ax.set_ylim(global_y_min - y_pad, global_y_max + y_pad)

    color_map = plt.cm.get_cmap("tab10", len(prepared))
    artists = []
    max_frames = max(len(item[1]) for item in prepared)

    for i, (filename, _, _) in enumerate(prepared):
        base_color = color_map(i)
        tail_segments = LineCollection([], linewidths=2.0, zorder=2)
        ax.add_collection(tail_segments)
        point, = ax.plot([], [], marker="o", ms=5, color=base_color, zorder=3, label=filename)
        artists.append((tail_segments, point, base_color))

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    def init():
        for tail_segments, point, _ in artists:
            tail_segments.set_segments([])
            tail_segments.set_color([])
            point.set_data([], [])
        return [artist for tail_segments, point, _ in artists for artist in (tail_segments, point)]

    def update(frame):
        for (_, x_vals, y_vals), (tail_segments, point, base_color) in zip(prepared, artists):
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
        return [artist for tail_segments, point, _ in artists for artist in (tail_segments, point)]

    anim = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )

    # Keep a reference alive for the life of the figure.
    fig._overlay_anim = anim

    plt.tight_layout()
    plt.show()


def main():
    print("CSV File Loader + Overlay Animator")
    print("-" * 40)
    
    # Requested files to load and animate.
    user_input = "flex_4.csv,flex_8.csv,flex_12.csv"
    
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

        create_overlay_animation(
            dataframes,
            x_col="sensor1_smooth_rot",
            y_col=" sensor2_smooth_rot",
        )
    else:
        print("No files were successfully loaded.")


if __name__ == "__main__":
    main()
