# ---------------------------------------------------------------------------
# speed_calculation.py — Interactive scan-timing tool
#
# Load a raw scan CSV (e.g. speed40_delay5.csv), plot a sensor trace against
# the computer timestamp, and click on the plot to drop vertical dashed
# markers. The elapsed time (seconds) between consecutive markers is drawn on
# the plot, so scan speed can be read off directly from feature-to-feature
# timing.
# ---------------------------------------------------------------------------

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# File selection + loading
# ---------------------------------------------------------------------------

def select_scan_csv(initial_dir="."):
    """
    Open a GUI file picker to choose a single scan CSV.

    Returns:
        str | None: The selected file path, or None if canceled/unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"GUI file picker unavailable ({exc}).")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            title="Select a scan CSV",
            initialdir=str(Path(initial_dir).resolve()),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    return selected or None


def load_scan_csv(path):
    """Load a scan CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scan CSV not found: {path}")
    return pd.read_csv(path)


def _resolve_column(df, requested):
    """
    Find a column in df matching `requested`, ignoring case, spaces, and
    underscores (so "Sensor 2" matches "sensor2").
    """
    def norm(name):
        return str(name).strip().lower().replace(" ", "").replace("_", "")

    if requested in df.columns:
        return requested

    target = norm(requested)
    for col in df.columns:
        if norm(col) == target:
            return col
    raise KeyError(
        f"Column {requested!r} not found. Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Interactive interval picker
# ---------------------------------------------------------------------------

class TimeIntervalPicker:
    """
    Plot a sensor trace and let the user click to place vertical dashed lines.
    The elapsed time (seconds) between consecutive lines is annotated on the
    plot to two decimal places.

    Controls:
        left click  — add a marker (snapped to the nearest sample)
        right click — remove the nearest marker
    """

    def __init__(self, df, x_col="timestamp_computer", y_col="sensor2", title=None,
                 snap_to_sample=False):
        import matplotlib.pyplot as plt

        self.snap_to_sample = snap_to_sample

        self.x_col = _resolve_column(df, x_col)
        self.y_col = _resolve_column(df, y_col)

        data = df[[self.x_col, self.y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        data = data.sort_values(self.x_col).reset_index(drop=True)
        if data.empty:
            raise ValueError("No numeric (x, y) samples to plot.")

        # Reference x to the first sample so the axis reads as elapsed seconds,
        # while intervals are still computed from the raw timestamps.
        self.x = data[self.x_col].to_numpy()
        self.y = data[self.y_col].to_numpy()
        self.x0 = float(self.x[0])

        self.marker_x = []          # raw x (timestamp) of each marker
        self._line_artists = []
        self._text_artists = []

        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.ax.plot(self.x - self.x0, self.y, lw=0.8, color="#1f77b4")
        self.ax.set_xlabel(f"Elapsed time (s)  [{self.x_col}]")
        self.ax.set_ylabel(self.y_col)
        self.ax.set_title(title or "Click to mark positions — time between markers shown in seconds")
        self.ax.grid(True, alpha=0.3)

        self._cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.tight_layout()

    # -- event handling -----------------------------------------------------
    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return

        clicked_raw = event.xdata + self.x0
        if self.snap_to_sample:
            clicked_raw = float(self.x[int((abs(self.x - clicked_raw)).argmin())])

        if event.button == 1:  # left click: add
            self.marker_x.append(clicked_raw)
        elif event.button == 3 and self.marker_x:  # right click: remove nearest
            idx = min(range(len(self.marker_x)),
                      key=lambda i: abs(self.marker_x[i] - clicked_raw))
            self.marker_x.pop(idx)
        else:
            return

        self._redraw()

    # -- drawing ------------------------------------------------------------
    def _redraw(self):
        for artist in self._line_artists + self._text_artists:
            artist.remove()
        self._line_artists.clear()
        self._text_artists.clear()

        self.marker_x.sort()
        y_lo, y_hi = self.ax.get_ylim()
        y_text = y_lo + 0.95 * (y_hi - y_lo)

        for raw_x in self.marker_x:
            line = self.ax.axvline(raw_x - self.x0, color="k", ls="--", lw=1)
            self._line_artists.append(line)

        for i in range(1, len(self.marker_x)):
            dt = self.marker_x[i] - self.marker_x[i - 1]
            mid = 0.5 * (self.marker_x[i] + self.marker_x[i - 1]) - self.x0
            text = self.ax.text(
                mid, y_text, f"{dt:.2f} s",
                ha="center", va="top", fontsize=9, color="k",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85),
            )
            self._text_artists.append(text)

        self.fig.canvas.draw_idle()

    # -- results ------------------------------------------------------------
    @property
    def intervals(self):
        """List of elapsed seconds between consecutive markers."""
        xs = sorted(self.marker_x)
        return [round(xs[i] - xs[i - 1], 2) for i in range(1, len(xs))]

    def print_intervals(self):
        """Print each marker-to-marker interval in seconds."""
        ivals = self.intervals
        if not ivals:
            print("No intervals yet — add at least two markers.")
            return
        for i, dt in enumerate(ivals, start=1):
            print(f"Interval {i}: {dt:.2f} s")
        print(f"Total marked span: {sum(ivals):.2f} s over {len(ivals)} interval(s).")


def launch_time_interval_picker(source, x_col="timestamp_computer", y_col="sensor2",
                                title=None, snap_to_sample=False):
    """
    Convenience entry point: accept a DataFrame or a CSV path and return a live
    TimeIntervalPicker. Requires an interactive matplotlib backend
    (e.g. run `%matplotlib qt` in the notebook first).

    Set snap_to_sample=True to snap each marker to the nearest recorded sample
    (note: with step-and-settle scans this can jump a marker across a timing gap).
    """
    df = source if isinstance(source, pd.DataFrame) else load_scan_csv(source)
    return TimeIntervalPicker(df, x_col=x_col, y_col=y_col, title=title,
                              snap_to_sample=snap_to_sample)
