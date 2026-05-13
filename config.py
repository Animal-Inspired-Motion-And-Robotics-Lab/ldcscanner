# ---------------------------------------------------------------------------
# config.py — Runtime toggles and shared constants
#
# This is the first place to look when tuning scan behaviour or adapting the
# script to a new CSV format.  Column names must match the CSV headers
# exactly, including any leading/trailing spaces.
# ---------------------------------------------------------------------------

# --- Execution speed -------------------------------------------------------

FAST_MODE = False
FRAME_STEP = 5 if FAST_MODE else 1   # 1 = highest quality GIF, 5 = faster
SHOW_PLOTS = False if FAST_MODE else True
SAVE_GIFS = True

# --- Sensor column names ---------------------------------------------------

# Edit these if the CSV headers change.
X_COL = "sensor1"
Y_COL = " sensor2"   # leading space is intentional — matches CSV
TIME_COL = "timestamp"

# --- SNR display mode ------------------------------------------------------
# Controls how SNR is shown across plots/animations.
# Allowed values: "db" or "linear"
SNR_DISPLAY_MODE = "linear"

# Controls grouping in static SNR charts only (analysis remains per-file).
# Allowed values: "grouped" or "per_file"
SNR_CHART_GROUP_MODE = "grouped"

# --- Fallback column candidates --------------------------------------------
# Searched in order when the exact column name is not known (e.g. raw signal
# detection in the labeling step).

X_CANDIDATES = ["timestamp", "time", "sample", "index"]
Y_CANDIDATES = [
    "sensor2_smooth_rot", " sensor2_smooth_rot",
    "sensor2",            " sensor2",
    "sensor1_smooth_rot", " sensor1_smooth_rot",
    "sensor1",            " sensor1",
]

# --- Crack label definitions -----------------------------------------------

CRACK_LABELS = ("Crack 1", "Crack 2", "Crack 3")

LABEL_COLOR_MAP = {
    "Crack 1":    "#d62728",
    "Crack 2":    "#2ca02c",
    "Crack 3":    "#ff7f0e",
    "Not a crack": "#7f7f7f",
}

LABEL_SHORT_MAP = {
    "Crack 1":    "C1",
    "Crack 2":    "C2",
    "Crack 3":    "C3",
    "Not a crack": "N",
}

# Maps keyboard / text input to canonical label strings.
LABEL_INPUT_MAP = {
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
