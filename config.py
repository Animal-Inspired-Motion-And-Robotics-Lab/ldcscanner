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

# --- Inductor group color map ---------------------------------------------
# Example: "flex_04": "#1f77b4" (blue), "flex_03": "#2ca02c" (green)
INDUCTOR_COLOR_MAP = {
    "flex_04": "#0fd4ea",
    "flex_08": "#0958c7",
    "flex_12": "#100386",
    "flex08_100": "#2ff119",
    "flex08_220": "#059e33",
    "flex08_440": "#03611A",
    "flex08_660": "#023712",
}

FREQUENCY_COLOR_MAP = {
    "1.1": "#100386",
    "2.4": "#0958c7",
    "3.2": "#0fd4ea"
}

# --- Material group color map -------------------------------------------------
MATERIAL_COLOR_MAP = {
    "Stainless Steel": "#b412f9",  # blue
    "Aluminum": "#801fee",  # green
    "Titanium": "#4E0696",  # orange
}

# --- Crack color map -------------------------------------------------
CRACK_COLOR_MAP = {
    "Crack 1": "#eed40c",
    "Crack 2": "#ff7f0e",
    "Crack 3": "#d0260f", 
}