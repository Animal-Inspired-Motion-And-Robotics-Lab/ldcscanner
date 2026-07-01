# ---------------------------------------------------------------------------
# scanner/config.py — Live scanner runtime constants (serial / CSV / plot)
#
# First place to look when tuning the live GUI or adapting it to new hardware.
# This is separate from the repo-root ``config.py`` (which configures the
# offline SNR/Scanimator analysis pipeline) — the two must not be confused.
# ---------------------------------------------------------------------------


# --- Plot line-width scaling (percent) ------------------------------------
# 100 means "use the current/original width".
SURFACE_TRACE_LINE_WIDTH_PERCENT = 100
RIGHT_PLOT_MAIN_LINE_WIDTH_PERCENT = 200
RIGHT_PLOT_RECENT_LINE_WIDTH_PERCENT = 200
CRACK_PLOT_LINE_WIDTH_PERCENT = 300


def scale_line_width(base_width, percent):
    """Return ``base_width`` scaled by a percentage (100 => unchanged)."""
    try:
        return float(base_width) * (float(percent) / 100.0)
    except (TypeError, ValueError):
        return float(base_width)


# --- Serial link -----------------------------------------------------------
SERIAL_PORT = "COM6"                # preselected port if detected at launch
BAUDRATE = 9600                     # preselected baud rate
BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400]

# --- CSV output ------------------------------------------------------------
CSV_FILE = "test.csv"               # default output file (editable in the UI)

# --- CSV replay (simulation) ----------------------------------------------
REPLAY_DEFAULT_SPEED = 1.0          # playback speed multiplier for CSV simulation
REPLAY_SCRUB_ACCEL = 1.3           # arrow-key hold: step multiplier per auto-repeat
REPLAY_SCRUB_MAX_STEP = 500        # max samples scrubbed per auto-repeat while holding

# --- Live data buffers -----------------------------------------------------
MAX_POINTS = 5000                   # ring-buffer length for every sample deque

# --- Plot / readout tuning -------------------------------------------------
DISPLAY_LAG_POINTS = 1              # skip newest N points in plots to reduce right-edge jitter
RECENT_FADE_POINTS = 100            # length of the highlighted red->white trajectory tail
AVERAGE_UPDATE_INTERVAL_SEC = 5.0
RP_ZERO_EPSILON = 1e-12             # R_p span at/below this counts as "flat/zero"
RP_ZERO_FALLBACK_WINDOW = 100       # samples inspected when deciding R_p is flat
SERIAL_RESPONSE_MAX_LINES = 20
SERIAL_RESPONSE_BOX_MAX_HEIGHT = 180


# Bounds the ``from scanner.config import *`` used by the entry file so the UI's
# bare-name constant references keep resolving after the refactor.
__all__ = [
    "SURFACE_TRACE_LINE_WIDTH_PERCENT",
    "RIGHT_PLOT_MAIN_LINE_WIDTH_PERCENT",
    "RIGHT_PLOT_RECENT_LINE_WIDTH_PERCENT",
    "CRACK_PLOT_LINE_WIDTH_PERCENT",
    "scale_line_width",
    "SERIAL_PORT",
    "BAUDRATE",
    "BAUD_RATES",
    "CSV_FILE",
    "REPLAY_DEFAULT_SPEED",
    "REPLAY_SCRUB_ACCEL",
    "REPLAY_SCRUB_MAX_STEP",
    "MAX_POINTS",
    "DISPLAY_LAG_POINTS",
    "RECENT_FADE_POINTS",
    "AVERAGE_UPDATE_INTERVAL_SEC",
    "RP_ZERO_EPSILON",
    "RP_ZERO_FALLBACK_WINDOW",
    "SERIAL_RESPONSE_MAX_LINES",
    "SERIAL_RESPONSE_BOX_MAX_HEIGHT",
]
