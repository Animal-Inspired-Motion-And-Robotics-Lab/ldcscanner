# ---------------------------------------------------------------------------
# scanner/state.py — All live runtime state + sample ingestion
#
# ``ScannerState`` is the central hub: the live sample ring buffers, derived
# readout text, latest non-zero values, view flags, and the cached snapshot
# window all live here.  No Qt — the UI reads/writes it through ``state.x``.
# ---------------------------------------------------------------------------

import time
from collections import deque

import numpy as np

from scanner.config import MAX_POINTS, RECENT_FADE_POINTS
from scanner.parsing import parse_crack_event


class ScannerState:
    """All mutable runtime state in one object: live sample buffers, derived
    readout text, latest non-zero values, and view flags.

    Holding it here lets handlers mutate shared state through ``state.x`` rather
    than scattering ``global`` declarations across the module.
    """

    def __init__(self):
        # Live sample ring buffers (sensor1 = R_p, sensor2 = L).
        self.timestamps = deque(maxlen=MAX_POINTS)
        self.sensor1 = deque(maxlen=MAX_POINTS)
        self.sensor2 = deque(maxlen=MAX_POINTS)
        self.crack_times = deque(maxlen=MAX_POINTS)
        self.crack_mags = deque(maxlen=MAX_POINTS)
        self.crack_sizes = deque(maxlen=MAX_POINTS)

        # Control flags.
        self.paused = False
        self.write_to_file_enabled = False

        # Readout text + most-recent non-zero values shown on the readout line.
        self.incoming_history = deque(maxlen=3)
        self.readout_text = "Incoming: waiting for data..."
        self.average_text = "Average: waiting for data..."
        self.reject_reason = None
        self.latest_nonzero_mag = None
        self.latest_nonzero_width = None
        self.latest_nonzero_crack_size = None
        self.pending_serial_out = ""
        self.pending_response = ""
        self.last_average_update_time = 0.0

        # View tracking.
        self.xy_start_index = 0
        self.right_x_mode = "RP"                 # "RP" or "TIME"
        self.right_plot_auto_time_fallback = False
        self.crack_y_mode = "mag"                # "mag" or "crack_size"
        self.ui_start_monotonic = time.monotonic()
        self.scrub_velocity = 1.0                # arrow-key time-scrub step (grows on hold)

        # Latest plotted data window, cached so the Snapshot export reproduces
        # exactly what is currently on screen (time, R_p, L + crack events).
        self.snap_time = np.array([], dtype=float)
        self.snap_rp = np.array([], dtype=float)
        self.snap_l = np.array([], dtype=float)
        self.snap_crack_times = np.array([], dtype=float)
        self.snap_crack_mags = np.array([], dtype=float)
        self.snap_crack_sizes = np.array([], dtype=float)

    def reset(self):
        """Clear buffered data so both plots restart from a clean state."""
        self.timestamps.clear()
        self.sensor1.clear()
        self.sensor2.clear()
        self.crack_times.clear()
        self.crack_mags.clear()
        self.crack_sizes.clear()
        self.xy_start_index = 0
        self.ui_start_monotonic = time.monotonic()
        self.average_text = f"Avg last {RECENT_FADE_POINTS}: waiting for data..."
        self.last_average_update_time = time.monotonic()
        self.latest_nonzero_mag = None
        self.latest_nonzero_width = None

    def ingest_crack_event(self, line):
        """Record a crack event from ``line`` (uses last sample time as fallback)."""
        fallback_t = self.timestamps[-1] if self.timestamps else None
        event = parse_crack_event(line, fallback_t)
        if event is None:
            return
        t_event, mag_val, crack_size_val = event
        self.crack_times.append(t_event)
        self.crack_mags.append(mag_val)
        self.crack_sizes.append(crack_size_val if crack_size_val is not None else 0.0)

    def stage_command_exchange(self, serial_out, response):
        """Queue one command/response pair to annotate the next logged sample."""
        self.pending_serial_out = str(serial_out or "")
        self.pending_response = str(response or "")

    def consume_pending_command_exchange(self):
        """Return and clear any queued command/response metadata."""
        serial_out = self.pending_serial_out
        response = self.pending_response
        self.pending_serial_out = ""
        self.pending_response = ""
        return serial_out, response

    def ingest_sample(self, t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val,
                      csv_logger, log=True):
        """Store one parsed sample, optionally log it, and refresh readout text.

        ``log=False`` ingests into the buffers without writing to the CSV — used
        when rebuilding the view during an arrow-key time scrub of a replay.
        """
        self.timestamps.append(t)
        self.sensor1.append(s1)
        self.sensor2.append(s2)

        if log and self.write_to_file_enabled:
            serial_out, response = self.consume_pending_command_exchange()
            csv_logger.write_sample(
                t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val,
                serial_out, response
            )

        if mag_val is not None and float(mag_val) != 0.0:
            self.latest_nonzero_mag = float(mag_val)
        if width_val is not None and float(width_val) != 0.0:
            self.latest_nonzero_width = float(width_val)
        if crack_size_val is not None and float(crack_size_val) != 0.0:
            self.latest_nonzero_crack_size = float(crack_size_val)

        mag_text = f"{self.latest_nonzero_mag:.6f}" if self.latest_nonzero_mag is not None else "n/a"
        width_text = f"{self.latest_nonzero_width:.6f}" if self.latest_nonzero_width is not None else "n/a"
        crack_size_text = f"{self.latest_nonzero_crack_size:.6f}" if self.latest_nonzero_crack_size is not None else "n/a"
        input_text = self.reject_reason if self.reject_reason is not None else "n/a"
        self.readout_text = (
            f"Incoming: t={t:.3f} | s1={s1:.6f} | s2={s2:.6f} | "
            f"mag={mag_text} | width={width_text} | "
            f"crack_size={crack_size_text} | rejected={input_text}"
        )
