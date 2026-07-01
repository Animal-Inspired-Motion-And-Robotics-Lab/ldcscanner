# ---------------------------------------------------------------------------
# scanner/data_sources.py — Live serial port + simulated CSV replay
#
# Both classes expose the same minimal interface the read loop relies on
# (``is_connected`` / ``in_waiting`` / ``readline`` / ``reset_input_buffer`` /
# ``write`` / ``disconnect`` / ``close``), so the UI's ``source`` pointer can
# swap between them transparently.  No Qt: they notify the UI via an optional
# ``on_change`` callback.
# ---------------------------------------------------------------------------

import csv
import os
import time
from collections import deque

import serial
from serial.tools import list_ports

from scanner.config import MAX_POINTS, REPLAY_DEFAULT_SPEED


class SerialManager:
    """Wraps a pyserial port that may or may not be open.

    The module-level reference (``serial_mgr``) never gets reassigned —
    ``connect()`` / ``disconnect()`` mutate ``self.port`` — so callers can hold
    onto the manager indefinitely.  Every I/O method is guarded: it is a safe
    no-op when disconnected, and any hardware exception (device unplugged
    mid-stream) auto-disconnects and fires ``on_change`` so the UI can react.
    """

    supports_commands = True            # live link accepts two-way console commands

    def __init__(self, baud_default):
        self.port = None                    # serial.Serial or None
        self.port_name = None
        self.baudrate = baud_default
        self.status_text = "Disconnected"
        self.on_change = None               # optional callback() set by the UI

    @property
    def is_connected(self):
        return self.port is not None and self.port.is_open

    @staticmethod
    def available_ports():
        """Return a list of ``(device, description)`` for all detected ports."""
        return [(p.device, p.description) for p in list_ports.comports()]

    def connect(self, port_name, baudrate):
        """Open ``port_name`` at ``baudrate``.  Returns True on success."""
        self.disconnect()
        try:
            self.port = serial.Serial(port_name, baudrate, timeout=1)
        except (serial.SerialException, ValueError, OSError) as exc:
            self.port = None
            self.status_text = f"Connect failed: {exc}"
            self._notify()
            return False
        self.port_name = port_name
        self.baudrate = baudrate
        self.status_text = f"Connected: {port_name} @ {baudrate}"
        self._notify()
        return True

    def disconnect(self, reason=None):
        """Close the port (if open) and update status text."""
        if self.port is not None:
            try:
                if self.port.is_open:
                    self.port.close()
            except Exception:
                pass
        self.port = None
        self.status_text = f"Disconnected ({reason})" if reason else "Disconnected"
        self._notify()

    @property
    def in_waiting(self):
        if not self.is_connected:
            return 0
        try:
            return self.port.in_waiting
        except (serial.SerialException, OSError):
            self.disconnect("device lost")
            return 0

    def readline(self):
        if not self.is_connected:
            return b""
        try:
            return self.port.readline()
        except (serial.SerialException, OSError):
            self.disconnect("device lost")
            return b""

    def write(self, data):
        """Write ``data`` and flush.  Raises SerialException if not connected."""
        if not self.is_connected:
            raise serial.SerialException("Not connected")
        self.port.write(data)
        self.port.flush()

    def reset_input_buffer(self):
        if not self.is_connected:
            return
        try:
            self.port.reset_input_buffer()
        except (serial.SerialException, OSError):
            self.disconnect("device lost")

    def close(self):
        self.disconnect()

    def _notify(self):
        if self.on_change is not None:
            self.on_change()


class CsvReplaySource:
    """Replays a recorded scanner CSV as if it were arriving live over serial.

    Exposes the same minimal interface the read loop relies on
    (``is_connected`` / ``in_waiting`` / ``readline`` / ``reset_input_buffer`` /
    ``write`` / ``disconnect`` / ``close``), so the rest of the program treats a
    replay exactly like a real port.  Each telemetry row is reconstructed into a
    keyed ``t:..>rp:..>l:..>...`` line — identical in shape to what the scanner
    emits — and released according to the recorded ``timestamp_computer``
    cadence, scaled by ``speed``.  Playback advances only while the read loop is
    polling, so the P-key pause (which stops polling) naturally freezes it.
    """

    supports_commands = False           # nothing to talk back to in a recording

    # CSV columns -> reconstructed keyed-line fields (sensor1=R_p, sensor2=L).
    _OPTIONAL_FIELDS = (("mag", "mag"), ("width", "width"),
                        ("crack_x", "crack_x"), ("crack_size", "crack_size"))

    def __init__(self, path, speed=REPLAY_DEFAULT_SPEED):
        self.path = str(path)
        self.speed = max(float(speed), 1e-6)
        self.port_name = os.path.basename(self.path)
        self.baudrate = None
        self.status_text = "CSV simulation: idle"
        self.on_change = None
        self.finished = False

        self._rows = self._load(self.path)          # list of (offset_sec, line_bytes)
        if not self._rows:
            raise ValueError("No replayable telemetry rows found in CSV")
        self._idx = 0                               # next row not yet released
        self._queue = deque()                       # released, not yet read
        self._playback_t = 0.0                      # virtual seconds into recording
        self._last_real = None                      # monotonic stamp of last advance
        self._active = False

    @classmethod
    def _load(cls, path):
        """Read ``path`` into ``[(offset_seconds, keyed_line_bytes), ...]``.

        Offsets come from the ``timestamp_computer`` column so playback matches
        the original real-world arrival cadence; rows missing the required
        ``timestamp``/``sensor2`` (L) values are skipped, and a row without a
        usable host timestamp inherits the previous offset (emitted together).
        """
        rows = []
        base_t = None
        prev_offset = 0.0
        with open(path, newline="") as handle:
            for raw in csv.DictReader(handle):
                t = (raw.get("timestamp") or "").strip()
                l_val = (raw.get("sensor2") or "").strip()
                if not t or not l_val:
                    continue
                rp = (raw.get("sensor1") or "").strip() or "0"
                parts = [f"t:{t}", f"rp:{rp}", f"l:{l_val}"]
                for key, column in cls._OPTIONAL_FIELDS:
                    value = (raw.get(column) or "").strip()
                    if value:
                        parts.append(f"{key}:{value}")
                line = ">".join(parts).encode("utf-8")

                tc_text = (raw.get("timestamp_computer") or "").strip()
                try:
                    tc_val = float(tc_text)
                except ValueError:
                    tc_val = None
                if tc_val is None:
                    offset = prev_offset
                else:
                    if base_t is None:
                        base_t = tc_val
                    # Clamp monotonic so out-of-order stamps never rewind playback.
                    offset = max(tc_val - base_t, prev_offset)
                prev_offset = offset
                rows.append((offset, line))
        return rows

    @property
    def is_connected(self):
        return self._active and not self.finished

    def start(self):
        """Begin (or restart) playback from the top of the recording."""
        self._idx = 0
        self._queue.clear()
        self._playback_t = 0.0
        self._last_real = None
        self._active = True
        self.finished = False
        self.status_text = (
            f"Simulating {self.port_name} @ {self.speed:g}x ({len(self._rows)} samples)"
        )
        self._notify()

    def _advance(self):
        """Move the playback clock forward and release any now-due rows."""
        now = time.monotonic()
        if self._last_real is None:
            self._last_real = now
            return
        self._playback_t += (now - self._last_real) * self.speed
        self._last_real = now
        while self._idx < len(self._rows) and self._rows[self._idx][0] <= self._playback_t:
            self._queue.append(self._rows[self._idx][1])
            self._idx += 1

    @property
    def in_waiting(self):
        if not self.is_connected:
            return 0
        self._advance()
        if not self._queue and self._idx >= len(self._rows):
            self._finish("end of file")
            return 0
        return len(self._queue)

    def readline(self):
        return self._queue.popleft() if self._queue else b""

    # --- Manual time scrubbing (arrow keys) --------------------------------
    @property
    def position(self):
        """Number of rows played so far (0..total)."""
        return self._idx

    @property
    def total(self):
        return len(self._rows)

    def window_lines(self, max_points=MAX_POINTS):
        """Keyed lines for the visible window ending at the current position."""
        start = max(0, self._idx - int(max_points))
        return [line for _, line in self._rows[start:self._idx]]

    def set_position(self, index):
        """Jump playback to ``index`` (clamped) and re-align the clock so that
        resuming auto-play continues seamlessly from the new spot."""
        index = max(0, min(int(index), len(self._rows)))
        self._idx = index
        self._queue.clear()
        self._playback_t = self._rows[index - 1][0] if index > 0 else 0.0
        self._last_real = None              # re-base on the next auto-advance poll
        if index < len(self._rows):
            # Scrubbing back into the recording re-arms it so P can resume play.
            self.finished = False
            self._active = True
        self.status_text = f"Scrubbed to sample {index}/{len(self._rows)}"
        self._notify()
        return index

    def reset_input_buffer(self):
        # Freeze the playback clock (used while paused) without dropping queued
        # rows; the next poll re-bases timing from "now" so no jump occurs.
        self._last_real = None

    def write(self, data):
        raise serial.SerialException("CSV simulation: command sending disabled")

    def disconnect(self, reason=None):
        self._finish(reason or "stopped")

    def _finish(self, reason):
        self._active = False
        self.finished = True
        self.status_text = f"Simulation finished ({reason})"
        self._notify()

    def close(self):
        self._active = False

    def _notify(self):
        if self.on_change is not None:
            self.on_change()
