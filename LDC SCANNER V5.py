# ---------------------------------------------------------------------------
# LDC SCANNER V5 — Live eddy-current scanner readout and serial console
#
# Reads sensor telemetry from the scanner robot over a serial port and streams
# it into three live views (a 3D R_p/L surface, a phase-space/time trace, and a
# crack-event plot), optionally logs every sample to CSV, and provides a
# two-way serial command console.
#
# Sections, top to bottom:
#   1. Configuration       — serial / CSV / plot constants gathered in one place
#   2. Serial parsing      — pure functions that turn a received line into numbers
#   3. Surface geometry    — pure helpers that build the 3D ribbon mesh
#   4. CsvLogger           — owns the output CSV file
#   5. SerialManager       — owns the serial port (connect/disconnect at runtime)
#   6. ScannerState        — all live runtime state + sample ingestion
#   7. Runtime objects     — create the logger / state / serial manager
#   8. Qt user interface   — widget construction (built once, top to bottom)
#   9. Handlers + loop     — key presses, serial read, command console, redraw timer
# ---------------------------------------------------------------------------

import csv
import os
import time
from collections import deque

import numpy as np
import serial
from serial.tools import list_ports
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

pg.setConfigOptions(antialias=True)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# --- Serial link -----------------------------------------------------------
SERIAL_PORT = "COM6"                # preselected port if detected at launch
BAUDRATE = 9600                     # preselected baud rate
BAUD_RATES = [9600, 19200, 38400, 57600, 115200, 230400]

# --- CSV output ------------------------------------------------------------
CSV_FILE = "test.csv"               # default output file (editable in the UI)

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


# ---------------------------------------------------------------------------
# 2. Serial parsing (pure functions)
# ---------------------------------------------------------------------------

def parse_keyed_fields(line):
    """Parse a ``key:value>key:value|...`` line into a {key: float} dict.

    Anything after the first ``|`` is ignored, and segments without a numeric
    value are skipped.  Keys are lower-cased and stripped.
    """
    fields = {}
    payload = line.split("|", 1)[0]
    for segment in payload.split(">"):
        if not segment or ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        try:
            fields[key.strip().lower()] = float(value.strip())
        except ValueError:
            continue
    return fields


def parse_serial_line(line):
    """Parse one telemetry line into ``(t, rp, l, mag, width, crack_x, crack_size)``.

    Supports the keyed format (``t:..>l:..>rp:..>...``) and the legacy
    whitespace format (``t s1 s2``).  ``t`` and ``l`` are required; ``rp``
    defaults to 0.  Optional fields are ``None`` when absent/non-finite.
    Raises ``ValueError`` when the line cannot be parsed.
    """
    if not line:
        raise ValueError("Empty serial line")

    if ">" in line and ":" in line:
        fields = parse_keyed_fields(line)

        t = fields.get("t")
        l_val = fields.get("l")
        rp_val = fields.get("rp", 0.0)
        mag_val = fields.get("mag")
        width_val = fields.get("width")
        crack_x_val = fields.get("crack_x")
        crack_size_val = fields.get("crack_size")
        if t is None or l_val is None:
            raise ValueError("Missing required keyed fields")
        if not np.isfinite(t) or not np.isfinite(l_val):
            raise ValueError("Non-finite required keyed fields")
        if not np.isfinite(rp_val):
            rp_val = 0.0
        if mag_val is not None and not np.isfinite(mag_val):
            mag_val = None
        if width_val is not None and not np.isfinite(width_val):
            width_val = None
        if crack_x_val is not None and not np.isfinite(crack_x_val):
            crack_x_val = None
        if crack_size_val is not None and not np.isfinite(crack_size_val):
            crack_size_val = None
        return float(t), float(rp_val), float(l_val), mag_val, width_val, crack_x_val, crack_size_val

    t, s1, s2 = map(float, line.split())
    if not np.isfinite(t) or not np.isfinite(s1) or not np.isfinite(s2):
        raise ValueError("Non-finite whitespace fields")
    return t, s1, s2, None, None, None, None


def extract_reject_reason(line):
    """Return the value following ``reject_reason=`` in a line, or ``None``.

    The value is trimmed at the first separator and stripped of brackets; a
    literal ``-`` (no reason) returns ``None``.
    """
    if not line:
        return None

    text = line.strip()
    marker = "reject_reason="
    idx = text.lower().find(marker)
    if idx < 0:
        return None

    value = text[idx + len(marker):].strip()
    if not value:
        return None

    for sep in ("|", ">", " ", "\t", ","):
        sep_index = value.find(sep)
        if sep_index >= 0:
            value = value[:sep_index]
            break

    parsed_value = value.strip().strip("[]")
    if parsed_value == "-":
        return None
    return parsed_value or None


def parse_crack_event(line, fallback_t):
    """Return ``(t, mag, crack_size)`` for a crack event line, or ``None``.

    A crack event is a line carrying a finite, non-zero ``mag``.  When the line
    has no ``t``, ``fallback_t`` (the most recent sample time) is used.
    """
    if ">" not in line or ":" not in line:
        return None

    fields = parse_keyed_fields(line)
    mag_val = fields.get("mag")
    if mag_val is None or not np.isfinite(mag_val) or float(mag_val) == 0.0:
        return None

    t_val = fields.get("t")
    if t_val is None:
        t_val = fallback_t
    if t_val is None or not np.isfinite(t_val):
        return None

    crack_size_val = fields.get("crack_size")
    if crack_size_val is None or not np.isfinite(crack_size_val):
        crack_size_val = None

    return float(t_val), float(mag_val), (float(crack_size_val) if crack_size_val is not None else None)


# ---------------------------------------------------------------------------
# 3. Surface geometry (pure functions)
# ---------------------------------------------------------------------------

def has_usable_rp(rp_vals, eps=RP_ZERO_EPSILON):
    """True when recent R_p values vary by more than ``eps`` (i.e. not flat)."""
    vals = np.asarray(rp_vals, dtype=float)
    if vals.size < 2:
        return False
    window = min(int(RP_ZERO_FALLBACK_WINDOW), vals.size)
    recent = vals[-window:]
    recent = recent[np.isfinite(recent)]
    if recent.size < 2:
        return False
    return float(np.ptp(recent)) > float(eps)


def build_surface_data(x_vals, rp_vals, l_vals):
    """Build the 3D ribbon "curtain" mesh between the live trace and a floor.

    Returns ``(vertices, faces, face_colors, line_pos)`` or ``None`` when there
    are too few points.  Each axis is robustly normalized so it stays active
    even with outliers present.
    """
    if len(x_vals) < 2:
        return None

    x_recent = np.asarray(x_vals, dtype=float)
    y_recent = np.asarray(rp_vals, dtype=float)
    z_recent = np.asarray(l_vals, dtype=float)
    if len(x_recent) < 2:
        return None

    def normalize_centered(vals):
        # Robust scaling keeps each axis active even when outliers are present.
        p_low = float(np.percentile(vals, 5.0))
        p_high = float(np.percentile(vals, 95.0))
        center = 0.5 * (p_low + p_high)
        robust_span = p_high - p_low
        std_span = float(np.std(vals)) * 6.0
        span = max(robust_span, std_span, 1e-9)
        norm = (vals - center) / span
        return np.clip(norm, -0.5, 0.5), center, span

    x_norm, _, _ = normalize_centered(x_recent)
    y_norm, _, _ = normalize_centered(y_recent)
    z_norm, _, _ = normalize_centered(z_recent)

    z_floor = -0.5
    n = len(x_recent)

    # Build a ribbon "curtain" mesh between the live trace and a floor plane.
    vertices = np.empty((2 * n, 3), dtype=np.float32)
    vertices[0::2, 0] = x_norm
    vertices[0::2, 1] = y_norm
    vertices[0::2, 2] = z_floor
    vertices[1::2, 0] = x_norm
    vertices[1::2, 1] = y_norm
    vertices[1::2, 2] = z_norm

    faces = np.empty((2 * (n - 1), 3), dtype=np.uint32)
    for i in range(n - 1):
        b = 2 * i
        faces[2 * i] = [b, b + 1, b + 2]
        faces[2 * i + 1] = [b + 1, b + 3, b + 2]

    z_color = z_norm + 0.5

    face_colors = np.empty((faces.shape[0], 4), dtype=np.float32)
    for i in range(n - 1):
        c = float(0.5 * (z_color[i] + z_color[i + 1]))
        r = 0.1 + 0.9 * c
        g = 0.5 * (1.0 - c)
        b = 1.0 - 0.8 * c
        face_colors[2 * i] = [r, g, b, 0.32]
        face_colors[2 * i + 1] = [r, g, b, 0.32]

    line_pos = np.column_stack((x_norm, y_norm, z_norm)).astype(np.float32)
    return vertices, faces, face_colors, line_pos


class ToggleAxisItem(pg.AxisItem):
    """Bottom axis that toggles x-mode when clicked."""
    toggled = QtCore.Signal()

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.toggled.emit()
            event.accept()
            return
        super().mouseClickEvent(event)


# ---------------------------------------------------------------------------
# 4. CsvLogger — owns the output CSV file
# ---------------------------------------------------------------------------

class CsvLogger:
    """Append-only CSV writer for sensor samples.

    Encapsulates filename normalization, (re)opening the output file, header
    writing, and per-sample rows so the rest of the program never touches the
    file handle directly.
    """

    HEADER = ["timestamp_computer", "timestamp", "sensor1", "sensor2",
              "mag", "width", "crack_x", "crack_size"]

    def __init__(self, filename):
        self._file = None
        self._writer = None
        self.path = ""
        self.set_output_file(filename)

    @staticmethod
    def _normalize(filename):
        name = str(filename).strip()
        if not name:
            name = "test.csv"
        if not name.lower().endswith(".csv"):
            name += ".csv"
        return name

    def set_output_file(self, filename):
        """Switch to ``filename`` (normalized), writing a header if it is new."""
        self.path = self._normalize(filename)

        if self._file is not None and not self._file.closed:
            self._file.close()

        self._file = open(self.path, "a", newline="")
        self._writer = csv.writer(self._file)
        if os.path.getsize(self.path) == 0:
            self._writer.writerow(self.HEADER)

    def write_sample(self, t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val):
        """Append one sample row, stamping the host clock, and flush."""
        timestamp_computer = f"{time.time():.3f}"
        self._writer.writerow([timestamp_computer, t, s1, s2,
                               mag_val if mag_val is not None else "",
                               width_val if width_val is not None else "",
                               crack_x_val if crack_x_val is not None else "",
                               crack_size_val if crack_size_val is not None else ""])
        self._file.flush()

    @property
    def basename(self):
        return os.path.basename(self.path)

    def close(self):
        if self._file is not None and not self._file.closed:
            self._file.close()


# ---------------------------------------------------------------------------
# 5. SerialManager — owns the serial port (connect/disconnect at runtime)
# ---------------------------------------------------------------------------

class SerialManager:
    """Wraps a pyserial port that may or may not be open.

    The module-level reference (``serial_mgr``) never gets reassigned —
    ``connect()`` / ``disconnect()`` mutate ``self.port`` — so callers can hold
    onto the manager indefinitely.  Every I/O method is guarded: it is a safe
    no-op when disconnected, and any hardware exception (device unplugged
    mid-stream) auto-disconnects and fires ``on_change`` so the UI can react.
    """

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


# ---------------------------------------------------------------------------
# 6. ScannerState — all live runtime state + sample ingestion
# ---------------------------------------------------------------------------

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
        self.last_average_update_time = 0.0

        # View tracking.
        self.xy_start_index = 0
        self.right_x_mode = "RP"                 # "RP" or "TIME"
        self.right_plot_auto_time_fallback = False
        self.crack_y_mode = "mag"                # "mag" or "crack_size"
        self.ui_start_monotonic = time.monotonic()

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

    def ingest_sample(self, t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val, csv_logger):
        """Store one parsed sample, optionally log it, and refresh readout text."""
        self.timestamps.append(t)
        self.sensor1.append(s1)
        self.sensor2.append(s2)

        if self.write_to_file_enabled:
            csv_logger.write_sample(t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val)

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


# ---------------------------------------------------------------------------
# 7. Runtime objects
# ---------------------------------------------------------------------------

# Serial port stays closed until the user connects via the GUI (no auto-open
# at import means the app launches cleanly even without a device attached).
serial_mgr = SerialManager(BAUDRATE)
csv_logger = CsvLogger(CSV_FILE)
state = ScannerState()


# ---------------------------------------------------------------------------
# 8. Qt user interface
# ---------------------------------------------------------------------------

# Keep UI scale consistent when moving between displays with different DPI.
# (These attributes must be set before the QApplication is constructed.)
if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy"):
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

app = QtWidgets.QApplication([])
main_widget = QtWidgets.QWidget()
main_widget.setWindowTitle("Eddy Current Scanner V5")
main_layout = QtWidgets.QVBoxLayout(main_widget)
main_layout.setContentsMargins(6, 6, 6, 6)
main_layout.setSpacing(6)

# Top row holds the live 3D surface (left) and phase-space plot (right).
top_row_layout = QtWidgets.QHBoxLayout()
top_row_layout.setContentsMargins(0, 0, 0, 0)
top_row_layout.setSpacing(6)
main_layout.addLayout(top_row_layout, 1)

# --- Left panel: live 3D surface -------------------------------------------
surface_container = QtWidgets.QWidget()
surface_layout = QtWidgets.QVBoxLayout(surface_container)
surface_layout.setContentsMargins(0, 0, 0, 0)
surface_layout.setSpacing(2)
surface_title = QtWidgets.QLabel("3D Surface: Time (x), R_p (y), L (z)")
surface_layout.addWidget(surface_title)

surface_view = gl.GLViewWidget()
surface_view.setMinimumSize(520, 340)
surface_view.opts['distance'] = 2.8
surface_view.opts['elevation'] = 22
surface_view.opts['azimuth'] = -35
surface_layout.addWidget(surface_view, 1)

surface_grid = gl.GLGridItem()
surface_grid.setSize(1.4, 1.4)
surface_grid.setSpacing(0.1, 0.1)
surface_view.addItem(surface_grid)

surface_axis = gl.GLAxisItem()
surface_axis.setSize(1.0, 1.0, 1.0)
surface_view.addItem(surface_axis)

_axis_font = QtGui.QFont('Helvetica', 11, QtGui.QFont.Bold)
_label_x = gl.GLTextItem(pos=np.array([0.56, 0.0, 0.0], dtype=float),
                         text='Time', color=QtGui.QColor(255, 80, 80), font=_axis_font)
_label_y = gl.GLTextItem(pos=np.array([0.0, 0.56, 0.0], dtype=float),
                         text='R_p', color=QtGui.QColor(80, 200, 80), font=_axis_font)
_label_z = gl.GLTextItem(pos=np.array([0.0, 0.0, 0.56], dtype=float),
                         text='L', color=QtGui.QColor(80, 160, 255), font=_axis_font)
surface_view.addItem(_label_x)
surface_view.addItem(_label_y)
surface_view.addItem(_label_z)

# Bootstrap quad shown until enough samples arrive to build a real ribbon.
_bootstrap_vertices = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_bootstrap_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
_bootstrap_face_colors = np.array(
    [[0.2, 0.6, 1.0, 0.32], [0.2, 0.6, 1.0, 0.32]],
    dtype=np.float32,
)

surface_meshdata = gl.MeshData(vertexes=_bootstrap_vertices, faces=_bootstrap_faces)
surface_meshdata.setFaceColors(_bootstrap_face_colors)
surface_item = gl.GLMeshItem(meshdata=surface_meshdata, smooth=False, drawEdges=False, drawFaces=True)
surface_item.setGLOptions('translucent')
surface_view.addItem(surface_item)

surface_trace = gl.GLLinePlotItem(
    pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32),
    color=(1.0, 0.65, 0.2, 1.0),
    width=2.0,
    antialias=True,
    mode='line_strip',
)
surface_view.addItem(surface_trace)

surface_head = gl.GLScatterPlotItem(
    pos=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
    color=(1.0, 1.0, 1.0, 1.0),
    size=8.0,
)
surface_view.addItem(surface_head)

top_row_layout.addWidget(surface_container, 1)

# --- Right panel: phase-space / time trace ---------------------------------
right_container = QtWidgets.QWidget()
right_layout = QtWidgets.QVBoxLayout(right_container)
right_layout.setContentsMargins(0, 0, 0, 0)
right_layout.setSpacing(2)
right_title = QtWidgets.QLabel("Phase Space / Time Trace")
right_layout.addWidget(right_title)

win = pg.GraphicsLayoutWidget()
right_layout.addWidget(win, 1)

top_row_layout.addWidget(right_container, 1)

win.setFocusPolicy(QtCore.Qt.StrongFocus)
win.setFocus()

bottom_axis = ToggleAxisItem(orientation='bottom')
plot_xy = win.addPlot(title="Phase Space", axisItems={'bottom': bottom_axis})
plot_xy.setLabel('bottom', 'R_p (ohm)')
plot_xy.setLabel('left', 'L (uH)')
bottom_axis.setToolTip("Click x-axis to toggle between R_p and Time")


def set_right_x_mode(mode):
    """Apply right-plot x-axis mode ("RP"/"TIME") and refresh its title/labels."""
    state.right_x_mode = mode
    if state.right_x_mode == "TIME":
        plot_xy.setTitle("Time Trace")
        plot_xy.setLabel('bottom', 'Time (timestamp)')
    elif state.right_plot_auto_time_fallback:
        plot_xy.setTitle("Time Trace (auto fallback)")
        plot_xy.setLabel('bottom', 'Time (timestamp, Rp flat/zero)')
    else:
        plot_xy.setTitle("Phase Space")
        plot_xy.setLabel('bottom', 'R_p (ohm)')


def toggle_right_x_mode():
    set_right_x_mode("TIME" if state.right_x_mode == "RP" else "RP")


bottom_axis.toggled.connect(toggle_right_x_mode)

initial_xy_view_state = plot_xy.getViewBox().getState(copy=True)
xy_curve = plot_xy.plot(pen='r')
recent_segment_curves = []
for _ in range(max(RECENT_FADE_POINTS - 1, 0)):
    recent_segment_curves.append(plot_xy.plot(pen=pg.mkPen((255, 0, 0), width=3)))

# Lower row: controls (left) and crack-event plot (right).
lower_row_layout = QtWidgets.QHBoxLayout()
lower_row_layout.setContentsMargins(0, 0, 0, 0)
lower_row_layout.setSpacing(6)

# --- Crack-event plot ------------------------------------------------------
crack_frame = QtWidgets.QFrame()
crack_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
crack_frame.setStyleSheet(
    "QFrame { border: 1px dashed #666; border-radius: 4px; background: #141414; }"
)
crack_frame_layout = QtWidgets.QVBoxLayout(crack_frame)
crack_frame_layout.setContentsMargins(8, 8, 8, 8)
crack_frame_layout.setSpacing(0)
crack_win = pg.GraphicsLayoutWidget()
crack_frame_layout.addWidget(crack_win)
crack_frame.setFixedHeight(242)

crack_left_axis = ToggleAxisItem(orientation='left')
crack_plot = crack_win.addPlot(axisItems={'left': crack_left_axis})
crack_plot.setLabel('bottom', 'Time (timestamp)')
crack_plot.setLabel('left', 'mag')
crack_plot.showGrid(x=True, y=True, alpha=0.25)
crack_plot.setYRange(0.0, 1.0, padding=0.0)
crack_curve = crack_plot.plot(
    [],
    [],
    pen=pg.mkPen((255, 190, 140, 230), width=1),
    connect='pairs',
)

crack_left_axis.setToolTip("Click y-axis to toggle between mag and crack_size")


def set_crack_y_mode(mode):
    state.crack_y_mode = mode
    crack_plot.setLabel('left', 'mag' if state.crack_y_mode == 'mag' else 'crack_size')


def toggle_crack_y_mode():
    set_crack_y_mode('crack_size' if state.crack_y_mode == 'mag' else 'mag')


crack_left_axis.toggled.connect(toggle_crack_y_mode)

# --- Connection controls (under the crack subplot) ------------------------
connection_frame = QtWidgets.QFrame()
connection_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
connection_frame.setStyleSheet(
    "QFrame { border: 1px dashed #666; border-radius: 4px; background: #141414; }"
)
connection_layout = QtWidgets.QVBoxLayout(connection_frame)
connection_layout.setContentsMargins(8, 6, 8, 6)
connection_layout.setSpacing(4)

port_combo = QtWidgets.QComboBox()
port_combo.setMinimumWidth(160)
port_combo.setToolTip("Detected serial ports")

refresh_button = QtWidgets.QPushButton("Refresh")
refresh_button.setMinimumWidth(80)

baud_combo = QtWidgets.QComboBox()
for _rate in BAUD_RATES:
    baud_combo.addItem(str(_rate), _rate)
_default_baud_idx = baud_combo.findData(BAUDRATE)
if _default_baud_idx >= 0:
    baud_combo.setCurrentIndex(_default_baud_idx)
baud_combo.setToolTip("Baud rate")

connect_button = QtWidgets.QPushButton("Connect")
connect_button.setMinimumWidth(100)

connection_row1 = QtWidgets.QHBoxLayout()
connection_row1.setContentsMargins(0, 0, 0, 0)
connection_row1.setSpacing(4)
connection_row1.addWidget(port_combo, 1)
connection_row1.addWidget(refresh_button)

connection_row2 = QtWidgets.QHBoxLayout()
connection_row2.setContentsMargins(0, 0, 0, 0)
connection_row2.setSpacing(4)
connection_row2.addWidget(baud_combo)
connection_row2.addWidget(connect_button, 1)

connection_status_label = QtWidgets.QLabel(serial_mgr.status_text)
connection_status_label.setStyleSheet("font-size: 10px; color: #bbbbbb;")

connection_layout.addLayout(connection_row1)
connection_layout.addLayout(connection_row2)
connection_layout.addWidget(connection_status_label)


def refresh_ports():
    """Re-scan the system for serial ports and repopulate the dropdown."""
    port_combo.clear()
    for device, desc in SerialManager.available_ports():
        port_combo.addItem(device, device)
        port_combo.setItemData(port_combo.count() - 1, desc, QtCore.Qt.ToolTipRole)
    default_idx = port_combo.findData(SERIAL_PORT)
    if default_idx >= 0:
        port_combo.setCurrentIndex(default_idx)


def toggle_connection():
    """Connect using the selected port + baud, or disconnect if already open."""
    if serial_mgr.is_connected:
        serial_mgr.disconnect()
        return
    port_name = port_combo.currentData() or port_combo.currentText()
    if not port_name:
        serial_mgr.status_text = "No port selected"
        set_connection_ui_state()
        return
    baud_value = baud_combo.currentData()
    if baud_value is None:
        baud_value = int(baud_combo.currentText())
    if not serial_mgr.connect(port_name, int(baud_value)):
        # Connection attempt failed — re-scan in case the port disappeared.
        refresh_ports()


def set_connection_ui_state():
    """Sync the connection cluster widgets with the current serial_mgr state."""
    connected = serial_mgr.is_connected
    connect_button.setText("Disconnect" if connected else "Connect")
    connection_status_label.setText(serial_mgr.status_text)
    # Lock port/baud selection while connected — disconnect first to change.
    port_combo.setEnabled(not connected)
    baud_combo.setEnabled(not connected)
    refresh_button.setEnabled(not connected)


refresh_button.clicked.connect(refresh_ports)
connect_button.clicked.connect(toggle_connection)
serial_mgr.on_change = set_connection_ui_state
refresh_ports()
set_connection_ui_state()

# Stack the crack plot and the connection panel into the lower-row right cell.
right_lower_container = QtWidgets.QWidget()
right_lower_layout = QtWidgets.QVBoxLayout(right_lower_container)
right_lower_layout.setContentsMargins(0, 0, 0, 0)
right_lower_layout.setSpacing(6)
right_lower_layout.addWidget(crack_frame)
right_lower_layout.addWidget(connection_frame)

lower_row_layout.addWidget(right_lower_container, 1)

# --- Controls (under the left 3D panel) ------------------------------------
controls_container = QtWidgets.QWidget()
controls_layout = QtWidgets.QVBoxLayout(controls_container)
controls_layout.setContentsMargins(0, 4, 0, 0)
controls_layout.setSpacing(6)

# Live incoming serial readout (last 3 lines + decoded sample + average).
incoming_line_box = QtWidgets.QPlainTextEdit()
incoming_line_box.setReadOnly(True)
incoming_line_box.setMinimumWidth(320)
incoming_line_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
incoming_line_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
incoming_line_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
line_height = incoming_line_box.fontMetrics().lineSpacing()
frame_height = incoming_line_box.frameWidth() * 2
doc_margin_height = int(incoming_line_box.document().documentMargin() * 2)
incoming_line_box.setFixedHeight(int(line_height * 3 + frame_height + doc_margin_height))
incoming_line_box.setPlainText("Incoming line: waiting for data...")


def append_incoming_line(line):
    """Push a raw incoming line into the rolling 3-line readout box."""
    if not line:
        return
    state.incoming_history.append(line)
    incoming_line_box.setPlainText("\n".join(state.incoming_history))


readout_label = QtWidgets.QLabel(state.readout_text)
readout_label.setMinimumWidth(320)
average_label = QtWidgets.QLabel(f"Avg last {RECENT_FADE_POINTS}: waiting for data...")
average_label.setMinimumWidth(320)

readout_layout = QtWidgets.QVBoxLayout()
readout_layout.setContentsMargins(0, 0, 0, 0)
readout_layout.setSpacing(2)
readout_layout.addWidget(incoming_line_box)
readout_layout.addWidget(readout_label)
readout_layout.addWidget(average_label)

readout_container = QtWidgets.QWidget()
readout_container.setLayout(readout_layout)
controls_layout.addWidget(readout_container)

# Two-way serial command controls.
serial_command_input = QtWidgets.QLineEdit()
serial_command_input.setPlaceholderText("Type serial command")
serial_command_input.setMinimumWidth(220)

serial_send_button = QtWidgets.QPushButton("Send")
serial_send_button.setMinimumWidth(80)

serial_command_row = QtWidgets.QHBoxLayout()
serial_command_row.setContentsMargins(0, 0, 0, 0)
serial_command_row.setSpacing(4)
serial_command_row.addWidget(serial_command_input)
serial_command_row.addWidget(serial_send_button)

serial_response_box = QtWidgets.QPlainTextEdit()
serial_response_box.setReadOnly(True)
serial_response_box.setMinimumWidth(320)
serial_response_box.setMaximumHeight(SERIAL_RESPONSE_BOX_MAX_HEIGHT)
serial_response_box.setPlainText("Response: waiting for command...")

serial_controls_layout = QtWidgets.QVBoxLayout()
serial_controls_layout.setContentsMargins(0, 0, 0, 0)
serial_controls_layout.setSpacing(2)
serial_controls_layout.addLayout(serial_command_row)
serial_controls_layout.addWidget(serial_response_box)

serial_controls_container = QtWidgets.QWidget()
serial_controls_container.setLayout(serial_controls_layout)
controls_layout.addWidget(serial_controls_container)

# Write-to-file toggle (defaults OFF) and output filename.
write_toggle_button = QtWidgets.QPushButton("Write to File: OFF")
write_toggle_button.setCheckable(True)
write_toggle_button.setChecked(False)
write_toggle_button.setMinimumWidth(140)


def write_toggle_changed(checked):
    state.write_to_file_enabled = checked
    write_toggle_button.setText("Write to File: ON" if checked else "Write to File: OFF")


write_toggle_button.toggled.connect(write_toggle_changed)

write_file_label = QtWidgets.QLabel(f"{csv_logger.basename}")
write_file_label.setAlignment(QtCore.Qt.AlignHCenter)
write_file_label.setStyleSheet("font-size: 10px; color: #bbbbbb;")
write_file_label.setToolTip(csv_logger.path)

write_file_input = QtWidgets.QLineEdit(csv_logger.path)
write_file_input.setPlaceholderText("CSV filename")
write_file_input.setMinimumWidth(160)
write_file_input.setToolTip("Output CSV file name (press Enter to apply)")


def apply_csv_filename():
    csv_logger.set_output_file(write_file_input.text())
    write_file_input.setText(csv_logger.path)
    write_file_label.setText(csv_logger.basename)
    write_file_label.setToolTip(csv_logger.path)


write_file_input.editingFinished.connect(apply_csv_filename)

write_controls_layout = QtWidgets.QVBoxLayout()
write_controls_layout.setContentsMargins(0, 0, 0, 0)
write_controls_layout.setSpacing(2)
write_controls_layout.addWidget(write_toggle_button)
write_controls_layout.addWidget(write_file_input)
write_controls_layout.addWidget(write_file_label)

write_controls_container = QtWidgets.QWidget()
write_controls_container.setLayout(write_controls_layout)
controls_layout.addWidget(write_controls_container)

lower_row_layout.insertWidget(0, controls_container, 1)
main_layout.addLayout(lower_row_layout, 0)
main_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
main_widget.setFocus()
main_widget.show()


# ---------------------------------------------------------------------------
# 9. Handlers + update loop
# ---------------------------------------------------------------------------

def keyPressEvent(event):
    """Keyboard shortcuts: Space=reset, P=pause, F=toggle CSV, 1/2/3=3D views."""
    if event.key() == QtCore.Qt.Key_Space:
        # Clear all buffered data so both plots restart from a clean state.
        state.reset()

        # Reset left 3D plot.
        meshdata = gl.MeshData(vertexes=_bootstrap_vertices, faces=_bootstrap_faces)
        meshdata.setFaceColors(_bootstrap_face_colors)
        surface_item.setMeshData(meshdata=meshdata)
        surface_trace.setData(pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        surface_head.setData(pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        surface_view.opts['center'] = QtGui.QVector3D(0.0, 0.0, 0.0)

        # Reset right XY plot.
        xy_curve.clear()
        for seg_curve in recent_segment_curves:
            seg_curve.setData([], [])
        plot_xy.getViewBox().setState(initial_xy_view_state)

        # Reset average label and crack plot.
        average_label.setText(state.average_text)
        crack_curve.setData([], [])
    elif event.key() == QtCore.Qt.Key_P:
        state.paused = not state.paused
        if state.paused:
            serial_mgr.reset_input_buffer()
        print("Paused" if state.paused else "Resumed")
    elif event.key() == QtCore.Qt.Key_F:
        write_toggle_button.setChecked(not write_toggle_button.isChecked())
        print("CSV write ON" if write_toggle_button.isChecked() else "CSV write OFF")
    elif event.key() == QtCore.Qt.Key_1:
        # Top-down view: look straight down the Z axis.
        surface_view.opts['elevation'] = 90
        surface_view.opts['azimuth'] = 0
        surface_view.update()
    elif event.key() == QtCore.Qt.Key_2:
        # Front view: look along the Y axis from the front.
        surface_view.opts['elevation'] = 0
        surface_view.opts['azimuth'] = 0
        surface_view.update()
    elif event.key() == QtCore.Qt.Key_3:
        # Left side view: look along the X axis from the left.
        surface_view.opts['elevation'] = 0
        surface_view.opts['azimuth'] = 90
        surface_view.update()


main_widget.keyPressEvent = keyPressEvent
win.keyPressEvent = keyPressEvent
surface_view.keyPressEvent = keyPressEvent


def consume_serial_line(line, responses=None):
    """Process one received serial line.

    Updates the incoming-line view, records any reject reason and crack event,
    then parses and ingests a sample.  Lines that are not parseable samples are
    appended to ``responses`` when a list is provided (used by the command
    console); empty lines are ignored.
    """
    if not line:
        return

    append_incoming_line(line)

    reason = extract_reject_reason(line)
    if reason is not None:
        state.reject_reason = reason

    state.ingest_crack_event(line)

    try:
        t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val = parse_serial_line(line)
    except (ValueError, KeyError):
        if responses is not None:
            responses.append(line)
        return

    state.ingest_sample(t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val, csv_logger)


def read_serial():
    """Drain any pending serial input into the live buffers (skipped while paused)."""
    if state.paused:
        serial_mgr.reset_input_buffer()
        return
    while serial_mgr.in_waiting:
        line = serial_mgr.readline().decode(errors='ignore').strip()
        consume_serial_line(line)


def send_serial_command():
    """Send the typed command, then collect non-sensor replies for ~400 ms."""
    command = serial_command_input.text().strip()
    if not command:
        serial_response_box.setPlainText("Response: command is empty")
        return

    if not serial_mgr.is_connected:
        serial_response_box.setPlainText("Response: not connected")
        return

    try:
        serial_mgr.write((command + "\n").encode("utf-8"))
    except serial.SerialException as exc:
        serial_mgr.disconnect("device lost")
        serial_response_box.setPlainText(f"Response error: {exc}")
        return

    responses = []
    deadline = time.monotonic() + 0.4

    while time.monotonic() < deadline:
        if serial_mgr.in_waiting <= 0:
            time.sleep(0.01)
            continue

        line = serial_mgr.readline().decode(errors='ignore').strip()
        consume_serial_line(line, responses)

    if responses:
        response_text = "Response:\n" + "\n".join(responses[-SERIAL_RESPONSE_MAX_LINES:])
    else:
        response_text = "Response: no non-sensor reply in 400 ms"

    serial_response_box.setPlainText(response_text)


serial_send_button.clicked.connect(send_serial_command)
serial_command_input.returnPressed.connect(send_serial_command)


def update():
    """Timer callback: read serial, recompute readouts, and redraw all plots."""
    read_serial()
    readout_label.setText(state.readout_text)

    x_all = np.array(state.timestamps)
    y1_all = np.array(state.sensor1)
    y2_all = np.array(state.sensor2)

    now = time.monotonic()
    if now - state.last_average_update_time >= AVERAGE_UPDATE_INTERVAL_SEC:
        avg_count = min(RECENT_FADE_POINTS, len(y1_all))
        if avg_count > 0:
            avg_s1 = float(np.mean(y1_all[-avg_count:]))
            avg_s2 = float(np.mean(y2_all[-avg_count:]))
            state.average_text = f"Avg last {avg_count}: s1={avg_s1:.6f} | s2={avg_s2:.6f}"
        else:
            state.average_text = f"Avg last {RECENT_FADE_POINTS}: waiting for data..."
        state.last_average_update_time = now
    average_label.setText(state.average_text)

    if len(x_all) > 0:
        t_min = float(x_all[0])
        t_now = float(x_all[-1])
    else:
        t_min = 0.0
        t_now = float(time.monotonic() - state.ui_start_monotonic)

    crack_plot.setXRange(t_min, max(t_now, t_min + 1e-6), padding=0.0)
    crack_count = min(len(state.crack_times), len(state.crack_mags), len(state.crack_sizes))
    if crack_count > 0:
        crack_times_arr = np.asarray(list(state.crack_times)[-crack_count:], dtype=float)
        crack_mags_arr = np.asarray(list(state.crack_mags)[-crack_count:], dtype=float)
        crack_sizes_arr = np.asarray(list(state.crack_sizes)[-crack_count:], dtype=float)
        crack_vals_arr = crack_mags_arr if state.crack_y_mode == 'mag' else crack_sizes_arr
        crack_x = np.repeat(crack_times_arr, 2)
        crack_y = np.empty(2 * crack_count, dtype=float)
        crack_y[0::2] = 0.0
        crack_y[1::2] = crack_vals_arr
        crack_curve.setData(crack_x, crack_y)
        crack_plot.setYRange(0.0, max(float(np.max(crack_vals_arr)) * 1.1, 1e-6), padding=0.0)
    else:
        crack_curve.setData([], [])
        crack_plot.setYRange(0.0, 1.0, padding=0.0)

    lag = max(0, int(DISPLAY_LAG_POINTS))
    if lag > 0 and len(x_all) > lag:
        x = x_all[:-lag]
        y1 = y1_all[:-lag]
        y2 = y2_all[:-lag]
    elif lag == 0:
        x = x_all
        y1 = y1_all
        y2 = y2_all
    else:
        x = np.array([], dtype=float)
        y1 = np.array([], dtype=float)
        y2 = np.array([], dtype=float)

    if len(x) == 0:
        return

    # Right and left views share the same visible data window after reset.
    xy_offset = max(0, state.xy_start_index)
    x_plot = x[xy_offset:]
    y1_plot = y1[xy_offset:]
    y2_plot = y2[xy_offset:]

    if len(x_plot) == 0:
        return

    surface_data = build_surface_data(x_plot, y1_plot, y2_plot)
    if surface_data is not None:
        vertices, faces, face_colors, line_pos = surface_data
        meshdata = gl.MeshData(vertexes=vertices, faces=faces)
        meshdata.setFaceColors(face_colors)
        surface_item.setMeshData(meshdata=meshdata)
        surface_trace.setData(pos=line_pos)
        surface_head.setData(pos=line_pos[-1:].copy())
        surface_view.opts['center'] = QtGui.QVector3D(0.0, 0.0, 0.0)

    # Clear previous text labels on the XY plot.
    for item in plot_xy.items[:]:
        if isinstance(item, pg.TextItem):
            plot_xy.removeItem(item)

    if state.right_x_mode == "TIME":
        state.right_plot_auto_time_fallback = False
        x_right = x_plot
        y_right = y2_plot
    else:
        if has_usable_rp(y1_plot):
            state.right_plot_auto_time_fallback = False
            x_right = y1_plot
            y_right = y2_plot
        else:
            # If Rp is flat/zero, preserve live L tracking by plotting against time.
            state.right_plot_auto_time_fallback = True
            x_right = x_plot
            y_right = y2_plot

    set_right_x_mode(state.right_x_mode)

    xy_curve.setData(x_right, y_right)

    # Highlight most recent trajectory with red -> white segment gradient.
    tail_count = min(RECENT_FADE_POINTS, len(x_right))
    if tail_count > 1:
        tail_x = x_right[-tail_count:]
        tail_y = y_right[-tail_count:]
        seg_count = tail_count - 1
        shades = np.linspace(0, 255, seg_count).astype(int)
        for i in range(seg_count):
            seg_curve = recent_segment_curves[i]
            seg_curve.setPen(pg.mkPen((255, int(shades[i]), int(shades[i]), 255), width=3))
            seg_curve.setData([tail_x[i], tail_x[i + 1]], [tail_y[i], tail_y[i + 1]])
        for i in range(seg_count, len(recent_segment_curves)):
            recent_segment_curves[i].setData([], [])
    else:
        for seg_curve in recent_segment_curves:
            seg_curve.setData([], [])


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


def close_resources():
    csv_logger.close()
    serial_mgr.close()


app.aboutToQuit.connect(close_resources)

app.exec()
