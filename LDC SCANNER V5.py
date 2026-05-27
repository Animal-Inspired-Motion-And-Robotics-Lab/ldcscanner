import serial
import numpy as np
import csv
import os
import time
from collections import deque
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

pg.setConfigOptions(antialias=True)

# -------------------------
# SERIAL CONFIGS
# ------------------------- 
SERIAL_PORT = "COM6"   
BAUDRATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

CSV_FILE = "test.csv"
csv_file = None
csv_writer = None

def normalize_csv_filename(filename):
    name = str(filename).strip()
    if not name:
        name = "test.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name

def set_csv_output_file(filename):
    global CSV_FILE, csv_file, csv_writer
    CSV_FILE = normalize_csv_filename(filename)

    if csv_file is not None and not csv_file.closed:
        csv_file.close()

    csv_file = open(CSV_FILE, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(CSV_FILE) == 0:
        csv_writer.writerow(["timestamp_computer", "timestamp", "sensor1", "sensor2", "mag", "width", "crack_x", "crack_size"])

set_csv_output_file(CSV_FILE)

# -------------------------
# DATA STORAGE
# -------------------------
MAX_POINTS = 5000
timestamps = deque(maxlen=MAX_POINTS)
sensor1 = deque(maxlen=MAX_POINTS)
sensor2 = deque(maxlen=MAX_POINTS)
crack_times = deque(maxlen=MAX_POINTS)
crack_sizes = deque(maxlen=MAX_POINTS)

# control flags
paused = False
write_to_file_enabled = False

# latest serial sample readout text
latest_readout_text = "Incoming: waiting for data..."
incoming_line_history = deque(maxlen=3)
latest_average_text = "Average: waiting for data..."
latest_command_response_text = "Response: waiting for command..."
latest_reason_input_value = None
latest_nonzero_mag_value = None
latest_nonzero_width_value = None
latest_nonzero_crack_size_value = None
last_average_update_time = 0.0

# XY plot tracking
xy_start_index = 0

# adjustable parameters
DISPLAY_LAG_POINTS = 1  # skip newest N points in plots to reduce right-edge jitter
RECENT_FADE_POINTS = 100
AVERAGE_UPDATE_INTERVAL_SEC = 5.0
RP_ZERO_EPSILON = 1e-12
RP_ZERO_FALLBACK_WINDOW = 100
SERIAL_RESPONSE_MAX_LINES = 20
SERIAL_RESPONSE_BOX_MAX_HEIGHT = 180
ui_start_monotonic = time.monotonic()

def has_usable_rp(rp_vals, eps=RP_ZERO_EPSILON):
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

# -------------------------
# QT SETUP
# -------------------------
# Keep UI scale consistent when moving between displays with different DPI.
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

# -------------------------
# RIGHT PLOT (XY)
# -------------------------
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

RIGHT_X_MODE = "RP"  # RP or TIME
right_plot_auto_time_fallback = False

bottom_axis = ToggleAxisItem(orientation='bottom')
plot_xy = win.addPlot(title="Phase Space", axisItems={'bottom': bottom_axis})
plot_xy.setLabel('bottom', 'R_p (ohm)')
plot_xy.setLabel('left', 'L (uH)')
bottom_axis.setToolTip("Click x-axis to toggle between R_p and Time")

def set_right_x_mode(mode):
    global RIGHT_X_MODE
    RIGHT_X_MODE = mode
    if RIGHT_X_MODE == "TIME":
        plot_xy.setTitle("Time Trace")
        plot_xy.setLabel('bottom', 'Time (timestamp)')
    elif right_plot_auto_time_fallback:
        plot_xy.setTitle("Time Trace (auto fallback)")
        plot_xy.setLabel('bottom', 'Time (timestamp, Rp flat/zero)')
    else:
        plot_xy.setTitle("Phase Space")
        plot_xy.setLabel('bottom', 'R_p (ohm)')

def toggle_right_x_mode():
    set_right_x_mode("TIME" if RIGHT_X_MODE == "RP" else "RP")

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

crack_plot = crack_win.addPlot()
crack_plot.setLabel('bottom', 'Time (timestamp)')
crack_plot.setLabel('left', 'Magnitude')
crack_plot.showGrid(x=True, y=True, alpha=0.25)
crack_plot.setYRange(0.0, 1.0, padding=0.0)
crack_curve = crack_plot.plot(
    [],
    [],
    pen=pg.mkPen((255, 190, 140, 230), width=1),
    connect='pairs',
)

lower_row_layout.addWidget(crack_frame, 1)

# -------------------------
# CONTROLS (UNDER LEFT 3D PANEL)
# -------------------------
controls_container = QtWidgets.QWidget()
controls_layout = QtWidgets.QVBoxLayout(controls_container)
controls_layout.setContentsMargins(0, 4, 0, 0)
controls_layout.setSpacing(6)

# Live incoming serial readout
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
    if not line:
        return
    incoming_line_history.append(line)
    incoming_line_box.setPlainText("\n".join(incoming_line_history))

readout_label = QtWidgets.QLabel(latest_readout_text)
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

# Two-way serial command controls
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
serial_response_box.setPlainText(latest_command_response_text)

serial_controls_layout = QtWidgets.QVBoxLayout()
serial_controls_layout.setContentsMargins(0, 0, 0, 0)
serial_controls_layout.setSpacing(2)
serial_controls_layout.addLayout(serial_command_row)
serial_controls_layout.addWidget(serial_response_box)

serial_controls_container = QtWidgets.QWidget()
serial_controls_container.setLayout(serial_controls_layout)
controls_layout.addWidget(serial_controls_container)

# Write-to-file toggle (defaults OFF)
write_toggle_button = QtWidgets.QPushButton("Write to File: OFF")
write_toggle_button.setCheckable(True)
write_toggle_button.setChecked(False)
write_toggle_button.setMinimumWidth(140)
def write_toggle_changed(checked):
    global write_to_file_enabled
    write_to_file_enabled = checked
    write_toggle_button.setText("Write to File: ON" if checked else "Write to File: OFF")
write_toggle_button.toggled.connect(write_toggle_changed)

write_file_label = QtWidgets.QLabel(f"{os.path.basename(CSV_FILE)}")
write_file_label.setAlignment(QtCore.Qt.AlignHCenter)
write_file_label.setStyleSheet("font-size: 10px; color: #bbbbbb;")
write_file_label.setToolTip(CSV_FILE)

write_file_input = QtWidgets.QLineEdit(CSV_FILE)
write_file_input.setPlaceholderText("CSV filename")
write_file_input.setMinimumWidth(160)
write_file_input.setToolTip("Output CSV file name (press Enter to apply)")

def apply_csv_filename():
    set_csv_output_file(write_file_input.text())
    write_file_input.setText(CSV_FILE)
    write_file_label.setText(os.path.basename(CSV_FILE))
    write_file_label.setToolTip(CSV_FILE)

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

# -------------------------
# KEY HANDLER
# -------------------------
def keyPressEvent(event):
    global paused, xy_start_index
    global latest_average_text, last_average_update_time, ui_start_monotonic
    global latest_nonzero_mag_value, latest_nonzero_width_value
    if event.key() == QtCore.Qt.Key_Space:
        # Clear all buffered data so both plots restart from a clean state.
        timestamps.clear()
        sensor1.clear()
        sensor2.clear()
        crack_times.clear()
        crack_sizes.clear()
        xy_start_index = 0
        ui_start_monotonic = time.monotonic()

        # Reset left 3D plot.
        surface_meshdata = gl.MeshData(vertexes=_bootstrap_vertices, faces=_bootstrap_faces)
        surface_meshdata.setFaceColors(_bootstrap_face_colors)
        surface_item.setMeshData(meshdata=surface_meshdata)
        surface_trace.setData(pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        surface_head.setData(pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        surface_view.opts['center'] = QtGui.QVector3D(0.0, 0.0, 0.0)

        # Reset right XY plot.
        xy_curve.clear()
        for seg_curve in recent_segment_curves:
            seg_curve.setData([], [])
        plot_xy.getViewBox().setState(initial_xy_view_state)

        # Reset average label.
        latest_average_text = f"Avg last {RECENT_FADE_POINTS}: waiting for data..."
        average_label.setText(latest_average_text)
        last_average_update_time = time.monotonic()
        latest_nonzero_mag_value = None
        latest_nonzero_width_value = None
        crack_curve.setData([], [])
    elif event.key() == QtCore.Qt.Key_P:
        paused = not paused
        if paused:
            ser.reset_input_buffer()
        print("Paused" if paused else "Resumed")
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

# -------------------------
# SERIAL READ
# -------------------------
def parse_keyed_fields(line):
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

def append_mag_sample(line):
    if ">" not in line or ":" not in line:
        return

    fields = parse_keyed_fields(line)
    mag_val = fields.get("mag")
    if mag_val is None or not np.isfinite(mag_val) or float(mag_val) == 0.0:
        return

    t_val = fields.get("t")
    if t_val is None and len(timestamps) > 0:
        t_val = float(timestamps[-1])
    if t_val is None or not np.isfinite(t_val):
        return

    t_event = float(t_val)
    crack_times.append(t_event)
    crack_sizes.append(float(mag_val))

def update_reason_input_from_line(line):
    global latest_reason_input_value
    if not line:
        return

    text = line.strip()
    marker = "reject_reason="
    idx = text.lower().find(marker)
    if idx < 0:
        return

    value = text[idx + len(marker):].strip()
    if not value:
        return

    for sep in ("|", ">", " ", "\t", ","):
        sep_index = value.find(sep)
        if sep_index >= 0:
            value = value[:sep_index]
            break

    parsed_value = value.strip().strip("[]")
    if parsed_value == "-":
        return
    if parsed_value:
        latest_reason_input_value = parsed_value

def parse_serial_line(line):
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

def append_sensor_sample(t, s1, s2, mag_val=None, width_val=None, crack_x_val=None, crack_size_val=None):
    global latest_readout_text, latest_nonzero_mag_value, latest_nonzero_width_value, latest_nonzero_crack_size_value, latest_reason_input_value
    timestamps.append(t)
    sensor1.append(s1)
    sensor2.append(s2)

    if write_to_file_enabled:
        timestamp_computer = f"{time.time():.3f}"
        csv_writer.writerow([timestamp_computer, t, s1, s2,
                              mag_val if mag_val is not None else "",
                              width_val if width_val is not None else "",
                              crack_x_val if crack_x_val is not None else "",
                              crack_size_val if crack_size_val is not None else ""])
        csv_file.flush()

    if mag_val is not None and float(mag_val) != 0.0:
        latest_nonzero_mag_value = float(mag_val)

    if width_val is not None and float(width_val) != 0.0:
        latest_nonzero_width_value = float(width_val)

    if crack_size_val is not None and float(crack_size_val) != 0.0:
        latest_nonzero_crack_size_value = float(crack_size_val)

    mag_text = f"{latest_nonzero_mag_value:.6f}" if latest_nonzero_mag_value is not None else "n/a"
    width_text = f"{latest_nonzero_width_value:.6f}" if latest_nonzero_width_value is not None else "n/a"
    crack_size_text = f"{latest_nonzero_crack_size_value:.6f}" if latest_nonzero_crack_size_value is not None else "n/a"
    input_text = latest_reason_input_value if latest_reason_input_value is not None else "n/a"
    latest_readout_text = f"Incoming: t={t:.3f} | s1={s1:.6f} | s2={s2:.6f} | mag={mag_text} | width={width_text} | crack_size={crack_size_text} | rejected={input_text}"

def send_serial_command():
    global latest_command_response_text

    command = serial_command_input.text().strip()
    if not command:
        latest_command_response_text = "Response: command is empty"
        serial_response_box.setPlainText(latest_command_response_text)
        return

    try:
        ser.write((command + "\n").encode("utf-8"))
        ser.flush()
    except serial.SerialException as exc:
        latest_command_response_text = f"Response error: {exc}"
        serial_response_box.setPlainText(latest_command_response_text)
        return

    responses = []
    deadline = time.monotonic() + 0.4

    while time.monotonic() < deadline:
        if ser.in_waiting <= 0:
            time.sleep(0.01)
            continue

        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        append_incoming_line(line)
        update_reason_input_from_line(line)
        append_mag_sample(line)

        try:
            t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val = parse_serial_line(line)
            append_sensor_sample(t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val)
        except (ValueError, KeyError):
            responses.append(line)

    if responses:
        latest_command_response_text = "Response:\n" + "\n".join(responses[-SERIAL_RESPONSE_MAX_LINES:])
    else:
        latest_command_response_text = "Response: no non-sensor reply in 400 ms"

    serial_response_box.setPlainText(latest_command_response_text)

serial_send_button.clicked.connect(send_serial_command)
serial_command_input.returnPressed.connect(send_serial_command)

def read_serial():
    if paused:
        ser.reset_input_buffer()
        return
    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        append_incoming_line(line)
        update_reason_input_from_line(line)
        append_mag_sample(line)
        try:
            t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val = parse_serial_line(line)
            append_sensor_sample(t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val)
        except (ValueError, KeyError):
            continue

# -------------------------
# UPDATE LOOP
# -------------------------
def update():
    read_serial()
    readout_label.setText(latest_readout_text)

    x_all = np.array(timestamps)
    y1_all = np.array(sensor1)
    y2_all = np.array(sensor2)

    global latest_average_text, last_average_update_time
    now = time.monotonic()
    if now - last_average_update_time >= AVERAGE_UPDATE_INTERVAL_SEC:
        avg_count = min(RECENT_FADE_POINTS, len(y1_all))
        if avg_count > 0:
            avg_s1 = float(np.mean(y1_all[-avg_count:]))
            avg_s2 = float(np.mean(y2_all[-avg_count:]))
            latest_average_text = f"Avg last {avg_count}: s1={avg_s1:.6f} | s2={avg_s2:.6f}"
        else:
            latest_average_text = f"Avg last {RECENT_FADE_POINTS}: waiting for data..."
        last_average_update_time = now
    average_label.setText(latest_average_text)

    if len(x_all) > 0:
        t_min = float(x_all[0])
        t_now = float(x_all[-1])
    else:
        t_min = 0.0
        t_now = float(time.monotonic() - ui_start_monotonic)

    crack_plot.setXRange(t_min, max(t_now, t_min + 1e-6), padding=0.0)
    crack_count = min(len(crack_times), len(crack_sizes))
    if crack_count > 0:
        crack_times_arr = np.asarray(list(crack_times)[-crack_count:], dtype=float)
        crack_sizes_arr = np.asarray(list(crack_sizes)[-crack_count:], dtype=float)
        crack_x = np.repeat(crack_times_arr, 2)
        crack_y = np.empty(2 * crack_count, dtype=float)
        crack_y[0::2] = 0.0
        crack_y[1::2] = crack_sizes_arr
        crack_curve.setData(crack_x, crack_y)
        crack_plot.setYRange(0.0, max(float(np.max(crack_sizes_arr)) * 1.1, 1e-6), padding=0.0)
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
    xy_offset = max(0, xy_start_index)
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

    # clear previous text labels on XY plot
    for item in plot_xy.items[:]:
        if isinstance(item, pg.TextItem):
            plot_xy.removeItem(item)

    global right_plot_auto_time_fallback
    if RIGHT_X_MODE == "TIME":
        right_plot_auto_time_fallback = False
        x_right = x_plot
        y_right = y2_plot
    else:
        if has_usable_rp(y1_plot):
            right_plot_auto_time_fallback = False
            x_right = y1_plot
            y_right = y2_plot
        else:
            # If Rp is flat/zero, preserve live L tracking by plotting against time.
            right_plot_auto_time_fallback = True
            x_right = x_plot
            y_right = y2_plot

    set_right_x_mode(RIGHT_X_MODE)

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

# -------------------------
# TIMER
# -------------------------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

def close_resources():
    if csv_file is not None and not csv_file.closed:
        csv_file.close()
    if ser is not None and ser.is_open:
        ser.close()

app.aboutToQuit.connect(close_resources)

app.exec()