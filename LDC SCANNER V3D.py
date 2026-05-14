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

CSV_FILE = input("Enter CSV filename (default: serial_data_log.csv): ").strip()
if not CSV_FILE:
    CSV_FILE = "serial_data_log.csv"
if not CSV_FILE.lower().endswith(".csv"):
    CSV_FILE += ".csv"

csv_file = open(CSV_FILE, "a", newline="")
csv_writer = csv.writer(csv_file)
if os.path.getsize(CSV_FILE) == 0:
    csv_writer.writerow(["timestamp", "sensor1", "sensor2", "sensor1_smooth", 
                         "sensor2_smooth", "sensor1_smooth_rot", "sensor2_smooth_rot", 
                         "rotation_angle_deg"])

# -------------------------
# DATA STORAGE
# -------------------------
MAX_POINTS = 5000
timestamps = deque(maxlen=MAX_POINTS)
sensor1 = deque(maxlen=MAX_POINTS)
sensor2 = deque(maxlen=MAX_POINTS)

# control flags
paused = False
write_to_file_enabled = False

# latest serial sample readout text
latest_readout_text = "Incoming: waiting for data..."
latest_average_text = "Average: waiting for data..."
last_average_update_time = 0.0

# XY plot tracking
xy_start_index = 0

# adjustable parameters
SMOOTH_WINDOW = 10
DISPLAY_LAG_POINTS = 1  # skip newest N points in plots to reduce right-edge jitter
ROTATION_ANGLE = 0.0  # degrees; applied to XY phase-space plot
RECENT_FADE_POINTS = 100
AVERAGE_UPDATE_INTERVAL_SEC = 5.0
PEAK_SEARCH_WINDOW = 50  # number of recent smoothed points to scan for peaks
PEAK_SLOPE_WINDOW = 5    # consecutive points that must be rising before AND falling after the peak
PEAK_MIN_PROMINENCE = 0.1  # minimum height of peak above both flanks
PEAK_HIGHLIGHT_FRAMES = 30  # how many update frames (~1.5 s) to keep peak green
PEAK_VALLEY_WINDOW = 25  # max points to search on each side of peak for local troughs
SURFACE_MAX_POINTS = 600
SURFACE_MAX_POINTS_MIN = SURFACE_MAX_POINTS
SURFACE_MAX_POINTS_MAX = SURFACE_MAX_POINTS * 10
SURFACE_DATA_MODE = "RAW"  # RAW or ROTATED for left 3D panel
SURFACE_SMOOTH_MODE = "UNSMOOTHED"  # UNSMOOTHED or SMOOTHED input for left 3D panel

# last detected peak height (persists until a new peak is found)
last_peak_s2 = float('nan')
peak_highlight_frames = 0
peak_abs_left = -1   # abs index in y2_rot of the peak's left flank
peak_abs_right = -1  # abs index in y2_rot of the peak's right flank

# -------------------------
# SMOOTHING FUNCTION
# -------------------------
def moving_average(data, window):
    arr = np.asarray(data, dtype=float)
    w = int(window)
    if len(arr) == 0 or w <= 1:
        return arr
    if w > len(arr):
        w = len(arr)

    # Edge padding prevents artificial dips/spikes at the start and end.
    left = w // 2
    right = w - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    smoothed = np.convolve(padded, kernel, mode="valid")

    # Keep the newest point causal to avoid right-edge amplification jitter.
    smoothed[-1] = np.mean(arr[-w:])
    return smoothed

def detect_last_peak(data):
    """Return the most recent qualified peak as (height, peak_i, left_i, right_i).

    A qualified peak at index i must have:
      - PEAK_SLOPE_WINDOW consecutive rising points leading up to it
      - PEAK_SLOPE_WINDOW consecutive falling points after it
            - height above both side troughs by at least PEAK_MIN_PROMINENCE
    """
    w = PEAK_SLOPE_WINDOW
    for i in range(len(data) - w - 1, w - 1, -1):
        # Check rising flank: all points in [i-w .. i] must be strictly increasing
        rising = all(data[i - w + k] < data[i - w + k + 1] for k in range(w))
        # Check falling flank: all points in [i .. i+w] must be strictly decreasing
        falling = all(data[i + k] > data[i + k + 1] for k in range(w))
        if not (rising and falling):
            continue
        # Find local troughs (valleys) around this peak so event span follows full shape.
        left_search_start = max(0, i - PEAK_VALLEY_WINDOW)
        right_search_end = min(len(data), i + PEAK_VALLEY_WINDOW + 1)
        left_segment = data[left_search_start:i + 1]
        right_segment = data[i:right_search_end]
        if len(left_segment) == 0 or len(right_segment) == 0:
            continue
        left_i = left_search_start + int(np.argmin(left_segment))
        right_i = i + int(np.argmin(right_segment))
        left_valley = float(data[left_i])
        right_valley = float(data[right_i])

        # Prominence gate: peak must exceed both side valleys by threshold.
        if data[i] - left_valley >= PEAK_MIN_PROMINENCE and data[i] - right_valley >= PEAK_MIN_PROMINENCE:
            # Report full event excursion as max-min over the detected peak span.
            event_segment = data[left_i:right_i + 1]
            height = float(np.max(event_segment) - np.min(event_segment))
            return height, i, left_i, right_i
    return float('nan'), None, None, None

def latest_smoothed_value(data, window):
    if not data:
        return np.nan
    if len(data) < window:
        return float(data[-1])
    return float(np.mean(list(data)[-window:]))

def rotate_points(points, degrees):
    """Rotates a set of 2D points by a given angle in degrees."""
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(points, R.T)

def rotate_xy_arrays(x_data, y_data, degrees, cx=0.0, cy=0.0, sx=1.0, sy=1.0):
    if len(x_data) == 0:
        return x_data, y_data
    points = np.column_stack((np.asarray(x_data, dtype=float), np.asarray(y_data, dtype=float)))
    # Normalize axes before rotation so angle behaves consistently when x/y scales differ.
    centered = points - np.array([cx, cy], dtype=float)
    normalized = centered / np.array([sx, sy], dtype=float)
    rotated_norm = rotate_points(normalized, -degrees)
    rotated = rotated_norm * np.array([sx, sy], dtype=float) + np.array([cx, cy], dtype=float)
    return rotated[:, 0], rotated[:, 1]

def get_rotation_params(y1_plot, y2_plot):
    if len(y1_plot) > 0:
        rot_cx = float(np.mean(y1_plot))
        rot_cy = float(np.mean(y2_plot))
        rot_sx = float(np.ptp(y1_plot))
        rot_sy = float(np.ptp(y2_plot))
        if rot_sx == 0.0:
            rot_sx = 1.0
        if rot_sy == 0.0:
            rot_sy = 1.0
        return rot_cx, rot_cy, rot_sx, rot_sy
    return 0.0, 0.0, 1.0, 1.0

def build_surface_data(x_vals, rp_vals, l_vals):
    if len(x_vals) < 2:
        return None

    x_recent = np.asarray(x_vals[-SURFACE_MAX_POINTS:], dtype=float)
    y_recent = np.asarray(rp_vals[-SURFACE_MAX_POINTS:], dtype=float)
    z_recent = np.asarray(l_vals[-SURFACE_MAX_POINTS:], dtype=float)
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

class ResettableDial(QtWidgets.QDial):
    """Dial that uses relative drag instead of click-to-jump, with scroll-wheel fine control."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_start_x = None
        self._drag_start_val = None

    def mouseDoubleClickEvent(self, event):
        self.setValue(0)
        # don't call super() so it doesn't also jump to clicked position

    def mousePressEvent(self, event):
        # Record start position; do NOT jump to clicked angle
        self._drag_start_x = event.x()
        self._drag_start_val = self.value()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_start_x is None:
            return
        # 1 step per pixel of horizontal drag (1 step = 0.1°)
        delta = event.x() - self._drag_start_x
        new_val = self._drag_start_val + delta
        new_val = max(self.minimum(), min(self.maximum(), new_val))
        self.setValue(new_val)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_start_x = None
        self._drag_start_val = None
        event.accept()

    def wheelEvent(self, event):
        # Scroll wheel: 1 step (0.1°) per wheel click
        delta = event.angleDelta().y()
        step = 1 if delta > 0 else -1
        self.setValue(self.value() + step)
        event.accept()

# -------------------------
# QT SETUP
# -------------------------
app = QtWidgets.QApplication([])
main_widget = QtWidgets.QWidget()
main_widget.setWindowTitle("Eddy Current Scanner")
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
win = pg.GraphicsLayoutWidget()
top_row_layout.addWidget(win, 1)

win.setFocusPolicy(QtCore.Qt.StrongFocus)
win.setFocus()

plot_xy = win.addPlot(title="Phase Space")
plot_xy.setLabel('bottom', 'R_p (ohm)')
plot_xy.setLabel('left', 'L (uH)')
initial_xy_view_state = plot_xy.getViewBox().getState(copy=True)
xy_curve = plot_xy.plot(pen='r')
recent_segment_curves = []
for _ in range(max(RECENT_FADE_POINTS - 1, 0)):
    recent_segment_curves.append(plot_xy.plot(pen=pg.mkPen((255, 0, 0), width=3)))

# -------------------------
# SLIDERS
# -------------------------
slider_layout = QtWidgets.QHBoxLayout()

# Smooth window + min peak height stacked vertically
surface_mode_label = QtWidgets.QLabel("3D data:")
surface_mode_switch = QtWidgets.QCheckBox()
surface_mode_switch.setChecked(SURFACE_DATA_MODE == "ROTATED")
surface_mode_switch.setToolTip("Toggle RAW/ROTATED for left 3D panel")
surface_mode_switch.setStyleSheet(
    "QCheckBox::indicator { width: 36px; height: 20px; border-radius: 10px;"
    " border: 1px solid #666; background: #2b2b2b; }"
    "QCheckBox::indicator:checked { background: #27ae60; border: 1px solid #1e8449; }"
)
surface_mode_value_label = QtWidgets.QLabel("ROTATED" if SURFACE_DATA_MODE == "ROTATED" else "RAW")
surface_mode_value_label.setMinimumWidth(60)

def surface_mode_toggled(checked):
    global SURFACE_DATA_MODE
    SURFACE_DATA_MODE = "ROTATED" if checked else "RAW"
    surface_mode_value_label.setText(SURFACE_DATA_MODE)
    surface_title.setText(f"3D Surface ({SURFACE_DATA_MODE}): Time (x), R_p (y), L (z)")

surface_mode_switch.toggled.connect(surface_mode_toggled)

# Smoothed vs Raw toggle for left 3D panel
surface_smooth_label = QtWidgets.QLabel("3D input:")
surface_smooth_switch = QtWidgets.QCheckBox()
surface_smooth_switch.setChecked(SURFACE_SMOOTH_MODE == "SMOOTHED")
surface_smooth_switch.setToolTip("Toggle UNSMOOTHED/SMOOTHED input for left 3D panel")
surface_smooth_switch.setStyleSheet(
    "QCheckBox::indicator { width: 36px; height: 20px; border-radius: 10px;"
    " border: 1px solid #666; background: #2b2b2b; }"
    "QCheckBox::indicator:checked { background: #27ae60; border: 1px solid #1e8449; }"
)
surface_smooth_value_label = QtWidgets.QLabel("SMOOTHED" if SURFACE_SMOOTH_MODE == "SMOOTHED" else "UNSMOOTHED")
surface_smooth_value_label.setMinimumWidth(68)

def surface_smooth_toggled(checked):
    global SURFACE_SMOOTH_MODE
    SURFACE_SMOOTH_MODE = "SMOOTHED" if checked else "UNSMOOTHED"
    surface_smooth_value_label.setText(SURFACE_SMOOTH_MODE)

surface_smooth_switch.toggled.connect(surface_smooth_toggled)

smooth_label = QtWidgets.QLabel(f"Smooth window: {SMOOTH_WINDOW}")
smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
smooth_slider.setMinimum(1)
smooth_slider.setMaximum(100)
smooth_slider.setValue(SMOOTH_WINDOW)
def smooth_changed(value):
    global SMOOTH_WINDOW
    SMOOTH_WINDOW = int(value)
    smooth_label.setText(f"Smooth window: {SMOOTH_WINDOW}")
smooth_slider.valueChanged.connect(smooth_changed)

surface_points_label = QtWidgets.QLabel(f"3D points: {SURFACE_MAX_POINTS}")
surface_points_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
surface_points_slider.setMinimum(SURFACE_MAX_POINTS_MIN)
surface_points_slider.setMaximum(SURFACE_MAX_POINTS_MAX)
surface_points_slider.setValue(SURFACE_MAX_POINTS)
surface_points_slider.setSingleStep(50)
surface_points_slider.setPageStep(300)
def surface_points_changed(value):
    global SURFACE_MAX_POINTS
    SURFACE_MAX_POINTS = int(value)
    surface_points_label.setText(f"3D points: {SURFACE_MAX_POINTS}")
surface_points_slider.valueChanged.connect(surface_points_changed)

# Min peak height slider: 0.001 – 1.0 in steps of 0.001 (stored as int * 1000)
_PEAK_PROM_SCALE = 1000
peak_prom_label = QtWidgets.QLabel(f"Min peak height: {PEAK_MIN_PROMINENCE:.3f} µH")
peak_prom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
peak_prom_slider.setMinimum(1)
peak_prom_slider.setMaximum(250)
peak_prom_slider.setValue(int(PEAK_MIN_PROMINENCE * _PEAK_PROM_SCALE))
def peak_prom_changed(value):
    global PEAK_MIN_PROMINENCE
    PEAK_MIN_PROMINENCE = value / _PEAK_PROM_SCALE
    peak_prom_label.setText(f"Min peak height: {PEAK_MIN_PROMINENCE:.3f} µH")
peak_prom_slider.valueChanged.connect(peak_prom_changed)

smooth_row = QtWidgets.QHBoxLayout()
smooth_row.setContentsMargins(0, 0, 0, 0)
smooth_row.addWidget(smooth_label)
smooth_row.addWidget(smooth_slider)

surface_points_row = QtWidgets.QHBoxLayout()
surface_points_row.setContentsMargins(0, 0, 0, 0)
surface_points_row.addWidget(surface_points_label)
surface_points_row.addWidget(surface_points_slider)

peak_prom_row = QtWidgets.QHBoxLayout()
peak_prom_row.setContentsMargins(0, 0, 0, 0)
peak_prom_row.addWidget(peak_prom_label)
peak_prom_row.addWidget(peak_prom_slider)

surface_mode_row = QtWidgets.QHBoxLayout()
surface_mode_row.setContentsMargins(0, 0, 0, 0)
surface_mode_row.addWidget(surface_mode_label)
surface_mode_row.addWidget(surface_mode_switch)
surface_mode_row.addWidget(surface_mode_value_label)

surface_smooth_row = QtWidgets.QHBoxLayout()
surface_smooth_row.setContentsMargins(0, 0, 0, 0)
surface_smooth_row.addWidget(surface_smooth_label)
surface_smooth_row.addWidget(surface_smooth_switch)
surface_smooth_row.addWidget(surface_smooth_value_label)

sliders_left_layout = QtWidgets.QVBoxLayout()
sliders_left_layout.setContentsMargins(0, 0, 0, 0)
sliders_left_layout.setSpacing(2)
sliders_left_layout.addLayout(surface_mode_row)
sliders_left_layout.addLayout(surface_smooth_row)
sliders_left_layout.addLayout(surface_points_row)
sliders_left_layout.addLayout(smooth_row)
sliders_left_layout.addLayout(peak_prom_row)

sliders_left_container = QtWidgets.QWidget()
sliders_left_container.setLayout(sliders_left_layout)
slider_layout.addWidget(sliders_left_container)
slider_layout.addSpacing(10)

peak_label = QtWidgets.QLabel("Peak: --")
peak_label.setMinimumWidth(220)
slider_layout.addWidget(peak_label)

slider_layout.addSpacing(30)

# Rotation dial
rotation_label = QtWidgets.QLabel("Rotation:")
rotation_dial = ResettableDial()
rotation_dial.setMinimum(-1800)
rotation_dial.setMaximum(1800)
rotation_dial.setValue(0)
rotation_dial.setNotchesVisible(True)
rotation_dial.setFixedSize(60, 60)
rotation_dial.setToolTip("Drag up/down to rotate · Scroll wheel for fine steps · Double-click to reset")

rotation_spinbox = QtWidgets.QDoubleSpinBox()
rotation_spinbox.setMinimum(-180.0)
rotation_spinbox.setMaximum(180.0)
rotation_spinbox.setSingleStep(0.1)
rotation_spinbox.setDecimals(1)
rotation_spinbox.setSuffix("°")
rotation_spinbox.setValue(0.0)
rotation_spinbox.setFixedWidth(80)
rotation_spinbox.setToolTip("Type an exact rotation angle")

_rotation_updating = False
def rotation_changed(value):
    global ROTATION_ANGLE, _rotation_updating
    if _rotation_updating:
        return
    _rotation_updating = True
    ROTATION_ANGLE = value / 10.0
    rotation_label.setText(f"Rotation: {ROTATION_ANGLE:.1f}°")
    rotation_spinbox.setValue(ROTATION_ANGLE)
    _rotation_updating = False

def rotation_spinbox_changed(value):
    global ROTATION_ANGLE, _rotation_updating
    if _rotation_updating:
        return
    _rotation_updating = True
    ROTATION_ANGLE = value
    rotation_label.setText(f"Rotation: {ROTATION_ANGLE:.1f}°")
    rotation_dial.setValue(int(round(value * 10)))
    _rotation_updating = False

rotation_dial.valueChanged.connect(rotation_changed)
rotation_spinbox.valueChanged.connect(rotation_spinbox_changed)

rotation_label.setText("Rotation: 0.0°")

rotation_widget = QtWidgets.QWidget()
rotation_layout = QtWidgets.QVBoxLayout(rotation_widget)
rotation_layout.setContentsMargins(0, 0, 0, 0)
rotation_layout.setSpacing(2)
rotation_layout.addWidget(rotation_label)
rotation_layout.addWidget(rotation_spinbox)

slider_layout.addWidget(rotation_widget)
slider_layout.addWidget(rotation_dial)

slider_layout.addSpacing(30)

# Live incoming serial readout
readout_label = QtWidgets.QLabel(latest_readout_text)
readout_label.setMinimumWidth(320)
average_label = QtWidgets.QLabel(f"Avg last {RECENT_FADE_POINTS}: waiting for data...")
average_label.setMinimumWidth(320)

readout_layout = QtWidgets.QVBoxLayout()
readout_layout.setContentsMargins(0, 0, 0, 0)
readout_layout.setSpacing(2)
readout_layout.addWidget(readout_label)
readout_layout.addWidget(average_label)

readout_container = QtWidgets.QWidget()
readout_container.setLayout(readout_layout)
slider_layout.addWidget(readout_container)

slider_layout.addStretch()

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

write_controls_layout = QtWidgets.QVBoxLayout()
write_controls_layout.setContentsMargins(0, 0, 0, 0)
write_controls_layout.setSpacing(2)
write_controls_layout.addWidget(write_toggle_button)
write_controls_layout.addWidget(write_file_label)

write_controls_container = QtWidgets.QWidget()
write_controls_container.setLayout(write_controls_layout)
slider_layout.addWidget(write_controls_container)

slider_container = QtWidgets.QWidget()
slider_container.setLayout(slider_layout)
main_layout.addWidget(slider_container, 0)
main_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
main_widget.setFocus()
main_widget.show()

# -------------------------
# KEY HANDLER
# -------------------------
def keyPressEvent(event):
    global paused, xy_start_index, last_peak_s2, peak_highlight_frames, peak_abs_left, peak_abs_right
    global latest_average_text, last_average_update_time
    if event.key() == QtCore.Qt.Key_Space:
        # Clear all buffered data so both plots restart from a clean state.
        timestamps.clear()
        sensor1.clear()
        sensor2.clear()
        xy_start_index = 0

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

        # Reset peak/average labels.
        last_peak_s2 = float('nan')
        peak_highlight_frames = 0
        peak_abs_left = -1
        peak_abs_right = -1
        peak_label.setText("Peak (vertical): --")
        latest_average_text = f"Avg last {RECENT_FADE_POINTS}: waiting for data..."
        average_label.setText(latest_average_text)
        last_average_update_time = time.monotonic()
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
def parse_serial_line(line):
    if not line:
        raise ValueError("Empty serial line")

    if ">" in line and ":" in line:
        fields = {}
        payload = line.split("|", 1)[0]
        for segment in payload.split(">"):
            if not segment or ":" not in segment:
                continue
            key, value = segment.split(":", 1)
            fields[key.strip().lower()] = float(value.strip())

        return fields["t"], fields["rp"], fields["l"]

    t, s1, s2 = map(float, line.split())
    return t, s1, s2

def read_serial():
    global latest_readout_text, write_to_file_enabled
    if paused:
        ser.reset_input_buffer()
        return
    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        try:
            t, s1, s2 = parse_serial_line(line)
            timestamps.append(t)
            sensor1.append(s1)
            sensor2.append(s2)

            s1_smooth = latest_smoothed_value(sensor1, SMOOTH_WINDOW)
            s2_smooth = latest_smoothed_value(sensor2, SMOOTH_WINDOW)

            y1_s = moving_average(np.array(sensor1), SMOOTH_WINDOW)
            y2_s = moving_average(np.array(sensor2), SMOOTH_WINDOW)
            xy_offset = max(0, xy_start_index - (len(sensor1) - len(y1_s)))
            y1_plot = y1_s[xy_offset:]
            y2_plot = y2_s[xy_offset:]
            rot_cx, rot_cy, rot_sx, rot_sy = get_rotation_params(y1_plot, y2_plot)
            s1_rot, s2_rot = rotate_xy_arrays(np.array([s1_smooth]), np.array([s2_smooth]), ROTATION_ANGLE, rot_cx, rot_cy, rot_sx, rot_sy)

            if write_to_file_enabled:
                csv_writer.writerow([t, s1, s2, s1_smooth, s2_smooth, float(s1_rot[0]), float(s2_rot[0]), ROTATION_ANGLE])
                csv_file.flush()
            latest_readout_text = f"Incoming: t={t:.3f} | s1={s1:.6f} | s2={s2:.6f}"
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

    global last_peak_s2, peak_highlight_frames, peak_abs_left, peak_abs_right

    # smoothing
    y1_s = moving_average(y1, SMOOTH_WINDOW)
    y2_s = moving_average(y2, SMOOTH_WINDOW)
    x_s = x[-len(y1_s):]

    # Live 3D surface source: choose smoothed or raw, then raw or rotated.
    if SURFACE_SMOOTH_MODE == "SMOOTHED":
        surface_base_rp = y1_s
        surface_base_l = y2_s
        surface_base_x = x_s
    else:
        surface_base_rp = y1
        surface_base_l = y2
        surface_base_x = x

    surface_rp = surface_base_rp
    surface_l = surface_base_l
    if SURFACE_DATA_MODE == "ROTATED":
        rot_cx_s, rot_cy_s, rot_sx_s, rot_sy_s = get_rotation_params(surface_base_rp, surface_base_l)
        surface_rp, surface_l = rotate_xy_arrays(surface_base_rp, surface_base_l, ROTATION_ANGLE, rot_cx_s, rot_cy_s, rot_sx_s, rot_sy_s)

    surface_data = build_surface_data(surface_base_x, surface_rp, surface_l)
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

    # XY plot (only new data after reset)
    xy_offset = max(0, xy_start_index - (len(y1) - len(y1_s)))
    y1_plot = y1_s[xy_offset:]
    y2_plot = y2_s[xy_offset:]

    rot_cx, rot_cy, rot_sx, rot_sy = get_rotation_params(y1_plot, y2_plot)

    y1_rot, y2_rot = rotate_xy_arrays(y1_plot, y2_plot, ROTATION_ANGLE, rot_cx, rot_cy, rot_sx, rot_sy)
    xy_curve.setData(y1_rot, y2_rot)

    # peak detection: look for upward peaks in the vertical axis (y2_rot) of phase-space
    if len(y2_rot) >= 3:
        slice_start = max(0, len(y2_rot) - PEAK_SEARCH_WINDOW)
        pk2, peak_i, peak_left_i, peak_right_i = detect_last_peak(y2_rot[-PEAK_SEARCH_WINDOW:])
        if not np.isnan(pk2):
            last_peak_s2 = pk2
            peak_abs_left = slice_start + peak_left_i
            peak_abs_right = slice_start + peak_right_i
            peak_highlight_frames = PEAK_HIGHLIGHT_FRAMES
    pk_str = f"{last_peak_s2:.6f}" if not np.isnan(last_peak_s2) else "--"
    peak_label.setText(f"Peak (vertical): {pk_str}")

    # Highlight most recent trajectory with red -> white segment gradient.
    tail_count = min(RECENT_FADE_POINTS, len(y1_rot))
    if tail_count > 1:
        tail_x = y1_rot[-tail_count:]
        tail_y = y2_rot[-tail_count:]
        seg_count = tail_count - 1
        shades = np.linspace(0, 255, seg_count).astype(int)
        tail_start_abs = len(y2_rot) - tail_count
        for i in range(seg_count):
            seg_curve = recent_segment_curves[i]
            seg_abs_start = tail_start_abs + i
            seg_abs_end = tail_start_abs + i + 1
            if (peak_highlight_frames > 0
                    and peak_abs_left >= 0
                    and seg_abs_end >= peak_abs_left
                    and seg_abs_start <= peak_abs_right):
                seg_curve.setPen(pg.mkPen((0, 255, 0, 255), width=3))
            else:
                seg_curve.setPen(pg.mkPen((255, int(shades[i]), int(shades[i]), 255), width=3))
            seg_curve.setData([tail_x[i], tail_x[i + 1]], [tail_y[i], tail_y[i + 1]])
        for i in range(seg_count, len(recent_segment_curves)):
            recent_segment_curves[i].setData([], [])
    else:
        for seg_curve in recent_segment_curves:
            seg_curve.setData([], [])
    if peak_highlight_frames > 0:
        peak_highlight_frames -= 1

# -------------------------
# TIMER
# -------------------------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

def close_resources():
    csv_file.close()

app.aboutToQuit.connect(close_resources)

app.exec()