import serial
import numpy as np
import csv
import os
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

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

# XY plot tracking
xy_start_index = 0

# adjustable parameters
SMOOTH_WINDOW = 10
THRESHOLD = 0.005
DISTANCE_LABEL_THRESHOLD = 5
CRACK_CONSTANT = 0.9  # multiplier for crack depth estimate
ROTATION_ANGLE = 0.0  # degrees; applied to XY phase-space plot
RECENT_FADE_POINTS = 100

# -------------------------
# SMOOTHING FUNCTION
# -------------------------
def moving_average(data, window):
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")

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

class ResettableDial(QtWidgets.QDial):
    def mouseDoubleClickEvent(self, event):
        self.setValue(0)
        super().mouseDoubleClickEvent(event)

# -------------------------
# EVENT DETECTION
# -------------------------
def detect_opposing(dx, dy, threshold):
    return (dx < -threshold) and (dy > threshold)

# -------------------------
# QT SETUP
# -------------------------
app = QtWidgets.QApplication([])
main_widget = QtWidgets.QWidget()
main_widget.setWindowTitle("Eddy Current Scanner")
main_layout = QtWidgets.QVBoxLayout(main_widget)
main_layout.setContentsMargins(6, 6, 6, 6)
main_layout.setSpacing(6)

win = pg.GraphicsLayoutWidget()
main_layout.addWidget(win, 1)

win.setFocusPolicy(QtCore.Qt.StrongFocus)
win.setFocus()

# -------------------------
# LEFT PLOT (TIME SERIES)
# -------------------------
plot_time = win.addPlot(title="Sensors vs Time")

# Remove default labels (we'll use colored TextItems)
plot_time.setLabel('left', '')
plot_time.setLabel('right', '')
plot_time.setLabel('bottom', 'Time')
plot_time.getAxis('left').setPen('w')
plot_time.getAxis('right').setPen('w')
plot_time.getAxis('bottom').setPen('w')

curve1 = plot_time.plot(pen='y')
plot_time.showAxis('right')
p2 = pg.ViewBox()
plot_time.scene().addItem(p2)
plot_time.getAxis('right').linkToView(p2)
p2.setXLink(plot_time)
curve2 = pg.PlotCurveItem(pen='c')
p2.addItem(curve2)

def updateViews():
    p2.setGeometry(plot_time.vb.sceneBoundingRect())
    p2.linkedViewChanged(plot_time.vb, p2.XAxis)
plot_time.vb.sigResized.connect(updateViews)

# Add custom colored axis labels as TextItems
label_left = pg.TextItem("R_p (ohm)", color=(255, 255, 0))   # yellow
label_right = pg.TextItem("L (uH)", color=(0, 255, 255))  # cyan
plot_time.addItem(label_left)
plot_time.addItem(label_right)

# -------------------------
# RIGHT PLOT (XY)
# -------------------------
win.nextColumn()
plot_xy = win.addPlot(title="Phase Space")
plot_xy.setLabel('bottom', 'R_p (ohm)')
plot_xy.setLabel('left', 'L (uH)')
xy_curve = plot_xy.plot(pen='r')
recent_segment_curves = []
for _ in range(max(RECENT_FADE_POINTS - 1, 0)):
    recent_segment_curves.append(plot_xy.plot(pen=pg.mkPen((255, 0, 0), width=3)))
event_curve = plot_xy.plot(pen=pg.mkPen(color=(0,255,0,128), width=3))  # semi-transparent green

# -------------------------
# SLIDERS
# -------------------------
win.nextRow()
slider_layout = QtWidgets.QHBoxLayout()

# Smooth window
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
slider_layout.addWidget(smooth_label)
slider_layout.addWidget(smooth_slider)

slider_layout.addSpacing(30)

# Threshold slider 0.00001 -> 0.01
threshold_label = QtWidgets.QLabel(f"Event threshold: {THRESHOLD:.5f}")
threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
threshold_slider.setMinimum(1)       # corresponds to 0.00001
threshold_slider.setMaximum(1000)    # corresponds to 0.01
threshold_slider.setValue(int(THRESHOLD * 100000))
def threshold_changed(value):
    global THRESHOLD
    THRESHOLD = value / 100000.0
    threshold_label.setText(f"Event threshold: {THRESHOLD:.5f}")
threshold_slider.valueChanged.connect(threshold_changed)
slider_layout.addWidget(threshold_label)
slider_layout.addWidget(threshold_slider)

slider_layout.addSpacing(30)

# Rotation dial
rotation_label = QtWidgets.QLabel(f"Rotation: 0.0°")
rotation_dial = ResettableDial()
rotation_dial.setMinimum(-1800)
rotation_dial.setMaximum(1800)
rotation_dial.setValue(0)
rotation_dial.setNotchesVisible(True)
rotation_dial.setFixedSize(60, 60)
def rotation_changed(value):
    global ROTATION_ANGLE
    ROTATION_ANGLE = value / 10.0
    rotation_label.setText(f"Rotation: {ROTATION_ANGLE:.1f}°")
rotation_dial.valueChanged.connect(rotation_changed)
slider_layout.addWidget(rotation_label)
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
main_widget.show()

# -------------------------
# KEY HANDLER
# -------------------------
def keyPressEvent(event):
    global paused, xy_start_index
    if event.key() == QtCore.Qt.Key_Space:
        xy_start_index = len(sensor1)
        xy_curve.clear()
        for seg_curve in recent_segment_curves:
            seg_curve.setData([], [])
        event_curve.clear()
        for item in plot_xy.items[:]:
            if isinstance(item, pg.TextItem):
                plot_xy.removeItem(item)
        plot_xy.enableAutoRange()
    elif event.key() == QtCore.Qt.Key_P:
        paused = not paused
        if paused:
            ser.reset_input_buffer()
        print("Paused" if paused else "Resumed")
    elif event.key() == QtCore.Qt.Key_F:
        write_toggle_button.setChecked(not write_toggle_button.isChecked())
        print("CSV write ON" if write_toggle_button.isChecked() else "CSV write OFF")

win.keyPressEvent = keyPressEvent

# -------------------------
# SERIAL READ
# -------------------------
def read_serial():
    global latest_readout_text, write_to_file_enabled
    if paused:
        ser.reset_input_buffer()
        return
    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        try:
            t, s1, s2 = map(float, line.split())
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
        except:
            pass

# -------------------------
# UPDATE LOOP
# -------------------------
def update():
    read_serial()
    readout_label.setText(latest_readout_text)

    x = np.array(timestamps)
    y1 = np.array(sensor1)
    y2 = np.array(sensor2)

    avg_count = min(RECENT_FADE_POINTS, len(y1))
    if avg_count > 0:
        avg_s1 = float(np.mean(y1[-avg_count:]))
        avg_s2 = float(np.mean(y2[-avg_count:]))
        average_label.setText(f"Avg last {avg_count}: s1={avg_s1:.6f} | s2={avg_s2:.6f}")
    else:
        average_label.setText(f"Avg last {RECENT_FADE_POINTS}: waiting for data...")

    if len(x) == 0:
        return

    # smoothing
    y1_s = moving_average(y1, SMOOTH_WINDOW)
    y2_s = moving_average(y2, SMOOTH_WINDOW)
    x_s = x[-len(y1_s):]

    # Update custom axis label positions dynamically
    if len(y1_s):
        label_left.setPos(x_s[0], y1_s[-1])
    if len(y2_s):
        label_right.setPos(x_s[0], y2_s[-1])

    # time plot
    curve1.setData(x_s, y1_s)
    curve2.setData(x_s, y2_s)

    # clear previous text labels on XY plot
    for item in plot_xy.items[:]:
        if isinstance(item, pg.TextItem):
            plot_xy.removeItem(item)

    # XY plot (only new data after reset)
    xy_offset = max(0, xy_start_index - (len(sensor1) - len(y1_s)))
    y1_plot = y1_s[xy_offset:]
    y2_plot = y2_s[xy_offset:]

    rot_cx, rot_cy, rot_sx, rot_sy = get_rotation_params(y1_plot, y2_plot)

    y1_rot, y2_rot = rotate_xy_arrays(y1_plot, y2_plot, ROTATION_ANGLE, rot_cx, rot_cy, rot_sx, rot_sy)
    xy_curve.setData(y1_rot, y2_rot)

    # Highlight most recent trajectory with red -> white segment gradient.
    tail_count = min(RECENT_FADE_POINTS, len(y1_rot))
    if tail_count > 1:
        tail_x = y1_rot[-tail_count:]
        tail_y = y2_rot[-tail_count:]
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

    # event highlighting + distance calculation
    event_x = []
    event_y = []
    current_event_dist = 0
    event_active = False

    for i in range(max(1, xy_offset), len(y1_s)):
        dx = y1_s[i] - y1_s[i-1]
        dy = y2_s[i] - y2_s[i-1]
        if detect_opposing(dx, dy, THRESHOLD):
            event_x.extend([y1_s[i-1], y1_s[i], np.nan])
            event_y.extend([y2_s[i-1], y2_s[i], np.nan])
            if not event_active:
                current_event_dist = 0
                event_active = True
            current_event_dist += np.sqrt(dx**2 + dy**2)
        else:
            if event_active:
                if current_event_dist > DISTANCE_LABEL_THRESHOLD:
                    crack_estimate = current_event_dist * CRACK_CONSTANT
                    label_text = f"Dist={current_event_dist:.2f}\nCrack={crack_estimate:.2f} thou"
                    label = pg.TextItem(text=label_text, color='w', anchor=(0.5, -0.5))
                    lx, ly = rotate_xy_arrays(np.array([y1_s[i-1]]), np.array([y2_s[i-1]]), ROTATION_ANGLE, rot_cx, rot_cy, rot_sx, rot_sy)
                    label.setPos(float(lx[0]), float(ly[0]))
                    plot_xy.addItem(label)
                event_active = False

    # ongoing event at end
    if event_active and current_event_dist > DISTANCE_LABEL_THRESHOLD:
        crack_estimate = current_event_dist * CRACK_CONSTANT
        label_text = f"Dist={current_event_dist:.2f}\nCrack={crack_estimate:.2f} thou"
        label = pg.TextItem(text=label_text, color='w', anchor=(0.5, -0.5))
        lx, ly = rotate_xy_arrays(np.array([y1_s[-1]]), np.array([y2_s[-1]]), ROTATION_ANGLE, rot_cx, rot_cy, rot_sx, rot_sy)
        label.setPos(float(lx[0]), float(ly[0]))
        plot_xy.addItem(label)

    if event_x:
        ex = np.array(event_x, dtype=float)
        ey = np.array(event_y, dtype=float)
        valid = ~np.isnan(ex) & ~np.isnan(ey)
        rx, ry = rotate_xy_arrays(ex[valid], ey[valid], ROTATION_ANGLE, rot_cx, rot_cy, rot_sx, rot_sy)
        ex[valid] = rx
        ey[valid] = ry
        event_curve.setData(ex.tolist(), ey.tolist())
    else:
        event_curve.setData([], [])

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