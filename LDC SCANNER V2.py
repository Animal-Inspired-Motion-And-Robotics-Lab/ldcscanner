import serial
import numpy as np
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

pg.setConfigOptions(antialias=True)

# -------------------------
# SERIAL CONFIG
# ------------------------- 
SERIAL_PORT = "COM6"  
BAUDRATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

# -------------------------
# DATA STORAGE
# -------------------------
MAX_POINTS = 5000
timestamps = deque(maxlen=MAX_POINTS)
sensor1 = deque(maxlen=MAX_POINTS)
sensor2 = deque(maxlen=MAX_POINTS)

# control flags
paused = False

# XY plot tracking
xy_start_index = 0

# adjustable parameters
SMOOTH_WINDOW = 10
THRESHOLD = 0.005
DISTANCE_LABEL_THRESHOLD = 5
CRACK_CONSTANT = 0.9  # multiplier for crack depth estimate

# -------------------------
# SMOOTHING FUNCTION
# -------------------------
def moving_average(data, window):
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")

# -------------------------
# EVENT DETECTION
# -------------------------
def detect_opposing(dx, dy, threshold):
    return (dx < -threshold) and (dy > threshold)

# -------------------------
# QT SETUP
# -------------------------
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Eddy Current Scanner")
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

slider_container = QtWidgets.QWidget()
slider_container.setLayout(slider_layout)
proxy = QtWidgets.QGraphicsProxyWidget()
proxy.setWidget(slider_container)
win.addItem(proxy, row=2, col=0, colspan=2)

# -------------------------
# KEY HANDLER
# -------------------------
def keyPressEvent(event):
    global paused, xy_start_index
    if event.key() == QtCore.Qt.Key_Space:
        xy_start_index = len(sensor1)
        xy_curve.clear()
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

win.keyPressEvent = keyPressEvent

# -------------------------
# SERIAL READ
# -------------------------
def read_serial():
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
        except:
            pass

# -------------------------
# UPDATE LOOP
# -------------------------
def update():
    read_serial()
    x = np.array(timestamps)
    y1 = np.array(sensor1)
    y2 = np.array(sensor2)
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
    xy_curve.setData(y1_plot, y2_plot)

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
                    label.setPos(y1_s[i-1], y2_s[i-1])
                    plot_xy.addItem(label)
                event_active = False

    # ongoing event at end
    if event_active and current_event_dist > DISTANCE_LABEL_THRESHOLD:
        crack_estimate = current_event_dist * CRACK_CONSTANT
        label_text = f"Dist={current_event_dist:.2f}\nCrack={crack_estimate:.2f} thou"
        label = pg.TextItem(text=label_text, color='w', anchor=(0.5, -0.5))
        label.setPos(y1_s[-1], y2_s[-1])
        plot_xy.addItem(label)

    event_curve.setData(event_x, event_y)

# -------------------------
# TIMER
# -------------------------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

app.exec()