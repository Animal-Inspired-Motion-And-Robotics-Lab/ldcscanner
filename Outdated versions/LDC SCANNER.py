import serial
import numpy as np
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

pg.setConfigOptions(antialias=True)

# -------------------------
# SERIAL CONFIG 
# -------------------------

SERIAL_PORT = "COM21"
BAUDRATE = 9600

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

# -------------------------
# DATA STORAGE
# -------------------------

MAX_POINTS = 1000

timestamps = deque(maxlen=MAX_POINTS)
sensor1 = deque(maxlen=MAX_POINTS)
sensor2 = deque(maxlen=MAX_POINTS)

# smoothing window (samples)
SMOOTH_WINDOW = 10

# -------------------------
# SMOOTHING FUNCTION
# -------------------------

def moving_average(data, window):

    if len(data) < window:
        return data

    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


# -------------------------
# QT APPLICATION
# -------------------------

app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(show=True, title="Live Sensor Data")

# -------------------------
# LEFT PLOT (TIME SERIES)
# -------------------------

plot_time = win.addPlot(title="Sensors vs Time")

plot_time.setLabel('bottom', 'Time')
plot_time.setLabel('left', 'Sensor 1')

plot_time.setClipToView(True)
plot_time.setDownsampling(mode='peak')

curve1 = plot_time.plot(pen='y', name="Sensor 1")

plot_time.showAxis('right')
plot_time.getAxis('right').setLabel('Sensor 2')

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

# -------------------------
# RIGHT PLOT (XY LINE)
# -------------------------

win.nextColumn()

plot_xy = win.addPlot(title="Phase Space")

plot_xy.setLabel('bottom', 'R_p (ohm)')
plot_xy.setLabel('left', 'L (uH)')

xy_curve = plot_xy.plot(pen='r')

# -------------------------
# SERIAL READ
# -------------------------

def read_serial():

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
# ANALYSIS
# -------------------------

def analyze():

    if len(sensor1) > 20:
        avg = np.mean(sensor1)
        print("Sensor1 avg:", avg)


# -------------------------
# UPDATE LOOP
# -------------------------

def update():

    read_serial()

    if len(timestamps) > 0:

        x = np.array(timestamps)
        y1 = np.array(sensor1)
        y2 = np.array(sensor2)

        # apply smoothing
        y1_smooth = moving_average(y1, SMOOTH_WINDOW)
        y2_smooth = moving_average(y2, SMOOTH_WINDOW)

        # timestamps must match smoothed data length
        x_smooth = x[-len(y1_smooth):]

        # time plot
        curve1.setData(x_smooth, y1_smooth)
        curve2.setData(x_smooth, y2_smooth)

        # XY plot
        xy_curve.setData(y1_smooth, y2_smooth)

    analyze()


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

app.exec()