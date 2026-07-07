# ---------------------------------------------------------------------------
# LDC SCANNER V6 - Live eddy-current scanner readout and serial console
#
# Reads sensor telemetry from the scanner robot over a serial port and streams
# it into three live views (a 3D R_p/L surface, a phase-space/time trace, and a
# crack-event plot), optionally logs every sample to CSV, and provides a
# two-way serial command console.
#
# CODE MAP - the Qt-free backend now lives in the ``scanner/`` package; this
# file is the runnable GUI + wiring.  Launch it directly from the repo root.
#
#   scanner/config.py        serial / CSV / replay / plot constants
#   scanner/parsing.py       telemetry-line parsing (pure functions)
#   scanner/surface.py       3D ribbon-mesh geometry (pure functions)
#   scanner/widgets.py       ToggleAxisItem (clickable-axis pyqtgraph subclass)
#   scanner/csv_logger.py    CsvLogger - owns the output CSV file
#   scanner/data_sources.py  SerialManager (live) + CsvReplaySource (simulated)
#   scanner/state.py         ScannerState - live buffers + sample ingestion
#   scanner/snapshot.py      make_snapshot - journal-quality PDF export
#
# This file, top to bottom:
#   7. Runtime objects     - create the logger / state / serial manager
#   8. Qt user interface   - widget construction (built once, top to bottom)
#   9. Handlers + loop     - key presses, serial read, command console, redraw
#
# Live data flow (see ARCHITECTURE.md for full flowcharts):
#   serial/replay -> read_serial -> consume_serial_line -> ScannerState buffers
#   -> update() @50ms -> 3D surface / phase-space / crack plots
# ---------------------------------------------------------------------------

import os
import time

import numpy as np
import serial
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from scanner.config import *          # runtime constants + scale_line_width
from scanner.parsing import parse_serial_line, extract_reject_reason
from scanner.surface import has_usable_rp, build_surface_data
from scanner.widgets import ToggleAxisItem
from scanner.csv_logger import CsvLogger
from scanner.data_sources import SerialManager, CsvReplaySource
from scanner.state import ScannerState
from scanner.snapshot import make_snapshot

pg.setConfigOptions(antialias=True)


# ---------------------------------------------------------------------------
# 7. Runtime objects
# ---------------------------------------------------------------------------

# Serial port stays closed until the user connects via the GUI (no auto-open
# at import means the app launches cleanly even without a device attached).
serial_mgr = SerialManager(BAUDRATE)
csv_logger = CsvLogger(CSV_FILE)
state = ScannerState()

# The read loop drains whichever source ``source`` points at: the live serial
# port in "Live Serial" mode, or a CsvReplaySource while simulating from a file.
replay_source = None                # created when a CSV simulation is started
source = serial_mgr                 # active data source for the read loop




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

# Force readable light-on-dark text for the combo boxes / spin box.  Without
# this the popup list (a separate QAbstractItemView) inherits dark-grey text on
# a black background and is illegible until an item is hovered.
app.setStyleSheet(
    """
    QComboBox, QDoubleSpinBox {
        background-color: #1e1e1e;
        color: #e6e6e6;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 2px 6px;
    }
    QComboBox:disabled, QDoubleSpinBox:disabled {
        color: #888888;
        background-color: #161616;
    }
    QComboBox QAbstractItemView {
        background-color: #1e1e1e;
        color: #e6e6e6;
        border: 1px solid #555555;
        outline: none;
        selection-background-color: #3a6ea5;
        selection-color: #ffffff;
    }
    """
)

main_widget = QtWidgets.QWidget()
main_widget.setWindowTitle("Eddy Current Scanner V6")
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
    width=scale_line_width(2.0, SURFACE_TRACE_LINE_WIDTH_PERCENT),
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

# Pens are rebuilt-free: created once here and reused every frame (constructing
# pg.mkPen per frame — once for the main curve and up to RECENT_FADE_POINTS more
# for the fade tail — was a needless per-frame cost that grew the redraw time).
XY_MAIN_PEN = pg.mkPen('r', width=scale_line_width(1.0, RIGHT_PLOT_MAIN_LINE_WIDTH_PERCENT))
CRACK_PEN = pg.mkPen((255, 190, 140, 230), width=scale_line_width(1.0, CRACK_PLOT_LINE_WIDTH_PERCENT))
RECENT_TAIL_WIDTH = scale_line_width(3.0, RIGHT_PLOT_RECENT_LINE_WIDTH_PERCENT)

# Downsampling / clip-to-view thin the ~5000-point main curve to what's visible,
# a big redraw saving — but pyqtgraph's peak downsampling and clip both assume x
# increases monotonically.  That holds in Time mode (x = timestamp) but NOT in
# phase-space mode (x = R_p weaves back and forth), where it produces sawtooth
# artifacts.  So update() toggles these per-frame to match the current x-axis.
xy_curve = plot_xy.plot(pen=XY_MAIN_PEN)
_xy_fast_draw = None                    # last-applied state, to avoid redundant toggles


def set_xy_fast_draw(enabled):
    """Enable clip/downsample on the main curve only when x is monotonic time."""
    global _xy_fast_draw
    if enabled == _xy_fast_draw:
        return
    _xy_fast_draw = enabled
    xy_curve.setClipToView(enabled)
    xy_curve.setDownsampling(auto=enabled, method='peak')

# Cache the red->white gradient pens per tail length.  The fade shade for
# segment i depends only on how many segments are drawn, so once the tail is at
# full length (the steady state) this builds the pen list exactly once.
_recent_pen_cache = {}


def _recent_tail_pens(seg_count):
    """Return the cached list of gradient pens for a tail of ``seg_count`` segs."""
    pens = _recent_pen_cache.get(seg_count)
    if pens is None:
        shades = np.linspace(0, 255, seg_count).astype(int)
        pens = [
            pg.mkPen((255, int(s), int(s), 255), width=RECENT_TAIL_WIDTH)
            for s in shades
        ]
        _recent_pen_cache[seg_count] = pens
    return pens


recent_segment_curves = []
for _ in range(max(RECENT_FADE_POINTS - 1, 0)):
    recent_segment_curves.append(plot_xy.plot(pen=pg.mkPen((255, 0, 0), width=RECENT_TAIL_WIDTH)))

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
crack_curve = crack_plot.plot([], [], pen=CRACK_PEN, connect='pairs')

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

# Snapshot: export all four plots as journal-quality PDFs (sits to the right of
# the Connect / Start Replay button, which rescales to share the row).
snapshot_button = QtWidgets.QPushButton("Snapshot")
snapshot_button.setMinimumWidth(100)
snapshot_button.setToolTip("Save all four plots as journal-quality PDFs")

# Data-source mode: live serial vs. replaying a recorded CSV.
mode_combo = QtWidgets.QComboBox()
mode_combo.addItem("Live Serial", "serial")
mode_combo.addItem("Simulate CSV", "csv")
mode_combo.setToolTip("Stream live from a serial port, or simulate from a recorded CSV")

# CSV-simulation widgets (shown only in "Simulate CSV" mode).
csv_path_input = QtWidgets.QLineEdit()
csv_path_input.setPlaceholderText("CSV file to replay")
csv_path_input.setMinimumWidth(160)
csv_path_input.setToolTip("Recorded scanner CSV to stream from")

csv_browse_button = QtWidgets.QPushButton("Browse…")
csv_browse_button.setMinimumWidth(80)

speed_label = QtWidgets.QLabel("Speed")
speed_label.setStyleSheet("font-size: 10px; color: #bbbbbb;")
speed_spin = QtWidgets.QDoubleSpinBox()
speed_spin.setRange(0.1, 100.0)
speed_spin.setSingleStep(0.5)
speed_spin.setValue(REPLAY_DEFAULT_SPEED)
speed_spin.setSuffix("x")
speed_spin.setToolTip("Playback speed multiplier (1x = original real-time cadence)")

mode_row = QtWidgets.QHBoxLayout()
mode_row.setContentsMargins(0, 0, 0, 0)
mode_row.setSpacing(4)
mode_row.addWidget(mode_combo, 1)

connection_row1 = QtWidgets.QHBoxLayout()
connection_row1.setContentsMargins(0, 0, 0, 0)
connection_row1.setSpacing(4)
connection_row1.addWidget(port_combo, 1)
connection_row1.addWidget(refresh_button)

csv_row = QtWidgets.QHBoxLayout()
csv_row.setContentsMargins(0, 0, 0, 0)
csv_row.setSpacing(4)
csv_row.addWidget(csv_path_input, 1)
csv_row.addWidget(csv_browse_button)

connection_row2 = QtWidgets.QHBoxLayout()
connection_row2.setContentsMargins(0, 0, 0, 0)
connection_row2.setSpacing(4)
connection_row2.addWidget(baud_combo)
connection_row2.addWidget(speed_label)
connection_row2.addWidget(speed_spin)
connection_row2.addWidget(connect_button, 1)
connection_row2.addWidget(snapshot_button)

connection_status_label = QtWidgets.QLabel(serial_mgr.status_text)
connection_status_label.setStyleSheet("font-size: 10px; color: #bbbbbb;")

connection_layout.addLayout(mode_row)
connection_layout.addLayout(connection_row1)
connection_layout.addLayout(csv_row)
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


def current_mode():
    """Return the active data-source mode: ``"serial"`` or ``"csv"``."""
    return mode_combo.currentData() or "serial"


def replay_active():
    """True when a CSV replay is currently streaming."""
    return replay_source is not None and replay_source.is_connected


def browse_csv():
    """Pick a CSV file to replay via a file dialog."""
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_widget, "Select CSV to simulate", csv_path_input.text(),
        "CSV files (*.csv);;All files (*)"
    )
    if path:
        csv_path_input.setText(path)


def start_replay():
    """Load the selected CSV and begin streaming it as a simulated source."""
    global source, replay_source
    path = csv_path_input.text().strip()
    if not path:
        connection_status_label.setText("No CSV selected")
        return
    try:
        new_source = CsvReplaySource(path, speed=speed_spin.value())
    except (OSError, ValueError) as exc:
        connection_status_label.setText(f"Replay failed: {exc}")
        return
    replay_source = new_source
    replay_source.on_change = set_connection_ui_state
    source = replay_source
    state.reset()                       # start the plots from a clean slate
    replay_source.start()


def stop_replay():
    """Stop the active CSV replay (leaves it parked, finished)."""
    if replay_source is not None:
        replay_source.disconnect("stopped")


def toggle_connection():
    """Connect/disconnect the serial link, or start/stop the CSV replay."""
    if current_mode() == "csv":
        stop_replay() if replay_active() else start_replay()
        return
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


def on_mode_changed():
    """Switch data-source mode, tearing down whatever the other mode was using."""
    global source
    if current_mode() == "serial":
        if replay_active():
            replay_source.disconnect("mode switch")
        source = serial_mgr
    else:
        # Drop any live link so two sources never feed the buffers at once.
        if serial_mgr.is_connected:
            serial_mgr.disconnect("simulation mode")
        source = replay_source if replay_active() else serial_mgr
    set_connection_ui_state()


def set_connection_ui_state():
    """Sync the connection cluster widgets with the active source's state."""
    is_serial = current_mode() == "serial"

    # Show only the widgets relevant to the current mode.
    for widget in (port_combo, refresh_button, baud_combo):
        widget.setVisible(is_serial)
    for widget in (csv_path_input, csv_browse_button, speed_label, speed_spin):
        widget.setVisible(not is_serial)

    if is_serial:
        connected = serial_mgr.is_connected
        connect_button.setText("Disconnect" if connected else "Connect")
        connection_status_label.setText(serial_mgr.status_text)
        # Lock port/baud selection while connected — disconnect first to change.
        port_combo.setEnabled(not connected)
        baud_combo.setEnabled(not connected)
        refresh_button.setEnabled(not connected)
    else:
        running = replay_active()
        connect_button.setText("Stop Replay" if running else "Start Replay")
        if replay_source is not None:
            connection_status_label.setText(replay_source.status_text)
        else:
            connection_status_label.setText("CSV simulation: idle")
        # Lock file/speed selection while a replay is running.
        csv_path_input.setEnabled(not running)
        csv_browse_button.setEnabled(not running)
        speed_spin.setEnabled(not running)


refresh_button.clicked.connect(refresh_ports)
connect_button.clicked.connect(toggle_connection)
csv_browse_button.clicked.connect(browse_csv)
mode_combo.currentIndexChanged.connect(on_mode_changed)
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

def scrub_replay(direction, accelerating):
    """Step the CSV replay back/forward in time by ``direction`` (-1/+1).

    A single tap moves one sample; holding the key auto-repeats, and each repeat
    grows the step geometrically so the scrub speeds up the longer it is held.
    Scrubbing freezes auto-playback (press P to resume from the new position).
    """
    if not isinstance(source, CsvReplaySource):
        return
    if accelerating:
        state.scrub_velocity = min(REPLAY_SCRUB_MAX_STEP,
                                   state.scrub_velocity * REPLAY_SCRUB_ACCEL)
    else:
        state.scrub_velocity = 1.0
    state.paused = True                 # manual control owns the position now
    step = max(1, int(state.scrub_velocity))
    before = source.position
    if source.set_position(before + direction * step) == before:
        return                          # already at an end — nothing to rebuild
    rebuild_from_replay()


def keyPressEvent(event):
    """Keyboard shortcuts: Space=reset, P=pause, F=toggle CSV, 1/2/3=3D views,
    Left/Right=scrub a CSV replay back/forward (tap=1 step, hold accelerates)."""
    if event.key() == QtCore.Qt.Key_Left:
        scrub_replay(-1, event.isAutoRepeat())
        return
    elif event.key() == QtCore.Qt.Key_Right:
        scrub_replay(+1, event.isAutoRepeat())
        return
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
            source.reset_input_buffer()
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


def keyReleaseEvent(event):
    """Reset the scrub acceleration when the arrow key is genuinely released."""
    if event.isAutoRepeat():
        return
    if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
        state.scrub_velocity = 1.0


main_widget.keyPressEvent = keyPressEvent
win.keyPressEvent = keyPressEvent
surface_view.keyPressEvent = keyPressEvent
main_widget.keyReleaseEvent = keyReleaseEvent
win.keyReleaseEvent = keyReleaseEvent
surface_view.keyReleaseEvent = keyReleaseEvent


def consume_serial_line(line, responses=None, log=True, show_incoming=True):
    """Process one received serial line.

    Updates the incoming-line view, records any reject reason and crack event,
    then parses and ingests a sample.  Lines that are not parseable samples are
    appended to ``responses`` when a list is provided (used by the command
    console); empty lines are ignored.  ``log=False`` suppresses CSV writing and
    ``show_incoming=False`` skips the per-line readout-box repaint — both used
    when re-ingesting many rows to rebuild the view during a replay time-scrub.
    """
    if not line:
        return

    if show_incoming:
        append_incoming_line(line)
    else:
        state.incoming_history.append(line)     # keep history; defer the repaint

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

    state.ingest_sample(t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val,
                        csv_logger, log=log)


def rebuild_from_replay():
    """Rebuild the live buffers from the replay's current scrub position.

    Clears the buffers and re-ingests the window of rows ending at the source's
    position (without logging), so the plots reflect exactly the recorded state
    up to that point.  The redraw timer repaints everything on its next tick.
    """
    if not isinstance(source, CsvReplaySource):
        return
    state.reset()
    for raw in source.window_lines():
        consume_serial_line(raw.decode(errors="ignore").strip(), log=False, show_incoming=False)
    # Repaint the rolling readout box once from the rebuilt history.
    incoming_line_box.setPlainText("\n".join(state.incoming_history))


def read_serial():
    """Drain the active source into the live buffers (skipped while paused).

    Works identically for a live serial port and a CSV replay — both expose the
    same ``in_waiting`` / ``readline`` interface via ``source``.
    """
    if state.paused:
        source.reset_input_buffer()
        return
    while source.in_waiting:
        line = source.readline().decode(errors='ignore').strip()
        consume_serial_line(line)


def send_serial_command():
    """Send the typed command, then collect non-sensor replies for ~400 ms."""
    command = serial_command_input.text().strip()
    if not command:
        serial_response_box.setPlainText("Response: command is empty")
        return

    if not getattr(source, "supports_commands", True):
        serial_response_box.setPlainText("Response: command sending is disabled during CSV simulation")
        return

    if not source.is_connected:
        serial_response_box.setPlainText("Response: not connected")
        return

    try:
        source.write((command + "\n").encode("utf-8"))
    except serial.SerialException as exc:
        source.disconnect("device lost")
        serial_response_box.setPlainText(f"Response error: {exc}")
        return

    responses = []
    deadline = time.monotonic() + 0.4

    while time.monotonic() < deadline:
        if source.in_waiting <= 0:
            time.sleep(0.01)
            continue

        line = source.readline().decode(errors='ignore').strip()
        consume_serial_line(line, responses)

    if responses:
        response_text = "Response:\n" + "\n".join(responses[-SERIAL_RESPONSE_MAX_LINES:])
    else:
        response_text = "Response: no non-sensor reply in 400 ms"

    first_response = responses[0] if responses else ""
    state.stage_command_exchange(command, first_response)

    serial_response_box.setPlainText(response_text)


serial_send_button.clicked.connect(send_serial_command)
serial_command_input.returnPressed.connect(send_serial_command)


# Timestamp of the last 3D mesh rebuild, used to throttle it inside update().
_last_surface_update = 0.0


def update():
    """Timer callback: read serial, recompute readouts, and redraw all plots."""
    global _last_surface_update
    read_serial()
    readout_label.setText(state.readout_text)

    # Pens are built once at construction (XY_MAIN_PEN / CRACK_PEN) and never
    # change at runtime, so there's no need to rebuild them here every frame.

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
        state.snap_crack_times = crack_times_arr
        state.snap_crack_mags = crack_mags_arr
        state.snap_crack_sizes = crack_sizes_arr
    else:
        crack_curve.setData([], [])
        crack_plot.setYRange(0.0, 1.0, padding=0.0)
        state.snap_crack_times = np.array([], dtype=float)
        state.snap_crack_mags = np.array([], dtype=float)
        state.snap_crack_sizes = np.array([], dtype=float)

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

    # Cache the live plot window for the Snapshot export (time, R_p, L).
    state.snap_time = x_plot
    state.snap_rp = y1_plot
    state.snap_l = y2_plot

    if len(x_plot) == 0:
        return

    # Rebuilding + re-uploading the ribbon mesh is the heaviest per-frame cost,
    # so throttle it to ~10 Hz.  The fast 2D plots below still refresh every read
    # tick, so the live trace stays responsive while the 3D view updates slightly
    # less often (imperceptible for a rotating surface).
    if now - _last_surface_update >= SURFACE_UPDATE_INTERVAL_SEC:
        surface_data = build_surface_data(x_plot, y1_plot, y2_plot)
        if surface_data is not None:
            vertices, faces, face_colors, line_pos = surface_data
            meshdata = gl.MeshData(vertexes=vertices, faces=faces)
            meshdata.setFaceColors(face_colors)
            surface_item.setMeshData(meshdata=meshdata)
            surface_trace.setData(
                pos=line_pos,
                width=scale_line_width(2.0, SURFACE_TRACE_LINE_WIDTH_PERCENT),
            )
            surface_head.setData(pos=line_pos[-1:].copy())
            surface_view.opts['center'] = QtGui.QVector3D(0.0, 0.0, 0.0)
        _last_surface_update = now

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

    # Keep crack-plot x-axis synced to the upper plot whenever the upper plot
    # is using timestamp on x (explicit TIME mode or auto time fallback).
    right_uses_time_x = (state.right_x_mode == "TIME") or state.right_plot_auto_time_fallback
    if right_uses_time_x:
        crack_plot.setXLink(plot_xy)
    else:
        crack_plot.setXLink(None)
        crack_plot.setXRange(t_min, max(t_now, t_min + 1e-6), padding=0.0)

    # Clip/downsample are only valid when x is monotonic (time); disable them in
    # phase-space mode, where non-monotonic R_p on x would otherwise alias.
    set_xy_fast_draw(right_uses_time_x)

    xy_curve.setData(x_right, y_right)

    # Highlight most recent trajectory with red -> white segment gradient.
    tail_count = min(RECENT_FADE_POINTS, len(x_right))
    if tail_count > 1:
        tail_x = x_right[-tail_count:]
        tail_y = y_right[-tail_count:]
        seg_count = tail_count - 1
        tail_pens = _recent_tail_pens(seg_count)
        for i in range(seg_count):
            seg_curve = recent_segment_curves[i]
            seg_curve.setPen(tail_pens[i])
            seg_curve.setData([tail_x[i], tail_x[i + 1]], [tail_y[i], tail_y[i + 1]])
        for i in range(seg_count, len(recent_segment_curves)):
            recent_segment_curves[i].setData([], [])
    else:
        for seg_curve in recent_segment_curves:
            seg_curve.setData([], [])


# --- Snapshot export -------------------------------------------------------
# The rendering lives in scanner/snapshot.py; here we just supply the output
# root (next to this file) and a status callback, then wire the button.
try:
    _SNAPSHOT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SNAPSHOT_SCRIPT_DIR = os.getcwd()
SNAPSHOT_ROOT = os.path.join(_SNAPSHOT_SCRIPT_DIR, "Snapshots")


def _snapshot_status(msg):
    """Surface snapshot progress/errors on the connection status label + stdout."""
    connection_status_label.setText(msg)
    print(msg)


def take_snapshot():
    """Export the current four plots as journal-quality PDFs under Snapshots/."""
    make_snapshot(state, SNAPSHOT_ROOT, on_status=_snapshot_status)


snapshot_button.clicked.connect(take_snapshot)



timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


def close_resources():
    csv_logger.close()
    serial_mgr.close()
    if replay_source is not None:
        replay_source.close()


app.aboutToQuit.connect(close_resources)

app.exec()
