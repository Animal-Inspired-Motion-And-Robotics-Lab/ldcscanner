"""Backend package for the live LDC scanner GUI (``LDC SCANNER V6.py``).

The Qt-free logic behind the real-time scanner readout, split out of the entry
file so colleagues can read and modify one concern at a time:

    config        runtime constants (serial / CSV / replay / plot tuning)
    parsing       telemetry-line parsing (pure functions)
    surface       3D ribbon-mesh geometry (pure functions)
    widgets       ToggleAxisItem (the one pyqtgraph subclass)
    csv_logger    CsvLogger — owns the output CSV file
    data_sources  SerialManager (live) + CsvReplaySource (simulated)
    state         ScannerState — live buffers + sample ingestion
    snapshot      make_snapshot — journal-quality PDF export

The GUI construction, the 50 ms redraw loop, and all Qt event handlers remain in
the ``LDC SCANNER V6.py`` entry file.  See ``ARCHITECTURE.md`` for flowcharts.
"""
