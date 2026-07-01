# ---------------------------------------------------------------------------
# scanner/csv_logger.py — Owns the output CSV file
#
# Encapsulates all file-handle handling so the rest of the program never touches
# it directly.  No Qt, no numpy.
# ---------------------------------------------------------------------------

import csv
import os
import time


class CsvLogger:
    """Append-only CSV writer for sensor samples.

    Encapsulates filename normalization, (re)opening the output file, header
    writing, and per-sample rows so the rest of the program never touches the
    file handle directly.
    """

    HEADER = ["timestamp_computer", "timestamp", "sensor1", "sensor2",
              "mag", "width", "crack_x", "crack_size", "serial out", "response"]

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

    def write_sample(self, t, s1, s2, mag_val, width_val, crack_x_val, crack_size_val,
                     serial_out="", response=""):
        """Append one sample row, stamping the host clock, and flush."""
        timestamp_computer = f"{time.time():.3f}"
        self._writer.writerow([timestamp_computer, t, s1, s2,
                               mag_val if mag_val is not None else "",
                               width_val if width_val is not None else "",
                               crack_x_val if crack_x_val is not None else "",
                               crack_size_val if crack_size_val is not None else "",
                               serial_out,
                               response])
        self._file.flush()

    @property
    def basename(self):
        return os.path.basename(self.path)

    def close(self):
        if self._file is not None and not self._file.closed:
            self._file.close()
