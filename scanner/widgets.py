# ---------------------------------------------------------------------------
# scanner/widgets.py — Custom pyqtgraph widgets
#
# ``ToggleAxisItem`` is the only piece of the extracted backend that subclasses
# pyqtgraph, so it lives here rather than in the Qt-free modules.
# ---------------------------------------------------------------------------

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class ToggleAxisItem(pg.AxisItem):
    """Bottom axis that toggles x-mode when clicked."""
    toggled = QtCore.Signal()

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.toggled.emit()
            event.accept()
            return
        super().mouseClickEvent(event)
