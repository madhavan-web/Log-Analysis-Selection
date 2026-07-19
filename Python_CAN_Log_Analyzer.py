"""
========================================================
Author: Marasu Madhavan
========================================================
MM_Pro - ANAlyzer  ·  CANalyzer-style CAN signal viewer
========================================================
A complete, offline Python application for visualizing CAN bus logs.
Supports loading .asc/.blf files, .dbc parsing, and time-synchronized 
stacked signal plotting with a measurement cursor.
"""

import sys
import os
import math
import can
import cantools
import numpy as np

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore    import Qt, QTimer, QRectF, QPointF, QSize

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

import pyqtgraph as pg
from pyqtgraph  import GraphicsObject

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QLabel, QMessageBox, QComboBox, QTableWidget, QTableWidgetItem,
    QColorDialog, QHeaderView, QSplitter, QLineEdit, QAbstractItemView
)
from PyQt5.QtGui  import QColor, QFont, QPainter, QPen

# ─────────────────────────────────────────────────────────────────────────────
MAX_DBC_FILES = 5
MAX_SIGNALS   = 10

# ─────────────────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow  { background:#f0f0f0; }
QWidget      { color:#000; font-family:'Segoe UI',Arial,sans-serif; font-size:9pt; }
QPushButton  { background:#e5e5e5; color:#000; border:1px solid #b0b0b0;
               padding:5px; font-weight:bold; border-radius:2px; }
QPushButton:hover             { background:#d5d5d5; border:1px solid #cc0000; }
QPushButton.PrimaryBtn        { background:#cc0000; color:#fff; font-size:10pt;
                                padding:8px; border:none; }
QPushButton.PrimaryBtn:hover  { background:#ee0000; }
QPushButton.ToolBtn           { background:#fff; font-size:14pt; padding:4px 12px;
                                border:1px solid #b0b0b0; }
QPushButton.ToolBtn:hover     { background:#f5f5f5; border:1px solid #cc0000; }
QPushButton.ToolBtn:checked   { background:#cc0000; color:#fff; }
QPushButton.CollapseBtn       { background:transparent; border:none; font-size:10pt;
                                color:#cc0000; padding:0 5px; }
QPushButton.CrossBtn          { background:transparent; color:#666; border:none; 
                                font-size:12pt; font-weight:bold; padding:0px; }
QPushButton.CrossBtn:hover    { color:#cc0000; }
QListWidget, QTableWidget, QComboBox, QLineEdit {
    background:#fff; border:1px solid #b0b0b0; color:#000;
    outline:none; padding:2px; }
QTableWidget::item:selected   { background:#cc0000; color:#fff; }
QHeaderView::section          { background:#e5e5e5; color:#000;
                                border:1px solid #b0b0b0; padding:4px;
                                font-weight:bold; }
QSplitter::handle             { background:#b0b0b0; width:3px; }
QSplitter::handle:hover       { background:#cc0000; }
QLabel                        { font-weight:bold; color:#cc0000; padding-bottom:2px; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Tick step table (seconds, finest → coarsest)
# ─────────────────────────────────────────────────────────────────────────────
_TICK_STEPS = [
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    0.01, 0.02, 0.05,
    0.1,  0.2,  0.5,
    1,    2,    5,
    10,   20,   30,
    60,   120,  300,
    600,  1800, 3600,
]
_MIN_PX_GAP = 80   # minimum pixels between adjacent major X tick labels


# ─────────────────────────────────────────────────────────────────────────────
# Reliable canvas size measurement (multi-path, DPI-aware)
# ─────────────────────────────────────────────────────────────────────────────
def _vb_pixel_size(view_box):
    try:
        geom = view_box.screenGeometry()
        if geom is not None and geom.width() > 10:
            return geom.width(), geom.height()
    except Exception:
        pass

    try:
        scene = view_box.scene()
        if scene is not None:
            views = scene.views()
            if views:
                qview = views[0]
                sr    = view_box.sceneBoundingRect()
                tl    = qview.mapFromScene(sr.topLeft())
                br    = qview.mapFromScene(sr.bottomRight())
                w     = abs(br.x() - tl.x())
                h     = abs(br.y() - tl.y())
                if w > 10:
                    return int(w), max(int(h), 10)
    except Exception:
        pass

    try:
        pi  = view_box.parentItem()
        br  = pi.boundingRect()
        w   = br.width()
        h   = br.height()
        if w > 10:
            return int(w), max(int(h), 10)
    except Exception:
        pass

    return 900, 150


def _canvas_px(view_box) -> int:
    return _vb_pixel_size(view_box)[0]


def _canvas_height_px(view_box) -> int:
    return _vb_pixel_size(view_box)[1]


# ─────────────────────────────────────────────────────────────────────────────
# Tick helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pick_tick(x_span_s: float, canvas_px: int):
    if canvas_px <= 0 or x_span_s <= 0:
        return 1.0, 0.2
    px_per_s = canvas_px / x_span_s
    for step in _TICK_STEPS:
        if step * px_per_s >= _MIN_PX_GAP:
            return step, step / 5.0
    return _TICK_STEPS[-1], _TICK_STEPS[-1] / 5.0


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    mag  = math.floor(math.log10(raw_step))
    base = 10 ** mag
    for mult in (1, 2, 5, 10):
        s = mult * base
        if s >= raw_step - base * 1e-9:
            return s
    return base * 10


def _build_tick_lists(step: float, lo: float, hi: float, max_count: int = 1000) -> list:
    if step <= 0 or hi <= lo:
        return []
    start = math.floor(lo / step) * step
    out, t = [], start
    while t <= hi + step * 1e-9:
        if t >= lo - step * 1e-9:
            out.append(t)
        t += step
        if len(out) >= max_count:
            break
    return out


def _tick_format(step: float) -> str:
    if step >= 1:     return "{:.0f}"
    if step >= 0.1:   return "{:.1f}"
    if step >= 0.01:  return "{:.2f}"
    if step >= 0.001: return "{:.3f}"
    return "{:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Stable grid item
# ─────────────────────────────────────────────────────────────────────────────
class _StableGridItem(GraphicsObject):
    _BIG = 1e9

    def __init__(self, plot_item: pg.PlotItem):
        super().__init__()
        self._pi       = plot_item
        self._vb       = plot_item.getViewBox()
        self._x_ticks  : list = []
        self._pen      = pg.mkPen(color=(230, 230, 230), width=1, style=Qt.SolidLine)
        self.setZValue(-100)

        self._vb.sigRangeChanged.connect(self._on_change)
        try:
            self._vb.sigResized.connect(self._on_change)
        except AttributeError:
            pass
        plot_item.addItem(self, ignoreBounds=True)

    def set_x_ticks(self, positions: list):
        self._x_ticks = list(positions)
        self.update()

    def detach(self):
        try:
            self._vb.sigRangeChanged.disconnect(self._on_change)
        except Exception:
            pass
        try:
            self._vb.sigResized.disconnect(self._on_change)
        except Exception:
            pass

    def _on_change(self, *_):
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(-self._BIG, -self._BIG, 2 * self._BIG, 2 * self._BIG)

    def paint(self, painter: QPainter, *_):
        try:
            xr, yr = self._vb.viewRange()
            x_lo, x_hi = xr[0], xr[1]
            y_lo, y_hi = yr[0], yr[1]
            if x_hi <= x_lo or y_hi <= y_lo:
                return
        except Exception:
            return

        painter.setPen(self._pen)
        for x in self._x_ticks:
            if x_lo <= x <= x_hi:
                painter.drawLine(QPointF(x, y_lo), QPointF(x, y_hi))

        y_span = y_hi - y_lo
        if y_span <= 0:
            return
        step = _nice_step(y_span / 5.0)
        if step <= 0:
            return
        y_start = math.floor(y_lo / step) * step
        y       = y_start
        count   = 0
        while y <= y_hi + step * 1e-9 and count < 200:
            if y_lo - step * 1e-9 <= y <= y_hi + step * 1e-9:
                painter.drawLine(QPointF(x_lo, y), QPointF(x_hi, y))
            y    += step
            count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Drag-and-drop legend table
# ─────────────────────────────────────────────────────────────────────────────
class LegendTableWidget(QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_delete_callback = None
        self.on_row_moved       = None

        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self.on_delete_callback:
            self.on_delete_callback()
        super().keyPressEvent(event)

    def dropEvent(self, event):
        # Prevent default messy drag-and-drop cell replacement 
        if event.source() == self:
            src_row = self.currentRow()
            
            # Find exact drop row index
            dst_row = self.indexAt(event.pos()).row()
            if dst_row == -1:
                # Dropped at the very bottom
                dst_row = self.rowCount() - 1
                
            # Discard the native drag drop execution
            event.setDropAction(Qt.IgnoreAction)
            event.accept()
            
            if src_row >= 0 and dst_row >= 0 and src_row != dst_row:
                if self.on_row_moved:
                    self.on_row_moved(src_row, dst_row)
        else:
            super().dropEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────────────────────────
class CANDataAnalyzer(QMainWindow):

    _COL_COLOR = 0
    _COL_SIG   = 1
    _COL_VAL   = 2
    _COL_MIN   = 3
    _COL_MAX   = 4

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MM_Pro - ANAlyzer")
        self.setStyleSheet(STYLE)

        self.db_dict            : dict  = {}
        self.log_messages       : list  = []
        self.selected_signals   : dict  = {}
        self.current_plot_data  : dict  = {}
        self.plots              : list  = []
        self._grids             : list  = []
        self.v_lines            : list  = []
        self._valid_keys        : list  = []
        self.max_x              : float = 0.0
        self._simulation_started        = False
        self._show_minmax               = True

        self._dpi_scale = QApplication.primaryScreen().devicePixelRatio()

        self._tick_timer = QTimer(self)
        self._tick_timer.setSingleShot(True)
        self._tick_timer.setInterval(60)
        self._tick_timer.timeout.connect(self._apply_ticks)

        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(120)
        self._layout_timer.timeout.connect(self._after_layout)

        self._build_ui()
        self._center_window()

    def _center_window(self):
        sg = QApplication.primaryScreen().availableGeometry()
        w  = int(sg.width()  * 0.85)
        h  = int(sg.height() * 0.85)
        self.setGeometry((sg.width() - w) // 2,
                         (sg.height() - h) // 2, w, h)

    @staticmethod
    def _collapsible(title: str, inner_layout) -> QWidget:
        outer = QWidget()
        vbox  = QVBoxLayout(outer)
        vbox.setContentsMargins(0, 0, 0, 0)
        hdr   = QWidget()
        hrow  = QHBoxLayout(hdr)
        hrow.setContentsMargins(0, 0, 0, 0)
        btn   = QPushButton("▼")
        btn.setProperty('class', 'CollapseBtn')
        lbl   = QLabel(title)
        hrow.addWidget(btn)
        hrow.addWidget(lbl)
        hrow.addStretch()
        body  = QWidget()
        body.setLayout(inner_layout)
        vbox.addWidget(hdr)
        vbox.addWidget(body)

        def _toggle():
            if body.isVisible():
                body.hide(); btn.setText("▶")
            else:
                body.show(); btn.setText("▼")
        btn.clicked.connect(_toggle)
        return outer

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QHBoxLayout(cw)
        ml.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        ml.addWidget(self.main_splitter)

        lw = QWidget()
        rw = QWidget()
        rw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        ll = QVBoxLayout(lw)
        ll.setContentsMargins(8, 8, 4, 8)
        self.left_splitter = QSplitter(Qt.Vertical)

        # Data sources
        fl = QVBoxLayout(); fl.setContentsMargins(0, 0, 0, 0)
        bl = QHBoxLayout()
        self.btn_load_log = QPushButton("Load Log (.asc/.blf)")
        self.btn_load_log.clicked.connect(self.load_log)
        
        self.btn_load_dbc = QPushButton("Load DBC")
        self.btn_load_dbc.clicked.connect(self.load_dbc)
        
        bl.addWidget(self.btn_load_log)
        bl.addWidget(self.btn_load_dbc)
        
        self.file_status = QLabel("Log: None  |  DBCs: 0")
        self.file_status.setStyleSheet("color:#666;font-weight:normal;font-size:8pt;")
        fl.addLayout(bl); fl.addWidget(self.file_status)

        # Signal selector
        sl = QVBoxLayout(); sl.setContentsMargins(0, 0, 0, 0)
        
        dbc_row = QHBoxLayout()
        dbc_row.setContentsMargins(0, 0, 0, 0)
        
        self.dbc_dropdown = QComboBox()
        self.dbc_dropdown.currentTextChanged.connect(self._populate_signals)
        dbc_row.addWidget(self.dbc_dropdown, 1)  
        
        self.btn_remove_dbc = QPushButton("✕")
        self.btn_remove_dbc.setProperty('class', 'CrossBtn')
        self.btn_remove_dbc.setToolTip("Remove selected DBC")
        self.btn_remove_dbc.setFixedSize(22, 22)
        self.btn_remove_dbc.clicked.connect(self.remove_dbc)
        dbc_row.addWidget(self.btn_remove_dbc, 0)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search signals…")
        self.search_bar.textChanged.connect(self._filter_signals)
        self.signal_list = QListWidget()
        self.signal_list.itemChanged.connect(self._on_item_checked)
        
        sl.addLayout(dbc_row)
        sl.addWidget(self.search_bar)
        sl.addWidget(self.signal_list)

        # Graphics / Legend table
        gl = QVBoxLayout(); gl.setContentsMargins(0, 0, 0, 0)
        minmax_bar = QHBoxLayout()
        self.btn_minmax = QPushButton("Min/Max ⇅")
        self.btn_minmax.setToolTip("Show / Hide Min and Max columns")
        self.btn_minmax.setCheckable(True)
        self.btn_minmax.setChecked(True)
        self.btn_minmax.clicked.connect(self._toggle_minmax_cols)
        minmax_bar.addWidget(self.btn_minmax)
        minmax_bar.addStretch()
        gl.addLayout(minmax_bar)

        self.legend_table = LegendTableWidget(0, 5)
        self.legend_table.on_delete_callback = self._delete_selected_signals
        self.legend_table.on_row_moved       = self._on_legend_row_moved
        self.legend_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.legend_table.setHorizontalHeaderLabels(["Color", "Signal", "Value", "Min", "Max"])
        self.legend_table.verticalHeader().setVisible(False)
        self.legend_table.setAlternatingRowColors(True)
        self.legend_table.setStyleSheet("alternate-background-color:#f9f9f9;")
        
        h = self.legend_table.horizontalHeader()
        h.setSectionResizeMode(self._COL_COLOR, QHeaderView.Interactive)
        h.setSectionResizeMode(self._COL_SIG,   QHeaderView.Interactive) 
        h.setSectionResizeMode(self._COL_VAL,   QHeaderView.Interactive)
        h.setSectionResizeMode(self._COL_MIN,   QHeaderView.Interactive)
        h.setSectionResizeMode(self._COL_MAX,   QHeaderView.Interactive)
        
        self.legend_table.setColumnWidth(self._COL_COLOR, 45)
        self.legend_table.setColumnWidth(self._COL_SIG,   140)
        self.legend_table.setColumnWidth(self._COL_VAL,   70)
        self.legend_table.setColumnWidth(self._COL_MIN,   65)
        self.legend_table.setColumnWidth(self._COL_MAX,   65)
        h.setStretchLastSection(True)
        
        gl.addWidget(self.legend_table)

        self.left_splitter.addWidget(self._collapsible("Data Sources", fl))
        self.left_splitter.addWidget(self._collapsible("Signals",      sl))
        self.left_splitter.addWidget(self._collapsible("Graphics",     gl))
        ll.addWidget(self.left_splitter)

        # Info strip
        iw = QWidget()
        iw.setStyleSheet("background:#e5e5e5;border:1px solid #b0b0b0;border-radius:3px;")
        il = QVBoxLayout(iw); il.setContentsMargins(5, 5, 5, 5)
        self.lbl_start  = QLabel("Log Start: --")
        self.lbl_end    = QLabel("Log End: --")
        self.lbl_cursor = QLabel("Cursor Time: --")
        _info_ss = "font-weight:bold;color:#333;font-size:8pt;"
        for lb in (self.lbl_start, self.lbl_end, self.lbl_cursor):
            lb.setStyleSheet(_info_ss); il.addWidget(lb)
        ll.addWidget(iw)

        self.btn_plot = QPushButton("Start Simulation")
        self.btn_plot.setProperty('class', 'PrimaryBtn')
        self.btn_plot.clicked.connect(self.plot)
        ll.addWidget(self.btn_plot)

        # Right Panel
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(4, 8, 8, 8)
        rl.setSpacing(0)

        tbw = QWidget()
        tbw.setStyleSheet("background:#e5e5e5;border:1px solid #b0b0b0;border-bottom:none;")
        tbl = QHBoxLayout(tbw)
        tbl.setContentsMargins(4, 4, 4, 4)
        tbl.setAlignment(Qt.AlignLeft)

        self.btn_cursor_toggle = QPushButton("⌖")
        self.btn_cursor_toggle.setToolTip("Toggle Measurement Cursor")
        self.btn_cursor_toggle.setProperty('class', 'ToolBtn')
        self.btn_cursor_toggle.setCheckable(True)
        self.btn_cursor_toggle.clicked.connect(self._toggle_cursor)

        self.btn_reset = QPushButton("⟲")
        self.btn_reset.setToolTip("Fit All Data (Reset Zoom)")
        self.btn_reset.setProperty('class', 'ToolBtn')
        self.btn_reset.clicked.connect(self._reset_zoom)

        self.btn_box_zoom = QPushButton("⛶")
        self.btn_box_zoom.setToolTip("Box-zoom region")
        self.btn_box_zoom.setProperty('class', 'ToolBtn')
        self.btn_box_zoom.setCheckable(True)
        self.btn_box_zoom.clicked.connect(self._toggle_zoom_mode)

        tbl.addWidget(self.btn_cursor_toggle)
        tbl.addWidget(self.btn_reset)
        tbl.addWidget(self.btn_box_zoom)

        pg.setConfigOptions(antialias=True, background='#ffffff', foreground='#000000')
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.graph_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_layout.setMinimumSize(QSize(100, 100))
        self.graph_layout.setStyleSheet("border:1px solid #b0b0b0;")
        self.graph_layout.ci.layout.setSpacing(2)
        self.graph_layout.ci.layout.setContentsMargins(4, 4, 4, 4)

        rl.addWidget(tbw, 0)
        rl.addWidget(self.graph_layout, 1)

        self.main_splitter.addWidget(lw)
        self.main_splitter.addWidget(rw)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([320, 1600])

        self._apply_minmax_visibility()

    def _toggle_minmax_cols(self):
        self._show_minmax = self.btn_minmax.isChecked()
        self._apply_minmax_visibility()

    def _apply_minmax_visibility(self):
        hide = not self._show_minmax
        self.legend_table.setColumnHidden(self._COL_MIN, hide)
        self.legend_table.setColumnHidden(self._COL_MAX, hide)

    def load_log(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select Log File", "", "CAN Logs (*.asc *.blf);;All Files (*)")
        if not fn:
            return
        try:
            self.log_messages = list(can.LogReader(fn))
            self._simulation_started = False
            self._refresh_status()
            if self.log_messages:
                dur = (self.log_messages[-1].timestamp - self.log_messages[0].timestamp)
                self.lbl_start.setText("Log Start: 0.000 s")
                self.lbl_end.setText(f"Log End:   {dur:.3f} s")
                self.lbl_cursor.setText("Cursor Time: --")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load log:\n{e}")

    def load_dbc(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select DBC Files", "", "DBC Files (*.dbc);;All Files (*)")
        for f in files:
            name = os.path.basename(f)
            if name not in self.db_dict:
                if len(self.db_dict) >= MAX_DBC_FILES:
                    QMessageBox.warning(self, "Limit Reached",
                        f"Maximum of {MAX_DBC_FILES} DBC files allowed.")
                    break
                try:
                    self.db_dict[name] = cantools.database.load_file(f)
                    self.dbc_dropdown.addItem(name)
                    self._refresh_status()
                except Exception:
                    pass

    def remove_dbc(self):
        dbc_name = self.dbc_dropdown.currentText()
        if not dbc_name:
            return

        self.db_dict.pop(dbc_name, None)
        unames_to_remove = [k for k in self.selected_signals.keys() if k.startswith(f"{dbc_name}_")]
        for uname in unames_to_remove:
            self.selected_signals.pop(uname, None)

        idx = self.dbc_dropdown.findText(dbc_name)
        if idx >= 0:
            self.dbc_dropdown.removeItem(idx)

        self._refresh_status()
        self._rebuild_legend()
        
        if self._simulation_started:
            self.plot()

    def _refresh_status(self):
        self.file_status.setText(
            f"Log: {'Loaded' if self.log_messages else 'None'}"
            f"  |  DBCs: {len(self.db_dict)}")

    def _populate_signals(self, dbc_name: str):
        self.signal_list.blockSignals(True)
        self.signal_list.clear()
        self.search_bar.clear()
        if dbc_name not in self.db_dict:
            self.signal_list.blockSignals(False)
            return
        rows = sorted(
            [(s.name, m.frame_id)
             for m in self.db_dict[dbc_name].messages
             for s in m.signals],
            key=lambda x: x[0])
        for sig_name, msg_id in rows:
            item = QListWidgetItem(sig_name)
            item.setData(Qt.UserRole, msg_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            uname = f"{dbc_name}_{sig_name}"
            item.setCheckState(
                Qt.Checked if uname in self.selected_signals else Qt.Unchecked)
            self.signal_list.addItem(item)
        self.signal_list.blockSignals(False)

    def _filter_signals(self, text: str):
        t = text.lower()
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            item.setHidden(t not in item.text().lower())

    def _on_item_checked(self, item: QListWidgetItem):
        dbc_name = self.dbc_dropdown.currentText()
        sig_name = item.text()
        msg_id   = item.data(Qt.UserRole)
        uname    = f"{dbc_name}_{sig_name}"
        if item.checkState() == Qt.Checked:
            if uname not in self.selected_signals:
                if len(self.selected_signals) >= MAX_SIGNALS:
                    QMessageBox.warning(self, "Limit Reached",
                        f"Maximum of {MAX_SIGNALS} signals can be monitored.")
                    self.signal_list.blockSignals(True)
                    item.setCheckState(Qt.Unchecked)
                    self.signal_list.blockSignals(False)
                    return
                
                try:
                    sig_obj = (self.db_dict[dbc_name]
                               .get_message_by_frame_id(msg_id)
                               .get_signal_by_name(sig_name))
                    unit = sig_obj.unit or ""
                    choices = sig_obj.choices 
                except Exception:
                    unit = ""
                    choices = None

                rng = np.random.default_rng(abs(hash(uname)) % (2**32))
                rgb = rng.integers(0, 150, 3) 
                
                self.selected_signals[uname] = {
                    'dbc'     : self.db_dict[dbc_name],
                    'msg_id'  : msg_id,
                    'sig_name': sig_name,
                    'unit'    : unit,
                    'choices' : choices,
                    'color'   : QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                }
        else:
            self.selected_signals.pop(uname, None)
        self._rebuild_legend()

    def _delete_selected_signals(self):
        selected_items = self.legend_table.selectedItems()
        if not selected_items:
            return
        rows = set(item.row() for item in selected_items)
        unames_to_remove = []
        for row in rows:
            ni = self.legend_table.item(row, self._COL_SIG)
            if ni is not None:
                unames_to_remove.append(ni.data(Qt.UserRole))
        if not unames_to_remove:
            return
        dbc_name = self.dbc_dropdown.currentText()
        for uname in unames_to_remove:
            self.selected_signals.pop(uname, None)
            for i in range(self.signal_list.count()):
                it = self.signal_list.item(i)
                if f"{dbc_name}_{it.text()}" == uname:
                    self.signal_list.blockSignals(True)
                    it.setCheckState(Qt.Unchecked)
                    self.signal_list.blockSignals(False)
                    break
        self._rebuild_legend()
        if self._simulation_started:
            self.plot()

    def _on_legend_row_moved(self, src_row: int, dst_row: int):
        keys = list(self.selected_signals.keys())
        if not (0 <= src_row < len(keys) and 0 <= dst_row < len(keys)):
            return
        
        # Pop the source key and insert it at the destination
        key = keys.pop(src_row)
        keys.insert(dst_row, key)
        
        # Rebuild dictionary keeping the new manual order
        self.selected_signals = {k: self.selected_signals[k] for k in keys}
        
        # Re-generate the UI entirely using the clean dictionary order
        self._rebuild_legend()
        if self._simulation_started:
            self.plot()

    def _rebuild_legend(self):
        self.legend_table.setRowCount(0)
        for uname, info in self.selected_signals.items():
            row = self.legend_table.rowCount()
            self.legend_table.insertRow(row)

            swatch = QPushButton()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background:{info['color'].name()};"
                "border:1px solid #000;border-radius:0;")
            swatch.clicked.connect(lambda _, n=uname: self._pick_color(n))
            sw  = QWidget()
            swl = QHBoxLayout(sw)
            swl.setContentsMargins(0, 0, 0, 0)
            swl.setAlignment(Qt.AlignCenter)
            swl.addWidget(swatch)
            self.legend_table.setCellWidget(row, self._COL_COLOR, sw)

            sig_display = (f"{info['sig_name']} [{info['unit']}]"
                           if info['unit'] else info['sig_name'])
            ni = QTableWidgetItem(sig_display)
            ni.setData(Qt.UserRole, uname)
            ni.setFlags(ni.flags() & ~Qt.ItemIsEditable)
            self.legend_table.setItem(row, self._COL_SIG, ni)

            for col in (self._COL_VAL, self._COL_MIN, self._COL_MAX):
                ci = QTableWidgetItem("--")
                ci.setFlags(ci.flags() & ~Qt.ItemIsEditable)
                self.legend_table.setItem(row, col, ci)

            self._update_minmax_cell(row, uname)

        self._apply_minmax_visibility()

    def _update_minmax_cell(self, row: int, uname: str):
        pd = self.current_plot_data.get(uname)
        if pd is None or len(pd['v']) == 0:
            return
        mn = float(pd['v'].min())
        mx = float(pd['v'].max())
        mi = QTableWidgetItem(f"{mn:.4g}")
        mi.setFlags(mi.flags() & ~Qt.ItemIsEditable)
        xi = QTableWidgetItem(f"{mx:.4g}")
        xi.setFlags(xi.flags() & ~Qt.ItemIsEditable)
        self.legend_table.setItem(row, self._COL_MIN, mi)
        self.legend_table.setItem(row, self._COL_MAX, xi)

    def _update_all_minmax(self):
        for row in range(self.legend_table.rowCount()):
            ni = self.legend_table.item(row, self._COL_SIG)
            if ni is None:
                continue
            self._update_minmax_cell(row, ni.data(Qt.UserRole))

    def _pick_color(self, uname: str):
        c = QColorDialog.getColor(self.selected_signals[uname]['color'], self)
        if c.isValid():
            self.selected_signals[uname]['color'] = c
            self._rebuild_legend()
            if self._simulation_started:
                self.plot()

    def _toggle_cursor(self):
        if self.btn_cursor_toggle.isChecked():
            if self.plots:
                vr  = self.plots[0].getViewBox().viewRange()[0]
                mid = max(0.0, min((vr[0] + vr[1]) / 2.0, self.max_x))
                for ln in self.v_lines:
                    ln.setValue(mid); ln.show()
                self._update_cursor_readouts(mid)
        else:
            for ln in self.v_lines:
                ln.hide()

    def _toggle_zoom_mode(self):
        mode = (pg.ViewBox.RectMode if self.btn_box_zoom.isChecked()
                else pg.ViewBox.PanMode)
        for p in self.plots:
            p.getViewBox().setMouseMode(mode)

    def _reset_zoom(self):
        if self.max_x > 0:
            self._set_x_range_exact(0.0, self.max_x)

    def _on_cursor_dragged(self, line):
        x = max(0.0, min(float(line.value()), self.max_x))
        for ln in self.v_lines:
            if ln is not line:
                ln.blockSignals(True)
                ln.setValue(x)
                ln.blockSignals(False)
        self.lbl_cursor.setText(f"Cursor Time: {x:.4f} s")
        self._update_cursor_readouts(x)

    def _update_cursor_readouts(self, x: float):
        for row in range(self.legend_table.rowCount()):
            ni = self.legend_table.item(row, self._COL_SIG)
            if ni is None:
                continue
            uname = ni.data(Qt.UserRole)
            pd    = self.current_plot_data.get(uname)
            
            if pd is None or len(pd['t']) == 0:
                continue
                
            idx = max(0, int(np.searchsorted(pd['t'], x, side='right')) - 1)
            idx = min(idx, len(pd['t']) - 1)
            
            val = float(pd['v'][idx])
            choices = self.selected_signals[uname].get('choices')
            
            display_str = f"{val:.4f}"
            
            if choices is not None and val.is_integer():
                int_val = int(val)
                if int_val in choices:
                    display_str = str(choices[int_val])
                    
            vi  = QTableWidgetItem(display_str)
            vi.setForeground(QColor('#cc0000'))
            self.legend_table.setItem(row, self._COL_VAL, vi)

    def _set_x_range_exact(self, x_lo: float, x_hi: float):
        for p in self.plots:
            vb = p.getViewBox()
            vb.setRange(xRange=(x_lo, x_hi), padding=0,
                        update=True, disableAutoRange=True)

    def _on_vb_resized(self):
        if self.max_x <= 0 or not self.plots:
            return
        vb = self.plots[0].getViewBox()
        xr = vb.viewRange()[0]
        lo, hi = xr[0], xr[1]
        tol = self.max_x * 0.01
        changed = False
        if hi > self.max_x + tol:
            hi = self.max_x
            changed = True
        if lo < -tol:
            lo = 0.0
            changed = True
        if changed:
            self._set_x_range_exact(lo, hi)
        self._schedule_tick_update()

    def _after_layout(self):
        self._dpi_scale = QApplication.primaryScreen().devicePixelRatio()
        if self.max_x > 0 and self.plots:
            vb = self.plots[0].getViewBox()
            xr = vb.viewRange()[0]
            if xr[1] > self.max_x * 1.005:
                self._set_x_range_exact(0.0, self.max_x)
        self._apply_ticks()
        self._update_y_label_fonts()

    def _update_y_label_fonts(self):
        if not self.plots or not self._valid_keys:
            return
        for p, key in zip(self.plots, self._valid_keys):
            if key not in self.selected_signals:
                continue
            si   = self.selected_signals[key]
            h_px = _canvas_height_px(p.getViewBox())

            lbl_full    = (f"{si['sig_name']} [{si['unit']}]"
                           if si['unit'] else si['sig_name'])
            n_chars     = max(len(lbl_full), 1)
            px_per_char = (h_px * 0.85) / n_chars
            font_pt     = max(5, min(10, int(px_per_char / 1.35)))

            left_ax = p.getAxis('left')
            left_ax.setLabel(lbl_full, color=si['color'].name())
            left_ax.label.setFont(QFont("Segoe UI", font_pt))
            left_ax.setWidth(int(70 * self._dpi_scale))

            try:
                yr     = p.getViewBox().viewRange()[1]
                y_lo, y_hi = yr[0], yr[1]
                y_span = y_hi - y_lo
                if y_span <= 0:
                    continue
                max_ticks = max(int(h_px / 40), 2)
                raw_step  = y_span / min(5, max_ticks)
                y_step    = _nice_step(raw_step)
                y_major   = _build_tick_lists(y_step, y_lo, y_hi,
                                              max_count=max_ticks + 2)
                y_minor   = _build_tick_lists(y_step / 5.0, y_lo, y_hi,
                                              max_count=max_ticks * 5 + 2)
                fmt_y     = _tick_format(y_step)
                left_ax.setTicks([
                    [(t, fmt_y.format(t)) for t in y_major],
                    [(t, "")              for t in y_minor],
                ])
            except Exception:
                pass

    def _schedule_tick_update(self):
        self._tick_timer.start()

    def _apply_ticks(self):
        if not self.plots:
            return

        vb     = self.plots[0].getViewBox()
        xr     = vb.viewRange()[0]
        x_lo   = max(0.0,        xr[0])
        x_hi   = min(self.max_x, xr[1]) if self.max_x > 0 else xr[1]
        x_span = x_hi - x_lo

        if x_span <= 0:
            QTimer.singleShot(100, self._apply_ticks)
            return

        cpx              = _canvas_px(vb)
        major_s, minor_s = _pick_tick(x_span, cpx)
        fmt              = _tick_format(major_s)

        major_pos = _build_tick_lists(major_s, x_lo, x_hi, max_count=500)
        minor_pos = _build_tick_lists(minor_s, x_lo, x_hi, max_count=2000)

        major_labelled = [(t, fmt.format(t)) for t in major_pos]
        minor_labelled = [(t, "")            for t in minor_pos]
        major_silent   = [(t, "")            for t in major_pos]

        n = len(self.plots)
        for i, p in enumerate(self.plots):
            ax = p.getAxis('bottom')
            if i == n - 1:
                ax.setStyle(showValues=True)
                ax.setTicks([major_labelled, minor_labelled])
            else:
                ax.setStyle(showValues=False, tickLength=0)
                ax.setTicks([major_silent, []])

        for grid in self._grids:
            grid.set_x_ticks(major_pos)

    def plot(self):
        for grid in self._grids:
            grid.detach()
        self._grids.clear()

        self.graph_layout.clear()
        self.plots.clear()
        self.v_lines.clear()
        self.current_plot_data.clear()
        self._valid_keys.clear()

        if not self.log_messages or not self.selected_signals:
            self._simulation_started = False
            return

        self._simulation_started = True

        t0     = self.log_messages[0].timestamp
        raw    = {k: {'time': [], 'val': []} for k in self.selected_signals}
        needed = {}
        for k, info in self.selected_signals.items():
            needed.setdefault(info['msg_id'], []).append(k)

        for msg in self.log_messages:
            if msg.arbitration_id not in needed:
                continue
            keys = needed[msg.arbitration_id]
            db   = self.selected_signals[keys[0]]['dbc']
            try:
                dec = db.decode_message(
                    msg.arbitration_id, msg.data, decode_choices=False)
                ts  = msg.timestamp - t0
                for k in keys:
                    sn = self.selected_signals[k]['sig_name']
                    if sn in dec and isinstance(dec[sn], (int, float)):
                        raw[k]['time'].append(ts)
                        raw[k]['val'].append(float(dec[sn]))
            except Exception:
                pass

        self.max_x = (self.log_messages[-1].timestamp
                      - self.log_messages[0].timestamp)
        if self.max_x <= 0:
            return

        valid_keys = []
        for k, d in raw.items():
            if d['time']:
                t_np = np.clip(
                    np.asarray(d['time'], dtype=np.float64), 0.0, self.max_x)
                v_np = np.asarray(d['val'],  dtype=np.float64)
                self.current_plot_data[k] = {'t': t_np, 'v': v_np}
                valid_keys.append(k)

        if not valid_keys:
            return

        self._valid_keys = valid_keys
        n_plots = len(valid_keys)
        self._update_all_minmax()

        border_pen    = pg.mkPen('#b0b0b0', width=1)
        time_axis_pen = pg.mkPen('#000000', width=2)
        cursor_pen    = pg.mkPen('#cc0000', width=2, style=Qt.DashLine)
        cursor_hover  = pg.mkPen('#ff0000', width=3)
        zero_ref_pen  = pg.mkPen('#999999', width=1, style=Qt.DashLine)

        first_plt = None

        for idx, key in enumerate(valid_keys):
            si      = self.selected_signals[key]
            pd_     = self.current_plot_data[key]
            t_arr   = pd_['t']
            v_arr   = pd_['v']
            is_last = (idx == n_plots - 1)

            n_s    = len(t_arr)
            t_step = np.empty(n_s * 2, dtype=np.float64)
            v_step = np.empty(n_s * 2, dtype=np.float64)
            t_step[0::2] = t_arr;  t_step[1::2] = t_arr
            v_step[0::2] = v_arr;  v_step[1::2] = v_arr
            t_step = t_step[1:];   v_step = v_step[:-1]
            if n_s > 0 and t_step[-1] < self.max_x:
                t_step = np.append(t_step, self.max_x)
                v_step = np.append(v_step, v_arr[-1])

            p = self.graph_layout.addPlot(row=idx, col=0)
            self.graph_layout.ci.layout.setRowStretchFactor(idx, 1)
            self.plots.append(p)

            if first_plt is None:
                first_plt = p
            else:
                p.setXLink(first_plt)

            vb = p.getViewBox()
            vb.disableAutoRange(pg.ViewBox.XAxis)
            vb.disableAutoRange(pg.ViewBox.YAxis)
            vb.setMouseEnabled(x=True, y=False)

            vb.suggestPadding = lambda axis: 0.0

            p.setLimits(xMin=0.0, xMax=self.max_x)
            p.sigXRangeChanged.connect(self._schedule_tick_update)

            try:
                vb.sigResized.connect(self._on_vb_resized)
            except AttributeError:
                pass

            for ax_name in ('top', 'left', 'right'):
                p.showAxis(ax_name)
                p.getAxis(ax_name).setPen(border_pen)
            p.getAxis('top').setStyle(showValues=False)
            p.getAxis('right').setStyle(showValues=False)

            p.showAxis('bottom')
            if is_last:
                p.getAxis('bottom').setPen(time_axis_pen)
                p.getAxis('bottom').setStyle(showValues=True)
                p.getAxis('bottom').setLabel("Time [s]", color='#000000')
            else:
                p.getAxis('bottom').setPen(border_pen)
                p.getAxis('bottom').setStyle(showValues=False, tickLength=0)

            grid = _StableGridItem(p)
            self._grids.append(grid)

            vmin, vmax = float(v_arr.min()), float(v_arr.max())
            span = vmax - vmin
            if span == 0:
                span = max(abs(vmin) * 0.1, 1.0)
            mg   = span * 0.05
            y_lo = vmin - mg
            y_hi = vmax + mg
            vb.setRange(yRange=(y_lo, y_hi), padding=0, disableAutoRange=True)

            if y_lo < 0.0 < y_hi:
                zero_line = pg.InfiniteLine(
                    pos=0.0, angle=0, movable=False, pen=zero_ref_pen)
                p.addItem(zero_line)

            lbl_full = (f"{si['sig_name']} [{si['unit']}]"
                        if si['unit'] else si['sig_name'])
            left_ax = p.getAxis('left')
            left_ax.setLabel(lbl_full, color=si['color'].name())
            left_ax.label.setFont(QFont("Segoe UI", 8))
            left_ax.setWidth(int(70 * self._dpi_scale))

            vl = pg.InfiniteLine(
                angle=90, movable=True,
                pen=cursor_pen, hoverPen=cursor_hover)
            vl.sigDragged.connect(self._on_cursor_dragged)
            if not self.btn_cursor_toggle.isChecked():
                vl.hide()
            p.addItem(vl)
            self.v_lines.append(vl)

            curve = pg.PlotCurveItem(
                x=t_step, y=v_step,
                pen=pg.mkPen(color=si['color'], width=1.5, cosmetic=True),
                antialias=True,
                connect='finite')
            p.addItem(curve)

        self._set_x_range_exact(0.0, self.max_x)

        QTimer.singleShot(0,   self._apply_ticks)
        QTimer.singleShot(150, self._apply_ticks)
        self._layout_timer.start()

        if self.btn_box_zoom.isChecked():
            for p in self.plots:
                p.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._dpi_scale = QApplication.primaryScreen().devicePixelRatio()
        self._schedule_tick_update()
        if self.max_x > 0 and self.plots:
            self._layout_timer.start()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CANDataAnalyzer()
    win.show()
    sys.exit(app.exec_())