"""
CTN_gui_plotting.py
Standalone GUI for plotting Facemap keypoints and SVD traces.

Usage:
    python CTN_gui_plotting.py
"""
import os
import sys
import glob

import h5py
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFileDialog, QGroupBox, QScrollArea,
    QFrame, QSizePolicy,
)

# ── Eye bodyparts and colours ──────────────────────────────────────────────────
EYE_BODYPARTS = ["eye(back)", "eye(bottom)", "eye(front)", "eye(top)"]
EYE_COLORS    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Colours cycled for external-input shading bands
SHADE_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ── External-input row widget ──────────────────────────────────────────────────
class ExternalInputRow(QWidget):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        layout.addWidget(QLabel(f"#{index + 1}  Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Input name")
        self.name_edit.setFixedWidth(130)
        layout.addWidget(self.name_edit)

        layout.addWidget(QLabel("From:"))
        self.from_edit = QLineEdit()
        self.from_edit.setPlaceholderText("0")
        self.from_edit.setFixedWidth(65)
        layout.addWidget(self.from_edit)

        layout.addWidget(QLabel("To:"))
        self.to_edit = QLineEdit()
        self.to_edit.setPlaceholderText("end")
        self.to_edit.setFixedWidth(65)
        layout.addWidget(self.to_edit)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["seconds", "frames"])
        self.unit_combo.setFixedWidth(85)
        layout.addWidget(self.unit_combo)

        layout.addStretch()

    def get_values(self, fps):
        """Return (name, t_from_sec, t_to_sec).  t_to_sec may be None."""
        name = self.name_edit.text().strip() or f"Input"
        unit = self.unit_combo.currentText()

        def _parse(text, fallback):
            try:
                return float(text.strip())
            except ValueError:
                return fallback

        raw_from = _parse(self.from_edit.text(), 0.0)
        raw_to   = _parse(self.to_edit.text(), None) if self.to_edit.text().strip() else None

        if unit == "frames":
            t_from = raw_from / fps
            t_to   = (raw_to / fps) if raw_to is not None else None
        else:
            t_from = raw_from
            t_to   = raw_to

        return name, t_from, t_to


# ── Main window ────────────────────────────────────────────────────────────────
class CTNGuiPlotting(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CTN GUI Plotting")
        self.proc_data      = None
        self.keypoints_data = None
        self.bodyparts      = None
        self.input_rows     = []

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(15, 15, 15, 15)

        # ── Data group ─────────────────────────────────────────────────────────
        data_group = QGroupBox("Data")
        dg = QGridLayout(data_group)

        dg.addWidget(QLabel("Folder:"), 0, 0)
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color: gray; font-style: italic;")
        dg.addWidget(self.folder_label, 0, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_folder)
        dg.addWidget(browse_btn, 0, 2)

        dg.addWidget(QLabel("File:"), 1, 0)
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(320)
        dg.addWidget(self.file_combo, 1, 1, 1, 2)

        dg.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_spinbox = QDoubleSpinBox()
        self.fps_spinbox.setRange(0.1, 100000.0)
        self.fps_spinbox.setValue(30.0)
        self.fps_spinbox.setDecimals(2)
        self.fps_spinbox.setFixedWidth(100)
        dg.addWidget(self.fps_spinbox, 2, 1)

        root.addWidget(data_group)

        # ── External inputs group ──────────────────────────────────────────────
        self.ext_group = QGroupBox("External Inputs")
        self.ext_group.setCheckable(True)
        self.ext_group.setChecked(False)
        self.ext_group.toggled.connect(self._on_ext_toggled)
        eg = QVBoxLayout(self.ext_group)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Number of inputs:"))
        self.n_inputs_spin = QSpinBox()
        self.n_inputs_spin.setRange(1, 20)
        self.n_inputs_spin.setValue(1)
        self.n_inputs_spin.setFixedWidth(60)
        self.n_inputs_spin.valueChanged.connect(self._update_input_rows)
        n_row.addWidget(self.n_inputs_spin)
        n_row.addStretch()
        eg.addLayout(n_row)

        # Scroll area holds the dynamic rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(190)
        self.rows_widget = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        scroll.setWidget(self.rows_widget)
        eg.addWidget(scroll)

        self._update_input_rows(1)
        root.addWidget(self.ext_group)

        # ── Plot buttons (lower right) ─────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.plot_eye_btn = QPushButton("Plot Eye")
        self.plot_eye_btn.setFixedSize(120, 36)
        self.plot_eye_btn.clicked.connect(self._plot_eye)
        btn_row.addWidget(self.plot_eye_btn)
        root.addLayout(btn_row)

        self.setMinimumWidth(560)
        self.adjustSize()

    # ── Slot: external-inputs group toggled ───────────────────────────────────
    def _on_ext_toggled(self, checked):
        # Children are automatically enabled/disabled by QGroupBox; nothing extra needed
        pass

    # ── Slot: browse for folder ────────────────────────────────────────────────
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select data folder")
        if not folder:
            return
        self.folder_label.setText(folder)
        self.folder_label.setStyleSheet("")
        self._populate_file_combo(folder)

    # ── Populate file dropdown ─────────────────────────────────────────────────
    def _populate_file_combo(self, folder):
        self.file_combo.clear()
        h5_files = sorted(glob.glob(os.path.join(folder, "*_FacemapPose.h5")))
        if not h5_files:
            self.file_combo.addItem("No *_FacemapPose.h5 files found")
            return
        for h5 in h5_files:
            basename = os.path.basename(h5).replace("_FacemapPose.h5", "")
            npy = os.path.join(folder, basename + "_proc.npy")
            has_npy = os.path.exists(npy)
            label = basename + (" [h5 + npy]" if has_npy else " [h5 only]")
            self.file_combo.addItem(label, userData=(h5, npy if has_npy else None))

    # ── Update external-input rows to match spinbox value ─────────────────────
    def _update_input_rows(self, n):
        while len(self.input_rows) < n:
            row = ExternalInputRow(index=len(self.input_rows))
            self.input_rows.append(row)
            self.rows_layout.addWidget(row)
        while len(self.input_rows) > n:
            row = self.input_rows.pop()
            self.rows_layout.removeWidget(row)
            row.deleteLater()

    # ── Load data from selected file ───────────────────────────────────────────
    def _load_data(self):
        idx = self.file_combo.currentIndex()
        if idx < 0:
            return False
        user_data = self.file_combo.itemData(idx)
        if user_data is None:
            return False
        h5_path, npy_path = user_data

        # Keypoints from .h5
        with h5py.File(h5_path, "r") as f:
            facemap = f["Facemap"]
            self.bodyparts = list(facemap.keys())
            self.keypoints_data = {}
            for bp in self.bodyparts:
                self.keypoints_data[bp] = {
                    "x":           np.array(facemap[bp]["x"]),
                    "y":           np.array(facemap[bp]["y"]),
                    "likelihood":  np.array(facemap[bp]["likelihood"]),
                }

        # Proc dict from .npy (optional)
        self.proc_data = None
        if npy_path and os.path.exists(npy_path):
            self.proc_data = np.load(npy_path, allow_pickle=True).item()

        return True

    # ── Collect external-input definitions ────────────────────────────────────
    def _get_ext_inputs(self, fps):
        if not self.ext_group.isChecked():
            return []
        return [row.get_values(fps) for row in self.input_rows]

    # ── Apply shading to a list of axes ───────────────────────────────────────
    def _shade_axes(self, axes_list, ext_inputs, t_max, legend_ax=None):
        for j, (name, t_from, t_to) in enumerate(ext_inputs):
            color  = SHADE_COLORS[j % len(SHADE_COLORS)]
            t_end  = t_to if t_to is not None else t_max
            for i, ax in enumerate(axes_list):
                ax.axvspan(t_from, t_end, alpha=0.15, color=color,
                           label=name if i == 0 else None)
        if legend_ax is not None and ext_inputs:
            # Add a compact shading legend separated from the trace legend
            from matplotlib.patches import Patch
            handles = [
                Patch(color=SHADE_COLORS[j % len(SHADE_COLORS)], alpha=0.4, label=name)
                for j, (name, _, _) in enumerate(ext_inputs)
            ]
            legend_ax.legend(handles=handles, loc="upper left",
                             fontsize=7, title="External inputs", title_fontsize=7)

    # ── Eye plot ───────────────────────────────────────────────────────────────
    def _plot_eye(self):
        if not self._load_data():
            QtWidgets.QMessageBox.warning(self, "Error", "No valid data file selected.")
            return

        missing = [bp for bp in EYE_BODYPARTS if bp not in self.keypoints_data]
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Missing bodyparts",
                f"The following eye bodyparts are not in the file:\n{missing}"
            )
            return

        fps      = self.fps_spinbox.value()
        ext_in   = self._get_ext_inputs(fps)
        n_frames = len(self.keypoints_data[EYE_BODYPARTS[0]]["x"])
        t        = np.arange(n_frames) / fps

        # ── Compute quantities ─────────────────────────────────────────────────
        eye_x = np.array([self.keypoints_data[bp]["x"] for bp in EYE_BODYPARTS])
        eye_y = np.array([self.keypoints_data[bp]["y"] for bp in EYE_BODYPARTS])

        center_x = eye_x.mean(axis=0)
        center_y = eye_y.mean(axis=0)
        ref_x    = center_x.mean()
        ref_y    = center_y.mean()

        dx = center_x - ref_x
        dy = center_y - ref_y
        center_dist  = np.sqrt(dx**2 + dy**2)
        center_angle = np.degrees(np.arctan2(dy, dx))

        kp_dist = np.array([
            np.sqrt((eye_x[i] - center_x)**2 + (eye_y[i] - center_y)**2)
            for i in range(4)
        ])

        win = EyePlotWindow(
            t=t, eye_x=eye_x, eye_y=eye_y,
            center_dist=center_dist, center_angle=center_angle,
            kp_dist=kp_dist, ext_inputs=ext_in,
            parent=self,
        )
        win.show()
        # Keep a reference so it isn't garbage-collected
        if not hasattr(self, "_plot_windows"):
            self._plot_windows = []
        self._plot_windows.append(win)


# ── Y-limit control panel ──────────────────────────────────────────────────────
class YLimitRow(QWidget):
    """One row in the y-limits sidebar: label + Min + Max fields."""
    def __init__(self, label, y_min, y_max, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)

        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        lbl.setWordWrap(True)
        row.addWidget(lbl)

        row.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setDecimals(2)
        self.min_spin.setValue(round(y_min, 2))
        self.min_spin.setFixedWidth(80)
        row.addWidget(self.min_spin)

        row.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setDecimals(2)
        self.max_spin.setValue(round(y_max, 2))
        self.max_spin.setFixedWidth(80)
        row.addWidget(self.max_spin)

    def get_limits(self):
        return self.min_spin.value(), self.max_spin.value()


# ── Eye plot window ────────────────────────────────────────────────────────────
class EyePlotWindow(QMainWindow):
    def __init__(self, t, eye_x, eye_y,
                 center_dist, center_angle, kp_dist,
                 ext_inputs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eye Analysis")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Store references for redrawing
        self._t             = t
        self._eye_x         = eye_x
        self._eye_y         = eye_y
        self._center_dist   = center_dist
        self._center_angle  = center_angle
        self._kp_dist       = kp_dist
        self._ext_inputs    = ext_inputs

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Left: matplotlib canvas ────────────────────────────────────────────
        self._fig = Figure(figsize=(13, 16), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas, stretch=1)

        # ── Right: y-limits sidebar ────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(390)
        sidebar.setStyleSheet("background: #f7f7f7;")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(8, 10, 8, 10)
        sb_layout.setSpacing(6)

        sb_layout.addWidget(QLabel("<b>Y-axis limits</b>"))

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sb_layout.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(rows_widget)
        self._rows_layout.setSpacing(4)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(rows_widget)
        sb_layout.addWidget(scroll, stretch=1)

        apply_btn = QPushButton("Apply limits")
        apply_btn.setFixedHeight(32)
        apply_btn.clicked.connect(self._apply_limits)
        sb_layout.addWidget(apply_btn)

        reset_btn = QPushButton("Reset to auto")
        reset_btn.setFixedHeight(28)
        reset_btn.clicked.connect(self._reset_limits)
        sb_layout.addWidget(reset_btn)

        layout.addWidget(sidebar)

        # Build figure and sidebar rows
        self._build_figure()
        self.resize(1400, 900)

    # ── Build / rebuild the figure ─────────────────────────────────────────────
    def _build_figure(self):
        self._fig.clear()
        t   = self._t
        gs  = gridspec.GridSpec(
            6, 1, figure=self._fig,
            height_ratios=[2, 1, 1, 1, 1, 1],
            hspace=0.55,
        )

        self._ax_upper = self._fig.add_subplot(gs[0])
        self._ax_ctr   = self._fig.add_subplot(gs[1])
        self._ax_kps   = [self._fig.add_subplot(gs[2 + i]) for i in range(4)]
        self._ax_ang   = self._ax_ctr.twinx()

        all_primary = [self._ax_upper, self._ax_ctr] + self._ax_kps

        # Upper: 8 traces
        for i, bp in enumerate(EYE_BODYPARTS):
            c = EYE_COLORS[i]
            self._ax_upper.plot(t, self._eye_x[i], color=c, linewidth=0.8, label=f"{bp}  x")
            self._ax_upper.plot(t, self._eye_y[i], color=c, linewidth=0.8,
                                linestyle="--", alpha=0.75, label=f"{bp}  y")
        self._ax_upper.set_title("Eye keypoint coordinates  (x — solid, y — dashed)", fontsize=10)
        self._ax_upper.set_ylabel("Position (px)")
        self._ax_upper.legend(loc="upper right", fontsize=6, ncol=4)
        self._ax_upper.tick_params(labelbottom=False)

        # Centre displacement
        self._ax_ctr.plot(t, self._center_dist, color="#1f77b4", linewidth=0.9)
        self._ax_ctr.set_ylabel("Distance (px)", color="#1f77b4")
        self._ax_ctr.tick_params(axis="y", labelcolor="#1f77b4")
        self._ax_ctr.set_title("Centre displacement from time-averaged reference", fontsize=10)
        self._ax_ctr.tick_params(labelbottom=False)

        self._ax_ang.plot(t, self._center_angle, color="#d62728", linewidth=0.6, alpha=0.7)
        self._ax_ang.set_ylabel("Angular displacement (°)", color="#d62728")
        self._ax_ang.set_ylim(-180, 180)
        self._ax_ang.tick_params(axis="y", labelcolor="#d62728")

        # Per-keypoint distances — only the last one gets an x-axis label
        for i, bp in enumerate(EYE_BODYPARTS):
            ax = self._ax_kps[i]
            ax.plot(t, self._kp_dist[i], color=EYE_COLORS[i], linewidth=0.8)
            ax.set_title(f"{bp}  —  distance to instantaneous centre", fontsize=10)
            ax.set_ylabel("Distance (px)")
            if i == len(EYE_BODYPARTS) - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.tick_params(labelbottom=False)

        # External-input shading
        from matplotlib.patches import Patch
        for j, (name, t_from, t_to) in enumerate(self._ext_inputs):
            color = SHADE_COLORS[j % len(SHADE_COLORS)]
            t_end = t_to if t_to is not None else t[-1]
            for ax in all_primary:
                ax.axvspan(t_from, t_end, alpha=0.15, color=color)
        if self._ext_inputs:
            handles = [
                Patch(color=SHADE_COLORS[j % len(SHADE_COLORS)], alpha=0.4, label=name)
                for j, (name, _, _) in enumerate(self._ext_inputs)
            ]
            self._ax_upper.legend(
                handles=handles, loc="upper left",
                fontsize=7, title="External inputs", title_fontsize=7
            )

        self._fig.suptitle("Eye Analysis", fontsize=14, fontweight="bold")
        self._canvas.draw()

        # Build (or rebuild) the sidebar y-limit rows
        self._build_sidebar_rows()

    # ── Build sidebar rows from current auto limits ────────────────────────────
    def _build_sidebar_rows(self):
        # Clear existing rows
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._ylimit_rows = {}   # key → YLimitRow

        def _add_row(key, label, ax, which="left"):
            if which == "left":
                lo, hi = ax.get_ylim()
            else:
                lo, hi = ax.get_ylim()
            row = YLimitRow(label, lo, hi)
            self._ylimit_rows[key] = row
            self._rows_layout.addWidget(row)

        _add_row("upper",     "Keypoint coordinates",             self._ax_upper)
        _add_row("ctr_dist",  "Centre dist. (left axis, px)",     self._ax_ctr)
        _add_row("ctr_angle", "Centre angle (right axis, °)",     self._ax_ang)
        for i, bp in enumerate(EYE_BODYPARTS):
            _add_row(f"kp_{i}", f"{bp}\n(dist. to centre)",       self._ax_kps[i])

        self._rows_layout.addStretch()

    # ── Apply user-entered limits ──────────────────────────────────────────────
    def _apply_limits(self):
        lo, hi = self._ylimit_rows["upper"].get_limits()
        self._ax_upper.set_ylim(lo, hi)

        lo, hi = self._ylimit_rows["ctr_dist"].get_limits()
        self._ax_ctr.set_ylim(lo, hi)

        lo, hi = self._ylimit_rows["ctr_angle"].get_limits()
        self._ax_ang.set_ylim(lo, hi)

        for i in range(4):
            lo, hi = self._ylimit_rows[f"kp_{i}"].get_limits()
            self._ax_kps[i].set_ylim(lo, hi)

        self._canvas.draw()

    # ── Reset all axes to matplotlib auto limits ───────────────────────────────
    def _reset_limits(self):
        for ax in [self._ax_upper, self._ax_ctr, self._ax_ang] + self._ax_kps:
            ax.autoscale(axis="y")
            ax.relim()
            ax.autoscale_view()
        self._canvas.draw()
        self._build_sidebar_rows()   # refresh displayed values


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = CTNGuiPlotting()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
