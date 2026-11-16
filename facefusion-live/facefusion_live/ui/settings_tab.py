"""Settings: hardware, models (download manager), output, diagnostics."""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QHeaderView, QScrollArea)

from facefusion_live.inference.session import available_providers
from facefusion_live.models.manifest import MODELS, human_size
from facefusion_live.ui.widgets import hbox, run_task, vbox
from facefusion_live.utils.gpu import diagnostics_text


def confirm_download(parent, names: list[str]) -> bool:
    infos = [MODELS[n] for n in names]
    total = sum(m.size for m in infos)
    lines = "\n".join(f"• {m.name} — {human_size(m.size)} — {m.license}" for m in infos)
    return QMessageBox.question(
        parent, "Download models",
        f"Download {len(infos)} model(s), {human_size(total)} in total, from GitHub?\n\n{lines}\n\n"
        "Several models are licensed for non-commercial research use only. Files are stored locally; "
        "none of your images or videos are uploaded.") == QMessageBox.Yes


class SettingsTab(QScrollArea):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        s = ctx.settings
        self._cancel = threading.Event()

        # hardware
        hw = QGroupBox("Processing device")
        f = QFormLayout(hw)
        self.provider = QComboBox()
        avail = available_providers()
        for label, key, prov in [("Automatic (best available)", "auto", None), ("NVIDIA CUDA", "cuda", "CUDAExecutionProvider"),
                                 ("DirectML (AMD / Intel / NVIDIA on Windows)", "directml", "DmlExecutionProvider"),
                                 ("CPU only", "cpu", "CPUExecutionProvider")]:
            if prov is None or prov in avail:
                self.provider.addItem(label, key)
        self.provider.setCurrentIndex(max(self.provider.findData(s.provider), 0))
        self.device = QSpinBox()
        self.device.setRange(0, 7)
        self.device.setValue(s.device_id)
        self.detector = QComboBox()
        self.detector.addItem("SCRFD-10G (accurate)", "retinaface_10g")
        self.detector.addItem("SCRFD-2.5G (fast)", "scrfd_2.5g")
        self.detector.setCurrentIndex(max(self.detector.findData(s.detector), 0))
        self.swapper = QComboBox()
        for t, d in [("Automatic (FP16 on GPU, FP32 on CPU)", "auto"), ("inswapper 128 FP32", "inswapper_128"),
                     ("inswapper 128 FP16", "inswapper_128_fp16")]:
            self.swapper.addItem(t, d)
        self.swapper.setCurrentIndex(max(self.swapper.findData(s.swapper), 0))
        self.det_size = QComboBox()
        for v in (320, 480, 640, 800):
            self.det_size.addItem(f"{v} px", v)
        self.det_size.setCurrentIndex(max(self.det_size.findData(s.det_size), 0))
        self.interval = QSpinBox()
        self.interval.setRange(1, 30)
        self.interval.setValue(s.detect_interval)
        self.interval.setToolTip("Full-frame face detection every N frames; tracking in between")
        f.addRow("Device", self.provider)
        f.addRow("GPU index", self.device)
        f.addRow("Face detector", self.detector)
        f.addRow("Swap model", self.swapper)
        f.addRow("Detection size", self.det_size)
        f.addRow("Re-detect every N frames", self.interval)
        f.addRow(QLabel(f"Available: {', '.join(p.replace('ExecutionProvider', '') for p in avail)}"))

        # output
        out = QGroupBox("Video output")
        of = QFormLayout(out)
        self.quality = QSpinBox()
        self.quality.setRange(10, 35)
        self.quality.setValue(s.video_quality)
        self.quality.setToolTip("CRF/CQ: 14 = near lossless, 18 = high (default), 23 = smaller files")
        self.encoder = QComboBox()
        self.encoder.addItem("Automatic (GPU encoder if available)", "auto")
        for e in ("h264_nvenc", "hevc_nvenc", "h264_amf", "h264_qsv", "libx264", "libx265"):
            self.encoder.addItem(e, e)
        self.encoder.setCurrentIndex(max(self.encoder.findData(s.video_encoder), 0))
        self.cam_fps = QSpinBox()
        self.cam_fps.setRange(10, 60)
        self.cam_fps.setValue(s.camera_fps)
        of.addRow("Quality (lower = better)", self.quality)
        of.addRow("Encoder", self.encoder)
        of.addRow("Webcam FPS request", self.cam_fps)
        privacy = QLabel("All processing is local. No telemetry is collected. Exported media always carry a "
                         "visible 'AI-generated' label and metadata.")
        privacy.setObjectName("muted")
        privacy.setWordWrap(True)
        of.addRow(privacy)

        apply = QPushButton("Apply settings")
        apply.setObjectName("primary")
        apply.clicked.connect(self.apply)

        # models
        mbox = QGroupBox("Models")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Model", "Size", "Status", "Purpose / license"])
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(330)
        self.mdir = QLabel()
        self.mdir.setObjectName("muted")
        self.mdir.setTextInteractionFlags(Qt.TextSelectableByMouse)
        chg, opn = QPushButton("Change folder…"), QPushButton("Open folder")
        self.dl_btn = QPushButton("Download selected…")
        self.dl_btn.setObjectName("primary")
        self.dl_cancel = QPushButton("Cancel download")
        self.dl_bar = QProgressBar()
        self.dl_lbl = QLabel("")
        self.dl_lbl.setObjectName("muted")
        mbox.setLayout(vbox(hbox(self.mdir, 1, chg, opn), self.table, hbox(self.dl_btn, self.dl_cancel, 1),
                            self.dl_bar, self.dl_lbl))
        chg.clicked.connect(self.change_dir)
        opn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ctx.settings.resolved_models_dir()))))
        self.dl_btn.clicked.connect(lambda: self.download_selected())
        self.dl_cancel.clicked.connect(self._cancel.set)

        # diagnostics
        dbox = QGroupBox("Hardware diagnostics")
        self.diag = QPlainTextEdit()
        self.diag.setReadOnly(True)
        self.diag.setMinimumHeight(150)
        rd = QPushButton("Run diagnostics")
        rd.clicked.connect(self.run_diag)
        dbox.setLayout(vbox(self.diag, hbox(rd, 1)))

        lay = QVBoxLayout(body)
        lay.addLayout(hbox(vbox(hw, out, hbox(apply, 1), 1), vbox(mbox, dbox)))
        self.refresh_models()

    def apply(self):
        s = self.ctx.settings
        old = (s.provider, s.device_id, s.detector, s.swapper)
        s.provider, s.device_id = self.provider.currentData(), self.device.value()
        s.detector, s.swapper = self.detector.currentData(), self.swapper.currentData()
        s.det_size, s.detect_interval = self.det_size.currentData(), self.interval.value()
        s.video_quality, s.video_encoder, s.camera_fps = self.quality.value(), self.encoder.currentData(), self.cam_fps.value()
        self.ctx.save()
        if old != (s.provider, s.device_id, s.detector, s.swapper):
            self.ctx.reset_engine()
        QMessageBox.information(self, "Settings", "Settings saved.")

    def refresh_models(self):
        store = self.ctx.store
        self.mdir.setText(f"Folder: {store.dir}")
        self.table.setRowCount(len(MODELS))
        for r, m in enumerate(MODELS.values()):
            inst = store.is_installed(m.name)
            cb = QCheckBox()
            cb.setChecked(not inst and m.required)
            cb.setEnabled(not inst)
            self.table.setCellWidget(r, 0, cb)
            self.table.setItem(r, 1, QTableWidgetItem(m.name + (" *" if m.required else "")))
            self.table.setItem(r, 2, QTableWidgetItem(human_size(m.size)))
            self.table.setItem(r, 3, QTableWidgetItem("Installed" if inst else "Not installed"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{m.purpose} — {m.license}"))
        self.table.resizeColumnsToContents()
        self.dl_lbl.setText("* required for basic swapping (plus one swap model)")

    def selected(self) -> list[str]:
        names = list(MODELS)
        return [names[r] for r in range(self.table.rowCount()) if self.table.cellWidget(r, 0).isChecked()]

    def change_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Models folder", str(self.ctx.settings.resolved_models_dir()))
        if d:
            self.ctx.settings.models_dir = d
            self.ctx.save()
            self.ctx.reset_engine()
            self.refresh_models()

    def download_selected(self, names: list[str] | None = None, confirmed: bool = False):
        names = names or self.selected()
        if not names:
            QMessageBox.information(self, "Models", "Tick the models to download first.")
            return
        if not confirmed and not confirm_download(self, names):
            return
        self._cancel.clear()
        self.dl_btn.setEnabled(False)
        store = self.ctx.store

        def work(emit):
            for i, n in enumerate(names):
                store.download(n, lambda d, t, n=n, i=i: emit((n, i, len(names), d, t)), self._cancel.is_set)
            return names

        def prog(p):
            n, i, k, d, t = p
            self.dl_bar.setMaximum(1000)
            self.dl_bar.setValue(int(d / max(t, 1) * 1000))
            self.dl_lbl.setText(f"[{i + 1}/{k}] {n}: {human_size(d)} / {human_size(t)}")

        def done(_):
            self.dl_btn.setEnabled(True)
            self.dl_lbl.setText("Download complete.")
            self.ctx.reset_engine()
            self.refresh_models()

        def fail(msg):
            self.dl_btn.setEnabled(True)
            self.dl_lbl.setText(f"Download stopped: {msg}. Partial files resume next time.")
            self.refresh_models()

        run_task(work, done, fail, prog)

    def run_diag(self):
        self.diag.setPlainText("Collecting…")
        run_task(lambda _: diagnostics_text(), self.diag.setPlainText, self.diag.setPlainText)
