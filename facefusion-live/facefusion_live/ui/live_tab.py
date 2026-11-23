"""LIVE SWAP tab: webcam -> real-time swap -> preview / virtual camera."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGroupBox, QLabel, QMessageBox, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from facefusion_live.camera.capture import list_cameras
from facefusion_live.camera.live import LivePipeline
from facefusion_live.camera.virtualcam import VirtualCamera
from facefusion_live.core.mapping import FaceMapping
from facefusion_live.ui.source_panel import SourcePanel
from facefusion_live.ui.widgets import VIDEO_FILTER, ConsentBox, ImageView, hbox, run_task, vbox
from facefusion_live.utils.gpu import GPUMonitor

RESOLUTIONS = [("1280×720 (720p)", (1280, 720)), ("1920×1080 (1080p)", (1920, 1080)), ("960×540", (960, 540)),
               ("640×480", (640, 480))]


class LiveTab(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.pipe: LivePipeline | None = None
        self.mapping = FaceMapping(mode="largest")
        self.gpu = GPUMonitor()
        self._seq = 0

        self.src = SourcePanel(ctx)
        cbox = QGroupBox("Camera")
        self.cam = QComboBox()
        refresh = QPushButton("↻")
        refresh.setToolTip("Refresh camera list")
        refresh.setFixedWidth(36)
        self.res = QComboBox()
        for t, wh in RESOLUTIONS:
            self.res.addItem(t, wh)
        self.all_faces = QCheckBox("Swap every face (not only the largest)")
        self.vcam = QCheckBox("Send to virtual camera (OBS Virtual Camera)")
        self.vcam.setToolTip("Requires OBS Studio 26+ on Windows. See docs/VIRTUAL_CAMERA.md")
        self.wm = QCheckBox("Show 'AI-generated' watermark")
        self.wm.setChecked(ctx.settings.live_watermark)
        self.adaptive = QCheckBox("Adaptive quality (keep frame rate)")
        self.adaptive.setChecked(ctx.settings.adaptive)
        cbox.setLayout(vbox(hbox(self.cam, refresh), self.res, self.all_faces, self.vcam, self.wm, self.adaptive))
        left = QWidget()
        left.setLayout(vbox(self.src, cbox))

        self.view = ImageView("Live preview — choose a source face and press Start Live Swap")
        self.consent = ConsentBox()
        self.start_btn = QPushButton("Start Live Swap")
        self.start_btn.setObjectName("primary")
        self.toggle_btn = QPushButton("Bypass swap")
        self.toggle_btn.setCheckable(True)
        self.stats = QLabel("")
        self.stats.setObjectName("stat")
        self.msg = QLabel("")
        self.msg.setObjectName("muted")
        self.msg.setWordWrap(True)
        right = QGroupBox("Live preview")
        right.setLayout(vbox(self.view, self.consent, hbox(self.start_btn, self.toggle_btn, 1), self.stats, self.msg))
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 820])
        QVBoxLayout(self).addWidget(split)

        refresh.clicked.connect(self.refresh_cameras)
        self.start_btn.clicked.connect(self.toggle)
        self.toggle_btn.toggled.connect(self._bypass)
        self.src.sourceChanged.connect(self._source)
        self.consent.toggled.connect(lambda _: self._refresh())
        self.all_faces.toggled.connect(lambda v: setattr(self.mapping, "mode", "all" if v else "largest"))
        self.wm.toggled.connect(lambda v: (setattr(ctx.settings, "live_watermark", v), ctx.save()))
        self.adaptive.toggled.connect(lambda v: (setattr(ctx.settings, "adaptive", v), ctx.save()))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.refresh_cameras()
        self._refresh()

    def refresh_cameras(self):
        self.cam.clear()
        cams = list_cameras()
        for c in cams:
            self.cam.addItem(c.name, c.index)
        self.cam.addItem("Video file (test input)…", "file")
        idx = self.cam.findData(self.ctx.settings.camera_index)
        self.cam.setCurrentIndex(max(idx, 0))
        if not cams:
            self.msg.setText("No webcam detected. Connect one and press ↻.")

    def _source(self, s):
        self.mapping.default = s
        self._refresh()

    def _refresh(self):
        running = self.pipe is not None
        self.start_btn.setText("Stop" if running else "Start Live Swap")
        self.start_btn.setObjectName("danger" if running else "primary")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.start_btn.setEnabled(running or (self.consent.isChecked() and self.src.source is not None))
        for w in (self.cam, self.res, self.vcam):
            w.setEnabled(not running)
        self.toggle_btn.setEnabled(running)

    def toggle(self):
        if self.pipe:
            self.stop()
        else:
            self.start()

    def start(self):
        data = self.cam.currentData()
        if data == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Video file as camera", "", VIDEO_FILTER)
            if not path:
                return
            source = path
        elif data is None:
            QMessageBox.warning(self, "Live", "No camera selected.")
            return
        else:
            source = int(data)
            self.ctx.settings.camera_index = source
        w, h = self.res.currentData()
        s = self.ctx.settings
        s.camera_width, s.camera_height = w, h
        self.ctx.save()
        if self.vcam.isChecked() and not VirtualCamera.available():
            QMessageBox.information(self, "Virtual camera", "pyvirtualcam is not installed; continuing without it.")
        self.start_btn.setEnabled(False)
        self.msg.setText("Starting camera and loading models…")

        def work(_):
            eng = self.ctx.engine()
            eng.warmup()
            p = LivePipeline(eng, self.mapping, source, virtual_cam=self.vcam.isChecked())
            p.start()
            return p

        run_task(work, self._started, self._failed)

    def _started(self, pipe):
        self.pipe = pipe
        self._seq = 0
        w, h = pipe.stats.resolution
        self.msg.setText(f"Running on {pipe.engine.provider} at {w}×{h}."
                         + (f" Virtual camera: {pipe.vcam.device}" if pipe.vcam else ""))
        if pipe.error:
            self.msg.setText(pipe.error)
        self.timer.start(15)
        self._refresh()

    def _failed(self, msg):
        self.msg.setText(msg)
        QMessageBox.warning(self, "Live swap", msg)
        self._refresh()

    def _bypass(self, on):
        if self.pipe:
            self.pipe.enabled = not on

    def stop(self):
        self.timer.stop()
        if self.pipe:
            self.pipe.stop()
            self.pipe = None
        self.msg.setText("Stopped. The camera is released.")
        self._refresh()

    def _tick(self):
        p = self.pipe
        if p is None:
            return
        r = p.result(self._seq)
        if r:
            frame, self._seq = r
            self.view.set_image(frame)
        st = p.stats
        g = self.gpu.info
        gpu = f" · GPU {g['util']}% {g['mem_used']}/{g['mem_total']} MB" if g else ""
        self.stats.setText(f"{st.fps:4.1f} FPS · latency {st.latency_ms:4.0f} ms · swap {st.process_ms:4.0f} ms · "
                           f"cam {st.capture_fps:3.0f} fps · faces {st.faces} · {p.engine.provider}"
                           f"{' · adaptive L' + str(st.adaptive_level) if st.adaptive_level else ''}{gpu}")
        if p.error:
            self.msg.setText(p.error)
            if not p.running:
                self.stop()
                self.msg.setText(p.error)

    def shutdown(self):
        self.stop()
        self.gpu.stop()
