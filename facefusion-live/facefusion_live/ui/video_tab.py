"""VIDEO SWAP tab: scan people, preview a segment, process with tracking, export with audio."""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
                               QProgressBar, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)

from facefusion_live.core.mapping import FaceMapping
from facefusion_live.ui.source_panel import SourcePanel
from facefusion_live.ui.widgets import VIDEO_FILTER, ConsentBox, ImageView, face_thumb_button, hbox, run_task, vbox
from facefusion_live.video import ffmpeg as ff
from facefusion_live.video.processor import Cancelled, default_output_path, process_video, scan_people


def fmt_time(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


class VideoTab(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.info = None
        self.people = []
        self.person_src: dict = {}   # person index -> SourceIdentity
        self.cancel = threading.Event()
        self.busy = False
        self.result_path = None
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ffl_ui_"))
        self._player = None
        self._latest_frame = None

        self.src = SourcePanel(ctx)
        # targets
        tbox = QGroupBox("Target video(s)")
        self.targets = QListWidget()
        self.targets.setMaximumHeight(90)
        add, rem = QPushButton("Add video…"), QPushButton("Remove")
        self.meta = QLabel("MP4, MOV, AVI, MKV. Add several for batch processing.")
        self.meta.setObjectName("muted")
        self.meta.setWordWrap(True)
        tbox.setLayout(vbox(hbox(add, rem, 1), self.targets, self.meta))
        # people
        pbox = QGroupBox("People in the video")
        self.scan_btn = QPushButton("Scan for faces")
        self.all_faces = QCheckBox("Replace every face")
        self.people_row = QHBoxLayout()
        self.people_hint = QLabel("Scan, then click the person(s) to replace. Others are left untouched.")
        self.people_hint.setObjectName("muted")
        self.people_hint.setWordWrap(True)
        pw = QWidget()
        pw.setLayout(self.people_row)
        pbox.setLayout(vbox(hbox(self.scan_btn, self.all_faces, 1), pw, self.people_hint))

        left = QWidget()
        left.setLayout(vbox(self.src, tbox, pbox))

        # right side
        self.view = ImageView("Preview")
        self.pos = QSlider(Qt.Horizontal)
        self.pos.setRange(0, 1000)
        self.pos_lbl = QLabel("Preview start 0:00")
        self.pos_lbl.setObjectName("stat")
        self.consent = ConsentBox()
        self.preview_btn = QPushButton("Preview 3 s")
        self.process_btn = QPushButton("Process Video")
        self.process_btn.setObjectName("primary")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.export_btn = QPushButton("Export Video…")
        self.batch_btn = QPushButton("Batch → folder…")
        self.bar = QProgressBar()
        self.eta = QLabel("")
        self.eta.setObjectName("stat")
        right = QGroupBox("Preview & processing")
        right.setLayout(vbox(self.view, hbox(self.pos_lbl, self.pos), self.consent,
                             hbox(self.preview_btn, self.process_btn, self.cancel_btn, self.export_btn, self.batch_btn, 1),
                             self.bar, self.eta))
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([400, 800])
        QVBoxLayout(self).addWidget(split)

        add.clicked.connect(self.add_videos)
        rem.clicked.connect(lambda: self.targets.takeItem(self.targets.currentRow()))
        self.targets.currentRowChanged.connect(self.target_changed)
        self.scan_btn.clicked.connect(self.scan)
        self.preview_btn.clicked.connect(self.preview)
        self.process_btn.clicked.connect(self.process)
        self.cancel_btn.clicked.connect(self.cancel.set)
        self.export_btn.clicked.connect(self.export)
        self.batch_btn.clicked.connect(self.batch)
        self.pos.valueChanged.connect(self._pos_changed)
        self.pos.sliderReleased.connect(self._show_still)
        for w in (self.consent, self.all_faces):
            w.toggled.connect(lambda _: self._refresh())
        self.src.sourceChanged.connect(lambda _: self._refresh())
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._tick)
        self.ui_timer.start(100)
        self._refresh()

    # ---- state -----------------------------------------------------------------
    def current(self):
        it = self.targets.currentItem()
        return it.text() if it else None

    def _mapping(self) -> FaceMapping:
        m = FaceMapping(threshold=self.ctx.settings.similarity_threshold)
        if self.all_faces.isChecked():
            m.mode, m.default = "all", self.src.source
        else:
            m.mode = "mapped"
            for i, s in self.person_src.items():
                m.add(self.people[i].embedding, s)
        return m

    def _ready(self) -> bool:
        has_map = self.all_faces.isChecked() and self.src.source is not None or bool(self.person_src)
        return bool(self.current() and self.consent.isChecked() and has_map and not self.busy)

    def _refresh(self):
        r = self._ready()
        self.preview_btn.setEnabled(r)
        self.process_btn.setEnabled(r)
        self.batch_btn.setEnabled(r and self.targets.count() > 0)
        self.cancel_btn.setEnabled(self.busy)
        self.export_btn.setEnabled(bool(self.result_path) and not self.busy)
        self.scan_btn.setEnabled(bool(self.current()) and not self.busy)
        self.all_faces.setEnabled(not self.busy)

    # ---- targets ------------------------------------------------------------------
    def add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", VIDEO_FILTER)
        for p in paths:
            self.targets.addItem(p)
        if paths:
            self.targets.setCurrentRow(self.targets.count() - len(paths))

    def target_changed(self, _row):
        path = self.current()
        self._clear_people()
        self.result_path = None
        if not path:
            self._refresh()
            return
        try:
            self.info = ff.probe(path)
        except Exception as e:  # noqa: BLE001
            self.meta.setText(str(e))
            self.info = None
            self._refresh()
            return
        i = self.info
        self.meta.setText(f"{i.width}×{i.height} · {float(i.fps):.2f} fps · {fmt_time(i.duration)} · "
                          f"{i.frames} frames · audio: {'yes' if i.has_audio else 'no'}")
        self._show_still()
        self._refresh()

    def _pos_changed(self, v):
        if self.info:
            self.pos_lbl.setText(f"Preview start {fmt_time(self.info.duration * v / 1000)}")

    def _start_time(self) -> float:
        return self.info.duration * self.pos.value() / 1000 if self.info else 0.0

    def _show_still(self):
        if not self.info:
            return
        from facefusion_live.video.processor import iter_frames
        fr = next(iter_frames(self.current(), self.info, self._start_time(), 0.2), None)
        if fr is not None:
            self.view.set_image(fr)

    # ---- people -------------------------------------------------------------------------
    def _clear_people(self):
        self.people, self.person_src = [], {}
        while self.people_row.count():
            w = self.people_row.takeAt(0).widget()
            if w:
                w.deleteLater()

    def scan(self):
        path = self.current()
        self._clear_people()
        self.people_hint.setText("Scanning…")
        self.scan_btn.setEnabled(False)
        run_task(lambda _: scan_people(self.ctx.engine(), path), self._scanned, self._fail)

    def _scanned(self, people):
        self.people = people
        for i, p in enumerate(people[:8]):
            b = face_thumb_button(p.thumbnail, f"Seen in {p.count} sampled frames")
            b.clicked.connect(lambda checked, i=i, b=b: self._toggle_person(i, b, checked))
            self.people_row.addWidget(b)
        self.people_row.addStretch(1)
        self.people_hint.setText(f"{len(people)} people found. Click the person(s) to replace with the current source."
                                 if people else "No faces found in the sampled frames.")
        if len(people) == 1 and self.src.source is not None:
            self.people_row.itemAt(0).widget().click()
        self._refresh()

    def _toggle_person(self, i, btn, checked):
        if checked and self.src.source is not None:
            self.person_src[i] = self.src.source
        else:
            self.person_src.pop(i, None)
            btn.setChecked(False)
        self._refresh()

    # ---- processing ----------------------------------------------------------------------
    def _run(self, target, out, start=0.0, duration=None, then=None):
        self.busy = True
        self.cancel.clear()
        self.bar.setValue(0)
        self._refresh()
        mapping = self._mapping()
        eng = self.ctx.engine()

        def work(emit):
            return process_video(eng, mapping, target, out, progress=emit, cancel=self.cancel,
                                 start=start, duration=duration, on_frame=self._set_latest)

        def done(path):
            self.busy = False
            self._refresh()
            if then:
                then(path)

        def fail(msg):
            self.busy = False
            self._refresh()
            if self.cancel.is_set():
                self.eta.setText("Cancelled. Partial output was deleted.")
            else:
                self._fail(msg)

        run_task(work, done, fail, self._progress)

    def _set_latest(self, frame):
        self._latest_frame = frame  # read by the UI timer; avoids flooding the event loop

    def _tick(self):
        if self._latest_frame is not None:
            self.view.set_image(self._latest_frame)
            self._latest_frame = None
        if self._player is not None:
            ok, fr = self._player.read()
            if ok:
                self.view.set_image(fr)
            else:
                self._player.release()
                self._player = None
                self.ui_timer.setInterval(100)

    def _progress(self, p):
        self.bar.setMaximum(max(p.total, 1))
        self.bar.setValue(p.done)
        self.eta.setText(f"{p.done}/{p.total} frames · {p.fps:.1f} fps · remaining {fmt_time(p.eta)}")

    def preview(self):
        out = str(self.tmpdir / f"preview_{int(time.time())}.mp4")
        self._run(self.current(), out, self._start_time(), 3.0, then=self._play)

    def _play(self, path):
        self.eta.setText("Preview ready (playing).")
        self._player = cv2.VideoCapture(path)
        self.ui_timer.setInterval(int(1000 / max(float(self.info.fps), 1)) if self.info else 40)

    def process(self):
        src = self.current()
        out = str(self.tmpdir / Path(default_output_path(src)).name)
        t0 = time.time()

        def done(path):
            self.result_path = path
            self.eta.setText(f"Finished in {fmt_time(time.time() - t0)}. Click Export Video to save it.")
            self._refresh()

        self._run(src, out, then=done)

    def export(self):
        if not self.result_path:
            return
        sug = default_output_path(self.current() or self.result_path)
        p, _ = QFileDialog.getSaveFileName(self, "Export video", sug, VIDEO_FILTER)
        if p:
            if not Path(p).suffix:
                p += Path(self.result_path).suffix
            shutil.copyfile(self.result_path, p)
            self.eta.setText(f"Exported {p}")

    def batch(self):
        folder = QFileDialog.getExistingDirectory(self, "Output folder for batch")
        if not folder:
            return
        items = [self.targets.item(i).text() for i in range(self.targets.count())]
        self.busy = True
        self.cancel.clear()
        self._refresh()
        mapping = self._mapping()
        eng = self.ctx.engine()

        def work(emit):
            outs = []
            for k, src in enumerate(items):
                if self.cancel.is_set():
                    raise Cancelled()
                outs.append(process_video(eng, mapping, src, default_output_path(src, folder), progress=emit,
                                          cancel=self.cancel, on_frame=self._set_latest))
            return outs

        def done(outs):
            self.busy = False
            self.eta.setText(f"Batch finished: {len(outs)} videos written to {folder}")
            self._refresh()

        def fail(msg):
            self.busy = False
            self._refresh()
            self.eta.setText("Batch cancelled." if self.cancel.is_set() else msg)

        run_task(work, done, fail, self._progress)

    def _fail(self, msg):
        self.people_hint.setText("")
        self.eta.setText(msg)
        QMessageBox.warning(self, "Video swap", msg)
        self._refresh()

    def shutdown(self):
        self.cancel.set()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
