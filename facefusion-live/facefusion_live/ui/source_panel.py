"""Source face picker shared by all three tabs."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout

from facefusion_live.core.engine import SourceIdentity
from facefusion_live.core.safety import read_image
from facefusion_live.ui.widgets import IMAGE_FILTER, FileRow, ImageView, run_task


class SourcePanel(QGroupBox):
    sourceChanged = Signal(object)  # SourceIdentity | None

    def __init__(self, ctx, title: str = "Source face", parent=None):
        super().__init__(title, parent)
        self.ctx = ctx
        self.img = None
        self.faces = []
        self.source: Optional[SourceIdentity] = None
        self.file = FileRow("Choose photo…", IMAGE_FILTER)
        self.view = ImageView("Photo of the face to use")
        self.view.setMinimumHeight(180)
        self.info = QLabel("")
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.file)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.info)
        self.file.chosen.connect(self.load)
        self.view.faceClicked.connect(self.select)

    def load(self, path: str) -> None:
        self.info.setText("Detecting faces…")
        self.source = None
        self.sourceChanged.emit(None)

        def work(_):
            img = read_image(path)
            eng = self.ctx.engine()
            eng.check_content(img, "source photo")
            return img, eng.analyze(img)

        run_task(work, self._loaded, self._failed)

    def _loaded(self, res) -> None:
        self.img, self.faces = res
        self.view.set_image(self.img, self.faces)
        if not self.faces:
            self.info.setText("No face found. Use a clear, well-lit, front-facing photo.")
            return
        self.select(0)

    def _failed(self, msg: str) -> None:
        self.info.setText(msg)
        self.info.setObjectName("error")
        self.info.setStyleSheet("color:#fca5a5")

    def select(self, i: int) -> None:
        if not self.faces:
            return
        self.info.setStyleSheet("")
        self.view.set_selected([i])
        self.source = self.ctx.engine().make_source(self.img, self.faces[i], f"face {i + 1}")
        n = len(self.faces)
        self.info.setText(f"Using face #{i + 1}" + (f" of {n} — click another box to change." if n > 1 else "."))
        self.sourceChanged.emit(self.source)
