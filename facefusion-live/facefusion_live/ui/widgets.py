"""Reusable Qt widgets and a background-task helper."""
from __future__ import annotations

import logging
import traceback
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from facefusion_live.core.face import Face

log = logging.getLogger(__name__)
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)"


def to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, img.strides[0], QImage.Format_BGR888).copy()


# ---- background tasks ---------------------------------------------------------
class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class Task(QRunnable):
    """Run fn(progress_emit) off the UI thread; result/failure delivered on the UI thread."""

    def __init__(self, fn: Callable, on_done=None, on_fail=None, on_progress=None):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()
        if on_done:
            self.signals.done.connect(on_done)
        if on_fail:
            self.signals.failed.connect(on_fail)
        if on_progress:
            self.signals.progress.connect(on_progress)

    def run(self):
        try:
            self.signals.done.emit(self.fn(self.signals.progress.emit))
        except Exception as e:  # noqa: BLE001
            log.error("Task failed: %s\n%s", e, traceback.format_exc())
            self.signals.failed.emit(str(e) or e.__class__.__name__)


_live_tasks: set = set()


def run_task(fn, on_done=None, on_fail=None, on_progress=None) -> Task:
    t = Task(fn, on_done, on_fail, on_progress)
    t.setAutoDelete(False)
    _live_tasks.add(t)
    t.signals.done.connect(lambda *_: _live_tasks.discard(t))
    t.signals.failed.connect(lambda *_: _live_tasks.discard(t))
    QThreadPool.globalInstance().start(t)
    return t


# ---- image display -------------------------------------------------------------
class ImageView(QWidget):
    """Aspect-correct image display with optional clickable face boxes."""
    faceClicked = Signal(int)

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 160)
        self._img: Optional[QImage] = None
        self._faces: list[Face] = []
        self._selected: set[int] = set()
        self.placeholder = placeholder
        self.labels: dict[int, str] = {}

    def set_image(self, img: Optional[np.ndarray], faces: Optional[list[Face]] = None) -> None:
        self._img = to_qimage(img) if img is not None else None
        self._faces = faces or []
        self.update()

    def set_selected(self, sel) -> None:
        self._selected = set(sel)
        self.update()

    def _rect(self) -> QRectF:
        if self._img is None:
            return QRectF()
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0f1115"))
        if self._img is None:
            p.setPen(QColor("#6b7280"))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        r = self._rect()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(r, self._img)
        s = r.width() / self._img.width()
        for i, f in enumerate(self._faces):
            sel = i in self._selected
            p.setPen(QPen(QColor("#22c55e" if sel else "#60a5fa"), 3 if sel else 2))
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            p.drawRect(QRectF(r.x() + x1 * s, r.y() + y1 * s, (x2 - x1) * s, (y2 - y1) * s))
            p.drawText(QPointF(r.x() + x1 * s + 3, r.y() + y1 * s - 4), self.labels.get(i, f"#{i + 1}"))

    def mousePressEvent(self, e):
        if not self._faces or self._img is None:
            return
        r = self._rect()
        s = r.width() / self._img.width()
        x, y = (e.position().x() - r.x()) / s, (e.position().y() - r.y()) / s
        for i, f in enumerate(self._faces):
            if f.bbox[0] <= x <= f.bbox[2] and f.bbox[1] <= y <= f.bbox[3]:
                self.faceClicked.emit(i)
                return


class CompareView(ImageView):
    """Before/after view with a draggable split line."""

    def __init__(self, parent=None):
        super().__init__("Result appears here", parent)
        self._before: Optional[QImage] = None
        self.split = 0.5

    def set_pair(self, before: Optional[np.ndarray], after: Optional[np.ndarray]) -> None:
        self._before = to_qimage(before) if before is not None else None
        self.set_image(after)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._img is None or self._before is None:
            return
        p = QPainter(self)
        r = self._rect()
        sx = r.x() + r.width() * self.split
        src_w = self._before.width() * self.split
        p.drawImage(QRectF(r.x(), r.y(), sx - r.x(), r.height()), self._before,
                    QRectF(0, 0, src_w, self._before.height()))
        p.setPen(QPen(QColor("#f8fafc"), 2))
        p.drawLine(QPointF(sx, r.y()), QPointF(sx, r.bottom()))
        p.setPen(QColor("#f8fafc"))
        p.drawText(QPointF(r.x() + 8, r.y() + 18), "Before")
        p.drawText(QPointF(r.right() - 44, r.y() + 18), "After")

    def mousePressEvent(self, e):
        self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        r = self._rect()
        if r.width() > 0:
            self.split = float(np.clip((e.position().x() - r.x()) / r.width(), 0, 1))
            self.update()


def face_thumb_button(img: np.ndarray, tip: str = "") -> QToolButton:
    b = QToolButton()
    b.setCheckable(True)
    b.setIcon(QPixmap.fromImage(to_qimage(cv2.resize(img, (72, 72)))))
    b.setIconSize(b.iconSize().expandedTo(b.iconSize().__class__(72, 72)))
    b.setToolTip(tip)
    return b


class FileRow(QWidget):
    """[label] [path text] [Browse]"""
    chosen = Signal(str)

    def __init__(self, title: str, filt: str, save: bool = False, parent=None):
        super().__init__(parent)
        self.filt, self.save, self.path = filt, save, ""
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn = QPushButton(title)
        self.lbl = QLabel("No file selected")
        self.lbl.setObjectName("muted")
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.btn)
        lay.addWidget(self.lbl, 1)
        self.btn.clicked.connect(self._browse)

    def _browse(self):
        if self.save:
            p, _ = QFileDialog.getSaveFileName(self, self.btn.text(), self.path, self.filt)
        else:
            p, _ = QFileDialog.getOpenFileName(self, self.btn.text(), self.path, self.filt)
        if p:
            self.set_path(p)

    def set_path(self, p: str) -> None:
        self.path = p
        self.lbl.setText(p)
        self.chosen.emit(p)


class ConsentBox(QCheckBox):
    def __init__(self, parent=None):
        super().__init__("I have the consent of every person whose face is used or replaced", parent)
        self.setToolTip("Required. Do not use someone's likeness without permission, and never for "
                        "deception, harassment, fraud or intimate imagery.")


def vbox(*widgets, margins=0, spacing=8) -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, int):
            lay.addStretch(w)
        elif isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return lay


def hbox(*widgets, margins=0, spacing=8) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, int):
            lay.addStretch(w)
        elif isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return lay
