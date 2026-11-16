"""IMAGE SWAP tab: source photo -> target photo, before/after, save."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QGroupBox, QLabel, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget

from facefusion_live.core.safety import read_image, save_labeled_image
from facefusion_live.ui.source_panel import SourcePanel
from facefusion_live.ui.widgets import IMAGE_FILTER, CompareView, ConsentBox, FileRow, ImageView, hbox, run_task, vbox


class ImageTab(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.target = None
        self.faces = []
        self.assign: dict = {}  # target face index -> SourceIdentity
        self.result = None

        self.src = SourcePanel(ctx)
        tgt_box = QGroupBox("Target photo")
        self.tfile = FileRow("Choose photo…", IMAGE_FILTER)
        self.tview = ImageView("Photo containing the face to replace")
        self.tinfo = QLabel("")
        self.tinfo.setObjectName("muted")
        self.tinfo.setWordWrap(True)
        tl = QVBoxLayout(tgt_box)
        tl.addWidget(self.tfile)
        tl.addWidget(self.tview, 1)
        tl.addWidget(self.tinfo)

        left = QSplitter()
        left.setOrientation(left.orientation().__class__.Vertical)
        left.addWidget(self.src)
        left.addWidget(tgt_box)

        self.compare = CompareView()
        self.consent = ConsentBox()
        self.swap_btn = QPushButton("Swap Face")
        self.swap_btn.setObjectName("primary")
        self.save_btn = QPushButton("Save Image…")
        self.use_btn = QPushButton("Use result as target")
        self.use_btn.setToolTip("Swap another face in the same photo with a different source")
        self.status = QLabel("")
        self.status.setObjectName("muted")
        res_box = QGroupBox("Result (drag to compare before/after)")
        res_box.setLayout(vbox(self.compare, self.consent, hbox(self.swap_btn, self.save_btn, self.use_btn, 1), self.status))

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(res_box)
        split.setSizes([380, 820])
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        self.tfile.chosen.connect(self.load_target)
        self.tview.faceClicked.connect(self.toggle_face)
        self.src.sourceChanged.connect(lambda _: self._refresh())
        self.consent.toggled.connect(lambda _: self._refresh())
        self.swap_btn.clicked.connect(self.swap)
        self.save_btn.clicked.connect(self.save)
        self.use_btn.clicked.connect(self.use_result)
        self._refresh()

    def _refresh(self):
        self.tview.labels = {i: f"#{i + 1} → {s.name}" for i, s in self.assign.items()}
        self.tview.set_selected(self.assign.keys())
        ready = bool(self.target is not None and self.consent.isChecked() and (self.assign or self.src.source))
        self.swap_btn.setEnabled(ready)
        self.save_btn.setEnabled(self.result is not None)
        self.use_btn.setEnabled(self.result is not None)

    def load_target(self, path: str):
        self.tinfo.setText("Detecting faces…")

        def work(_):
            img = read_image(path)
            eng = self.ctx.engine()
            eng.check_content(img, "target photo")
            return img, eng.analyze(img, embed=False)

        run_task(work, self._target_loaded, self._fail)

    def _target_loaded(self, res, keep_assign=False):
        self.target, self.faces = res
        if not keep_assign:
            self.assign.clear()
        self.result = None
        self.tview.set_image(self.target, self.faces)
        self.compare.set_pair(None, None)
        n = len(self.faces)
        if n == 0:
            self.tinfo.setText("No face found in the target photo.")
        elif n == 1:
            self.tinfo.setText("1 face found.")
        else:
            self.tinfo.setText(f"{n} faces found. Click a face to assign the current source to it "
                               "(click again to clear). Unselected faces stay untouched.")
        self._refresh()

    def toggle_face(self, i: int):
        src = self.src.source
        if i in self.assign and (src is None or self.assign[i] is src):
            del self.assign[i]
        elif src is not None:
            self.assign[i] = src
        self._refresh()

    def swap(self):
        if not self.faces:
            self.status.setText("No face in target photo.")
            return
        pairs_idx = dict(self.assign)
        if not pairs_idx:  # default: largest face gets the source
            pairs_idx = {0: self.src.source}
        pairs = [(self.faces[i], s) for i, s in pairs_idx.items()]
        self.swap_btn.setEnabled(False)
        self.status.setText("Swapping…")
        target = self.target
        run_task(lambda _: self.ctx.engine().process(target, pairs), self._swapped, self._fail)

    def _swapped(self, out):
        self.result = out
        self.compare.set_pair(self.target, out)
        self.status.setText("Done. AI label is added when saving.")
        self._refresh()

    def save(self):
        base = Path(self.tfile.path).stem + "_swapped.png" if self.tfile.path else "swapped.png"
        p, _ = QFileDialog.getSaveFileName(self, "Save image", str(Path(self.tfile.path).parent / base),
                                           "PNG (*.png);;JPEG (*.jpg);;WEBP (*.webp)")
        if not p:
            return
        if not Path(p).suffix:
            p += ".png"
        try:
            save_labeled_image(self.result, p)
            self.status.setText(f"Saved {p}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(e))

    def use_result(self):
        faces = self.faces
        self._target_loaded((self.result, faces), keep_assign=False)

    def _fail(self, msg):
        self.status.setText(msg)
        self.tinfo.setText("")
        QMessageBox.warning(self, "Image swap", msg)
        self._refresh()
