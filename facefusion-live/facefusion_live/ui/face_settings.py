"""Quality controls (shared dock). Changes apply live to all modes."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QScrollArea, QSlider, QWidget


class FaceSettingsPanel(QScrollArea):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWidgetResizable(True)
        body = QWidget()
        self.form = QFormLayout(body)
        self.form.setLabelAlignment(Qt.AlignLeft)
        self.setWidget(body)
        s = ctx.settings

        self._combo("Enhancer", "enhancer", [("Off", "none"), ("GFPGAN 1.4", "gfpgan_1.4"), ("CodeFormer", "codeformer")])
        self._slider("Enhance strength", "enhance_strength", 0, 1)
        self._slider("Blend strength", "blend", 0, 1)
        self._slider("Edge feather", "mask_blur", 0, 1)
        self._slider("Skin tone match", "color_match", 0, 1)
        self._combo("Detail (pixel boost)", "pixel_boost", [("128 px (fastest)", 1), ("256 px", 2), ("512 px (best)", 4)])
        self._check("Preserve mouth", "mouth_preserve", "Keep the original mouth interior (teeth, tongue) for natural speech")
        self._check("Occlusion mask", "occlusion_mask", "Keep hands/objects in front of the face (XSeg model)")
        self._check("Face-region mask", "region_mask", "Only replace skin/eyes/nose/mouth; keep hair and beard edges (BiSeNet)")
        self._check("Face tracking", "tracking", "Track faces between detections (faster, steadier)")
        self._slider("Temporal smoothing", "smoothing", 0, 1)
        note = QLabel("Optional masks/enhancers need their models (Settings > Models).")
        note.setObjectName("muted")
        note.setWordWrap(True)
        self.form.addRow(note)
        _ = s

    def _changed(self):
        self.ctx.save()

    def _slider(self, label, key, lo, hi):
        sl = QSlider(Qt.Horizontal)
        sl.setRange(0, 100)
        sl.setValue(int((getattr(self.ctx.settings, key) - lo) / (hi - lo) * 100))
        val = QLabel(f"{getattr(self.ctx.settings, key):.2f}")
        val.setObjectName("stat")

        def on(v):
            x = lo + (hi - lo) * v / 100
            setattr(self.ctx.settings, key, x)
            val.setText(f"{x:.2f}")

        sl.valueChanged.connect(on)
        sl.sliderReleased.connect(self._changed)
        w = QWidget()
        from facefusion_live.ui.widgets import hbox
        w.setLayout(hbox(sl, val))
        self.form.addRow(label, w)

    def _check(self, label, key, tip):
        cb = QCheckBox()
        cb.setChecked(bool(getattr(self.ctx.settings, key)))
        cb.setToolTip(tip)

        def on(v):
            setattr(self.ctx.settings, key, bool(v))
            self._changed()

        cb.toggled.connect(on)
        lbl = QLabel(label)
        lbl.setToolTip(tip)
        self.form.addRow(lbl, cb)

    def _combo(self, label, key, items):
        cb = QComboBox()
        for text, data in items:
            cb.addItem(text, data)
        idx = cb.findData(getattr(self.ctx.settings, key))
        cb.setCurrentIndex(max(idx, 0))

        def on(i):
            setattr(self.ctx.settings, key, cb.itemData(i))
            self._changed()

        cb.currentIndexChanged.connect(on)
        self.form.addRow(label, cb)
