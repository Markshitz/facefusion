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
        self._sync: dict = {}  # key -> function updating the widget from settings

        preset = QComboBox()
        for text, key in [("Choose a preset\u2026", ""), ("Best quality (photos, final videos)", "best"),
                          ("Balanced (video, default)", "balanced"), ("Fast (live webcam on CPU)", "fast")]:
            preset.addItem(text, key)
        preset.setToolTip("Sets detail, restoration and blending together. You can fine-tune below afterwards.")
        preset.currentIndexChanged.connect(lambda i: self.apply_preset(preset.itemData(i)))
        self.form.addRow("Quality preset", preset)

        self._combo("Enhancer", "enhancer", [("Off", "none"), ("GPEN 256 (sharp, default)", "gpen_bfr_256"), ("GPEN 512 (large faces)", "gpen_bfr_512"),
                                            ("GFPGAN 1.4", "gfpgan_1.4"), ("CodeFormer", "codeformer")])
        self._slider("Enhance strength", "enhance_strength", 0, 1)
        self._slider("Blend strength", "blend", 0, 1)
        self._slider("Edge feather", "mask_blur", 0, 1)
        self._slider("Skin tone match", "color_match", 0, 1)
        self._slider("Forehead blend", "forehead_blend", 0, 0.5)
        self._combo("Detail (pixel boost)", "pixel_boost", [("128 px (fastest)", 1), ("256 px", 2), ("512 px (best)", 4)])
        self._check("Keep original lips", "mouth_preserve", "Exact lip sync: the target's own lips and mouth opening follow speech precisely")
        self._slider("Lip sync strength", "mouth_strength", 0, 1)
        self._combo("Mouth region", "mouth_region", [("Reshape swapped lips to the speech (best)", "warp"), ("Original lips + opening", "lips"), ("Teeth/opening only", "inner")])
        self._combo("Landmarks", "landmarker", [("Precise 68-point (follows lips/eyes)", "2dfan4"), ("Fast (detector points)", "none")])
        self._check("Zero-lag video smoothing", "offline_smoothing", "Video export: smooth using neighbouring frames, no trailing behind movement")
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

    def apply_preset(self, name: str) -> None:
        from facefusion_live.app.config import PRESETS
        if name not in PRESETS:
            return
        for key, value in PRESETS[name].items():
            setattr(self.ctx.settings, key, value)
            if key in self._sync:
                self._sync[key](value)
        self.ctx.save()

    def _slider(self, label, key, lo, hi):
        sl = QSlider(Qt.Horizontal)
        sl.setRange(0, 100)
        sl.setValue(int((getattr(self.ctx.settings, key) - lo) / (hi - lo) * 100))
        self._sync[key] = lambda v, sl=sl: sl.setValue(int((v - lo) / (hi - lo) * 100))
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
        self._sync[key] = lambda v, cb=cb: cb.setChecked(bool(v))
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
        self._sync[key] = lambda v, cb=cb: cb.setCurrentIndex(max(cb.findData(v), 0))

        def on(i):
            setattr(self.ctx.settings, key, cb.itemData(i))
            self._changed()

        cb.currentIndexChanged.connect(on)
        self.form.addRow(label, cb)
