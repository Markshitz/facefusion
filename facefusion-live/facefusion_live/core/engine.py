"""Shared face-swap engine used by image, video and live modes."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from facefusion_live.app.config import Settings
from facefusion_live.core.align import warp_face
from facefusion_live.core.blend import box_mask, color_transfer, forehead_fade, paste_back, restore_mouth
from facefusion_live.core.face import Face
from facefusion_live.inference.session import Sessions, provider_label, resolve_providers
from facefusion_live.models.store import ModelStore

log = logging.getLogger(__name__)


@dataclass
class SourceIdentity:
    embedding: np.ndarray
    latent: np.ndarray
    thumbnail: np.ndarray
    name: str = ""


class FaceEngine:
    def __init__(self, settings: Settings, store: Optional[ModelStore] = None):
        self.settings = settings
        self.store = store or ModelStore(settings.resolved_models_dir())
        self.providers = resolve_providers(settings.provider, settings.device_id)
        self.sessions = Sessions(self.providers)
        self._lock = threading.RLock()
        self._mods: dict = {}

    # ---- lazy model access -------------------------------------------------
    @property
    def provider(self) -> str:
        return provider_label(self.providers)

    def swapper_name(self) -> str:
        s = self.settings.swapper
        if s != "auto":
            return s
        gpu = self.provider in ("CUDA", "Tensorrt", "Dml")
        pref = ["inswapper_128_fp16", "inswapper_128"] if gpu else ["inswapper_128", "inswapper_128_fp16"]
        for name in pref:
            if self.store.is_installed(name):
                return name
        return pref[0]

    def required_models(self) -> list[str]:
        return [self.settings.detector, "arcface_w600k_r50", self.swapper_name()]

    def _mod(self, key: str, factory):
        m = self._mods.get(key)
        if m is None:
            with self._lock:
                m = self._mods.get(key)
                if m is None:
                    m = factory()
                    self._mods[key] = m
        return m

    @property
    def detector(self):
        from facefusion_live.face_detection.scrfd import SCRFD
        name = self.settings.detector
        return self._mod("det:" + name, lambda: SCRFD(self.sessions, self.store.require(name)))

    @property
    def embedder(self):
        from facefusion_live.face_swap.embedder import ArcFace
        return self._mod("arcface", lambda: ArcFace(self.sessions, self.store.require("arcface_w600k_r50")))

    @property
    def swapper(self):
        from facefusion_live.face_swap.inswapper import InSwapper
        name = self.swapper_name()
        return self._mod("swap:" + name, lambda: InSwapper(self.sessions, self.store.require(name)))

    @property
    def landmarker(self):
        from facefusion_live.face_detection.landmarks import Landmark68From5
        return self._mod("lm68", lambda: Landmark68From5(self.sessions, self.store.require("fan_68_5")))

    @property
    def fan68(self):
        from facefusion_live.face_detection.fan68 import FAN68
        return self._mod("fan68", lambda: FAN68(self.sessions, self.store.require("2dfan4")))

    def refine(self, img: np.ndarray, face: Face) -> Face:
        """Measure 68 landmarks and align with points derived from them (eyes, nose tip, mouth corners).

        The detector's 5 points are coarse; measured landmarks follow lips and head motion precisely.
        """
        if self.settings.landmarker == "2dfan4" and self.store.is_installed("2dfan4"):
            from facefusion_live.face_detection.fan68 import five_from_68
            lm, score = self.fan68(img, face.bbox)
            if score > 0.25:  # unreliable on heavy occlusion / extreme pose: keep detector points
                face.landmarks68 = lm
                face.kps = five_from_68(lm)
        return face

    def restorer(self, name: str):
        from facefusion_live.enhancement.restorer import FaceRestorer
        return self._mod("enh:" + name, lambda: FaceRestorer(self.sessions, self.store.require(name), name))

    @property
    def occluder(self):
        from facefusion_live.core.masks import OcclusionMasker
        return self._mod("xseg", lambda: OcclusionMasker(self.sessions, self.store.require("xseg_1")))

    @property
    def parser(self):
        from facefusion_live.core.masks import RegionMasker
        return self._mod("bisenet", lambda: RegionMasker(self.sessions, self.store.require("bisenet_resnet_34")))

    @property
    def nsfw(self):
        from facefusion_live.core.safety import NSFWChecker
        return self._mod("nsfw", lambda: NSFWChecker(self.sessions, self.store.require("open_nsfw")))

    def check_content(self, img: np.ndarray, what: str) -> None:
        if self.settings.nsfw_check and self.store.is_installed("open_nsfw"):
            self.nsfw.check(img, what)

    def warmup(self) -> None:
        dummy = np.zeros((256, 256, 3), np.uint8)
        cv2.circle(dummy, (128, 128), 60, (180, 180, 200), -1)
        self.detector.detect(dummy, 320)
        kps = np.array([[100, 110], [156, 110], [128, 140], [105, 170], [151, 170]], np.float32)
        emb = self.embedder.embed(dummy, kps)
        self.swap_face(dummy, Face(np.array([60, 60, 196, 196], np.float32), kps, 1.0), self.make_source_from_embedding(emb))

    # ---- analysis ------------------------------------------------------------
    def detect(self, img: np.ndarray, size: Optional[int] = None, max_faces: int = 0) -> list[Face]:
        return self.detector.detect(img, size or self.settings.det_size, self.settings.det_threshold, max_faces)

    def analyze(self, img: np.ndarray, embed: bool = True) -> list[Face]:
        faces = self.detect(img)
        if embed:
            for f in faces:
                f.embedding = self.embedder.embed(img, f.kps)
        return faces

    def make_source_from_embedding(self, emb: np.ndarray, thumb: Optional[np.ndarray] = None, name: str = "") -> SourceIdentity:
        return SourceIdentity(emb, self.swapper.source_latent(emb),
                              thumb if thumb is not None else np.zeros((112, 112, 3), np.uint8), name)

    def make_source(self, img: np.ndarray, face: Face, name: str = "") -> SourceIdentity:
        emb = face.embedding if face.embedding is not None else self.embedder.embed(img, face.kps)
        thumb, _ = warp_face(img, face.kps, "arcface_128", 128)
        return self.make_source_from_embedding(emb, thumb, name)

    def make_average_source(self, items: list[tuple[np.ndarray, Face]], name: str = "") -> SourceIdentity:
        """Average identity from several photos of the same person (more robust identity)."""
        embs = [f.embedding if f.embedding is not None else self.embedder.embed(i, f.kps) for i, f in items]
        e = np.mean(embs, 0)
        e /= np.linalg.norm(e)
        thumb, _ = warp_face(items[0][0], items[0][1].kps, "arcface_128", 128)
        return self.make_source_from_embedding(e, thumb, name)

    # ---- swapping --------------------------------------------------------------
    def swap_face(self, frame: np.ndarray, target: Face, source: SourceIdentity,
                  original: Optional[np.ndarray] = None) -> None:
        """Swap one face in-place. `original` is the untouched frame used for masks."""
        s = self.settings
        original = frame if original is None else original
        boost = s.pixel_boost if s.pixel_boost in (1, 2, 4) else 1
        size = 128 * boost
        crop, m = warp_face(original, target.kps, "arcface_128", size)
        out = self.swapper.swap(crop, source.latent, boost)

        mask = box_mask(size, s.mask_blur, 0.0).copy()
        if s.forehead_blend > 0:
            mask *= forehead_fade(size, s.forehead_blend)
        if s.occlusion_mask and self.store.is_installed("xseg_1"):
            mask = np.minimum(mask, self.occluder(crop))
        if s.region_mask and self.store.is_installed("bisenet_resnet_34"):
            mask = np.minimum(mask, self.parser(crop))
        if s.color_match > 0:
            out = color_transfer(out, crop.astype(np.float32), mask, s.color_match)
        mask *= float(np.clip(s.blend, 0, 1))
        paste_back(frame, out, mask, m)

        if s.enhancer != "none" and self.store.is_installed(s.enhancer):
            self.restorer(s.enhancer).enhance(frame, target.kps, s.enhance_strength, s.codeformer_fidelity)

        # Real mouth last (after restoration, which could otherwise reshape it): the original teeth, tongue and
        # mouth opening follow speech exactly.
        if s.mouth_preserve:
            lm = target.landmarks68
            if lm is None and self.store.is_installed("fan_68_5"):
                lm = self.landmarker(target.kps)
            if lm is not None:
                restore_mouth(frame, original, lm, s.mouth_region, s.mouth_strength)

    def process(self, frame: np.ndarray, pairs: list[tuple[Face, SourceIdentity]]) -> np.ndarray:
        """Return a new frame with each (target, source) pair swapped."""
        if not pairs:
            return frame
        out = frame.copy()
        for target, source in pairs:
            self.swap_face(out, target, source, original=frame)
        return out
