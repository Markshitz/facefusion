"""Decides which source identity goes onto which detected face."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from facefusion_live.core.engine import SourceIdentity
from facefusion_live.core.face import Face


@dataclass
class FaceMapping:
    """mode:
    - "all":      every detected face gets `default`
    - "largest":  only the largest face gets `default` (live default)
    - "mapped":   each face is matched by identity to a reference embedding;
                  unmatched faces are left untouched (no accidental swaps)
    """
    mode: str = "largest"
    default: Optional[SourceIdentity] = None
    references: list[tuple[np.ndarray, SourceIdentity]] = field(default_factory=list)
    threshold: float = 0.35
    _sticky: dict = field(default_factory=dict)  # track_id -> reference index

    def add(self, reference_embedding: np.ndarray, source: SourceIdentity) -> None:
        self.references.append((reference_embedding, source))
        self._sticky.clear()

    def reset(self) -> None:
        self._sticky.clear()

    def needs_embeddings(self) -> bool:
        return self.mode == "mapped"

    def assign(self, faces: list[Face]) -> list[tuple[Face, SourceIdentity]]:
        if self.mode == "all":
            return [(f, self.default) for f in faces] if self.default else []
        if self.mode == "largest":
            return [(max(faces, key=lambda f: f.area), self.default)] if faces and self.default else []
        out, taken = [], set()
        scored = []
        for f in faces:
            if f.embedding is None:
                continue
            for ri, (ref, _) in enumerate(self.references):
                sim = float(np.dot(f.embedding, ref))
                if f.track_id >= 0 and self._sticky.get(f.track_id) == ri:
                    sim += 0.1  # hysteresis keeps assignments stable between frames
                scored.append((sim, id(f), ri, f))
        used_faces: set = set()
        for sim, fid, ri, f in sorted(scored, key=lambda x: -x[0]):
            if sim < self.threshold or fid in used_faces or ri in taken:
                continue
            used_faces.add(fid)
            taken.add(ri)
            if f.track_id >= 0:
                self._sticky[f.track_id] = ri
            out.append((f, self.references[ri][1]))
        return out
