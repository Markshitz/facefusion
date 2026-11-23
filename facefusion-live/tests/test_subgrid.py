"""Pixel boost sub-grid assembly: no mesh pattern, broken batches are detected."""
import numpy as np

from facefusion_live.face_swap import subgrid


def _face(n=128, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    base = 120 + 60 * np.sin(xx / 9.0) * np.cos(yy / 13.0)
    img = np.repeat(base[..., None], 3, 2) + rng.normal(0, 3, (n, n, 3))
    return img.astype(np.float32)


def _checker(img):
    r = img.mean(2)
    return float(np.abs(r[0::2, 0::2] - r[0::2, 1::2] - r[1::2, 0::2] + r[1::2, 1::2]).mean())


def test_assemble_removes_mesh_from_offset_subgrids():
    ys = np.stack([_face(seed=k) + off for k, off in enumerate([0, 0, 0, 25])])
    naive = np.empty((256, 256, 3), np.float32)
    k = 0
    for i in range(2):
        for j in range(2):
            naive[i::2, j::2] = ys[k]
            k += 1
    fixed = subgrid.assemble((256, 256, 3), ys, 2)
    assert _checker(fixed) < _checker(naive) * 0.5


def test_mismatched_flags_only_broken_items():
    good = np.stack([_face(seed=k) for k in range(4)])
    assert subgrid.mismatched(good) == []
    broken = good.copy()
    broken[2] = np.roll(good[2], 8, axis=1)
    broken[3] = good[3] * 0.6
    assert subgrid.mismatched(broken) == [2, 3]
