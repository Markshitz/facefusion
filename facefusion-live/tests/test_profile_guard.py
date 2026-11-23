"""Turned heads: no mouth warp on unreliable landmarks, swap clipped to the face outline."""
import numpy as np

from facefusion_live.core.blend import PROFILE, face_hull_mask, frontalness, mouth_landmarks_ok


def _face(nose_x=50.0):
    jaw = np.stack([np.linspace(10, 90, 17), 40 + 40 * np.sin(np.linspace(0, np.pi, 17))], 1)
    lm = np.zeros((68, 2), np.float32)
    lm[0:17] = jaw
    lm[17:27] = np.stack([np.linspace(15, 85, 10), np.full(10, 30)], 1)
    lm[27:36] = [nose_x, 50]
    mouth = np.stack([35 + 30 * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, 20))), 70 + 4 * np.sin(np.linspace(0, 2 * np.pi, 20))], 1)
    lm[48:68] = mouth
    lm[48], lm[54] = [35, 70], [65, 70]
    return lm


def test_frontalness_separates_frontal_from_profile():
    assert frontalness(_face(50)) > 0.9
    assert frontalness(_face(84)) < PROFILE


def test_mouth_guard_rejects_far_off_landmarks():
    lm = _face()
    assert mouth_landmarks_ok(lm + 2, lm)
    bad = lm.copy()
    bad[48:68] += [25, 10]
    assert not mouth_landmarks_ok(bad, lm)


def test_hull_mask_covers_face_not_background():
    lm = _face() * 1.28  # into a 128 px crop
    m = face_hull_mask(lm, 128)
    assert m[int(lm[30, 1]), int(lm[30, 0])] > 0.95  # nose
    assert m[122, 5] < 0.01 and m[122, 122] < 0.01  # bottom corners (beside the jaw) are background
    assert m[2, 64] > 0.95  # open to the crop top: the forehead stays swapped
