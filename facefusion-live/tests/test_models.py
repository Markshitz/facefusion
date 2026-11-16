"""Real-inference tests. Need models (FFL_MODELS_DIR) and media (FFL_TEST_MEDIA with t1.jpg, lena.jpg)."""
import time

import cv2
import numpy as np
import pytest

from conftest import have_models, make_video, media, needs_models
from facefusion_live.core.mapping import FaceMapping
from facefusion_live.core.safety import read_image

pytestmark = needs_models


@pytest.fixture(scope="module")
def group():
    return read_image(media("t1.jpg"))


@pytest.fixture(scope="module")
def source(engine):
    img = read_image(media("lena.jpg"))
    return engine.make_source(img, engine.analyze(img)[0])


def test_detects_all_faces_in_group(engine, group):
    faces = engine.analyze(group)
    assert len(faces) == 6
    sims = np.array([f.embedding for f in faces]) @ np.array([f.embedding for f in faces]).T
    assert (sims[~np.eye(6, dtype=bool)] < 0.4).all()  # distinct identities


def test_no_face(engine):
    assert engine.analyze(np.full((480, 640, 3), 90, np.uint8)) == []


def test_image_swap_changes_identity_and_keeps_others(engine, group, source):
    faces = engine.analyze(group)
    tgt = faces[2]
    out = engine.process(group, [(tgt, source)])
    new = {round(float(f.center[0])): f for f in engine.analyze(out)}
    swapped = min(new.values(), key=lambda f: np.linalg.norm(f.center - tgt.center))
    assert float(np.dot(swapped.embedding, source.embedding)) > 0.6 > float(np.dot(tgt.embedding, source.embedding))
    # pixels outside the swapped face are untouched
    x1, y1, x2, y2 = tgt.bbox.astype(int)
    mask = np.ones(group.shape[:2], bool)
    pad = int((x2 - x1) * 0.6)
    mask[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = False
    assert (out[mask] == group[mask]).all()


def test_multi_face_different_sources(engine, group, source):
    faces = engine.analyze(group)
    src2 = engine.make_source(group, faces[5])
    out = engine.process(group, [(faces[0], source), (faces[1], src2)])
    after = engine.analyze(out)
    near = lambda f: min(after, key=lambda a: np.linalg.norm(a.center - f.center))  # noqa: E731
    assert float(np.dot(near(faces[0]).embedding, source.embedding)) > 0.6
    assert float(np.dot(near(faces[1]).embedding, src2.embedding)) > 0.6


@pytest.mark.parametrize("variant", ["dark", "rotated", "occluded"])
def test_robustness(engine, group, source, variant):
    img = group.copy()
    if variant == "dark":
        img = (img.astype(np.float32) * 0.3).astype(np.uint8)
    elif variant == "rotated":
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    else:
        f = engine.analyze(img)[2]
        x1, y1, x2, y2 = f.bbox.astype(int)
        img[(y1 + y2) // 2:y2, x1:x2] = (40, 120, 200)  # cover lower half of face
    faces = engine.analyze(img)
    assert len(faces) >= 4, variant
    out = engine.process(img, [(faces[0], source)])
    assert float(np.dot(min(engine.analyze(out), key=lambda a: np.linalg.norm(a.center - faces[0].center)).embedding,
                        source.embedding)) > 0.45


@pytest.mark.skipif(not have_models("fan_68_5", "xseg_1", "bisenet_resnet_34"), reason="mask models missing")
def test_all_quality_options(engine, group, source):
    s = engine.settings
    old = dict(vars(s))
    try:
        s.pixel_boost, s.mouth_preserve, s.occlusion_mask, s.region_mask, s.color_match = 2, True, True, True, 0.5
        f = engine.analyze(group)[2]
        out = engine.process(group, [(f, source)])
        assert (out != group).any()
    finally:
        vars(s).update(old)


@pytest.mark.skipif(not have_models("gfpgan_1.4"), reason="gfpgan missing")
def test_enhancer(engine, group, source):
    s = engine.settings
    s.enhancer = "gfpgan_1.4"
    try:
        f = engine.analyze(group)[2]
        a = engine.process(group, [(f, source)])
        s.enhance_strength = 0.0
        b = engine.process(group, [(f, source)])
        assert (a != b).any()
    finally:
        s.enhancer, s.enhance_strength = "none", 0.8


@pytest.mark.skipif(not have_models("open_nsfw"), reason="nsfw model missing")
def test_nsfw_check_passes_normal_photo(engine, group):
    engine.check_content(group, "photo")  # must not raise


def test_video_swaps_only_selected_person_with_tracking(engine, group, source, tmp_path):
    from facefusion_live.video import ffmpeg as ff
    from facefusion_live.video.processor import process_video, scan_people
    vid = make_video(tmp_path / "g.mp4", group, frames=24, fps=12, size=(960, 540))
    people = scan_people(engine, vid, samples=4)
    assert len(people) >= 5
    m = FaceMapping(mode="mapped")
    m.add(people[0].embedding, source)
    out = process_video(engine, m, vid, str(tmp_path / "o.mp4"))
    info = ff.probe(out)
    assert info.has_audio and info.frames == 24 and float(info.fps) == 12
    cap = cv2.VideoCapture(out)
    hits = 0
    for _ in range(24):
        ok, fr = cap.read()
        faces = engine.analyze(fr)
        sims = sorted((float(np.dot(f.embedding, source.embedding)) for f in faces), reverse=True)
        hits += sims[0] > 0.5 and sims[1] < 0.3  # exactly one face carries the source identity
    assert hits >= 22


def test_live_pipeline_real_inference(engine, group, source, tmp_path):
    from facefusion_live.camera.live import LivePipeline
    vid = make_video(tmp_path / "cam.mp4", group, frames=30, fps=15, size=(960, 540), audio=False)
    lp = LivePipeline(engine, FaceMapping(mode="largest", default=source), vid)
    lp.start()
    deadline = time.time() + 30
    while lp.result() is None and time.time() < deadline:
        time.sleep(0.2)
    time.sleep(2)
    res = lp.result()
    lp.stop()
    assert res is not None and lp.error is None
    faces = engine.analyze(res[0])
    assert max(float(np.dot(f.embedding, source.embedding)) for f in faces) > 0.5
