"""Entry point.

  python -m facefusion_live                      # desktop app
  python -m facefusion_live swap-image  -s src.jpg -t target.jpg -o out.png --consent
  python -m facefusion_live swap-video  -s src.jpg -t in.mp4 -o out.mp4 --consent [--all-faces]
  python -m facefusion_live models [--download NAME ... | --required] [--yes]
  python -m facefusion_live benchmark [-t video.mp4]
  python -m facefusion_live diagnostics
  python -m facefusion_live selftest              # used by the Windows build smoke test
"""
from __future__ import annotations

import argparse
import logging
import sys

from facefusion_live import APP_NAME, __version__

log = logging.getLogger("facefusion_live")


def _engine(args):
    from facefusion_live.app.config import load_settings
    from facefusion_live.core.engine import FaceEngine
    s = load_settings()
    for k in ("provider", "enhancer", "pixel_boost", "blend", "detector"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(s, k, v)
    for k in ("mouth_preserve", "occlusion_mask", "region_mask"):
        if getattr(args, k, False):
            setattr(s, k, True)
    return FaceEngine(s)


def _need_consent(args):
    if not args.consent:
        sys.exit("Refusing to run: pass --consent to confirm every person whose face is used or replaced has consented.")


def _source(eng, path, index):
    from facefusion_live.core.safety import read_image
    img = read_image(path)
    eng.check_content(img, "source photo")
    faces = eng.analyze(img)
    if not faces:
        sys.exit(f"No face found in source image {path}")
    return eng.make_source(img, faces[min(index, len(faces) - 1)], "source")


def cmd_swap_image(args):
    _need_consent(args)
    from facefusion_live.core.safety import read_image, save_labeled_image
    eng = _engine(args)
    src = _source(eng, args.source, args.source_face)
    img = read_image(args.target)
    eng.check_content(img, "target photo")
    faces = eng.analyze(img, embed=False)
    if not faces:
        sys.exit("No face found in target image")
    idx = range(len(faces)) if args.all_faces else [min(args.target_face, len(faces) - 1)]
    out = eng.process(img, [(faces[i], src) for i in idx])
    save_labeled_image(out, args.output)
    print(f"Saved {args.output} ({len(list(idx))} face(s) swapped)")


def cmd_swap_video(args):
    _need_consent(args)
    from facefusion_live.core.mapping import FaceMapping
    from facefusion_live.video.processor import process_video, scan_people
    eng = _engine(args)
    src = _source(eng, args.source, args.source_face)
    if args.all_faces:
        m = FaceMapping(mode="all", default=src)
    else:
        people = scan_people(eng, args.target)
        if not people:
            sys.exit("No faces found in the video")
        p = people[min(args.person, len(people) - 1)]
        print(f"Found {len(people)} people; replacing person #{args.person} (seen in {p.count} samples)")
        m = FaceMapping(mode="mapped", threshold=eng.settings.similarity_threshold)
        m.add(p.embedding, src)

    def prog(p):
        print(f"\r{p.done}/{p.total} frames  {p.fps:.1f} fps  ETA {p.eta:.0f}s   ", end="", flush=True)

    process_video(eng, m, args.target, args.output, progress=prog)
    print(f"\nSaved {args.output}")


def cmd_models(args):
    from facefusion_live.app.config import load_settings
    from facefusion_live.models.manifest import MODELS, human_size
    from facefusion_live.models.store import ModelStore
    s = load_settings()
    store = ModelStore(s.resolved_models_dir())
    names = list(args.download or [])
    if args.required:
        names += [m.name for m in MODELS.values() if m.required] + ["inswapper_128_fp16" if args.gpu else "inswapper_128"]
    if not names:
        print(f"Models folder: {store.dir}")
        for m in MODELS.values():
            print(f"  [{'x' if store.is_installed(m.name) else ' '}] {m.name:20s} {human_size(m.size):>9s}  {m.purpose}  ({m.license})")
        return
    names = [n for n in dict.fromkeys(names) if not store.is_installed(n)]
    if not names:
        print("Already installed.")
        return
    total = sum(MODELS[n].size for n in names)
    print("To download:\n" + "\n".join(f"  {n} ({human_size(MODELS[n].size)})" for n in names) + f"\nTotal {human_size(total)}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return
    for n in names:
        store.download(n, lambda d, t, n=n: print(f"\r{n}: {d * 100 // t}%", end="", flush=True))
        print()


def cmd_benchmark(args):
    from facefusion_live.utils.benchmark import run_benchmark
    run_benchmark(_engine(args), args.target, args.frames)


def cmd_diag(_):
    from facefusion_live.utils.gpu import diagnostics_text
    print(diagnostics_text())


def cmd_selftest(args):
    """Launch the GUI headlessly, build all tabs, run one real swap if models exist, exit 0."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from facefusion_live.utils.selftest import run_selftest
    sys.exit(run_selftest(args.image, report=args.report))


def main(argv=None):
    from facefusion_live.utils.log import setup_logging
    p = argparse.ArgumentParser(prog="facefusion_live", description=f"{APP_NAME} {__version__}")
    p.add_argument("--debug", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--provider", choices=["auto", "cuda", "directml", "cpu"])
        sp.add_argument("--enhancer", choices=["none", "gpen_bfr_256", "gpen_bfr_512", "gfpgan_1.4", "codeformer"])
        sp.add_argument("--pixel-boost", dest="pixel_boost", type=int, choices=[1, 2, 4])
        sp.add_argument("--blend", type=float)
        sp.add_argument("--detector", choices=["retinaface_10g", "scrfd_2.5g"])
        sp.add_argument("--mouth-preserve", dest="mouth_preserve", action="store_true")
        sp.add_argument("--occlusion-mask", dest="occlusion_mask", action="store_true")
        sp.add_argument("--region-mask", dest="region_mask", action="store_true")

    for name, fn in (("swap-image", cmd_swap_image), ("swap-video", cmd_swap_video)):
        sp = sub.add_parser(name)
        sp.add_argument("-s", "--source", required=True)
        sp.add_argument("-t", "--target", required=True)
        sp.add_argument("-o", "--output", required=True)
        sp.add_argument("--source-face", type=int, default=0)
        sp.add_argument("--target-face", type=int, default=0)
        sp.add_argument("--person", type=int, default=0, help="video: index of person from scan (0 = most frequent)")
        sp.add_argument("--all-faces", action="store_true")
        sp.add_argument("--consent", action="store_true")
        common(sp)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("models")
    sp.add_argument("--download", nargs="*")
    sp.add_argument("--required", action="store_true")
    sp.add_argument("--gpu", action="store_true")
    sp.add_argument("-y", "--yes", action="store_true")
    sp.set_defaults(fn=cmd_models)
    sp = sub.add_parser("benchmark")
    sp.add_argument("-t", "--target")
    sp.add_argument("--frames", type=int, default=60)
    common(sp)
    sp.set_defaults(fn=cmd_benchmark)
    sub.add_parser("diagnostics").set_defaults(fn=cmd_diag)
    sp = sub.add_parser("selftest")
    sp.add_argument("--image")
    sp.add_argument("--report", help="also write results to this file")
    sp.set_defaults(fn=cmd_selftest)

    args = p.parse_args(argv)
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    if not args.cmd:
        from facefusion_live.app.gui import run_gui
        sys.exit(run_gui())
    args.fn(args)


if __name__ == "__main__":
    main()
