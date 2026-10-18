"""A small folder to hear the identity test: the same clip before and after the pipeline.

For each case, a few clips drawn at random (seeded -- not picked by ear), each as
  1_real.wav                 the real recording that went in
  2_identity_handoff100.wav  after the pipeline, target model on both sides, handoff 100
  3_identity_handoff5.wav    the same at handoff 5
plus a few real transfers (flute + bassoon -> violin + cello, cello -> bassoon) at
the same handoffs, from other clips, for comparison. Files are copied unchanged
-- no level matching -- so what you hear is what the pipeline wrote.

  venv_yuval/bin/python evaluation/make_identity_listen.py
  -> Outputs/identity_listen.zip
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IDENTITY = "Outputs/WAV_files/identity_test/audio"
TIMBRE = "Outputs/WAV_files/clap_validation_1000/fad"
CASES = {
    "poly_violin_cello": dict(pair="identity_vc_midi_s100", ens="violin_cello",
                              transfer_case="poly", transfer="original_midi_s100"),
    "mono_bassoon": dict(pair="identity_bsn_midi_s100", ens="bassoon",
                         transfer_case="mono", transfer="mono_c2b_original_midi_s100"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5, help="identity clips per case")
    p.add_argument("--n-transfer", type=int, default=2, help="transfer examples per case and handoff")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="Outputs/identity_listen")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    audio = Path(IDENTITY)
    lines = []
    for name, c in CASES.items():
        pre = c["pair"] + "_"
        tail_in = "_input_%s.wav" % c["ens"]
        tracks = sorted(p.name[len(pre):-len(tail_in)] for p in audio.glob(pre + "*" + tail_in)
                        if not p.name.startswith(c["pair"] + "_h"))
        if not tracks:
            print("%s: no identity outputs in %s -- skipped" % (name, audio))
            continue
        gen = "_generated_%s_from_%s.wav" % (c["ens"], c["ens"])
        for i, t in enumerate(sorted(rng.sample(tracks, min(args.n, len(tracks)))), 1):
            d = out / name / ("%02d_%s" % (i, t))
            d.mkdir(parents=True)
            shutil.copyfile(audio / (pre + t + tail_in), d / "1_real.wav")
            shutil.copyfile(audio / (pre + t + gen), d / "2_identity_handoff100.wav")
            shutil.copyfile(audio / (c["pair"] + "_h5_" + t + gen), d / "3_identity_handoff5.wav")
        tdir = out / name / "transfer_examples"
        tdir.mkdir(parents=True)
        for h, suffix in ((100, ""), (5, "_h5")):
            src = Path(TIMBRE) / c["transfer_case"] / "generated" / (c["transfer"] + suffix)
            clips = sorted(src.glob("*.wav"))
            for p in rng.sample(clips, min(args.n_transfer, len(clips))):
                shutil.copyfile(p.resolve(), tdir / ("handoff%d_%s.wav" % (h, p.stem)))
        lines.append("%s: %d clips x 3 versions, %d transfer examples" % (name, min(args.n, len(tracks)),
                                                                          len(list(tdir.glob("*.wav")))))
    (out / "README.txt").write_text(
        "Identity test - the same clip before and after the pipeline (instruments unchanged)\n\n"
        "1_real.wav                 the real recording that went in\n"
        "2_identity_handoff100.wav  after the pipeline, handoff 100\n"
        "3_identity_handoff5.wav    after the pipeline, handoff 5\n"
        "transfer_examples/         real transfers (other clips) at the same handoffs, for comparison\n\n"
        "Clips drawn at random (seed %d). Files unchanged, no level matching.\n\n%s\n"
        % (args.seed, "\n".join(lines)), encoding="utf-8")
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("\n".join(lines))
    print("-> %s (%.0f MB)" % (zip_path, Path(zip_path).stat().st_size / 1e6))


if __name__ == "__main__":
    main()
