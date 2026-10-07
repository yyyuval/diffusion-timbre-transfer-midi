"""CLAP-FAD for every experiment in the 200-clip bridge CSV.

One FAD per experiment: its generated clips vs a real reference set of the
TARGET ensemble -- violin+cello mixtures for poly, bassoon stems for mono --
drawn (seeded) from tracks that are not among any input list, and preprocessed
exactly like the bridge inputs (24 kHz, summed, 0.95 rescale only if a mixture
clips, crop from the start). Reference and generated embeddings are cached
next to the audio, so re-runs only embed what is new.

The embedder and the distance are imported from compute_clap_fad.py unchanged.

**On sample size.** CLAP embeddings are 512-D. A Frechet distance fits a full
covariance per side, so with fewer than 512 clips the estimate is singular and
the score is biased upward by an amount that depends mostly on n. Two
consequences: absolute values are meaningless below ~1000 per side, and
differences between conditions survive anyway *as long as every condition uses
the same n*. The bridge lists hold 1000 clips for exactly this reason. The
reference side is the one that may fall short, because the clips neither target
model trained on are finite -- --pad-reference-to tops it up and says so.

EnCodec-FAD is 128-D and needs far less, which is one reason it, not this, is
the number to lead with. Never compare the two scales.

  python evaluation/compute_clap_fad_200.py --track-list-dir evaluation
  python evaluation/compute_clap_fad_200.py --cases mono --pad-reference-to 1000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bridge_200 import (  # noqa: E402
    CASES,
    DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    SAMPLING_RATE,
    list_family_tracks,
    load_input_waveform,
    pick_tracks,
    read_track_list,
    stem_paths,
)

FIELDS = ["case", "experiment", "pair", "use_midi", "trained_sigma", "sigma_handoff",
          "reference", "n_reference", "n_generated", "clap_fad"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--bridge-csv", default=None, help="default: <out-root>/bridge_metrics_all.csv")
    p.add_argument("--dataset-root", default=DATASET_ROOT)
    p.add_argument("--cases", nargs="+", default=["poly", "mono"], choices=["poly", "mono"])
    p.add_argument("--n-ref", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--reference-from", default="tools/pools/reference_{case}.txt",
                   help="candidate real target tracks, one per line -- by default "
                        "the clips BOTH target architectures held out, so no "
                        "condition is measured against audio its own generator "
                        "memorised. {case} is substituted. A missing file falls "
                        "back to the whole family, which is mostly training data.")
    p.add_argument("--pad-reference-to", type=int, default=None,
                   help="if the clean reference pool is smaller than this, top it "
                        "up from the reference family at large. The padding WAS "
                        "seen in training; it biases every condition by the same "
                        "amount, so use it to reach the n CLAP needs and then read "
                        "only the differences.")
    p.add_argument("--track-list-dir", default=None,
                   help="also exclude every track_list_*.txt in here from the "
                        "reference, so no clip is both an input and a reference")
    p.add_argument("--exclude-track-lists", nargs="*",
                   default=[CASES["poly"]["track_list"], CASES["mono"]["track_list"]])
    p.add_argument("--experiments", nargs="*", default=None)
    p.add_argument("--out-csv", default=None, help="default: <out-root>/fad/clap_fad_200.csv")
    p.add_argument("--device", default=None)
    return p.parse_args()


def choose_reference_tracks(case_name, dataset_root, n_ref, seed, exclude,
                            reference_from=None, pad_to=None):
    """Which real target recordings the reference set is built from.

    Clean pool first -- clips both target architectures held out, so no
    condition is scored against audio its own generator may have memorised. If
    that pool cannot fill n_ref, stop there unless --pad-reference-to was
    given, and either way say which it was: that distinction decides whether
    the absolute FAD means anything.
    """
    case = CASES[case_name]
    stems = case["reference_stems"]

    def accept(track):
        return all(os.path.isfile(p) and os.path.getsize(p) > 0
                   for p in stem_paths(dataset_root, track, stems))

    clean_path = Path(reference_from.replace("{case}", case_name)) if reference_from else None
    if clean_path is None or not clean_path.is_file():
        if clean_path is not None:
            print("  no %s -- drawing the reference from the whole %s family, which is\n"
                  "  mostly training data of the target models."
                  % (clean_path, case["reference_family"]))
        return pick_tracks(list_family_tracks(dataset_root, case["reference_family"]),
                           n_ref, seed, accept=accept, exclude=exclude)

    tracks = pick_tracks(read_track_list(clean_path), n_ref, seed,
                         accept=accept, exclude=exclude, allow_short=True)
    print("  reference pool: %d clean clips from %s" % (len(tracks), clean_path))
    if len(tracks) >= n_ref:
        return tracks

    if not pad_to or pad_to <= len(tracks):
        print("  only %d clean reference clips, %d asked for. Keeping it clean.\n"
              "  Under 512 the CLAP covariance is singular -- lead with EnCodec-FAD,\n"
              "  or pass --pad-reference-to %d." % (len(tracks), n_ref, n_ref))
        return tracks

    extra = pick_tracks(list_family_tracks(dataset_root, case["reference_family"]),
                        pad_to - len(tracks), seed, accept=accept,
                        exclude=set(exclude) | set(tracks), allow_short=True)
    print("  PADDED with %d %s clips from outside the clean pool, reaching %d.\n"
          "  Those were in the target models' training data. Every condition shares\n"
          "  this one reference, so the bias is common and the ranking holds; the\n"
          "  absolute value does not."
          % (len(extra), case["reference_family"], len(tracks) + len(extra)))
    return sorted(set(tracks) | set(extra))


def build_reference(case_name, ref_dir, dataset_root, n_ref, seed, exclude,
                    reference_from=None, pad_to=None):
    """Write the real target clips; idempotent, keeps what already exists."""
    import torchaudio

    case = CASES[case_name]
    stems = case["reference_stems"]
    ref_dir.mkdir(parents=True, exist_ok=True)
    list_path = ref_dir.parent / "reference_tracks.txt"
    if list_path.is_file():
        tracks = read_track_list(list_path)
        print("  reference list was frozen earlier: %s (%d clips)" % (list_path, len(tracks)))
    else:
        tracks = choose_reference_tracks(case_name, dataset_root, n_ref, seed, exclude,
                                         reference_from, pad_to)
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(tracks) + "\n")
    for i, track in enumerate(tracks, 1):
        out = ref_dir / ("%s.wav" % track)
        if out.is_file():
            continue
        wav = load_input_waveform(stem_paths(dataset_root, track, stems))
        torchaudio.save(str(out), wav.squeeze(0), SAMPLING_RATE)
        if i % 25 == 0:
            print("  reference %d/%d" % (i, len(tracks)), flush=True)
    print("  reference: %d %s clips in %s (list: %s)"
          % (len(tracks), case["reference_ensemble"], ref_dir, list_path))
    return tracks


def link_generated(rows, gen_dir):
    gen_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        src = Path(row["output_audio"])
        if not src.is_absolute():
            src = Path.cwd() / src
        dst = gen_dir / ("%s.wav" % row["sample_id"])
        if dst.is_symlink() or dst.exists():
            continue
        if not src.is_file():
            print("  MISSING %s -- skipped" % src)
            continue
        os.symlink(src, dst)


def cached_embeddings(audio_dir, cache_path, model, processor, device):
    """compute_clap_fad.get_embeddings, reused while the file list is unchanged."""
    import numpy as np
    from evaluation.compute_clap_fad import get_embeddings

    names = sorted(p.name for p in audio_dir.glob("*.wav"))
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if list(cached["names"]) == names:
            return cached["emb"]
    emb = get_embeddings(audio_dir, model, processor, device)
    np.savez(cache_path, emb=emb, names=np.array(names))
    return emb


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    bridge_csv = Path(args.bridge_csv) if args.bridge_csv else out_root / "bridge_metrics_all.csv"
    out_csv = Path(args.out_csv) if args.out_csv else out_root / "fad" / "clap_fad_200.csv"
    if not bridge_csv.is_file():
        raise SystemExit("no bridge CSV at %s -- run run_bridge_batched.py --merge first" % bridge_csv)

    with bridge_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if args.experiments:
        rows = [r for r in rows if r["experiment"] in args.experiments]

    # A clip must never be both an input and a reference: it would then be
    # compared against its own transformation.
    exclude = {r["sample_id"] for r in rows}
    exclude_lists = list(args.exclude_track_lists or [])
    if args.track_list_dir:
        exclude_lists += [str(q) for q in
                          sorted(Path(args.track_list_dir).glob("track_list_*.txt"))]
    for path in exclude_lists:
        if Path(path).is_file():
            exclude.update(read_track_list(path))

    import torch
    from transformers import ClapModel, ClapProcessor
    from evaluation.compute_clap_fad import MODEL_NAME, frechet_distance

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s | CLAP model %s" % (device, MODEL_NAME))
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    results = []
    for case_name in args.cases:
        case_rows = [r for r in rows if r["case"] == case_name]
        if not case_rows:
            print("no %s rows in %s -- skipped" % (case_name, bridge_csv))
            continue
        fad_dir = out_root / "fad" / case_name
        ref_dir = fad_dir / "reference"
        print("=" * 70)
        print("  %s: reference = real %s" % (case_name, CASES[case_name]["reference_ensemble"]))
        print("=" * 70, flush=True)
        build_reference(case_name, ref_dir, args.dataset_root, args.n_ref, args.seed,
                        exclude, args.reference_from, args.pad_reference_to)
        ref_emb = cached_embeddings(ref_dir, fad_dir / "reference_clap.npz", model, processor, device)

        mid = ref_emb.shape[0] // 2
        results.append({
            "case": case_name, "experiment": "sanity_reference_halves", "pair": "",
            "use_midi": "", "trained_sigma": "", "sigma_handoff": "",
            "reference": str(ref_dir), "n_reference": mid, "n_generated": ref_emb.shape[0] - mid,
            "clap_fad": frechet_distance(ref_emb[:mid], ref_emb[mid:]),
        })
        print("  sanity, reference half vs half (n=%d each): %.4f"
              % (mid, results[-1]["clap_fad"]), flush=True)

        by_exp = {}
        for r in case_rows:
            by_exp.setdefault(r["experiment"], []).append(r)
        for exp in sorted(by_exp):
            exp_rows = by_exp[exp]
            gen_dir = fad_dir / "generated" / exp
            link_generated(exp_rows, gen_dir)
            gen_emb = cached_embeddings(gen_dir, fad_dir / "generated" / ("%s_clap.npz" % exp),
                                        model, processor, device)
            score = frechet_distance(ref_emb, gen_emb)
            first = exp_rows[0]
            results.append({
                "case": case_name, "experiment": exp, "pair": first["pair"],
                "use_midi": first["use_midi"], "trained_sigma": first["trained_sigma"],
                "sigma_handoff": first["sigma_handoff"], "reference": str(ref_dir),
                "n_reference": int(ref_emb.shape[0]), "n_generated": int(gen_emb.shape[0]),
                "clap_fad": score,
            })
            print("  %-40s CLAP-FAD %.4f  (n_ref %d, n_gen %d)"
                  % (exp, score, ref_emb.shape[0], gen_emb.shape[0]), flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print("\nSaved %s" % out_csv)
    print("Note: CLAP-FAD fits a 512-D covariance. Check n_reference and n_generated in\n"
          "      the CSV: below ~1000 per side, compare conditions only -- never absolute\n"
          "      values -- and lead with EnCodec-FAD (128-D), which is the paper's metric.")


if __name__ == "__main__":
    main()
