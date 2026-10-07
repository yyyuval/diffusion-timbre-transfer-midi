"""CLAP-FAD for every experiment in the 200-clip bridge CSV.

One FAD per experiment: its generated clips vs a real reference set of the
TARGET ensemble -- violin+cello mixtures for poly, bassoon stems for mono --
drawn (seeded) from tracks that are not among any input list, and preprocessed
exactly like the bridge inputs (24 kHz, summed, 0.95 rescale only if a mixture
clips, crop from the start). Reference and generated embeddings are cached
next to the audio, so re-runs only embed what is new.

The embedder and the distance are imported from compute_clap_fad.py unchanged.
FAD fits a 512-D Gaussian; at n=200 per side it is at the low end of usable and
still noisy -- read differences between conditions, not absolute values, and
never compare these numbers with EnCodec-FAD or the paper's table.

  python evaluation/compute_clap_fad_200.py
  python evaluation/compute_clap_fad_200.py --cases mono --n-ref 200
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
    p.add_argument("--n-ref", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--exclude-track-lists", nargs="*",
                   default=[CASES["poly"]["track_list"], CASES["mono"]["track_list"]])
    p.add_argument("--experiments", nargs="*", default=None)
    p.add_argument("--out-csv", default=None, help="default: <out-root>/fad/clap_fad_200.csv")
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_reference(case_name, ref_dir, dataset_root, n_ref, seed, exclude):
    """Write n_ref real target clips; idempotent, keeps what already exists."""
    import torchaudio

    case = CASES[case_name]
    stems = case["reference_stems"]
    ref_dir.mkdir(parents=True, exist_ok=True)
    list_path = ref_dir.parent / "reference_tracks.txt"
    if list_path.is_file():
        tracks = read_track_list(list_path)
    else:
        def accept(track):
            return all(os.path.isfile(p) and os.path.getsize(p) > 0
                       for p in stem_paths(dataset_root, track, stems))
        tracks = pick_tracks(list_family_tracks(dataset_root, case["reference_family"]),
                             n_ref, seed, accept=accept, exclude=exclude)
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

    exclude = {r["sample_id"] for r in rows}
    for path in args.exclude_track_lists or []:
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
        build_reference(case_name, ref_dir, args.dataset_root, args.n_ref, args.seed, exclude)
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
    print("Note: FAD at n=200 in 512-D is noisy. Compare conditions; do not read absolute values.")


if __name__ == "__main__":
    main()
