"""Select fixed, shared-validation inputs for pitch evaluation."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bridge_200 import CASES, DATASET_ROOT
from evaluation.make_track_lists import make_accept

def candidates(rows, base, stems):
    columns = [base + "_midi", base + "_nomidi"]
    validation = [{r[c].strip() for r in rows if r["block"] == "VALIDATION" and r[c].strip()} for c in columns]
    training = set().union(*[{r[c].strip() for r in rows if r["block"] == "TRAINING" and r[c].strip()} for c in columns])
    common = validation[0] & validation[1]
    if common & training:
        raise ValueError("Shared validation entries also occur in training")
    # Keep exact stem layouts supported by the current bridge and metric readers.
    required = {Path(s).stem for s in stems}
    return sorted({entry.split("/")[0] for entry in common
                   if len(entry.split("/")) == 2 and set(entry.split("/")[1].split("+")) == required}), len(common)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, default=ROOT / "evaluation/splits.csv")
    p.add_argument("--dataset-root", default=DATASET_ROOT)
    p.add_argument("--out-dir", type=Path, default=ROOT / "evaluation")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--inventory-only", action="store_true", help="Count candidates without accessing audio or writing lists")
    args = p.parse_args()
    if args.n <= 0:
        p.error("--n must be positive")
    with args.splits.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selections = {}
    audit = {"splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
             "seed": args.seed, "n": args.n, "cases": {},
             "qualification": "Reconstructed validation split; not an independent test set"}
    for name, base in [("poly", "flute_bassoon"), ("mono", "cello")]:
        case = CASES[name]
        pool, common_count = candidates(rows, base, case["stems"])
        print("%s: shared validation=%d; supported stem layouts=%d" % (name, common_count, len(pool)))
        if args.inventory_only:
            continue
        random.Random(args.seed).shuffle(pool)
        accept, rejected = make_accept(args.dataset_root, case["stems"], True)
        picked = []
        for track in pool:
            if accept(track):
                picked.append(track)
            if len(picked) == args.n:
                break
        if len(picked) != args.n:
            raise SystemExit("%s: only %d valid inputs; need %d. No lists written." % (name, len(picked), args.n))
        selections[name] = picked
        audit["cases"][name] = {"source": base, "stems": case["stems"],
                                "tracks": picked, "rejected": rejected,
                                "shared_validation": common_count}
    if args.inventory_only:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, picked in selections.items():
        path = args.out_dir / ("track_list_%d_%s.txt" % (args.n, name))
        path.write_text("\n".join(picked) + "\n")
        print("Wrote", path)
    (args.out_dir / "pitch_selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

if __name__ == "__main__":
    main()
