from pathlib import Path
import numpy as np


SRC_CACHE = Path("/home/shared_workspace/diffusion-timbre-transfer/midi_cache_bassoon_fps75")
DST_CACHE = Path("/home/shared_workspace/diffusion-timbre-transfer/midi_cache_bassoon_fps75_addfifth")

INTERVAL = 7  # fifth = 7 semitones
OVERWRITE = False


def add_fifth_to_roll(roll: np.ndarray, interval: int = 7) -> np.ndarray:
    """
    roll shape: [128, T]
    Adds a fifth above every active note.
    """
    out = roll.copy()

    # note n active -> also activate note n + interval
    out[interval:, :] = np.maximum(out[interval:, :], roll[:-interval, :])

    return out.astype(np.uint8)


def main():
    files = sorted(SRC_CACHE.glob("*/*_pianoroll.npy"))

    print("source cache:", SRC_CACHE)
    print("destination cache:", DST_CACHE)
    print("found pianoroll files:", len(files))
    print("interval:", INTERVAL)

    success = 0
    skipped = 0
    failed = 0

    for i, src_path in enumerate(files, start=1):
        rel_path = src_path.relative_to(SRC_CACHE)
        dst_path = DST_CACHE / rel_path

        if dst_path.exists() and not OVERWRITE:
            skipped += 1
            if i % 500 == 0:
                print(f"[{i}/{len(files)}] SKIP existing")
            continue

        try:
            roll = np.load(src_path)

            if roll.shape[0] != 128:
                raise ValueError(f"Expected roll shape [128, T], got {roll.shape}")

            roll_fifth = add_fifth_to_roll(roll, interval=INTERVAL)

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst_path, roll_fifth)

            success += 1

            if i % 100 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] OK {rel_path} shape={roll.shape}")

        except Exception as e:
            failed += 1
            print(f"[{i}/{len(files)}] FAILED {src_path}: {e}")

    print()
    print("Done")
    print("success:", success)
    print("skipped:", skipped)
    print("failed:", failed)


if __name__ == "__main__":
    main()