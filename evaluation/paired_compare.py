from pathlib import Path
import pandas as pd

metrics_csv = Path("Outputs/WAV_files/metrics/inference_metrics.csv")
out_dir = Path("Outputs/WAV_files/metrics")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(metrics_csv)

# Keep only the two experiments we compare
df = df[df["experiment"].isin(["no_midi", "original_midi"])].copy()

# Pivot DPD and JD so each sample has both conditions side by side
dpd = df.pivot_table(
    index=["sample_id", "source", "target"],
    columns="experiment",
    values="DPD",
    aggfunc="first",
).reset_index()

jd = df.pivot_table(
    index=["sample_id", "source", "target"],
    columns="experiment",
    values="JD",
    aggfunc="first",
).reset_index()

paired = dpd.merge(
    jd,
    on=["sample_id", "source", "target"],
    suffixes=("_DPD", "_JD"),
)

paired["DPD_delta_no_minus_midi"] = paired["no_midi_DPD"] - paired["original_midi_DPD"]
paired["JD_delta_no_minus_midi"] = paired["no_midi_JD"] - paired["original_midi_JD"]

paired["DPD_midi_better"] = paired["DPD_delta_no_minus_midi"] > 0
paired["JD_midi_better"] = paired["JD_delta_no_minus_midi"] > 0

paired_csv = out_dir / "paired_comparison.csv"
paired.to_csv(paired_csv, index=False)

summary = {
    "n_pairs": len(paired),
    "DPD_mean_delta_no_minus_midi": paired["DPD_delta_no_minus_midi"].mean(),
    "DPD_median_delta_no_minus_midi": paired["DPD_delta_no_minus_midi"].median(),
    "DPD_midi_better_count": int(paired["DPD_midi_better"].sum()),
    "DPD_midi_better_percent": 100 * paired["DPD_midi_better"].mean(),
    "JD_mean_delta_no_minus_midi": paired["JD_delta_no_minus_midi"].mean(),
    "JD_midi_better_count": int(paired["JD_midi_better"].sum()),
    "JD_midi_better_percent": 100 * paired["JD_midi_better"].mean(),
}

summary_df = pd.DataFrame([summary])
summary_csv = out_dir / "paired_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\nPaired summary:")
print(summary_df.T)

print("\nBest DPD improvements:")
print(
    paired.sort_values("DPD_delta_no_minus_midi", ascending=False)
    [["sample_id", "no_midi_DPD", "original_midi_DPD", "DPD_delta_no_minus_midi"]]
    .head(5)
)

print("\nWorst / negative DPD changes:")
print(
    paired.sort_values("DPD_delta_no_minus_midi", ascending=True)
    [["sample_id", "no_midi_DPD", "original_midi_DPD", "DPD_delta_no_minus_midi"]]
    .head(5)
)

print("\nSaved:")
print(" ", paired_csv)
print(" ", summary_csv)
