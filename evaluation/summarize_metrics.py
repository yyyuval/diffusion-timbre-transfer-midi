import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_csv",
        default="Outputs/WAV_files/metrics/inference_metrics.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="Outputs/WAV_files/metrics",
    )
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")

    df = pd.read_csv(metrics_csv)

    summary = (
        df
        .groupby(["experiment", "source", "target"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            DPD_mean=("DPD", "mean"),
            DPD_std=("DPD", "std"),
            JD_mean=("JD", "mean"),
            JD_std=("JD", "std"),
        )
        .reset_index()
    )

    for col in ["DPD_mean", "DPD_std", "JD_mean", "JD_std"]:
        summary[col] = summary[col].round(4)

    summary_csv = out_dir / "summary_table.csv"
    summary_md = out_dir / "summary_table.md"

    summary.to_csv(summary_csv, index=False)

    # Write markdown table manually, without requiring the tabulate package
    with open(summary_md, "w") as f:
        cols = list(summary.columns)

        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")

        for _, row in summary.iterrows():
            values = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    values.append("")
                else:
                    values.append(str(val))
            f.write("| " + " | ".join(values) + " |\n")

    print("\nSummary table:")
    print(summary)

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {summary_md}")


if __name__ == "__main__":
    main()
