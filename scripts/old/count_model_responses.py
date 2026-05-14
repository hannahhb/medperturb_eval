import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASKS = ("manage", "visit", "resource")
MODELS = {
    "llama": "llama",
    "gpt": "gpt-4o",
    "medgemma": "medgemma",
}


def has_response(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & text.ne("nan")


def answered_mask(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [f"{prefix}_{task}" for task in TASKS]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {prefix}: {missing}")
    masks = [has_response(df[col]) for col in cols]
    return masks[0] & masks[1] & masks[2]


def clinician_mask(df: pd.DataFrame) -> pd.Series:
    cols = [f"clinician_consensus_{task}" for task in TASKS]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing clinician columns: {missing}")
    return df[cols].notna().all(axis=1)


def summarize(df: pd.DataFrame) -> dict:
    summary = {}
    dataset_order = sorted(df["dataset"].dropna().astype(str).unique())
    model_masks = {label: answered_mask(df, prefix) for label, prefix in MODELS.items()}
    model_masks["clinicians"] = clinician_mask(df)

    for dataset in dataset_order:
        dataset_df = df[df["dataset"].astype(str) == dataset].copy()
        dataset_summary = {
            "total": len(dataset_df),
            "perturbations": {},
        }
        perturbation_order = sorted(dataset_df["perturbation"].dropna().astype(str).unique())
        for perturbation in perturbation_order:
            sub = dataset_df[dataset_df["perturbation"].astype(str) == perturbation].copy()
            counts = {"total": len(sub)}
            for label, mask in model_masks.items():
                counts[label] = int(mask.loc[sub.index].sum())
            dataset_summary["perturbations"][perturbation] = counts
        summary[dataset] = dataset_summary
    return summary


def print_summary(summary: dict) -> None:
    grand_total = sum(block["total"] for block in summary.values())
    print(f"Total rows: {grand_total:,}")
    print()
    for dataset, dataset_info in summary.items():
        print(f"{dataset}: {dataset_info['total']:,}")
        for perturbation, counts in dataset_info["perturbations"].items():
            print(f"  {perturbation}: {counts['total']:,}")
            print(
                "    "
                f"llama={counts['llama']:,} "
                f"gpt={counts['gpt']:,} "
                f"medgemma={counts['medgemma']:,} "
                f"clinicians={counts['clinicians']:,}"
            )


def save_overlay_pies(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, dataset_info in summary.items():
        perturbations = dataset_info["perturbations"]
        if not perturbations:
            continue

        middle_sizes = []
        middle_labels = []
        outer_sizes = []
        outer_labels = []
        outer_colors = []

        cmap = plt.get_cmap("tab20c")
        color_idx = 0

        for perturbation, counts in perturbations.items():
            total = counts["total"]
            if total == 0:
                continue

            middle_sizes.append(total)
            middle_labels.append(f"{perturbation}\n{total}")

            coverage = [
                ("llama", counts["llama"]),
                ("gpt", counts["gpt"]),
                ("medgemma", counts["medgemma"]),
                ("clinicians", counts["clinicians"]),
            ]

            for label, value in coverage:
                outer_sizes.append(value)
                outer_labels.append(f"{perturbation}\n{label}\n{value}")
                outer_colors.append(cmap(color_idx % cmap.N))
                color_idx += 1

        if not middle_sizes or not outer_sizes:
            continue

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

        ax.pie(
            [dataset_info["total"]],
            radius=0.45,
            labels=[f"{dataset}\n{dataset_info['total']}"],
            labeldistance=0.0,
            colors=["#d9d9d9"],
            textprops={"fontsize": 12, "weight": "bold"},
            wedgeprops=dict(width=0.45, edgecolor="white"),
        )

        ax.pie(
            middle_sizes,
            radius=0.8,
            labels=middle_labels,
            labeldistance=0.82,
            textprops={"fontsize": 9},
            wedgeprops=dict(width=0.3, edgecolor="white"),
        )

        ax.pie(
            outer_sizes,
            radius=1.15,
            labels=outer_labels,
            labeldistance=1.05,
            colors=outer_colors,
            textprops={"fontsize": 7},
            wedgeprops=dict(width=0.3, edgecolor="white"),
        )

        ax.set_title(
            f"{dataset}: perturbation counts and response coverage",
            fontsize=13,
            pad=20,
        )

        safe_name = dataset.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"{safe_name}_overlay_pie.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize MedPerturb dataset and response coverage by dataset and perturbation."
    )
    parser.add_argument(
        "--csv",
        default="data/medperturb_data.csv",
        help="Path to the MedPerturb CSV (default: data/medperturb_data.csv)",
    )
    parser.add_argument(
        "--output_dir",
        default="plots/dataset_coverage",
        help="Directory to save nested donut charts (default: plots/dataset_coverage)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    print(f"CSV: {csv_path}")
    summary = summarize(df)
    print_summary(summary)

    output_dir = Path(args.output_dir)
    save_overlay_pies(summary, output_dir)
    print()
    print(f"Saved charts to: {output_dir}")


if __name__ == "__main__":
    main()
