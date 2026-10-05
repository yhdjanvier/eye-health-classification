"""
explore_data.py - Exploratory Data Analysis (EDA).

Run:  python src/explore_data.py
Outputs (results/): eda_class_distribution.png, eda_sample_images.png,
                    eda_image_sizes.png, eda_summary.json
WHY: understanding class balance, image sizes and appearance BEFORE modelling drives the
     preprocessing choices (resize target, class weights, augmentation type).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config
from src import data_utils


def main():
    df, class_names = data_utils.prepare_splits(verbose=True)
    config.ensure_dirs()

    # 1. Class distribution
    counts = df["label_name"].value_counts().reindex(class_names)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#1f77b4")
    ax.bar_label(bars)
    ax.set_title("Class distribution (after cleaning)")
    ax.set_ylabel("Number of images")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Sample images (4 per class)
    n = 4
    fig, axes = plt.subplots(len(class_names), n, figsize=(3 * n, 3 * len(class_names)))
    for r, cname in enumerate(class_names):
        sample = df[df["label_name"] == cname].sample(n, random_state=config.SEED)
        for c, (_, row) in enumerate(sample.iterrows()):
            ax = axes[r, c]
            ax.imshow(Image.open(config.PROJECT_ROOT / row["filepath"]).convert("RGB"))
            ax.axis("off")
            if c == 0:
                ax.set_title(config.get_class_info(cname)["display"], loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "eda_sample_images.png", dpi=120)
    plt.close(fig)

    # 3. Image-size statistics
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df["width"], df["height"], s=6, alpha=0.4)
    axes[0].set_xlabel("width (px)"); axes[0].set_ylabel("height (px)")
    axes[0].set_title("Image dimensions")
    axes[1].hist(df["size_kb"], bins=40, color="#2ca02c")
    axes[1].set_xlabel("file size (KB)"); axes[1].set_title("File sizes")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "eda_image_sizes.png", dpi=150)
    plt.close(fig)

    summary = {
        "total_images": int(len(df)),
        "classes": class_names,
        "images_per_class": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 3),
        "unique_image_sizes": int(df[["width", "height"]].drop_duplicates().shape[0]),
        "width_min_max": [int(df["width"].min()), int(df["width"].max())],
        "height_min_max": [int(df["height"].min()), int(df["height"].max())],
        "color_modes": df["mode"].value_counts().to_dict(),
        "split_sizes": df["split"].value_counts().to_dict(),
    }
    (config.RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[eda] plots saved to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
