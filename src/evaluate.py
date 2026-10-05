"""
evaluate.py - evaluate the saved model on the held-out TEST set.

Run:  python src/evaluate.py

Outputs (results/): metrics.json, per_class_metrics.csv, classification_report.txt,
  confusion_matrix.png, confusion_matrix_normalized.png, per_class_metrics.png,
  confidence_histogram.png, misclassified_examples.png, evaluation_interpretation.md
Also: chooses a confidence threshold on the VALIDATION set and stores it in
models/model_config.json (used by the app to flag uncertain predictions), and copies
a few test images to sample_images/ for demonstrations.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from tensorflow import keras

import config
from src import data_utils


def predict_probs(model, df):
    ds = data_utils.make_dataset(df, training=False)
    probs = model.predict(ds, verbose=0)
    return probs / probs.sum(axis=1, keepdims=True)


def plot_confusion(cm, labels, path, normalized):
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalized else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalized else " (counts)"))
    thresh = data.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{data[i, j]:.2f}" if normalized else f"{int(data[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def choose_threshold(val_true, val_prob, target_acc=0.95, min_coverage=0.30):
    """Lowest confidence threshold whose accepted predictions reach `target_acc` on VALIDATION data."""
    conf, pred = val_prob.max(axis=1), val_prob.argmax(axis=1)
    for t in np.arange(0.50, 0.96, 0.05):
        mask = conf >= t
        if mask.mean() >= min_coverage and (pred[mask] == val_true[mask]).mean() >= target_acc:
            return round(float(t), 2)
    return config.DEFAULT_CONFIDENCE_THRESHOLD


def selective_table(y_true, y_prob):
    conf, pred = y_prob.max(axis=1), y_prob.argmax(axis=1)
    rows = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        mask = conf >= t
        rows.append({"threshold": t, "coverage": float(mask.mean()),
                     "accuracy_on_accepted": float((pred[mask] == y_true[mask]).mean()) if mask.any() else None})
    return rows


def main():
    config.ensure_dirs()
    if not config.MODEL_PATH.exists():
        sys.exit("Model not found. Train first:  python src/train.py")
    model = keras.models.load_model(config.MODEL_PATH)
    df, class_names = data_utils.load_splits()
    val_df, test_df = df[df["split"] == "val"], df[df["split"] == "test"]
    display = [config.get_class_info(c)["display"] for c in class_names]

    val_true, test_true = val_df["label_index"].to_numpy(), test_df["label_index"].to_numpy()
    val_prob, test_prob = predict_probs(model, val_df), predict_probs(model, test_df)
    val_pred, test_pred = val_prob.argmax(1), test_prob.argmax(1)

    # ---- core metrics ----------------------------------------------------------
    acc = accuracy_score(test_true, test_pred)
    val_acc = accuracy_score(val_true, val_pred)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(test_true, test_pred, labels=range(len(class_names)), zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(test_true, test_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(test_true, test_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(test_true, test_pred, labels=range(len(class_names)))
    try:
        auc = float(roc_auc_score(test_true, test_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None

    report_txt = classification_report(test_true, test_pred, target_names=display, digits=4, zero_division=0)
    (config.RESULTS_DIR / "classification_report.txt").write_text(report_txt)
    per_class = pd.DataFrame({"class": display, "precision": p_c, "recall": r_c, "f1_score": f_c, "support": sup})
    per_class.to_csv(config.PER_CLASS_PATH, index=False)

    # ---- Healthy vs Diseased (screening) view -------------------------------------
    screening = None
    if config.HEALTHY_CLASS in class_names:
        h = class_names.index(config.HEALTHY_CLASS)
        t_dis, p_dis = test_true != h, test_pred != h
        tp, fn = int((t_dis & p_dis).sum()), int((t_dis & ~p_dis).sum())
        tn, fp = int((~t_dis & ~p_dis).sum()), int((~t_dis & p_dis).sum())
        screening = {
            "sensitivity_disease_detected": tp / max(tp + fn, 1),
            "specificity_healthy_confirmed": tn / max(tn + fp, 1),
            "precision_disease": tp / max(tp + fp, 1),
            "binary_accuracy": (tp + tn) / len(test_true),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        }

    # ---- confidence analysis + threshold --------------------------------------------
    threshold = choose_threshold(val_true, val_prob)
    selective = selective_table(test_true, test_prob)
    conf = test_prob.max(1)

    metrics = {
        "test": {"accuracy": acc, "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
                 "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
                 "roc_auc_ovr_macro": auc, "n_samples": int(len(test_true))},
        "validation": {"accuracy": val_acc, "n_samples": int(len(val_true))},
        "confusion_matrix": cm.tolist(), "classes": class_names,
        "screening_healthy_vs_diseased": screening,
        "chosen_confidence_threshold": threshold,
        "selective_prediction_test": selective,
        "mean_confidence_correct": float(conf[test_pred == test_true].mean()),
        "mean_confidence_wrong": float(conf[test_pred != test_true].mean()) if (test_pred != test_true).any() else None,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    cfg = json.loads(config.MODEL_CONFIG_PATH.read_text())
    cfg["confidence_threshold"] = threshold
    cfg["test_accuracy"] = acc
    cfg["test_f1_macro"] = f_m
    config.MODEL_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

    # ---- plots ----------------------------------------------------------------------
    plot_confusion(cm, display, config.RESULTS_DIR / "confusion_matrix.png", normalized=False)
    plot_confusion(cm, display, config.RESULTS_DIR / "confusion_matrix_normalized.png", normalized=True)

    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(display)); w = 0.26
    ax.bar(x - w, p_c, w, label="Precision"); ax.bar(x, r_c, w, label="Recall"); ax.bar(x + w, f_c, w, label="F1-score")
    ax.set_xticks(x); ax.set_xticklabels(display, rotation=15); ax.set_ylim(0, 1.05)
    ax.set_title("Per-class test metrics"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "per_class_metrics.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf[test_pred == test_true], bins=20, alpha=0.7, label="correct", color="#2ca02c")
    if (test_pred != test_true).any():
        ax.hist(conf[test_pred != test_true], bins=20, alpha=0.7, label="wrong", color="#d62728")
    ax.axvline(threshold, color="k", linestyle="--", label=f"chosen threshold {threshold:.2f}")
    ax.set_xlabel("Model confidence"); ax.set_ylabel("Images"); ax.set_title("Confidence: correct vs wrong"); ax.legend()
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "confidence_histogram.png", dpi=150); plt.close(fig)

    wrong_idx = np.where(test_pred != test_true)[0][:12]
    if len(wrong_idx):
        cols = 4; rows = int(np.ceil(len(wrong_idx) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows)); axes = np.atleast_1d(axes).ravel()
        for ax in axes: ax.axis("off")
        paths = data_utils.abs_paths(test_df)
        for ax, i in zip(axes, wrong_idx):
            ax.imshow(Image.open(paths[i]).convert("RGB"))
            ax.set_title(f"True: {display[test_true[i]]}\nPred: {display[test_pred[i]]} ({conf[i]:.0%})", fontsize=8)
        fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "misclassified_examples.png", dpi=120); plt.close(fig)

    # ---- sample images for the demo ---------------------------------------------------
    for old in config.SAMPLES_DIR.glob("*"):
        if old.name != ".gitkeep":
            old.unlink()
    for cname in class_names:
        for n, (_, row) in enumerate(test_df[test_df["label_name"] == cname].head(2).iterrows(), start=1):
            src = config.PROJECT_ROOT / row["filepath"]
            shutil.copy(src, config.SAMPLES_DIR / f"{cname}_{n}{src.suffix.lower()}")

    # ---- automatic interpretation ---------------------------------------------------------
    write_interpretation(metrics, per_class, cm, display, test_true, test_pred, val_acc, acc)

    print("\n" + report_txt)
    print(f"Test accuracy : {acc:.4f}   Validation accuracy: {val_acc:.4f}")
    print(f"Macro precision/recall/F1: {p_m:.4f} / {r_m:.4f} / {f_m:.4f}   ROC-AUC (OvR): {auc}")
    print(f"Chosen confidence threshold (from validation): {threshold}")
    print("[evaluate] outputs saved to", config.RESULTS_DIR)


def write_interpretation(metrics, per_class, cm, display, y_true, y_pred, val_acc, acc):
    t = metrics["test"]
    level = ("excellent" if acc >= 0.95 else "good" if acc >= 0.90 else "acceptable" if acc >= 0.80
             else "weak - the model needs improvement")
    best, worst = per_class.loc[per_class["f1_score"].idxmax()], per_class.loc[per_class["f1_score"].idxmin()]
    off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    gap = val_acc - acc
    lines = [
        "# Evaluation interpretation (auto-generated)", "",
        f"* **Overall accuracy on the unseen test set: {acc:.2%}** - judged {level}. "
        f"Macro precision {t['precision_macro']:.2%}, macro recall {t['recall_macro']:.2%}, macro F1 {t['f1_macro']:.2%}.",
        f"* **Best class:** {best['class']} (F1 {best['f1_score']:.2%}). "
        f"**Weakest class:** {worst['class']} (F1 {worst['f1_score']:.2%}, recall {worst['recall']:.2%}).",
    ]
    if off.max() > 0:
        lines.append(f"* **Most frequent confusion:** {display[i]} images predicted as {display[j]} ({int(off[i, j])} cases) - "
                     "these classes look most similar to the network.")
    lines.append(f"* **Generalisation check:** validation accuracy {val_acc:.2%} vs test accuracy {acc:.2%} "
                 f"(gap {gap:+.2%}). " + ("The small gap shows the model is not over-fitted to the validation set."
                                         if abs(gap) < 0.03 else "A gap above 3 points suggests some over-fitting or an unlucky split - consider more augmentation/regularisation."))
    s = metrics["screening_healthy_vs_diseased"]
    if s:
        lines.append(f"* **Healthy-vs-Diseased screening view:** sensitivity (diseased eyes detected) {s['sensitivity_disease_detected']:.2%}, "
                     f"specificity (healthy eyes confirmed) {s['specificity_healthy_confirmed']:.2%}. "
                     f"{s['fn']} diseased images were missed (called healthy) - in screening, missed disease is the costliest error.")
    lines.append(f"* **Confidence behaviour:** mean confidence when correct {metrics['mean_confidence_correct']:.2%}"
                 + (f" vs when wrong {metrics['mean_confidence_wrong']:.2%}" if metrics['mean_confidence_wrong'] is not None else "")
                 + f". A threshold of {metrics['chosen_confidence_threshold']:.2f} (chosen on validation data) marks low-confidence results as 'uncertain' in the app.")
    lines += ["", "## Suggested improvements if results are below target", "",
              "1. Weakest-class recall low -> train longer, raise `--unfreeze` (e.g. 60), or keep class weights on.",
              "2. Large train/validation gap -> raise `--dropout` (0.4-0.5) or reduce `--lr-finetune`.",
              "3. Persistent confusion between two classes -> collect more/cleaner images for them or use a larger backbone.",
              "4. Re-run `python src/train.py` then `python src/evaluate.py` and compare rows in `results/experiments.csv`."]
    (config.RESULTS_DIR / "evaluation_interpretation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
