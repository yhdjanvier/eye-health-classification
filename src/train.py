"""
train.py - train the CNN and save every artefact needed to reproduce the application.

Run (default = MobileNetV2 transfer learning, two phases):
    python src/train.py
Baseline custom CNN from scratch:
    python src/train.py --model custom
Useful overrides:
    python src/train.py --epochs-head 10 --epochs-finetune 15 --batch-size 16

Phase 1 (head)     : backbone frozen, only the new classifier head learns (lr 1e-3).
Phase 2 (fine-tune): top backbone layers unfrozen, very small lr (1e-5) to adapt features
                     to retinal images. The better of the two phases (by validation accuracy)
                     is kept, so fine-tuning can never make the saved model worse on validation.

Saved to models/ : eye_health_cnn.keras, class_names.json, model_config.json
Saved to results/: training_history.csv, training_curves.png, experiments.csv, splits.csv
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import config
from src import data_utils, model as model_lib


def parse_args():
    p = argparse.ArgumentParser(description="Train the eye-health CNN")
    p.add_argument("--model", choices=["mobilenetv2", "custom"], default="mobilenetv2")
    p.add_argument("--epochs-head", type=int, default=config.EPOCHS_HEAD)
    p.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE)
    p.add_argument("--epochs-custom", type=int, default=config.EPOCHS_CUSTOM)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr-head", type=float, default=config.LR_HEAD)
    p.add_argument("--lr-finetune", type=float, default=config.LR_FINETUNE)
    p.add_argument("--unfreeze", type=int, default=config.UNFREEZE_LAYERS)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--no-class-weights", action="store_true")
    return p.parse_args()


def callbacks(patience):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(patience // 2, 2),
                                          min_lr=1e-7, verbose=1),
    ]


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def history_frame(hist, phase):
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoch_in_phase", range(1, len(df) + 1))
    df.insert(0, "phase", phase)
    return df


def plot_curves(history_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(1, len(history_df) + 1)
    axes[0].plot(x, history_df["accuracy"], label="train"); axes[0].plot(x, history_df["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(x, history_df["loss"], label="train"); axes[1].plot(x, history_df["val_loss"], label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    phases = history_df["phase"].tolist()
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            for ax in axes:
                ax.axvline(i + 0.5, color="grey", linestyle="--", label=None)
                ax.text(i + 0.6, ax.get_ylim()[0], " fine-tuning starts", fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    keras.utils.set_random_seed(config.SEED)
    config.ensure_dirs()

    # ---- data -------------------------------------------------------------
    df, class_names = data_utils.prepare_splits(verbose=True)
    train_df, val_df, test_df = (df[df["split"] == s] for s in ("train", "val", "test"))
    train_ds = data_utils.make_dataset(train_df, training=True, batch_size=args.batch_size)
    val_ds = data_utils.make_dataset(val_df, training=False, batch_size=args.batch_size)
    class_weights = None if args.no_class_weights else data_utils.compute_class_weights(train_df)
    print("[train] class weights:", class_weights)

    # ---- model ------------------------------------------------------------
    model = model_lib.build_model(args.model, len(class_names), dropout=args.dropout)
    model.summary()
    histories, val_head = [], None
    t0 = datetime.now()

    if args.model == "mobilenetv2":
        print("\n=== Phase 1: training the classification head (backbone frozen) ===")
        compile_model(model, args.lr_head)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                       class_weight=class_weights, callbacks=callbacks(config.EARLY_STOP_PATIENCE))
        histories.append(history_frame(h1, "head"))
        val_head = model.evaluate(val_ds, verbose=0, return_dict=True)
        weights_head = model.get_weights()
        print(f"[train] phase 1 validation: {val_head}")

        print(f"\n=== Phase 2: fine-tuning the top {args.unfreeze} backbone layers ===")
        model_lib.unfreeze_top_layers(model, args.unfreeze)
        compile_model(model, args.lr_finetune)          # re-compile after changing trainable flags
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        print(f"[train] trainable parameters now: {trainable:,}")
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                       class_weight=class_weights, callbacks=callbacks(config.EARLY_STOP_PATIENCE))
        histories.append(history_frame(h2, "finetune"))
        val_final = model.evaluate(val_ds, verbose=0, return_dict=True)
        print(f"[train] phase 2 validation: {val_final}")
        if val_final["accuracy"] < val_head["accuracy"]:
            print("[train] fine-tuning did not improve validation accuracy -> keeping phase-1 weights")
            model.set_weights(weights_head)
            val_final = val_head
    else:
        print("\n=== Training the custom CNN from scratch ===")
        compile_model(model, args.lr_head)
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_custom,
                      class_weight=class_weights, callbacks=callbacks(config.EARLY_STOP_PATIENCE))
        histories.append(history_frame(h, "scratch"))
        val_final = model.evaluate(val_ds, verbose=0, return_dict=True)

    minutes = (datetime.now() - t0).total_seconds() / 60

    # ---- save everything needed to reproduce the app ------------------------
    model.save(config.MODEL_PATH)
    config.CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2))
    model_config = {
        "architecture": args.model,
        "img_size": list(config.IMG_SIZE),
        "channels": 3,
        "input_note": "Raw RGB float values in 0-255. Rescaling to [-1,1] happens INSIDE the model; "
                      "resizing uses src/preprocessing.resize_to_uint8 (bilinear).",
        "class_names": class_names,
        "healthy_class": config.HEALTHY_CLASS,
        "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
        "hyperparameters": {
            "batch_size": args.batch_size, "dropout": args.dropout, "dense_units": config.DENSE_UNITS,
            "l2_weight": config.L2_WEIGHT, "lr_head": args.lr_head, "lr_finetune": args.lr_finetune,
            "unfreeze_layers": args.unfreeze, "class_weights": class_weights, "seed": config.SEED,
        },
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "validation_accuracy": float(val_final["accuracy"]),
        "validation_loss": float(val_final["loss"]),
        "tensorflow_version": tf.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    config.MODEL_CONFIG_PATH.write_text(json.dumps(model_config, indent=2))

    history_df = pd.concat(histories, ignore_index=True)
    history_df.to_csv(config.HISTORY_PATH, index=False)
    plot_curves(history_df, config.RESULTS_DIR / "training_curves.png")

    row = pd.DataFrame([{
        "timestamp": model_config["trained_at"], "architecture": args.model,
        "lr_head": args.lr_head, "lr_finetune": args.lr_finetune if args.model == "mobilenetv2" else "",
        "dropout": args.dropout, "batch_size": args.batch_size,
        "class_weights": class_weights is not None,
        "val_acc_after_phase1": val_head["accuracy"] if val_head else "",
        "val_acc_final": val_final["accuracy"], "val_loss_final": val_final["loss"],
        "epochs_run": len(history_df), "minutes": round(minutes, 1),
    }])
    row.to_csv(config.EXPERIMENTS_PATH, mode="a", header=not config.EXPERIMENTS_PATH.exists(), index=False)

    print(f"\n[train] DONE in {minutes:.1f} min. Final validation accuracy: {val_final['accuracy']:.4f}")
    print(f"[train] model saved to {config.MODEL_PATH}")
    print("[train] next step: python src/evaluate.py")


if __name__ == "__main__":
    main()
