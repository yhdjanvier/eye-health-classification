"""
data_utils.py - data acquisition, validation, de-duplication, splitting and tf.data pipelines.

WHAT : finds the Kaggle dataset on disk, verifies every image, removes corrupt files and
       exact duplicates, creates a stratified 70/15/15 split and builds fast tf.data pipelines.
WHY  : medical datasets often contain corrupt/duplicate files; duplicates that land in both
       train and test inflate accuracy (data leakage). A stratified split keeps class balance.
WHERE: used by explore_data.py, train.py and evaluate.py.
"""
import hashlib
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from src.preprocessing import resize_to_uint8

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# 1. Data acquisition: locate the dataset inside data/raw
# ---------------------------------------------------------------------------
def _image_files(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def _has_images(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMG_EXT for p in folder.iterdir())


def locate_dataset_root(raw_dir: Path = config.RAW_DATA_DIR) -> Path:
    """Find the folder whose sub-folders are the class folders (works for any zip layout)."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} does not exist. See data/README.md for download steps.")
    candidates = [raw_dir] + [p for p in raw_dir.rglob("*") if p.is_dir()]
    best, best_n = None, 0
    for cand in candidates:
        n = sum(1 for s in cand.iterdir() if s.is_dir() and _has_images(s))
        if n > best_n:
            best, best_n = cand, n
    if best is None or best_n < 2:
        raise FileNotFoundError(
            "Dataset not found. Download it from Kaggle and extract it so that you get one folder per "
            "class inside data/raw/ (e.g. data/raw/dataset/cataract, .../glaucoma ...). "
            "See data/README.md for exact steps."
        )
    return best


# ---------------------------------------------------------------------------
# 2. Manifest: verify images, drop corrupt + duplicate files
# ---------------------------------------------------------------------------
def build_manifest(root: Path, verbose: bool = True) -> pd.DataFrame:
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and _has_images(d))
    class_names = [d.name for d in class_dirs]
    rows, corrupt = [], []
    for idx, cdir in enumerate(class_dirs):
        for path in _image_files(cdir):
            try:
                data = path.read_bytes()
                with Image.open(io.BytesIO(data)) as im:
                    im.load()                      # full decode => catches truncated files
                    width, height, mode = im.width, im.height, im.mode
                rows.append({
                    "filepath": os.path.relpath(path, config.PROJECT_ROOT),
                    "label_name": cdir.name, "label_index": idx,
                    "width": width, "height": height, "mode": mode,
                    "size_kb": round(len(data) / 1024, 1),
                    "md5": hashlib.md5(data).hexdigest(),
                })
            except Exception as exc:               # noqa: BLE001 - any decode failure => skip file
                corrupt.append((str(path), str(exc)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No readable images were found in the dataset folder.")
    n_before = len(df)
    df = df.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    if verbose:
        print(f"[data] dataset root     : {root}")
        print(f"[data] classes          : {class_names}")
        print(f"[data] readable images  : {n_before}")
        print(f"[data] corrupt removed  : {len(corrupt)}")
        print(f"[data] duplicates removed: {n_before - len(df)}")
        print(f"[data] final images     : {len(df)}")
    return df


# ---------------------------------------------------------------------------
# 3. Stratified train / validation / test split
# ---------------------------------------------------------------------------
def split_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df, test_size=1 - config.TRAIN_FRAC, stratify=df["label_index"], random_state=config.SEED)
    val_share = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_share, stratify=temp_df["label_index"], random_state=config.SEED)
    out = pd.concat([
        train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test"),
    ]).reset_index(drop=True)
    return out


def prepare_splits(verbose: bool = True):
    """Full pipeline: locate -> verify -> de-duplicate -> split -> save CSVs. Returns (df, class_names)."""
    config.ensure_dirs()
    try:
        root = locate_dataset_root()
    except FileNotFoundError:
        print("[data] dataset not found locally -> downloading it from Kaggle (one time only)")
        from src import download_data
        try:
            download_data.download()
        except Exception as exc:                        # noqa: BLE001
            raise FileNotFoundError(
                f"Automatic download failed ({exc}). Download manually - see data/README.md") from exc
        root = locate_dataset_root()
    df = build_manifest(root, verbose)
    df.drop(columns=["md5"]).to_csv(config.MANIFEST_PATH, index=False)
    df = split_dataframe(df)
    df.drop(columns=["md5"]).to_csv(config.SPLITS_PATH, index=False)
    class_names = class_names_from_df(df)
    if verbose:
        print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0))
    return df, class_names


def class_names_from_df(df: pd.DataFrame):
    return (df.drop_duplicates("label_index").sort_values("label_index")["label_name"].tolist())


def load_splits():
    """Load the saved split (so evaluation uses exactly the same test set as training)."""
    if not config.SPLITS_PATH.exists():
        raise FileNotFoundError("results/splits.csv not found. Run `python src/train.py` first.")
    df = pd.read_csv(config.SPLITS_PATH)
    return df, class_names_from_df(df)


def abs_paths(df: pd.DataFrame):
    return [str(config.PROJECT_ROOT / p) for p in df["filepath"]]


# ---------------------------------------------------------------------------
# 4. tf.data pipelines
# ---------------------------------------------------------------------------
def _load_and_resize(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return resize_to_uint8(img, config.IMG_SIZE), label


def make_dataset(df: pd.DataFrame, training: bool, batch_size: int = config.BATCH_SIZE):
    """
    Build a batched tf.data pipeline of (float32 image 0-255, int label).
    Images are cached as uint8 (small) after the first epoch. Shuffle only when training;
    evaluation datasets keep file order so labels line up with predictions.
    """
    paths = np.array(abs_paths(df), dtype=str)
    labels = df["label_index"].to_numpy().astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_and_resize, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(min(len(df), 2048), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def compute_class_weights(train_df: pd.DataFrame):
    """'Balanced' class weights: rare classes count more in the loss."""
    classes = np.sort(train_df["label_index"].unique())
    weights = compute_class_weight("balanced", classes=classes, y=train_df["label_index"].to_numpy())
    return {int(c): float(w) for c, w in zip(classes, weights)}
