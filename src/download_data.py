"""
download_data.py - download the Kaggle dataset automatically into data/raw/.

Run manually:  python src/download_data.py
It also runs automatically from train.py / explore_data.py when the dataset is missing.

Uses the `kagglehub` package. KAGGLEHUB_CACHE is pointed at data/raw so the files land inside
the project (data/raw/datasets/...), where data_utils.locate_dataset_root() finds them.
If Kaggle asks for credentials: create a token at Kaggle > Settings > "Create New Token",
then either place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

DATASET_HANDLE = "gunavenkatdoddi/eye-diseases-classification"


def download():
    config.ensure_dirs()
    os.environ["KAGGLEHUB_CACHE"] = str(config.RAW_DATA_DIR)
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed. Run:  pip install kagglehub") from exc
    print(f"[download] downloading '{DATASET_HANDLE}' (~770 MB) into {config.RAW_DATA_DIR} ...")
    print("[download] if it is interrupted, just run the command again - finished files are kept.")
    path = kagglehub.dataset_download(DATASET_HANDLE)
    print(f"[download] done: {path}")
    return Path(path)


if __name__ == "__main__":
    download()
