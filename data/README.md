# Dataset preparation

**Dataset:** *Eye Diseases Classification* (Kaggle, author `gunavenkatdoddi`)
**URL:** https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
**Content:** about 4,200 retinal fundus images in 4 folders: `cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`.

## Option 0 - automatic (no manual steps)
Just run `python src/train.py` (or `python src/explore_data.py`). If the dataset is missing, it is downloaded into
`data/raw/` automatically via `kagglehub` (~770 MB). To download first and train later: `python src/download_data.py`.
If it is interrupted, run it again. If Kaggle asks for credentials, create a token (Kaggle > Settings > Create New Token)
and put `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`), or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

## Option A - manual download (easiest, works in the ICT room)
1. Log in to Kaggle (free account) and open the URL above.
2. Click **Download** -> you get `archive.zip`.
3. Extract it **inside `data/raw/`** so you get something like:
```
data/raw/dataset/cataract/*.jpg
data/raw/dataset/diabetic_retinopathy/*.jpg
data/raw/dataset/glaucoma/*.jpg
data/raw/dataset/normal/*.jpg
```
The code searches `data/raw/` automatically for the folder that contains the class folders,
so an extra wrapper folder (e.g. `data/raw/archive/dataset/...`) is also fine.

## Option B - Kaggle command line
```bash
pip install kaggle
# put kaggle.json (Kaggle > Settings > Create New Token) in ~/.kaggle/ (Windows: C:\Users\<you>\.kaggle\)
kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification -p data/raw --unzip
```

## Check it worked
```bash
python src/explore_data.py
```
It prints the number of images per class and writes the EDA plots to `results/`.
Folder names become the class labels; the code needs at least 2 class folders.
