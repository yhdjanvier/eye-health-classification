# Eye Health Classification System

CNN-based deep-learning application that classifies **retinal fundus images** as **Healthy (Normal)** or one of three
diseases (**Cataract, Diabetic Retinopathy, Glaucoma**) and serves the result through a **Streamlit** web app.
Built for the **CAT2 Practical Assessment 2026** - *Python and Fundamentals of AI (ITLPA701)*, Rwanda Polytechnic - Ngoma College.

> ⚠️ **AI screening / research tool - NOT a medical diagnosis.** See [Responsible use](#8-responsible-use).

---------------------------------------------------------------------------------------------------

## 1. Problem, approach and dataset

| Item | Choice | Why |
|---|---|---|
| Problem domain | Image recognition - eye-disease screening | Early detection of cataract, diabetic retinopathy and glaucoma prevents blindness; screening specialists are scarce. |
| AI approach | **Deep learning - CNN** | Images need spatial feature extraction; CNNs are the standard. (Allowed by the CAT2 sheet.) |
| Model | **MobileNetV2 transfer learning** (default) + custom 4-block CNN (baseline) | Pre-trained ImageNet filters work with only ~4k images, train on a laptop CPU, and the saved model is small (~10 MB) for deployment. The custom CNN lets you *compare* against a from-scratch model. |
| Dataset | Kaggle **Eye Diseases Classification** (~4,200 fundus images, 4 balanced classes) | Public, labelled, roughly balanced, right size for a 2-hour practical, one-folder-per-class layout. Download steps: [`data/README.md`](data/README.md). |

**Important:** the data are *fundus* photographs (taken with a retinal camera), not phone photos of the outside of the eye.

## 2. Technologies (what, why, where)

| Technology | What it is | Why chosen | Where used |
|---|---|---|---|
| Python 3.10-3.12 | Programming language | Required by the module | everywhere |
| TensorFlow / Keras | Deep-learning framework | Simple high-level API, `.keras` model saving, MobileNetV2 built in | `src/model.py`, `train.py`, `evaluate.py`, `predict.py` |
| MobileNetV2 | Efficient pre-trained CNN (depthwise-separable convolutions) | High accuracy for few parameters; fast CPU training/inference | `src/model.py` |
| tf.data | Input-pipeline API | Fast decode/resize/cache/prefetch | `src/data_utils.py` |
| scikit-learn | ML utilities | stratified split, class weights, precision/recall/F1/confusion matrix/ROC-AUC | `data_utils.py`, `evaluate.py` |
| Pillow | Image library | validates/opens uploads, EXIF rotation | `data_utils.py`, `predict.py` |
| NumPy / pandas | Arrays / tables | metrics, manifests, CSV logs | all |
| Matplotlib | Plotting | EDA, curves, confusion matrices | `explore_data.py`, `train.py`, `evaluate.py` |
| Streamlit | Python web-app framework | Fast, professional UI, free deployment | `app.py` |

## 3. Project structure

```
eye_health_classification/
├── app.py                    # Streamlit web application
├── config.py                 # paths, constants, class descriptions (single source of truth)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml    # theme + upload limit
├── data/
│   ├── README.md             # dataset download instructions
│   └── raw/                  # <- extract Kaggle dataset here (not committed to Git)
├── src/
│   ├── preprocessing.py      # resize function shared by training & app + image-quality checks
│   ├── data_utils.py         # acquisition, validation, de-duplication, split, tf.data pipelines
│   ├── download_data.py      # automatic Kaggle download (kagglehub)
│   ├── explore_data.py       # EDA
│   ├── model.py              # CNN architectures, augmentation, fine-tuning helper
│   ├── train.py              # 2-phase training + saving artefacts
│   ├── evaluate.py           # test-set evaluation, plots, interpretation, threshold selection
│   └── predict.py            # upload validation, inference, explanation (used by app.py)
├── models/                   # eye_health_cnn.keras, class_names.json, model_config.json (created by train.py)
├── results/                  # plots, metrics.json, CSV logs (created by scripts)
└── sample_images/            # demo images exported by evaluate.py
```

## 4. Installation and running (PyCharm / terminal)

```bash
# 1. open the folder in PyCharm (File > Open) and create a virtual environment (Python 3.10-3.12)
python -m venv .venv
.venv\Scripts\activate            # Windows      (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. dataset: downloaded automatically on first run (or: python src/download_data.py).
#    Manual alternative: data/README.md

# 3. run the pipeline in this order
python src/explore_data.py        # EDA plots + dataset summary          (~1 min)
python src/train.py               # train + save model                    (~15-40 min CPU)
python src/evaluate.py            # metrics, confusion matrix, report     (~1 min)
streamlit run app.py              # launch the web app -> http://localhost:8501
```
PyCharm: right-click each script > *Run*, and create a Run Configuration with module `streamlit` and parameters `run app.py` for the app
(or just use the Terminal tab). Internet is needed **once** to download the MobileNetV2 ImageNet weights (cached afterwards).

Optional: `python src/train.py --model custom` trains the from-scratch CNN baseline (compare rows in `results/experiments.csv`).
If training is too slow in the ICT room: `python src/train.py --epochs-head 5 --epochs-finetune 8`.

## 5. Architecture and workflow

```
Kaggle dataset ──► verify images (decode test) ──► drop corrupt + exact duplicates ──► stratified split 70/15/15
        │
        ▼
 tf.data: read ► decode ► resize 224x224 ► cache ► shuffle ► batch ► prefetch
        │
        ▼  (inside the saved model)
 Augmentation (train only) ► Rescaling [-1,1] ► MobileNetV2 backbone ► GAP ► BatchNorm ► Dropout
        ► Dense(256, L2) ► Dropout ► Softmax(4)
        │
 Phase 1: backbone frozen, lr 1e-3  ──►  Phase 2: top 40 layers unfrozen, lr 1e-5 (best-of-two kept)
        │
        ▼
 evaluate.py: accuracy, precision, recall, F1, confusion matrix, ROC-AUC, confidence threshold
        │
        ▼
 models/ (.keras + class_names.json + model_config.json)  ──►  app.py (Streamlit) ──► prediction + probabilities + explanation
```

**Consistency guarantee:** resizing is one function (`src/preprocessing.resize_to_uint8`) used by both training and the app; pixel
scaling and augmentation are layers *inside* the saved model, so the app only sends raw 0-255 pixels. Class order is saved in `class_names.json`.

## 6. Important parameters and why

| Parameter | Value | Reason |
|---|---|---|
| Image size | 224×224×3 | MobileNetV2 native size; good detail/speed balance |
| Split | 70 / 15 / 15 stratified, seed 42 | Test set stays unseen; classes proportionally represented; reproducible |
| Augmentation | flip, rotation ±18°, zoom 10 %, contrast 10 %, brightness 10 % | Simulates camera/lighting variation, reduces over-fitting; fundus images have no fixed orientation |
| Batch size | 32 | Stable gradients, fits ordinary RAM |
| Optimizer | Adam | Robust default for CNNs |
| Learning rate | 1e-3 (head) → 1e-5 (fine-tune) | Large steps for new layers, tiny steps so pre-trained filters are adapted, not destroyed |
| Dropout 0.3, L2 1e-4 | regularisation | Reduces over-fitting on ~3k training images |
| Dense units | 256 | Enough capacity for 4 classes without many parameters |
| Loss | sparse categorical cross-entropy | Multi-class, integer labels |
| Class weights | balanced | Small imbalances do not bias the model |
| EarlyStopping (val_loss, patience 6, restore best) + ReduceLROnPlateau (×0.3) | callbacks | Stops at the best epoch; automatic learning-rate reduction |
| Unfrozen layers | 40 (BatchNorm kept frozen) | Adapts high-level features while keeping BN statistics stable |
| Confidence threshold | chosen on validation set by `evaluate.py` | Flags unreliable predictions as "inconclusive" |

## 7. Evaluation, interpretation and improvement

`python src/evaluate.py` evaluates on the untouched test set and writes to `results/`:
`metrics.json`, `classification_report.txt`, `per_class_metrics.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png`,
`per_class_metrics.png`, `confidence_histogram.png`, `misclassified_examples.png`, `training_curves.png`, and an **auto-generated
`evaluation_interpretation.md`** (best/weakest class, most-confused pair, validation-vs-test gap, healthy-vs-diseased sensitivity/specificity).

**Metrics:** accuracy (overall correctness), precision (of predicted X, how many are X), recall (of real X, how many found - the
critical one for disease), F1 (balance of the two), confusion matrix (which classes are mixed up), ROC-AUC (ranking quality).

**Improvements built in (evidence for "parameters adjusted based on evaluation results"):**
1. Class weights; 2. two-phase training with automatic keep-the-better-phase check; 3. LR scheduling + early stopping;
4. confidence threshold tuned from validation results; 5. every run is logged in `results/experiments.csv`
(phase-1 vs final validation accuracy, learning rates, dropout) so you can show before/after.
**For the assessor:** run `evaluate.py`; if the weakest class recall is low, re-train with e.g. `--unfreeze 60` or `--dropout 0.4`,
re-evaluate and point to the two rows in `experiments.csv`.

## 8. Responsible use

* **Not a medical diagnosis.** It is a screening/research/learning tool; every result must be reviewed by an ophthalmologist.
* **Limitations:** fundus images only; only 4 classes (other diseases are not recognised); dataset bias (limited sources, devices, populations);
  no clinical context; neural networks can be confidently wrong; public datasets can contain label noise.
* **Risks:** false negatives delay treatment, false positives cause anxiety, users may over-trust AI.
* **Risk reduction:** permanent disclaimers; plain-language explanations; "inconclusive" label below the confidence threshold;
  image-quality and non-fundus warnings; strict upload validation; per-class recall reporting; uploaded images are not stored;
  clinical validation, bias audit and regulatory approval required before any real-world use.

## 9. Deployment

### GitHub
```bash
git init
git add .
git commit -m "Eye Health Classification System"
git branch -M main
git remote add origin https://github.com/<your-username>/eye-health-classification.git
git push -u origin main
```
`data/raw/` is ignored (too big). **`models/` and `results/` MUST be committed** (the model is ~10 MB, below GitHub's 100 MB file limit) so the app runs online.
Also commit `sample_images/` for demos.

### Streamlit Community Cloud (free)
1. Push the repo to GitHub (public, or private with access granted) after running training + evaluation locally.
2. Go to https://share.streamlit.io -> sign in with GitHub -> **Create app**.
3. Select repository, branch `main`, main file path `app.py`.
4. *Advanced settings* -> choose **Python 3.11** (or 3.12).
5. **Deploy**. First start takes a few minutes while TensorFlow installs. If the build is too heavy, change `tensorflow` to `tensorflow-cpu` in `requirements.txt`.

### Alternative: Hugging Face Spaces
Create a new Space with SDK **Streamlit**, upload all files (use Git LFS only if you add large files), keep `app.py` and `requirements.txt` at the root.

## 10. CAT2 assessment mapping (30 marks)

| Assessable outcome | Criterion (marks) | Where satisfied |
|---|---|---|
| **Data preprocessing (30 % = 9)** | Environment properly configured, functionalities specified (1.5) | §2 technologies table, `requirements.txt`, `config.py` (functions of every module are documented at the top of each file) |
| | Environment configured (1.5) | §4 virtual environment + `pip install -r requirements.txt`, PyCharm run steps |
| | Data acquired from the given source (3) | Kaggle dataset, `data/README.md`, `data_utils.locate_dataset_root` / `build_manifest`, `explore_data.py` |
| | Data pre-processed for the model (3) | `data_utils.py` (decode test, corrupt/duplicate removal, stratified split, resize, class weights), `preprocessing.py` |
| **Deep learning algorithms (50 % = 15)** | Data features engineered (3) | tf.data pipeline: resize 224×224, uint8 caching, augmentation layers (`model.augmentation_block`), pixel scaling |
| | Model features engineered (3) | `model.py`: MobileNetV2 backbone + GAP/BN/Dropout/Dense head, fine-tuning (`unfreeze_top_layers`), custom CNN alternative |
| | Approach and model selected and justified (1.5) | §1 table, `model.py` docstring |
| | Important parameters selected and explained (1.5) | §6 table, `config.py` comments |
| | Application correctly implemented and tested (6) | `train.py`, `evaluate.py` (test set), `app.py`, `predict.py` with error handling |
| **Model evaluation (20 % = 6)** | Appropriate metrics selected (1) | accuracy, precision, recall, F1, confusion matrix, ROC-AUC (`evaluate.py`) |
| | Evaluation implemented and results interpreted (2) | `results/` outputs + `evaluation_interpretation.md`, Streamlit *Model performance* tab |
| | Parameters adjusted from evaluation results (1) | §7 improvements, `experiments.csv`, validation-chosen confidence threshold |
| | Model/configuration saved for reproducibility (1) | `models/eye_health_cnn.keras`, `class_names.json`, `model_config.json`, `results/splits.csv`, fixed seed |
| | Deployed/demonstrated via web application (1) | `app.py` (Streamlit), §9 deployment |
| Task 8 (responsible use) | at least one limitation/risk + mitigation | §8 and the app's *Responsible use* tab |

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | Extract the Kaggle zip inside `data/raw/` (see `data/README.md`) |
| Weights download error | Connect to the internet once; MobileNetV2 weights are then cached |
| `ModuleNotFoundError: config` / `src` | Run commands from the project root; scripts add it to the path automatically |
| App says model not found | Run `python src/train.py` (and `evaluate.py`) first |
| Out-of-memory while training | `python src/train.py --batch-size 16` |
| Streamlit Cloud build fails | Use Python 3.11 and `tensorflow-cpu` in `requirements.txt` |
| Loading error after upgrading TensorFlow | Re-train with the installed version (model saved in Keras 3 `.keras` format) |

## 12. Suggested 5-minute demo script
1. Show `results/eda_class_distribution.png` and `eda_sample_images.png` (data understanding).
2. Explain the architecture diagram (§5) and parameters (§6).
3. Show `training_curves.png`, `confusion_matrix.png` and the metrics table.
4. Run `streamlit run app.py`; upload a file from `sample_images/`; point out prediction, probabilities, explanation.
5. Upload a `.txt` renamed to `.jpg` to show error handling; end on the *Responsible use* tab.
