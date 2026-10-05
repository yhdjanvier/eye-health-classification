"""
app.py - Streamlit web application: Eye Health Classification System.

Run locally:  streamlit run app.py
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from src import predict as pred
from src.preprocessing import assess_image_quality

st.set_page_config(page_title="Eye Health Classification System", page_icon="👁️", layout="wide")

st.markdown(
    """
    <style>
    .result-card {padding: 1.1rem 1.3rem; border-radius: 12px; margin-bottom: 0.8rem; border: 1px solid #d0d7de;}
    .result-healthy {background:#e8f6ec; border-left: 8px solid #2e9e4f;}
    .result-disease {background:#fff4e5; border-left: 8px solid #f08c00;}
    .result-uncertain {background:#eef1f5; border-left: 8px solid #6c757d;}
    .result-card h2 {margin: 0 0 0.2rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the CNN model...")
def get_artifacts():
    return pred.load_artifacts()


def disclaimer_banner():
    st.warning(
        "**Not a medical diagnosis.** This is an AI screening / research tool built for learning and demonstration. "
        "It cannot replace an examination by a qualified eye-care professional. If you have symptoms or concerns, see an ophthalmologist.",
        icon="⚠️",
    )


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- header
st.title("👁️ Eye Health Classification System")
st.caption("Deep-learning (CNN) screening of retinal fundus images · TensorFlow / Keras + Streamlit")
disclaimer_banner()

if not pred.artifacts_available():
    st.error(
        "The trained model was not found in the `models/` folder.\n\n"
        "1. Put the dataset in `data/raw/` (see `data/README.md`)\n"
        "2. Run `python src/train.py` and then `python src/evaluate.py`\n"
        "3. Restart this app."
    )
    st.stop()

try:
    model, class_names, cfg = get_artifacts()
except Exception as exc:  # noqa: BLE001
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

metrics = load_json(config.METRICS_PATH)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model information")
    st.write(f"**Architecture:** {cfg.get('architecture', 'CNN')}")
    st.write(f"**Input size:** {cfg['img_size'][0]}×{cfg['img_size'][1]} RGB")
    st.write("**Classes:**")
    for c in class_names:
        st.write(f"- {config.get_class_info(c)['display']}")
    if metrics:
        st.metric("Test accuracy", f"{metrics['test']['accuracy']:.1%}")
        st.metric("Macro F1-score", f"{metrics['test']['f1_macro']:.1%}")
    st.write(f"**Uncertainty threshold:** {cfg.get('confidence_threshold', config.DEFAULT_CONFIDENCE_THRESHOLD):.0%}")
    st.caption(f"Trained: {cfg.get('trained_at', 'n/a')}")
    st.divider()
    st.caption("CAT2 Practical 2026 · Python and Fundamentals of AI (ITLPA701)")

tab_classify, tab_perf, tab_resp, tab_how = st.tabs(
    ["🔍 Classify an image", "📊 Model performance", "⚖️ Responsible use", "🧠 How it works"])

# --------------------------------------------------------------------------- classify
with tab_classify:
    st.subheader("Upload a retinal (fundus) image")
    st.info("The model was trained on **retinal fundus photographs** (the image of the back of the eye taken with a "
            "fundus camera). Ordinary photos of the outside of the eye are not supported.")
    uploaded = st.file_uploader("Choose an image (JPG, JPEG or PNG, max 10 MB)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.write("Waiting for an image. Sample images (if available) are in the `sample_images/` folder.")
    else:
        try:
            image = pred.read_image_bytes(uploaded.getvalue(), uploaded.name)
        except pred.InvalidImageError as err:
            st.error(f"❌ {err}")
            image = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Unexpected problem while reading the file: {exc}")
            image = None

        if image is not None:
            col_img, col_res = st.columns([1, 1.2], gap="large")
            with col_img:
                st.image(image, caption=f"{uploaded.name} ({image.width}×{image.height}px)", use_container_width=True)
                for w in assess_image_quality(image):
                    st.warning(w, icon="📷")
            with col_res:
                try:
                    with st.spinner("Analysing image..."):
                        result = pred.predict(model, class_names, cfg, image)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Prediction failed: {exc}")
                    st.stop()

                if result.uncertain:
                    css, icon, headline = "result-uncertain", "❓", "Inconclusive result"
                elif result.is_healthy:
                    css, icon, headline = "result-healthy", "✅", result.top_display
                else:
                    css, icon, headline = "result-disease", "🩺", f"Possible {result.top_display}"
                st.markdown(
                    f"<div class='result-card {css}'><h2>{icon} {headline}</h2>"
                    f"Confidence: <b>{result.confidence:.1%}</b> ({result.confidence_level})"
                    f"{'' if not result.uncertain else ' · most likely: ' + result.top_display}</div>",
                    unsafe_allow_html=True)

                st.markdown("#### Class probabilities")
                prob_df = pd.DataFrame({
                    "Class": [config.get_class_info(c)["display"] for c in class_names],
                    "Probability (%)": (result.probs * 100).round(2)})
                for _, row in prob_df.sort_values("Probability (%)", ascending=False).iterrows():
                    st.progress(float(row["Probability (%)"]) / 100, text=f"{row['Class']}: {row['Probability (%)']:.2f}%")
                st.bar_chart(prob_df.set_index("Class"))

                st.markdown("#### Explanation")
                explanation = pred.build_explanation(result, cfg)
                st.markdown(explanation)

                report = (f"Eye Health Classification System - result\nGenerated: {datetime.now():%Y-%m-%d %H:%M}\n"
                          f"File: {uploaded.name}\nPrediction: {result.top_display}\n"
                          f"Confidence: {result.confidence:.1%} ({result.confidence_level})\n"
                          f"Uncertain: {result.uncertain}\n\nProbabilities:\n"
                          + "\n".join(f"  {r['Class']}: {r['Probability (%)']:.2f}%" for _, r in prob_df.iterrows())
                          + "\n\nDISCLAIMER: AI screening/research tool only - not a medical diagnosis.\n")
                st.download_button("⬇️ Download result (TXT)", report, file_name="eye_screening_result.txt")
            disclaimer_banner()

# --------------------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Model performance on the unseen test set")
    if not metrics:
        st.info("No evaluation results yet. Run `python src/evaluate.py` to generate them.")
    else:
        t = metrics["test"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{t['accuracy']:.2%}")
        c2.metric("Precision (macro)", f"{t['precision_macro']:.2%}")
        c3.metric("Recall (macro)", f"{t['recall_macro']:.2%}")
        c4.metric("F1-score (macro)", f"{t['f1_macro']:.2%}")
        st.caption(f"Computed on {t['n_samples']} test images never used for training or model selection.")
        if config.PER_CLASS_PATH.exists():
            st.markdown("**Per-class metrics**")
            st.dataframe(pd.read_csv(config.PER_CLASS_PATH).style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1_score": "{:.3f}"}), use_container_width=True)
        s = metrics.get("screening_healthy_vs_diseased")
        if s:
            st.markdown(f"**Healthy-vs-Diseased view:** sensitivity {s['sensitivity_disease_detected']:.2%} · "
                        f"specificity {s['specificity_healthy_confirmed']:.2%}")
        left, right = st.columns(2)
        for col, name, cap in ((left, "confusion_matrix.png", "Confusion matrix"),
                               (right, "confusion_matrix_normalized.png", "Normalised confusion matrix"),
                               (left, "training_curves.png", "Training curves"),
                               (right, "per_class_metrics.png", "Per-class metrics")):
            p = config.RESULTS_DIR / name
            if p.exists():
                col.image(str(p), caption=cap, use_container_width=True)
        interp = config.RESULTS_DIR / "evaluation_interpretation.md"
        if interp.exists():
            with st.expander("Interpretation of the results", expanded=True):
                st.markdown(interp.read_text())

# --------------------------------------------------------------------------- responsible use
with tab_resp:
    st.subheader("Responsible use, limitations and risk reduction")
    st.error("This system is an **AI screening / research tool**. It is **not a medical device** and does **not provide a medical diagnosis**.")
    st.markdown(
        """
**Limitations**
- **Image type:** trained only on retinal fundus photographs; other eye photos give meaningless results.
- **Only four classes:** cataract, diabetic retinopathy, glaucoma and normal. Other eye diseases (e.g. macular degeneration) are *not* recognised and may be mislabelled.
- **Dataset bias:** the data come from limited sources and populations, cameras and imaging conditions; accuracy may drop for other groups or devices.
- **No clinical context:** age, symptoms, eye pressure and medical history are ignored.
- **Over-confidence:** neural networks can be confidently wrong, especially for unusual or poor-quality images.
- **Possible label noise / duplicate-like images** in public datasets can make test scores look better than real-world performance.

**Risks**
- A *false negative* (disease predicted as healthy) may delay treatment; a *false positive* may cause unnecessary worry.
- Users may over-trust an AI answer.

**How the risks are reduced in this project**
- Persistent disclaimers and a plain-language explanation of every result.
- A confidence threshold chosen on validation data: low-confidence results are labelled **inconclusive**.
- Image-quality and "does not look like a fundus photo" warnings, plus strict file validation.
- Class weights and per-class recall reporting, so weak classes are visible instead of hidden by overall accuracy.
- Uploaded images are processed in memory only and are **not stored**.
- Recommended before any real use: clinical validation on local data, ophthalmologist review of every result, bias audits, and regulatory approval.
        """
    )

# --------------------------------------------------------------------------- how it works
with tab_how:
    st.subheader("System architecture and workflow")
    st.code(
        "Fundus image (JPG/PNG)\n"
        "   -> validation (type, size, corrupt-file check)\n"
        "   -> resize to 224x224 (same function as training)\n"
        "   -> [inside the model] pixel scaling to [-1, 1]\n"
        "   -> MobileNetV2 CNN backbone (ImageNet pre-trained, fine-tuned)\n"
        "   -> Global average pooling -> BatchNorm -> Dropout -> Dense(256) -> Dropout\n"
        "   -> Softmax over the disease classes\n"
        "   -> probabilities + confidence threshold + plain-language explanation",
        language="text")
    st.markdown(
        """
**Training workflow:** download dataset → verify images & remove duplicates → stratified 70/15/15 split → augmentation
(flip, rotation, zoom, contrast, brightness) → phase 1 (train new head) → phase 2 (fine-tune top layers) → evaluate on test set
(accuracy, precision, recall, F1, confusion matrix) → choose confidence threshold on validation data → save model + configuration.
        """
    )
