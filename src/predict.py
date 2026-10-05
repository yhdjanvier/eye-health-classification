"""
predict.py - inference helpers used by the Streamlit app (no Streamlit imports here,
so this module can also be unit-tested or used from a notebook).

WHAT : validates an uploaded file, preprocesses it exactly like training, runs the CNN and
       builds a plain-language explanation of the result.
"""
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

import config
from src.preprocessing import resize_to_uint8

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_FILE_MB = 10
MIN_SIDE = 64


class InvalidImageError(Exception):
    """Raised with a user-friendly message when an uploaded file cannot be used."""


class ModelNotFoundError(Exception):
    """Raised when the trained model or its configuration files are missing."""


@dataclass
class PredictionResult:
    class_names: list
    probs: np.ndarray
    top_index: int
    top_class: str
    top_display: str
    confidence: float
    second_display: str
    second_confidence: float
    is_healthy: bool
    uncertain: bool
    confidence_level: str


def artifacts_available() -> bool:
    return all(p.exists() for p in (config.MODEL_PATH, config.CLASS_NAMES_PATH, config.MODEL_CONFIG_PATH))


def load_artifacts():
    """Load (model, class_names, model_config). Raises ModelNotFoundError if training was not run."""
    if not artifacts_available():
        raise ModelNotFoundError(
            "Trained model files were not found in the 'models/' folder. "
            "Run `python src/train.py` first (see README.md)."
        )
    from tensorflow import keras
    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    class_names = json.loads(config.CLASS_NAMES_PATH.read_text())
    cfg = json.loads(config.MODEL_CONFIG_PATH.read_text())
    return model, class_names, cfg


def read_image_bytes(data: bytes, filename: str = "upload") -> Image.Image:
    """Validate raw bytes and return an RGB PIL image, or raise InvalidImageError."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported file type '{ext or 'unknown'}'. Please upload a JPG, JPEG or PNG image.")
    if len(data) == 0:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise InvalidImageError(f"The file is larger than {MAX_FILE_MB} MB. Please upload a smaller image.")
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()                               # detects corrupt / truncated files
        img = Image.open(BytesIO(data))
        fmt = img.format
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise InvalidImageError("This file could not be read as an image. It may be corrupted or not a real image.")
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(f"The file content is '{fmt}', not JPEG/PNG. Please upload a valid JPG or PNG image.")
    if min(img.size) < MIN_SIDE:
        raise InvalidImageError(f"The image is too small ({img.width}x{img.height}px). Minimum side is {MIN_SIDE}px.")
    return img


def preprocess(img: Image.Image, img_size) -> tf.Tensor:
    """Identical to training: bilinear resize -> uint8 -> float32 (0-255); scaling is inside the model."""
    arr = tf.convert_to_tensor(np.asarray(img, dtype=np.uint8))
    resized = resize_to_uint8(arr, tuple(img_size))
    return tf.cast(resized, tf.float32)[tf.newaxis, ...]


def predict(model, class_names, cfg, img: Image.Image) -> PredictionResult:
    batch = preprocess(img, cfg["img_size"])
    probs = np.asarray(model(batch, training=False))[0].astype(float)
    probs = probs / probs.sum()
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    threshold = float(cfg.get("confidence_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD))
    conf = float(probs[top])
    level = "High" if conf >= 0.90 else "Moderate" if conf >= threshold else "Low"
    return PredictionResult(
        class_names=class_names, probs=probs, top_index=top, top_class=class_names[top],
        top_display=config.get_class_info(class_names[top])["display"], confidence=conf,
        second_display=config.get_class_info(class_names[second])["display"],
        second_confidence=float(probs[second]),
        is_healthy=bool(config.get_class_info(class_names[top])["healthy"]),
        uncertain=conf < threshold, confidence_level=level,
    )


def build_explanation(result: PredictionResult, cfg) -> str:
    """Plain-language explanation of the prediction (markdown)."""
    info = config.get_class_info(result.top_class)
    threshold = float(cfg.get("confidence_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD))
    parts = [
        f"**What the model saw:** the image is most similar to the **{result.top_display}** examples it learned from, "
        f"with **{result.confidence:.1%}** confidence ({result.confidence_level.lower()} confidence).",
        f"**About this result:** {info['description']}",
    ]
    margin = result.confidence - result.second_confidence
    if result.uncertain:
        parts.append(
            f"**Why it is marked uncertain:** confidence is below the reliability threshold ({threshold:.0%}) and the "
            f"runner-up was **{result.second_display}** ({result.second_confidence:.1%}). Try a clearer, well-lit fundus photo "
            "and consult a professional rather than relying on this result.")
    elif margin < 0.25:
        parts.append(
            f"**Close call:** the second most likely class was **{result.second_display}** ({result.second_confidence:.1%}), "
            "so the model found the image partly similar to both.")
    parts.append(f"**Suggested next step:** {info['advice']}")
    return "\n\n".join(parts)
