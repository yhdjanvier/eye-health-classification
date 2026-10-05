"""
preprocessing.py - the ONE image-preprocessing definition used by training AND the app.

WHAT : resize helper (TensorFlow) + a light image-quality checker (NumPy only).
WHY  : if training and inference resize images differently, accuracy silently drops.
       Both import `resize_to_uint8` from here, so they cannot disagree.
       Pixel scaling to [-1, 1] lives INSIDE the saved model (Rescaling layer).
WHERE: src/data_utils.py (training pipeline), src/predict.py (Streamlit inference).
"""
import numpy as np
import tensorflow as tf


def resize_to_uint8(image, img_size):
    """Bilinear-resize an HxWx3 image tensor to `img_size`, returned as uint8 (0-255)."""
    image = tf.image.resize(image, img_size, method="bilinear")
    return tf.cast(tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8)


def assess_image_quality(pil_img):
    """
    Return a list of human-readable warnings about the uploaded image.
    These are soft heuristics (they never block a prediction); they help the user
    understand why a result might be unreliable.
    """
    warnings = []
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    if min(h, w) < 150:
        warnings.append(f"The image is small ({w}x{h}px). Higher resolution gives more reliable results.")

    gray = arr.mean(axis=2)
    mean_brightness = float(gray.mean())
    if mean_brightness < 30:
        warnings.append("The image is very dark.")
    elif mean_brightness > 225:
        warnings.append("The image is over-exposed (very bright).")

    # Sharpness: variance of a Laplacian filter on a 256-px thumbnail
    step_h, step_w = max(h // 256, 1), max(w // 256, 1)
    g = gray[::step_h, ::step_w]
    if g.shape[0] > 3 and g.shape[1] > 3:
        lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        if float(lap.var()) < 5.0:
            warnings.append("The image looks very blurry or low-detail.")

    # Fundus-likeness heuristic: retinal photos are reddish/orange with darker corners.
    r, b = float(arr[..., 0].mean()), float(arr[..., 2].mean())
    ch, cw = max(h // 10, 1), max(w // 10, 1)
    corners = np.concatenate([
        gray[:ch, :cw].ravel(), gray[:ch, -cw:].ravel(),
        gray[-ch:, :cw].ravel(), gray[-ch:, -cw:].ravel(),
    ]).mean()
    centre = gray[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw].mean()
    if not (r > b * 1.15 and corners < centre * 0.8):
        warnings.append(
            "This does not look like a typical retinal (fundus) photograph. The model was trained "
            "only on fundus images, so its prediction may be meaningless for other kinds of eye photos."
        )
    return warnings
