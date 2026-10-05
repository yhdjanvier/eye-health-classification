"""
model.py - CNN architectures (feature/model engineering).

Two architectures are provided; the default is transfer learning with MobileNetV2:

1. MobileNetV2 (ImageNet pre-trained) + new classification head  -> default, most accurate
2. Custom 4-block CNN trained from scratch                        -> baseline / comparison

Both models contain their own augmentation + pixel-scaling layers, so the saved .keras file
accepts RAW float images in the 0-255 range. The Streamlit app therefore needs no extra scaling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorflow import keras
from tensorflow.keras import layers, regularizers

import config


def augmentation_block():
    """
    Data augmentation (active ONLY during training, automatically disabled at inference).
    Simulates natural variation in camera angle, position, lighting and contrast.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.05),                                   # about +/-18 degrees
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
        layers.RandomBrightness(0.10, value_range=(0, 255)),
    ], name="augmentation")


def build_transfer_model(num_classes, img_size=config.IMG_SIZE, dropout=config.DROPOUT,
                         dense_units=config.DENSE_UNITS, l2_weight=config.L2_WEIGHT):
    inputs = keras.Input(shape=(*img_size, 3), name="image")
    x = augmentation_block()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)   # 0-255 -> [-1, 1]
    try:
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    except Exception as exc:                                               # noqa: BLE001
        raise RuntimeError(
            "Could not download the ImageNet weights for MobileNetV2. Check your internet "
            "connection (needed only once; weights are cached in ~/.keras)."
        ) from exc
    base.trainable = False                                                 # phase 1: frozen
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="head_dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_weight), name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="eye_health_mobilenetv2")


def build_custom_cnn(num_classes, img_size=config.IMG_SIZE, dropout=config.DROPOUT,
                     dense_units=config.DENSE_UNITS, l2_weight=config.L2_WEIGHT):
    inputs = keras.Input(shape=(*img_size, 3), name="image")
    x = augmentation_block()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)
    for i, filters in enumerate((32, 64, 128, 256), start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}a")(x)
        x = layers.BatchNormalization(name=f"bn{i}a")(x)
        x = layers.Activation("relu", name=f"relu{i}a")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}b")(x)
        x = layers.BatchNormalization(name=f"bn{i}b")(x)
        x = layers.Activation("relu", name=f"relu{i}b")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        x = layers.Dropout(0.15 + 0.05 * i, name=f"drop{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_weight), name="head_dense")(x)
    x = layers.Dropout(0.5, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="eye_health_custom_cnn")


def build_model(architecture, num_classes, **kwargs):
    if architecture == "mobilenetv2":
        return build_transfer_model(num_classes, **kwargs)
    if architecture == "custom":
        return build_custom_cnn(num_classes, **kwargs)
    raise ValueError(f"Unknown architecture '{architecture}'. Use 'mobilenetv2' or 'custom'.")


def get_base_model(model):
    """Return the nested MobileNetV2 backbone of the transfer model."""
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2"):
            return layer
    raise ValueError("This model has no MobileNetV2 backbone.")


def unfreeze_top_layers(model, n_layers=config.UNFREEZE_LAYERS):
    """
    Phase-2 fine-tuning: unfreeze the last `n_layers` of the backbone but keep every
    BatchNormalization layer frozen (standard practice - protects the learned statistics
    when the dataset is small). The model must be re-compiled afterwards.
    """
    base = get_base_model(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return base
