"""Export the trained CNN to TensorFlow.js Layers format for in-browser inference.

Writes model.json + weight shard + classes.json to the portfolio's public folder
(override with the first CLI argument). Re-run after every retraining:

    python src/export_tfjs.py

The exported model contains only the inference layers (no augmentation, no
Rescaling) — the web page divides pixel values by 255 and applies softmax itself.
Weights are stored as float16, halving the download size.
"""
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "output" / "cnn-model.h5"
CLASSES_PATH = BASE_DIR / "output" / "class_names.json"
DEFAULT_OUT = BASE_DIR.parent.parent / "enes-portfolio" / "public" / "models" / "document-classifier"


def layer_configs(convs, denses, num_classes):
    def conv(name, filters):
        return {"class_name": "Conv2D", "config": {
            "name": name, "trainable": True, "filters": filters,
            "kernel_size": [3, 3], "strides": [1, 1], "padding": "same",
            "data_format": "channels_last", "dilation_rate": [1, 1],
            "activation": "relu", "use_bias": True,
            "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": None, "bias_regularizer": None,
            "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None}}

    def pool(name):
        return {"class_name": "MaxPooling2D", "config": {
            "name": name, "trainable": True, "pool_size": [2, 2], "padding": "valid",
            "strides": [2, 2], "data_format": "channels_last"}}

    def dense(name, units, activation):
        return {"class_name": "Dense", "config": {
            "name": name, "trainable": True, "units": units, "activation": activation,
            "use_bias": True,
            "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": None, "bias_regularizer": None,
            "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None}}

    return [
        {"class_name": "InputLayer", "config": {
            "batch_input_shape": [None, 180, 180, 3], "dtype": "float32",
            "sparse": False, "name": "input_image"}},
        conv("conv1", convs[0].filters), pool("pool1"),
        conv("conv2", convs[1].filters), pool("pool2"),
        conv("conv3", convs[2].filters), pool("pool3"),
        {"class_name": "Flatten", "config": {
            "name": "flatten", "trainable": True, "data_format": "channels_last"}},
        dense("dense1", denses[0].units, "relu"),
        dense("dense2", num_classes, "linear"),
    ]


def main():
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    num_classes = denses[-1].units

    if CLASSES_PATH.exists():
        classes = json.loads(CLASSES_PATH.read_text())
    else:
        classes = ['driving_license', 'others', 'social_security']
    assert len(classes) == num_classes, (
        f"Model has {num_classes} outputs but class list is {classes}. "
        "Retrain via the web app (which writes output/class_names.json) and re-run.")

    # weights, float16-quantized, concatenated in manifest order
    manifest_weights, blobs = [], []
    named = ([("conv1", convs[0]), ("conv2", convs[1]), ("conv3", convs[2]),
              ("dense1", denses[0]), ("dense2", denses[1])])
    for name, layer in named:
        kernel, bias = layer.get_weights()
        for suffix, arr in (("kernel", kernel), ("bias", bias)):
            manifest_weights.append({
                "name": f"{name}/{suffix}", "shape": list(arr.shape), "dtype": "float32",
                "quantization": {"dtype": "float16"}})
            blobs.append(arr.astype("<f2").tobytes())

    shard = "group1-shard1of1.bin"
    (out_dir / shard).write_bytes(b"".join(blobs))
    model_json = {
        "format": "layers-model",
        "generatedBy": "DocumentTypeClassifier-CNN export_tfjs.py",
        "convertedBy": None,
        "modelTopology": {
            "keras_version": "2.15.0", "backend": "tensorflow",
            "model_config": {"class_name": "Sequential", "config": {
                "name": "document_classifier",
                "layers": layer_configs(convs, denses, num_classes)}}},
        "weightsManifest": [{"paths": [shard], "weights": manifest_weights}],
    }
    (out_dir / "model.json").write_text(json.dumps(model_json))
    (out_dir / "classes.json").write_text(json.dumps(classes))

    size_mb = (out_dir / shard).stat().st_size / 1e6
    print(f"Exported {num_classes}-class model to {out_dir} ({size_mb:.1f} MB weights)")
    print("Classes:", classes)


if __name__ == "__main__":
    main()
