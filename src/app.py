"""Single-command web app: `python src/app.py` serves the classifier UI at http://localhost:5001

Everything Engine.py's CLI menu offered (train / predict / deploy) is available
from the web page instead: a Classify tab for predictions, a Test tab that
evaluates the model on the held-out test set, and a Train tab that retrains
the CNN with live epoch progress.
"""
import json
import os
import pathlib
import shutil
import tempfile
import threading
import webbrowser

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from ML_Pipeline import Preprocess, Train_Model

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "output" / "cnn-model.h5"
BACKUP_PATH = BASE_DIR / "output" / "cnn-model-prev.h5"
TRAIN_DIR = BASE_DIR / "input" / "Training_data"
TEST_DIR = BASE_DIR / "input" / "Testing_Data"

IMG_HEIGHT, IMG_WIDTH = 180, 180
BATCH_SIZE = 32
EPOCHS = 12
CLASS_NAMES = ['driving_license', 'others', 'passport', 'social_security', 'turkish_id']
PORT = 5001

app = Flask(__name__, template_folder="ML_Pipeline/templates")

ml_model = None
model_lock = threading.Lock()
train_status = {"state": "idle"}
eval_status = {"state": "idle"}


def load_model_if_present():
    global ml_model
    if MODEL_PATH.exists():
        with model_lock:
            ml_model = tf.keras.models.load_model(MODEL_PATH, compile=False)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/status")
def status():
    return {"model_loaded": ml_model is not None,
            "train": train_status, "evaluate": eval_status}


@app.post("/get-image-class")
def get_image_class():
    if ml_model is None:
        return jsonify({"error": "No trained model found. Run training first."}), 409
    image = request.files['file']
    # Per-request temp file so concurrent uploads can't overwrite each other
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    try:
        img = tf.keras.utils.load_img(tmp_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    finally:
        os.unlink(tmp_path)
    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    with model_lock:
        predictions = ml_model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    return {"class": CLASS_NAMES[int(np.argmax(score))],
            "confidence(%)": float(100 * np.max(score)),
            "scores": {name: float(100 * s) for name, s in zip(CLASS_NAMES, score)}}


def run_evaluation():
    try:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR, shuffle=False,
            image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
        class_names = test_ds.class_names
        file_paths = test_ds.file_paths
        true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])

        with model_lock:
            scores = tf.nn.softmax(ml_model.predict(test_ds, verbose=0)).numpy()
        pred_labels = scores.argmax(axis=1)

        misclassified = [
            {"file": "/".join(path.split("/")[-2:]),
             "true": class_names[t], "predicted": class_names[p],
             "confidence": float(100 * s[p])}
            for path, t, p, s in zip(file_paths, true_labels, pred_labels, scores)
            if t != p]
        per_class = [
            {"name": name,
             "correct": int((pred_labels[true_labels == i] == i).sum()),
             "total": int((true_labels == i).sum())}
            for i, name in enumerate(class_names)]
        correct = int((pred_labels == true_labels).sum())
        eval_status.update({
            "state": "done",
            "overall": {"correct": correct, "total": len(true_labels),
                        "accuracy": float(100 * correct / len(true_labels))},
            "per_class": per_class, "misclassified": misclassified})
    except Exception as e:
        eval_status.update({"state": "error", "message": str(e)})


@app.post("/evaluate")
def evaluate():
    if ml_model is None:
        return jsonify({"error": "No trained model found. Run training first."}), 409
    if eval_status.get("state") == "running":
        return jsonify({"error": "Evaluation already running."}), 409
    eval_status.clear()
    eval_status["state"] = "running"
    threading.Thread(target=run_evaluation, daemon=True).start()
    return {"started": True}


class EpochProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_status["epoch"] = epoch + 1
        train_status["history"].append({
            "epoch": epoch + 1,
            "accuracy": round(float(logs.get("accuracy", 0)), 4),
            "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
            "loss": round(float(logs.get("loss", 0)), 4),
            "val_loss": round(float(logs.get("val_loss", 0)), 4)})


def run_training():
    global ml_model
    try:
        train_ds, val_ds, class_names = Preprocess.apply(TRAIN_DIR)
        new_model = Train_Model.fit(train_ds, val_ds, class_names,
                                    epochs=EPOCHS, callbacks=[EpochProgress()])
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, BACKUP_PATH)  # keep previous model as -prev
        Train_Model.inference_model(new_model).save(MODEL_PATH)
        # record the class order this model was trained with (used by export_tfjs.py)
        (MODEL_PATH.parent / "class_names.json").write_text(json.dumps(class_names))
        with model_lock:
            ml_model = new_model
        train_status["state"] = "done"
    except Exception as e:
        train_status.update({"state": "error", "message": str(e)})


@app.post("/train")
def train():
    if train_status.get("state") == "running":
        return jsonify({"error": "Training already running."}), 409
    train_status.clear()
    train_status.update({"state": "running", "epoch": 0, "epochs": EPOCHS, "history": []})
    threading.Thread(target=run_training, daemon=True).start()
    return {"started": True}


if __name__ == '__main__':
    load_model_if_present()
    if ml_model is None:
        print("No model found at", MODEL_PATH, "- use the Train tab to create one.")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    print(f"Document Type Classifier running at http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT)
