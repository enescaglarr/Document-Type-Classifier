# Document Type Classifier

Classifies document images into five types for automated document routing:

| Class | Contents |
|---|---|
| `driving_license` | Driving license cards |
| `passport` | Passport data pages (multi-country) |
| `social_security` | Social security cards |
| `turkish_id` | Turkish identity cards (front & back) |
| `others` | Invoices, receipts, utility bills, other documents |

**Test set accuracy: 97.5%** (278/285 held-out images; per-class: driving license 46/50 ·
others 47/50 · passport 60/60 · social security 50/50 · Turkish ID 75/75). The test set shares no
files with the training set (exact duplicates were removed), and training is seeded, so
retraining from the same data reproduces this number.

## Model

The project went through two architectures:

1. **V1 — from-scratch CNN** (3 classes): three `Conv2D`/`MaxPooling2D` blocks (16→32→64) with
   dropout and flip/rotation/zoom augmentation. Reached 92.7% on the original 3-class problem.
2. **V2 — transfer learning** (5 classes, current): when visually similar classes were added
   (Turkish ID vs. passport data pages), the small CNN collapsed to 74%. The current model is an
   ImageNet-pretrained **MobileNetV2** backbone with a trained classification head, followed by a
   fine-tuning phase that unfreezes the top 50 base layers at a low learning rate (BatchNorm kept
   frozen). Augmentation (flip/rotation/zoom/brightness/contrast) lives inside the training model;
   an inference-only copy (without the augmentation block) is what gets saved to
   `output/cnn-model.h5`. Training is deterministic (fixed seed for the train/val split, shuffling,
   augmentation and weight init) and each phase keeps the weights of its best-validation epoch
   rather than the last one, so two runs on the same data produce the same model.

### Prerequisites

- **Python 3.10 or newer** (developed and tested on 3.13) — check with `python3 --version`
- **Git**
- ~2 GB free disk space (TensorFlow wheel + virtual environment)
- Optional: a GPU is *not* required. Inference runs fine on CPU; retraining on CPU takes
  several minutes per epoch, on Apple Silicon or an NVIDIA GPU it is considerably faster.

### 1. Clone the repository

```bash
git clone https://github.com/enescaglarr/Document-Type-Classifier.git
cd Document-Type-Classifier
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now be prefixed with `(.venv)`.

### 3. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow, Flask, gunicorn, Pillow and NumPy. The TensorFlow download is large
(several hundred MB), so the first install can take a few minutes.

### 4. Verify the installation

```bash
python -c "import tensorflow as tf, flask; print('TensorFlow', tf.__version__, '| Flask', flask.__version__)"
```

The **trained model ships with the repository** (`output/cnn-model.h5` and
`output/class_names.json`), so you can classify images immediately — no dataset or training
step is needed to get started. The dataset is only required for the *Test* and *Train* features
(see [Data](#data)).

## Run

All commands below assume the virtual environment is activated and you are in the repository
root.

### Web app (recommended)

```bash
python src/app.py
```

What happens:

1. The model is loaded from `output/cnn-model.h5`.
2. A local server starts on port **5001** and your default browser opens
   http://localhost:5001 automatically (open it manually if it doesn't).
3. Stop the server with `Ctrl+C`.

The UI has three tabs:

- **Classify** — drag & drop an image (JPG/PNG), get the predicted class with per-class
  confidence bars. Works out of the box with the shipped model.
- **Test** — one-click evaluation on the held-out test set in `input/Testing_Data/`: overall
  accuracy, per-class breakdown, and the list of misclassified images. Requires the dataset.
- **Train** — retrains the model from `input/Training_data/` with live epoch progress. The
  previous model is backed up to `output/cnn-model-prev.h5` and the new one goes live without a
  restart. Requires the dataset.

To change the port, edit `PORT = 5001` near the top of `src/app.py`. The server binds to
`0.0.0.0`, so it is also reachable from other machines on your network at
`http://<your-ip>:5001`.

### REST API

The web app also exposes a JSON endpoint. `POST /get-image-class` with a multipart `file`
field:

```bash
curl -X POST http://localhost:5001/get-image-class -F "file=@path/to/document.jpg"
# {"class": "turkish_id", "confidence(%)": 99.8, "scores": {...}}
```

Python example:

```python
import requests

with open("path/to/document.jpg", "rb") as f:
    r = requests.post("http://localhost:5001/get-image-class", files={"file": f})
print(r.json())
```

### Production-style server (gunicorn)

For a multi-worker server without the browser auto-open:

```bash
cd src/ML_Pipeline
./wsgi.sh          # gunicorn -b 0.0.0.0:5001 -w 2 -t 60 wsgi:app
```

Each worker loads its own copy of the model. On Windows (where gunicorn is not available) use
the web app command above instead.

### CLI (legacy)

```bash
cd src
python Engine.py
```

Interactive menu with train (`0`), predict/evaluate (`1`), and deploy (`2`) modes. The web app
supersedes it but it still works.

### TensorFlow.js export

```bash
python src/export_tfjs.py [output_dir]
```

Exports the trained model to TensorFlow.js Layers format (float16-quantized, ~5 MB) for fully
in-browser inference — no server sees the uploaded image. Re-run after every retraining.
(The MobileNetV2 export bridges Keras 3 weights into a tf-keras rebuild; see the script.)

### Troubleshooting

| Problem | Fix |
|---|---|
| `No model found at .../output/cnn-model.h5` | Make sure you cloned the full repo (the model is tracked in git). Otherwise add the dataset and retrain via the **Train** tab. |
| `Address already in use` on port 5001 | Another process is using the port — stop it, or change `PORT` in `src/app.py`. |
| Test/Train tab reports missing data | Create the `input/` directory with the layout described in [Data](#data). |
| Slow first prediction | TensorFlow warms up on the first inference; subsequent requests are fast. |
| `pip install` fails on TensorFlow | Confirm you are on Python 3.10–3.13 and a 64-bit interpreter; upgrade pip first. |

## Data

The `input/` directory is **not included in the repo** (see `.gitignore`): it contains third-party
dataset images, some with real-looking personal data, and none of it should be redistributed.
Expected layout:

```
input/
  Training_data/
    driving_license/  others/  passport/  social_security/  turkish_id/
  Testing_Data/
    driving_license/  others/  passport/  social_security/  turkish_id/
```

Sources used locally: web-collected samples for the US-document classes, and specimen documents
from the [MIDV-500 / MIDV-2019](https://arxiv.org/abs/1807.05786) academic datasets for
`turkish_id` and `passport` (test splits held out at the video-clip level), plus synthetic
composition/lighting variants generated from specimen cards.

## Project structure

```
src/
  app.py                  # single-command web app (serve + train + evaluate)
  export_tfjs.py          # TensorFlow.js export
  Engine.py               # legacy CLI (train / predict / deploy)
  ML_Pipeline/
    Preprocess.py         # dataset loading, train/val split, caching
    Train_Model.py        # MobileNetV2 transfer-learning model + fine-tuning
    Utils.py              # shared constants, model save/load
    deploy.py             # bare Flask API (used by Engine.py deploy mode)
    templates/index.html  # web UI (Classify / Test / Train tabs)
    wsgi.py, wsgi.sh      # gunicorn entry point
output/
  cnn-model.h5            # trained model (inference-only)
  class_names.json        # class order the model was trained with
```

## Known limits

The `turkish_id` class is trained on specimen cards, so the model generalizes across shooting
conditions (lighting, angle, background) but has seen few distinct physical card instances —
a genuinely unseen card can land near the decision boundary. More distinct real instances would
fix this; privacy makes that data (rightly) hard to obtain.

## License

[MIT](LICENSE)
