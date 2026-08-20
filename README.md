# Document Type Classifier

Classifies document images into five types for automated document routing:

| Class | Contents |
|---|---|
| `driving_license` | Driving license cards |
| `passport` | Passport data pages (multi-country) |
| `social_security` | Social security cards |
| `turkish_id` | Turkish identity cards (front & back) |
| `others` | Invoices, receipts, utility bills, other documents |

**Test set accuracy: 97.5%** (278/285 held-out images, per-class: 45/50 · 60/60 · 50/50 · 75/75 · 48/50).

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
   `output/cnn-model.h5`.

## Setup

```bash
python3 -m venv .venv           # Python 3.10+ works; developed on 3.13
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### Web app (recommended)

```bash
python src/app.py
```

Starts a local server at http://localhost:5001 (opens your browser automatically) with three tabs:

- **Classify** — drag & drop an image, get the predicted class with per-class confidence bars.
- **Test** — one-click evaluation on the held-out test set: overall accuracy, per-class breakdown,
  and the list of misclassified images.
- **Train** — retrains the model from `input/Training_data/` with live epoch progress. The previous
  model is backed up to `output/cnn-model-prev.h5` and the new one goes live without a restart.

### REST API

`POST /get-image-class` with a multipart `file` field:

```bash
curl -X POST http://localhost:5001/get-image-class -F "file=@path/to/document.jpg"
# {"class": "turkish_id", "confidence(%)": 99.8, "scores": {...}}
```

For a production-style server: `cd src/ML_Pipeline && ./wsgi.sh` (gunicorn).

### CLI (legacy)

`cd src && python Engine.py` — interactive menu with train (0), predict/evaluate (1),
and deploy (2) modes. The web app supersedes it but it still works.

### TensorFlow.js export

```bash
python src/export_tfjs.py [output_dir]
```

Exports the trained model to TensorFlow.js Layers format (float16-quantized, ~5 MB) for fully
in-browser inference — no server sees the uploaded image. Re-run after every retraining.
(The MobileNetV2 export bridges Keras 3 weights into a tf-keras rebuild; see the script.)

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
