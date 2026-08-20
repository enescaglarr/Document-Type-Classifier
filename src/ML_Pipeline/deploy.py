import pathlib

import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
import Utils

app = Flask(__name__)

# resolve relative to this file so the app works from any working directory
_BASE = pathlib.Path(__file__).resolve().parents[2]
model_path = str(_BASE / 'output' / 'cnn-model.h5')
input_image_path = str(_BASE / 'output' / 'api_input.jpg')
ml_model = Utils.load_model(model_path)
img_height = 180
img_width = 180
class_names = ['driving_license', 'others', 'passport', 'social_security', 'turkish_id']


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/get-image-class")
def get_image_class():
    image = request.files['file']
    image.save(input_image_path)
    img = tf.keras.utils.load_img(input_image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = ml_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    output = {"class": class_names[np.argmax(score)],
              "confidence(%)": float(100 * np.max(score)),
              "scores": {name: float(100 * s) for name, s in zip(class_names, score)}}
    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
