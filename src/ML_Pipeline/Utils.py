import keras

batch_size = 32
img_height = 180
img_width = 180


def save_model(model):
    model.save("../output/cnn-model.h5")
    return True


def load_model(model_path):
    try:
        model = keras.models.load_model(model_path, compile=False)
    except OSError as e:
        print(f"Could not find or open model file at '{model_path}': {e}")
        exit(0)

    return model
