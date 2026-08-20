from ML_Pipeline import Utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf


# Build an inference-only copy sharing the trained layers. The augmentation
# block is skipped: it is inactive at inference time anyway, and its random
# color layers cannot be serialized to the legacy H5 format.
def inference_model(model):
    inf = keras.Sequential([keras.Input(shape=(Utils.img_height, Utils.img_width, 3))])
    for layer in model.layers[1:]:
        inf.add(layer)
    return inf


# Function to train ML model
def train(model, train_ds, val_ds, epochs=12, callbacks=None):
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model


# Function to initiate model and training data
# Transfer learning: a frozen ImageNet-pretrained MobileNetV2 base extracts
# features, and only the classification head is trained. The original
# from-scratch 3-block CNN plateaued once visually similar classes were added
# (Turkish ID vs. passport data pages).
def fit(train_ds, val_ds, class_names, epochs=12, callbacks=None):
    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(Utils.img_height,
                                           Utils.img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.25, value_range=(0.0, 255.0)),
            layers.RandomContrast(0.3),
        ]
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(Utils.img_height, Utils.img_width, 3),
        include_top=False,
        weights="imagenet")
    base.trainable = False

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 127.5, offset=-1),  # MobileNetV2 expects [-1, 1]
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())

    model = train(model, train_ds, val_ds, epochs=epochs, callbacks=callbacks)

    # Phase 2 fine-tuning: unfreeze the top of the base at a low learning rate
    # so the network can learn document-specific cues beyond generic ImageNet
    # features. BatchNorm layers stay frozen to keep training stable.
    for layer in base.layers[-50:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    fine_tune_epochs = max(3, epochs // 3)
    print(f"Fine-tuning top layers for {fine_tune_epochs} epochs...")
    model = train(model, train_ds, val_ds, epochs=fine_tune_epochs, callbacks=callbacks)

    return model
