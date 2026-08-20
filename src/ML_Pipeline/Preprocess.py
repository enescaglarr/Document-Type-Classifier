import tensorflow as tf
from ML_Pipeline import Utils


# Function to cache data for tensorflow
def cache_data(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


# Function to call dependent functions
def apply(data_dir):
    print("Preprocessing started....")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(Utils.img_height, Utils.img_width),
        batch_size=Utils.batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(Utils.img_height, Utils.img_width),
        batch_size=Utils.batch_size)

    class_names = train_ds.class_names
    print("Class Names: ", class_names)
    print("Data loading completed....")

    train_ds, val_ds = cache_data(train_ds, val_ds)

    print("Preprocessing completed....")
    return train_ds, val_ds, class_names
