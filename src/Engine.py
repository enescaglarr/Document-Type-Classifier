import pathlib
import subprocess

import numpy as np
import tensorflow as tf

from ML_Pipeline import Train_Model
from ML_Pipeline import Utils
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Utils import load_model, save_model


val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))
if val == 0:
    data_dir = pathlib.Path("../input/Training_data/")
    image_count = len(list(data_dir.glob('*/*')))
    print("Number of images for training: ", image_count)

    train_ds, val_ds, class_names = apply(data_dir)
    ml_model = Train_Model.fit(train_ds, val_ds, class_names)
    # save the inference-only model (augmentation layers are not serializable)
    model_path = save_model(Train_Model.inference_model(ml_model))
    print("Model saved in: ", "../output/cnn-model.h5")
elif val == 1:
    model_path = "../output/cnn-model.h5"
    # model_path = input("Enter full model path: ")
    ml_model = load_model(model_path)

    test_data_dir = pathlib.Path("../input/Testing_Data/")
    image_count = len(list(test_data_dir.glob('*/*')))
    print("Number of images for testing: ", image_count)

    # shuffle=False keeps predictions aligned with test_ds.file_paths
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        shuffle=False,
        image_size=(Utils.img_height, Utils.img_width),
        batch_size=Utils.batch_size)

    class_names = test_ds.class_names
    file_paths = test_ds.file_paths
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])

    scores = tf.nn.softmax(ml_model.predict(test_ds)).numpy()
    pred_labels = scores.argmax(axis=1)

    print("\nMisclassified images:")
    for path, true_label, pred_label, score in zip(file_paths, true_labels, pred_labels, scores):
        if true_label != pred_label:
            name = "/".join(path.split("/")[-2:])
            print(f"  {name}  ->  {class_names[pred_label]} ({100 * score[pred_label]:.0f}%)")

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        in_class = true_labels == i
        print(f"  {class_name}: {int((pred_labels[in_class] == i).sum())}/{int(in_class.sum())}")

    correct = int((pred_labels == true_labels).sum())
    print(f"\nOverall accuracy: {correct / len(true_labels):.1%} ({correct}/{len(true_labels)})")
else:
    # For prod deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For dev deployment
    process = subprocess.Popen(['python', 'ML_Pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)
