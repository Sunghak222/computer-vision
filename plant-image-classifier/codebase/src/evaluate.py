import json
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from feature_extraction import FeatureExtractor
import csv
import shutil
import pickle

from train import build_feature_dataset_from_json, load_class_names

TEST_DIR = "data/test"

MODEL_PATH = "models/experiments/20260322_174659__rf__hsv__aug__tuned/model.pkl"
CLASS_NAMES_PATH = "models/class_names.json"

RESULT_DIR = "results"
MISCLASS_DIR = "results/misclassified"

def save_prediction_records(
    image_paths,
    y_true,
    y_pred,
    class_names,
    csv_path: Path,
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_class", "pred_class", "correct"])

        for img_path, yt, yp in zip(image_paths, y_true, y_pred):
            true_class = class_names[yt]
            pred_class = class_names[yp]
            correct = int(yt == yp)
            writer.writerow([img_path, true_class, pred_class, correct])


def export_misclassified_images(
    image_paths,
    y_true,
    y_pred,
    class_names,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, yt, yp in zip(image_paths, y_true, y_pred):
        if yt == yp:
            continue

        true_class = class_names[yt]
        pred_class = class_names[yp]

        src = Path(img_path)
        dst_name = f"true_{true_class}__pred_{pred_class}__{src.name}"
        dst = output_dir / dst_name
        shutil.copy2(src, dst)

def ensure_dirs():
    Path(RESULT_DIR).mkdir(exist_ok=True)
    Path(MISCLASS_DIR).mkdir(exist_ok=True)


def load_model():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    class_names = load_class_names(Path(CLASS_NAMES_PATH))
    return model, class_names


def save_confusion_matrix(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, xticks_rotation=45)

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

def main():

    ensure_dirs()

    print("Loading model...")
    model, class_names = load_model()

    feature_set = ["hsv"]

    extractor = FeatureExtractor(
        img_size=(128, 128),
        feature_set=feature_set
    )


    print("Extracting test features...")


    X_test, y_test, test_used_classes, test_image_paths = build_feature_dataset_from_json(
        Path(TEST_DIR),
        extractor,
        class_names
    )

    if class_names != test_used_classes:
        raise ValueError("Test class names do not match training class names")

    print("Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}")

    save_confusion_matrix(y_test, y_pred, class_names)

    prediction_csv_path = Path(RESULT_DIR) / "prediction_records.csv"

    save_prediction_records(
        test_image_paths,
        y_test,
        y_pred,
        class_names,
        prediction_csv_path,
    )

    export_misclassified_images(
        test_image_paths,
        y_test,
        y_pred,
        class_names,
        Path(MISCLASS_DIR),
    )

    print("\nSaved confusion matrix to results/confusion_matrix.png")
    print("Saved prediction records to results/prediction_records.csv")
    print("Saved misclassified images to results/misclassified/")


if __name__ == "__main__":
    main()