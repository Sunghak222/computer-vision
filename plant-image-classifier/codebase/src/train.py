# src/train.py

import time
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import FeatureExtractor
from experiment_utils import make_experiment_output_paths, save_json, append_jsonl
from experiment_config import ExperimentConfig

# =========================
# Config
# =========================
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "svm_model.pkl"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

RESULTS_DIR = Path("results")
LOG_PATH = RESULTS_DIR / "train_log.json"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# Utils
# =========================
def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"class_names.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not all(isinstance(x, str) for x in class_names):
        raise ValueError("class_names.json must be a JSON list of strings.")

    return class_names


def is_valid_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def build_feature_dataset_from_json(data_dir: Path, extractor: FeatureExtractor, class_names: list[str]):
    """
    Build dataset using ONLY the class names listed in class_names.json.
    Label indices strictly follow the order of class_names.
    Also returns image_paths for later error analysis.
    """


    X = []
    y = []
    image_paths = []

    used_class_names = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name

        if not class_dir.exists() or not class_dir.is_dir():
            print(f"[WARNING] Missing class folder in {data_dir}: {class_name}")
            continue

        image_paths_in_class = [p for p in sorted(class_dir.iterdir()) if is_valid_image_file(p)]

        if len(image_paths_in_class) == 0:
            print(f"[WARNING] Empty class folder in {data_dir}: {class_name}")
            continue

        used_class_names.append(class_name)

        for img_path in image_paths_in_class:
            feat = extractor.extract(img_path)
            X.append(feat)
            y.append(class_idx)
            image_paths.append(str(img_path))

    X = np.array(X)
    y = np.array(y)

    return X, y, used_class_names, image_paths


def filter_dataset_to_used_classes(X, y, original_class_names, used_class_names):
    """
    Remap labels so that dataset labels follow the order of used_class_names only.
    """
    new_label_map = {cls_name: idx for idx, cls_name in enumerate(used_class_names)}

    filtered_X = []
    filtered_y = []

    for xi, yi in zip(X, y):
        cls_name = original_class_names[yi]

        if cls_name in new_label_map:
            filtered_X.append(xi)
            filtered_y.append(new_label_map[cls_name])

    return np.array(filtered_X), np.array(filtered_y)


def build_model(model_name, model_params=None):
    model_params = model_params or {}

    if model_name == "svm":
        c = model_params.get("C", 10)
        gamma = model_params.get("gamma", "scale")
        kernel = model_params.get("kernel", "rbf")
        probability = model_params.get("probability", True)

        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel=kernel,
                C=c,
                gamma=gamma,
                probability=probability
            ))
        ])

    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 200),
            max_depth=model_params.get("max_depth", None),
            min_samples_split=model_params.get("min_samples_split", 2),
            random_state=model_params.get("random_state", 42)
        )

    elif model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=model_params.get("n_estimators", 300),
            max_depth=model_params.get("max_depth", 6),
            learning_rate=model_params.get("learning_rate", 0.1),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            eval_metric="mlogloss"
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

def append_json(obj, path: Path):  
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []

    if not isinstance(data, list):  # safety
        data = []

    data.append(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_experiment(config: ExperimentConfig):
    output_paths = make_experiment_output_paths(config)

    class_names_from_json = load_class_names(CLASS_NAMES_PATH)
    print(f"Class list from JSON: {class_names_from_json}")

    extractor = FeatureExtractor(
        img_size=config.img_size,
        feature_set=config.feature_set
    )

    print(f"Running experiment: {config.experiment_tag()}")
    print(f"Using feature set: {config.feature_set}")
    print(f"Using model: {config.model_name}")

    print("Loading training features...")
    X_train, y_train, train_used_classes, train_image_paths = build_feature_dataset_from_json(
        TRAIN_DIR,
        extractor,
        class_names_from_json
    )

    print("Loading validation features...")
    X_val, y_val, val_used_classes, val_image_paths = build_feature_dataset_from_json(
        VAL_DIR,
        extractor,
        class_names_from_json
    )

    class_names = [
        cls for cls in class_names_from_json
        if cls in train_used_classes and cls in val_used_classes
    ]

    if len(class_names) == 0:
        raise ValueError("No overlapping non-empty classes between train and val.")

    X_train, y_train = filter_dataset_to_used_classes(
        X_train, y_train, class_names_from_json, class_names
    )
    X_val, y_val = filter_dataset_to_used_classes(
        X_val, y_val, class_names_from_json, class_names
    )

    model = build_model(
        config.model_name,
        getattr(config, "model_params", None)
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time_sec = time.time() - start_time

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)

    val_report = classification_report(
        y_val,
        val_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    with open(output_paths["model_path"], "wb") as f:
        pickle.dump(model, f)

    save_json(config.to_dict(), output_paths["config_path"])
    save_json(val_report, output_paths["report_path"])

    summary = {
        "run_name": output_paths["run_name"],
        "model": config.model_name,
        "feature_set": config.feature_tag(),
        "augmentation": config.augmentation,
        "dataset_tag": config.dataset_tag,
        "notes": config.notes,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "num_classes": len(class_names),
        "class_names": class_names,
        "feature_dim": int(X_train.shape[1]),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "val_weighted_f1": float(val_weighted_f1),
        "train_time_sec": float(train_time_sec),
        "model_path": str(output_paths["model_path"]),
        "report_path": str(output_paths["report_path"]),
        "timestamp": output_paths["run_name"].split("__")[0],
    }

    save_json(summary, output_paths["summary_path"])
    append_jsonl(summary, RESULTS_DIR / "experiment_history.jsonl")

    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1: {val_macro_f1:.4f}")
    print(f"Validation Weighted F1: {val_weighted_f1:.4f}")
    print(f"Saved experiment to: {output_paths['result_dir']}")

    return summary


# =========================
# Main
# =========================
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = input("Which model do you want to train? (svm, rf, xgb) ")
    feature_input = input("which feature set do you want to use? Select features from (hog, lbp, hsv, glcm). Separate multiple entries with a space: ")
    selected_features = feature_input.lower().split()

    config = ExperimentConfig(
        model_name=model_name,                   # "svm", "rf", "xgb"
        feature_set=selected_features,  # "hog", "lbp", "hsv", "glcm"
        augmentation=True,                  # augmentation = True or False. this should be manually set.
        dataset_tag="base",
        notes="Single run baseline"
    )

    run_experiment(config)


if __name__ == "__main__":
    main()