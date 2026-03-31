# split_dataset.py

import json
import shutil
from pathlib import Path
from collections import defaultdict


# =========================
# Global Config
# =========================
RAW_DIR = Path("data/raw")
OUTPUT_BASE_DIR = Path("data")
CLASS_NAMES_PATH = Path("models/class_names.json")

SPLITS = ["train", "val", "test"]

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

COPY_FILES = True  # True: copy, False: move

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# Utility Functions
# =========================
def load_class_names(json_path: Path) -> list[str]:
    if not json_path.exists():
        raise FileNotFoundError(f"class_names.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not all(isinstance(x, str) for x in class_names):
        raise ValueError("class_names.json must be a JSON list of strings.")

    return class_names


def detect_lighting_group(filename: str) -> str | None:
    """
    Infer lighting group from filename.

    Supported:
    - BrightLighting_XXXX.png
    - BrightLight_XXXX.png
    - MidLight_XXXX.png
    - DarkLight_XXXX.png

    Returns:
        "BrightLighting", "MidLight", "DarkLight", or None
    """
    lower_name = filename.lower()

    # Exclude label image
    if lower_name == "label.png" or lower_name.startswith("label."):
        return None

    if lower_name.startswith("brightlighting_") or lower_name.startswith("brightlight_"):
        return "BrightLighting"

    if lower_name.startswith("midlight_"):
        return "MidLight"

    if lower_name.startswith("darklight_"):
        return "DarkLight"

    return None


def list_images_by_lighting(class_dir: Path) -> dict[str, list[Path]]:
    grouped = defaultdict(list)

    for file_path in sorted(class_dir.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        lighting_group = detect_lighting_group(file_path.name)
        if lighting_group is None:
            continue

        grouped[lighting_group].append(file_path)

    return dict(grouped)


def split_one_group(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split one lighting group using sorted block split.

    Example:
        files sorted by filename
        first 60% -> train
        next 20% -> val
        last 20% -> test
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")

    n = len(files)
    if n == 0:
        return [], [], []

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # Small-group safeguard
    if n >= 3:
        if len(val_files) == 0 and len(train_files) > 1:
            val_files = [train_files.pop()]

        if len(test_files) == 0:
            if len(train_files) > 1:
                test_files = [train_files.pop()]
            elif len(val_files) > 1:
                test_files = [val_files.pop()]

    return train_files, val_files, test_files


def ensure_output_dirs(class_names: list[str]) -> None:
    for split in SPLITS:
        split_dir = OUTPUT_BASE_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for class_name in class_names:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def clear_existing_split_files(class_names: list[str]) -> None:
    """
    Remove existing files under data/train, data/val, data/test
    only for the classes listed in class_names.
    """
    for split in SPLITS:
        for class_name in class_names:
            class_dir = OUTPUT_BASE_DIR / split / class_name
            if not class_dir.exists():
                continue

            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()


def copy_or_move_file(src: Path, dst: Path) -> None:
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def print_summary(summary: dict[str, dict[str, dict[str, int]]]) -> None:
    print("\nSPLIT SUMMARY")
    print("-" * 82)
    print(f"{'Class':25} {'Lighting':15} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    print("-" * 82)

    for class_name in sorted(summary.keys()):
        for lighting in ["BrightLighting", "MidLight", "DarkLight"]:
            counts = summary[class_name].get(
                lighting,
                {"train": 0, "val": 0, "test": 0, "total": 0}
            )

            print(
                f"{class_name:25} {lighting:15} "
                f"{counts['train']:6} {counts['val']:6} {counts['test']:6} {counts['total']:6}"
            )

    print("-" * 82)

    print("\nTOTAL PER CLASS")
    print("-" * 55)
    print(f"{'Class':25} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    print("-" * 55)

    for class_name in sorted(summary.keys()):
        train_total = sum(summary[class_name][k]["train"] for k in summary[class_name])
        val_total = sum(summary[class_name][k]["val"] for k in summary[class_name])
        test_total = sum(summary[class_name][k]["test"] for k in summary[class_name])
        total = sum(summary[class_name][k]["total"] for k in summary[class_name])

        print(f"{class_name:25} {train_total:6} {val_total:6} {test_total:6} {total:6}")

    print("-" * 55)


# =========================
# Main
# =========================
def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    class_names = load_class_names(CLASS_NAMES_PATH)

    class_dirs = []
    for class_name in class_names:
        class_path = RAW_DIR / class_name
        if class_path.exists() and class_path.is_dir():
            class_dirs.append(class_path)
        else:
            print(f"[WARNING] Missing class folder in raw: {class_name}")

    if not class_dirs:
        raise ValueError("No valid class folders found from class_names.json")

    used_class_names = [p.name for p in class_dirs]

    ensure_output_dirs(used_class_names)
    clear_existing_split_files(used_class_names)

    summary = defaultdict(dict)

    for class_dir in class_dirs:
        class_name = class_dir.name
        grouped_images = list_images_by_lighting(class_dir)

        for lighting in ["BrightLighting", "MidLight", "DarkLight"]:
            files = sorted(grouped_images.get(lighting, []))

            train_files, val_files, test_files = split_one_group(
                files,
                TRAIN_RATIO,
                VAL_RATIO,
                TEST_RATIO
            )

            for split_name, split_files in [
                ("train", train_files),
                ("val", val_files),
                ("test", test_files),
            ]:
                for src_path in split_files:
                    dst_path = OUTPUT_BASE_DIR / split_name / class_name / src_path.name
                    copy_or_move_file(src_path, dst_path)

            summary[class_name][lighting] = {
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files),
                "total": len(files),
            }

    print_summary(summary)

    print("\nDone.")
    print(f"Source: {RAW_DIR}")
    print(f"Class list: {CLASS_NAMES_PATH}")
    print(f"Output dirs: {OUTPUT_BASE_DIR / 'train'}, {OUTPUT_BASE_DIR / 'val'}, {OUTPUT_BASE_DIR / 'test'}")
    print(f"Mode: {'COPY' if COPY_FILES else 'MOVE'}")


if __name__ == "__main__":
    main()