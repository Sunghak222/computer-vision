# sanity_check.py

import json
import hashlib
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2


# =========================
# Global Config
# =========================
DATA_DIR = Path("data")
CLASS_NAMES_PATH = Path("models/class_names.json")

SPLITS = ["train", "val", "test"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SAMPLE_OUTPUT_DIR = Path("results/sample_images")
SAMPLE_PER_CLASS_PER_SPLIT = 3

CLASS_COL_WIDTH = 28
SPLIT_COL_WIDTH = 8


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


def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS


def check_corrupted_image(file_path: Path) -> bool:
    """
    Returns True if corrupted.
    """
    img = cv2.imread(str(file_path))
    return img is None


def collect_dataset(class_names: list[str]) -> dict[str, dict[str, list[Path]]]:
    """
    dataset[split][class_name] = list of image paths
    """
    dataset = defaultdict(dict)

    for split in SPLITS:
        split_dir = DATA_DIR / split

        for class_name in class_names:
            class_dir = split_dir / class_name

            if not class_dir.exists() or not class_dir.is_dir():
                dataset[split][class_name] = []
                continue

            image_paths = [
                p for p in sorted(class_dir.iterdir())
                if is_valid_image_file(p)
            ]

            dataset[split][class_name] = image_paths

    return dataset


def print_image_count_table(dataset: dict[str, dict[str, list[Path]]], class_names: list[str]) -> None:
    print("\nIMAGE COUNT TABLE")
    print("-" * 72)
    print(
        f"{'Class':{CLASS_COL_WIDTH}}"
        f"{'Train':>{SPLIT_COL_WIDTH}}"
        f"{'Val':>{SPLIT_COL_WIDTH}}"
        f"{'Test':>{SPLIT_COL_WIDTH}}"
        f"{'Total':>{SPLIT_COL_WIDTH}}"
    )
    print("-" * 72)

    for class_name in class_names:
        train_n = len(dataset["train"].get(class_name, []))
        val_n = len(dataset["val"].get(class_name, []))
        test_n = len(dataset["test"].get(class_name, []))
        total_n = train_n + val_n + test_n

        print(
            f"{class_name:{CLASS_COL_WIDTH}}"
            f"{train_n:>{SPLIT_COL_WIDTH}}"
            f"{val_n:>{SPLIT_COL_WIDTH}}"
            f"{test_n:>{SPLIT_COL_WIDTH}}"
            f"{total_n:>{SPLIT_COL_WIDTH}}"
        )

    print("-" * 72)


def print_split_totals(dataset: dict[str, dict[str, list[Path]]], class_names: list[str]) -> None:
    train_total = sum(len(dataset["train"].get(c, [])) for c in class_names)
    val_total = sum(len(dataset["val"].get(c, [])) for c in class_names)
    test_total = sum(len(dataset["test"].get(c, [])) for c in class_names)

    print("\nTOTAL IMAGES PER SPLIT")
    print("-" * 40)
    print(f"{'Train':10}: {train_total}")
    print(f"{'Val':10}: {val_total}")
    print(f"{'Test':10}: {test_total}")
    print("-" * 40)


def find_corrupted_files(dataset: dict[str, dict[str, list[Path]]], class_names: list[str]) -> list[tuple[str, str, Path]]:
    corrupted = []

    for split in SPLITS:
        for class_name in class_names:
            for file_path in dataset[split].get(class_name, []):
                if check_corrupted_image(file_path):
                    corrupted.append((split, class_name, file_path))

    return corrupted


def print_corrupted_report(corrupted: list[tuple[str, str, Path]]) -> None:
    print("\nCORRUPTED FILE CHECK")
    print("-" * 72)

    if not corrupted:
        print("No corrupted images detected.")
    else:
        print(f"Found {len(corrupted)} corrupted image(s):")
        for split, class_name, file_path in corrupted:
            print(f"[{split}] {class_name} -> {file_path}")

    print("-" * 72)


def find_exact_duplicates(
    dataset: dict[str, dict[str, list[Path]]],
    class_names: list[str]
) -> dict[str, list[Path]]:
    """
    Exact duplicates across all splits/classes based on MD5 hash.
    """
    hash_map = defaultdict(list)

    for split in SPLITS:
        for class_name in class_names:
            for file_path in dataset[split].get(class_name, []):
                file_hash = compute_md5(file_path)
                hash_map[file_hash].append(file_path)

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    return duplicates


def print_duplicate_report(duplicates: dict[str, list[Path]]) -> None:
    print("\nEXACT DUPLICATE CHECK (MD5)")
    print("-" * 72)

    if not duplicates:
        print("No exact duplicates detected.")
    else:
        print(f"Found {len(duplicates)} duplicate group(s):")
        for idx, (file_hash, paths) in enumerate(duplicates.items(), start=1):
            print(f"\nDuplicate Group {idx} | hash={file_hash}")
            for p in paths:
                print(f"  - {p}")

    print("-" * 72)


def clear_sample_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def save_sample_images(
    dataset: dict[str, dict[str, list[Path]]],
    class_names: list[str],
    output_dir: Path,
    sample_per_class_per_split: int
) -> None:
    """
    Save a few images per class per split for report inspection.
    results/sample_images/<split>/<class_name>/
    """
    clear_sample_output_dir(output_dir)

    for split in SPLITS:
        for class_name in class_names:
            image_paths = dataset[split].get(class_name, [])

            if not image_paths:
                continue

            split_class_dir = output_dir / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            sample_count = min(sample_per_class_per_split, len(image_paths))
            sampled_paths = random.sample(image_paths, sample_count)

            for src_path in sampled_paths:
                dst_path = split_class_dir / src_path.name
                shutil.copy2(src_path, dst_path)


def print_missing_class_warning(dataset: dict[str, dict[str, list[Path]]], class_names: list[str]) -> None:
    print("\nMISSING / EMPTY CLASS CHECK")
    print("-" * 72)

    has_warning = False

    for split in SPLITS:
        for class_name in class_names:
            if len(dataset[split].get(class_name, [])) == 0:
                print(f"[WARNING] Empty class in {split}: {class_name}")
                has_warning = True

    if not has_warning:
        print("No empty class folders detected in train/val/test.")

    print("-" * 72)


# =========================
# Main
# =========================
def main() -> None:
    random.seed(42)

    class_names = load_class_names(CLASS_NAMES_PATH)
    dataset = collect_dataset(class_names)

    print_image_count_table(dataset, class_names)
    print_split_totals(dataset, class_names)
    print_missing_class_warning(dataset, class_names)

    corrupted = find_corrupted_files(dataset, class_names)
    print_corrupted_report(corrupted)

    duplicates = find_exact_duplicates(dataset, class_names)
    print_duplicate_report(duplicates)

    save_sample_images(
        dataset=dataset,
        class_names=class_names,
        output_dir=SAMPLE_OUTPUT_DIR,
        sample_per_class_per_split=SAMPLE_PER_CLASS_PER_SPLIT
    )

    print("\nSAMPLE IMAGE EXPORT")
    print("-" * 72)
    print(f"Saved sample images to: {SAMPLE_OUTPUT_DIR}")
    print("-" * 72)

    print("\nDone.")


if __name__ == "__main__":
    main()