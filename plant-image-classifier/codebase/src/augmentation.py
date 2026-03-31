"""
Data augmentation functions to increase the diversity of training data
randomly flip, rotate, adjust brightness, and crop/zoom images in the training set.
"""

import random
from pathlib import Path
from PIL import Image, ImageEnhance


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_image_file(path):
    return path.suffix.lower() in VALID_EXTENSIONS


# =========================
# Augmentation functions
# =========================
def random_flip(img):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_brightness(img):
    factor = random.uniform(0.8, 1.2)  
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def random_rotation(img):
    angle = random.uniform(-10, 10) 
    return img.rotate(angle)


def random_crop_zoom(img):
    w, h = img.size

    scale = random.uniform(0.9, 0.95)  
    new_w, new_h = int(w * scale), int(h * scale)

    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)

    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h))


# =========================
# Main augmentation pipeline
# =========================
def augment_image(img):
    img = random_flip(img)
    img = random_brightness(img)
    img = random_rotation(img)
    img = random_crop_zoom(img)
    return img


def augment_dataset(train_dir: Path, num_aug=3):
    """
    For each image in train_dir/class_name/,
    create num_aug augmented images
    """
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.iterdir():
            if not is_image_file(img_path):
                continue

            img = Image.open(img_path).convert("RGB")

            for i in range(num_aug):
                aug_img = augment_image(img)

                new_name = img_path.stem + f"_aug{i}.jpg"
                new_path = class_dir / new_name

                aug_img.save(new_path)

    print("Augmentation completed.")


if __name__ == "__main__":
    augment_dataset(Path("data/train"), num_aug=3)