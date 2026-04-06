import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import random

# 1. Load your current dataset
dataset = fo.load_dataset("pet_mischief_dataset")

# 2. Add more samples for underrepresented classes (Goal: ~100+ each)
# Specifically targeting potted plant, bowl, and laptop
short_classes = ["potted plant", "bowl", "laptop", "knife"]
print(f"--- Step 1: Adding samples for {short_classes} ---")

additional_samples = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=short_classes,
    max_samples=200, 
)
dataset.merge_samples(additional_samples)

# 3. Reduce overrepresented classes (cup, bottle) to ~120 samples
# This prevents the model from being biased towards cups/bottles
print("--- Step 2: Downsampling 'cup' and 'bottle' for balance ---")

for label in ["cup", "bottle"]:
    # Find samples containing the overrepresented label
    view = dataset.filter_labels("ground_truth", F("label") == label)
    
    # If there are too many, randomly select some to remove from the dataset
    if len(view) > 120:
        samples_to_remove = view.take(len(view) - 120)
        dataset.delete_samples(samples_to_remove)

# 4. Final cleaning: Keep ONLY our target classes
final_classes = ["cat", "dog", "cup", "bottle", "potted plant", "bowl", "laptop", "knife"]
final_view = dataset.filter_labels("ground_truth", F("label").is_in(final_classes))

print(f"--- Final Dataset Balance Complete. Total samples: {len(final_view)} ---")

# 5. Launch app to see the beautiful, flat histogram
session = fo.launch_app(final_view)
session.wait()