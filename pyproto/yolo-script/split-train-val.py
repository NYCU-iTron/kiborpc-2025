import os
import random
import shutil
from pathlib import Path

DATASET_DIR = Path("../../assets/dataset")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

if (
    (IMAGES_DIR / "train").exists() and
    (IMAGES_DIR / "val").exists() and
    (LABELS_DIR / "train").exists() and
    (LABELS_DIR / "val").exists()
):
    print("Dataset already split into train and val sets.")
    exit(0)

# read images
image_files = list(IMAGES_DIR.glob("*.*"))
random.shuffle(image_files)

# split 80% train, 20% val
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# create folders
for subset in ['train', 'val']:
    (DATASET_DIR / 'images' / subset).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / 'labels' / subset).mkdir(parents=True, exist_ok=True)

# Move files
def move_files(files, subset):
    for img_file in files:
        # Move image
        shutil.move(str(img_file), str(DATASET_DIR / 'images' / subset / img_file.name))
        # Move corresponding label
        label_file = LABELS_DIR / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.move(str(label_file), str(DATASET_DIR / 'labels' / subset / label_file.name))

move_files(train_files, 'train')
move_files(val_files, 'val')