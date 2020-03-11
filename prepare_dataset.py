from pathlib import Path
import shutil
from tqdm import tqdm

root_dir = Path.cwd()
dataset_path = root_dir / "Dataset" / "tiny-imagenet-200"

temp_dataset_path = dataset_path / "temp"

if not temp_dataset_path.exists():
    temp_dataset_path.mkdir()

## train dataset

train_path = dataset_path / "train"
wordnet_ids = [d.name for d in train_path.iterdir() if d.is_dir()]
temp_train_path = temp_dataset_path / "train"

if not temp_train_path.exists():
    temp_train_path.mkdir()

for wid in tqdm(wordnet_ids):
    class_path = train_path / wid / "images"
    temp_class_path = temp_train_path / wid
    temp_class_path.mkdir()
    for img in class_path.glob("*.JPEG"):
        shutil.move(img, temp_class_path/img.name)

## val dataset

val_path = dataset_path / "val"
temp_val_path = temp_dataset_path / "val"
if not temp_val_path.exists():
    temp_val_path.mkdir()

val_annotation_file = dataset_path / "val" / "val_annotations.txt"

with open(val_annotation_file, "r") as f:
    content = f.read()

content_lines = content.split("\n")
content_lines = [d.split("\t") for d in content_lines]

errors = [[i, d] for i, d in enumerate(content_lines) if len(d) != 6]
print("error in validation file skipping entries with index :", errors)

content_lines = [d for d in content_lines if len(d) == 6]

class_path = val_path / "images"
for img in tqdm(content_lines):
    temp_class_path = temp_val_path / img[1]
    if not temp_class_path.exists():
        temp_class_path.mkdir()
    shutil.move(class_path / img[0], temp_class_path / img[0])
