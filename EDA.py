from pathlib import Path
import numpy as np
from _collections import defaultdict

root_dir = Path.cwd()
dataset_path = root_dir / "Dataset" / "tiny-imagenet-200"
wnid_to_class_file = dataset_path / "words.txt"

with open(wnid_to_class_file, "r") as f:
    content = f.read()

content_lines = content.split("\n")
content_lines = [d.split("\t") for d in content_lines]
wnid_to_class = {d[0]: d[1] for d in content_lines}

data_splits = ["train"]

for split in data_splits:

    split_path = dataset_path / split
    wordnet_ids = [d.name for d in split_path.iterdir() if d.is_dir()]
    image_per_class = dict()
    for wids in wordnet_ids:
        images = list((split_path / wids / "images").glob("*.JPEG"))
        image_per_class[wids] = len(images)

    unique_image_count = np.unique(list(image_per_class.values()))

    print("No of unique image counts in ", split, " dataset :", unique_image_count)
    #print(wordnet_ids, "\n", len(wordnet_ids))

val_annotation_file = dataset_path / "val" / "val_annotations.txt"

with open(val_annotation_file, "r") as f:
    content = f.read()

content_lines = content.split("\n")
content_lines = [d.split("\t") for d in content_lines]

errors = [[i, d] for i, d in enumerate(content_lines) if len(d) != 6]
print("error in validation file skipping entries with index :", errors)

content_lines = [d for d in content_lines if len(d) == 6]
val_images_by_class = defaultdict(list)

for cont in content_lines:
    val_images_by_class[cont[1]].append(cont[0])

unique_image_count = np.unique([len(v) for v in val_images_by_class.values()])
print("No of unique image counts in validation split :", unique_image_count)

