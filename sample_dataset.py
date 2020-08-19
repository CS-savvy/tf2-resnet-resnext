from pathlib import Path
import shutil
from tqdm import tqdm
from random import sample, shuffle


this_dir = Path.cwd()
dataset_dir = this_dir / "Dataset" / "tiny-imagenet-200" / "large_200" / "train"
train_output_dir = this_dir / "Dataset" / "tiny-imagenet-200" / "small" / "train"
val_output_dir = this_dir / "Dataset" / "tiny-imagenet-200" / "small" / "val"

if not train_output_dir.exists():
    train_output_dir.mkdir(parents=True)

if not val_output_dir.exists():
    val_output_dir.mkdir(parents=True)

sample_class = 10
image_per_class_train = 80
image_per_class_val = 20

available_class = [c.name for c in list(dataset_dir.iterdir()) if c.is_dir()]
selected_class = sample(available_class, sample_class)

for cls in tqdm(selected_class):
    class_dir = dataset_dir / cls
    images = list(class_dir.glob("*.JPEG"))
    shuffle(images)
    train_out_cls_dir = train_output_dir / cls
    val_out_cls_dir = val_output_dir / cls
    train_out_cls_dir.mkdir()
    val_out_cls_dir.mkdir()

    for img in images[:image_per_class_train]:
        shutil.copy(img, train_out_cls_dir / img.name)
    for img in images[image_per_class_train :image_per_class_train + image_per_class_val]:
        shutil.copy(img, val_out_cls_dir / img.name)

print("completed")