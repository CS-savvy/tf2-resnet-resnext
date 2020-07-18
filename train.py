from networks.resnet_34 import ResNext
from input_pipeline import keras_datagen
from pathlib import Path


dataset_path = Path("Dataset/tiny-imagenet-200/temp/")
train_dataset_path = dataset_path / "train"
val_dataset_path = dataset_path / "val"

batch_size = 64

train_generator = keras_datagen.get_train_generator(train_dataset_path, batch_size)
print("class mappings : ", train_generator.class_indices)

val_generator = keras_datagen.get_val_generator(val_dataset_path, batch_size)

