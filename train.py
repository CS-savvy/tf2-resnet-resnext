from networks.resnet_34 import ResNext
from input_pipeline import keras_datagen
from pathlib import Path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import config

gpus = config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


dataset_path = Path("Dataset/tiny-imagenet-200/temp/")
train_dataset_path = dataset_path / "train"
val_dataset_path = dataset_path / "val"

batch_size = 64
#
train_generator = keras_datagen.get_train_generator(train_dataset_path, batch_size)
print("class mappings : ", train_generator.class_indices)

val_generator = keras_datagen.get_val_generator(val_dataset_path, batch_size)

train_steps = train_generator.__len__()
val_steps = val_generator.__len__()

resnext = ResNext(input_shape=(64, 64, 3))

# Get resnet with 34 layers
resnet34 = resnext.resnet34()
print(resnet34.summary())

# save model layers visualization
plot_model(resnet34, "model.png", show_shapes=True, show_layer_names=True)

# compile model with optimizer, loss and eval metrics

resnet34.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
#print("Metrics : ", resnet34)

history = resnet34.fit_generator(train_generator, steps_per_epoch=train_steps, validation_data=val_generator,
                                 validation_steps=val_steps, epochs=10)

resnet34.save("model.h5")

