from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_dataset(dataset_path, batch_size):
    train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(dataset_path, target_size=(224, 224),
                                                        batch_size = batch_size, class_mode='categorical')

    return train_generator

def get_val_dataset(dataset_path, batch_size):
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(dataset_path, target_size=(224, 224),
                                                    batch_size = batch_size, class_mode='categorical')

    return  val_generator