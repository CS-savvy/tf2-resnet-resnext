from tensorflow.keras import layers
from tensorflow.keras import models


def resnet_block(shape=(224, 224, 3)):

    image_tensor = layers.Input(shape=shape)

    # intial conv layer
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(image_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # First conv layer

    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    shortcut = x

    x = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)

    model = models.Model(inputs=[image_tensor], outputs=[x], name='Resnet_single')
    return model