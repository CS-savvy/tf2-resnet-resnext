from networks.resnet_single_block import resnet_block
from pathlib import Path
from tensorflow.keras.utils import plot_model
from tensorflow import config


# gpus = config.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#         config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)


resnet = resnet_block()

print(resnet.summary())

plot_model(resnet, "block.png", show_shapes=True, show_layer_names=True)