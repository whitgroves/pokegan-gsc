import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore bugged CUDA errors; must precede tf import
import tensorflow as tf
keras = tf.keras # autocomplete workaround for lazy-loading
from keras import layers
if len(tf.config.list_physical_devices('GPU')) > 0:
    keras.mixed_precision.set_global_policy("mixed_float16") # NVIDIA speed optimization

DATA_ROOT = './.data'
IMG_SCALE = 256
BATCH_SIZE = 32
CONV_KERNEL =  (5, 5)

def load_dataset(image_dir:str|os.PathLike) -> tf.data.Dataset:
    dataset = keras.preprocessing \
        .image_dataset_from_directory(image_dir, label_mode=None, image_size=(IMG_SCALE, IMG_SCALE))
    dataset = dataset.map(lambda image : image / IMG_SCALE) # normalize to [0, 1]
    return dataset

# wraps tf.data.Dataset since it doesn't seem to work with indexing or next()
def get_sample(dataset:tf.data.Dataset, display=True) -> np.ndarray:
    for image in dataset:
        image = (image.numpy() * IMG_SCALE).astype("int16")[0] # rescale
        if display:
            plt.axis('off')
            plt.imshow(image)
        yield image

def make_generator() -> keras.Sequential:
    model = keras.Sequential([
        layers.Dense(16*16*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((16, 16, 256)),
        layers.Conv2DTranspose(128, CONV_KERNEL, strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, CONV_KERNEL, strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, CONV_KERNEL, strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
    ])
    return model