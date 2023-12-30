import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore bugged CUDA errors; must precede tf import
import tensorflow as tf
keras = tf.keras # autocomplete workaround for lazy-loading
keras.utils.set_random_seed(1998)
if len(tf.config.list_physical_devices('GPU')) > 0:
    keras.mixed_precision.set_global_policy("mixed_float16") # NVIDIA speed optimization

DATA_ROOT = './.data'
IMG_SCALE = 256
BATCH_SIZE = 32

def load_dataset(image_dir:str|os.PathLike) -> tf.data.Dataset:
    dataset = keras.preprocessing \
        .image_dataset_from_directory(image_dir, label_mode=None, image_size=(IMG_SCALE, IMG_SCALE))
    dataset = dataset.map(lambda x : x / IMG_SCALE) # normalize to [0, 1]
    return dataset

def display_sample(dataset:tf.data.Dataset) -> None:
    sample = None
    for x in dataset: # tf.data.Dataset doesn't work with indexing or next()
        sample = x
        break
    plt.axis('off')
    plt.imshow((sample.numpy() * IMG_SCALE).astype("int16")[0]) # rescale