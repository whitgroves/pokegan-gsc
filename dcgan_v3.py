import os
import time
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore bugged CUDA errors; must precede tf import
import tensorflow as tf
keras = tf.keras # autocomplete workaround for lazy-loading
from keras import preprocessing, layers, losses, optimizers, Sequential
if len(tf.config.list_physical_devices('GPU')) > 0:
    keras.mixed_precision.set_global_policy("mixed_float16") # NVIDIA speed optimization
from IPython import display

IMG_CHANNELS = 4 # rgba
IMG_SCALE = 128 # 256 causes OOM
BATCH_SIZE = 64 # default = 32; 256 causes OOM
NOISE_DIM = 100
CONV_KERNEL = (5, 5)
EPOCHS = 50

def load_dataset(image_dir:str|os.PathLike) -> tf.data.Dataset:
    dataset = preprocessing.image_dataset_from_directory(image_dir, label_mode=None,
                                                         color_mode=('rgba' if IMG_CHANNELS == 4 else 'rgb'),
                                                         image_size=(IMG_SCALE, IMG_SCALE),
                                                         batch_size=BATCH_SIZE)
    dataset = dataset.map(lambda image : image / 255) # [0, 255] -> [0, 1]
    return dataset

# wraps tf.data.Dataset since it doesn't seem to work with indexing or next()
def get_sample(dataset:tf.data.Dataset, display:bool=True) -> np.ndarray:
    for image in dataset:
        image = (image.numpy() * 255).astype("uint8")[0] # [0, 1] -> [0, 255]
        if display:
            plt.axis('off')
            plt.imshow(image)
        yield image

def make_generator() -> Sequential:
    seed_scale = IMG_SCALE // 4 # used for the initial dim so final image is correct scale
    seed_channels = IMG_CHANNELS * 256 # not actually channels but upscaled to maintain total array size
    model = Sequential([
        layers.Dense(seed_scale**2 * seed_channels, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((seed_scale, seed_scale, seed_channels)),
        layers.Conv2DTranspose(128 * IMG_CHANNELS, CONV_KERNEL, strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64 * IMG_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(IMG_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    assert model.output_shape == (None, IMG_SCALE, IMG_SCALE, IMG_CHANNELS)
    return model

def make_discriminator() -> Sequential:
    model = Sequential([
        layers.Conv2D(64 * IMG_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same', input_shape=[IMG_SCALE, IMG_SCALE, IMG_CHANNELS]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128 * IMG_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# helper function to compute model loss
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake:tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(fake), fake)

def discriminator_loss(real:tf.Tensor, fake:tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(real), real) + cross_entropy(tf.zeros_like(fake), fake)

# these need to persist
generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

def make_checkpoint(generator:Sequential, discriminator:Sequential) -> tf.train.Checkpoint:
    return tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer,
                               generator=generator, discriminator=discriminator)

@tf.function # auto-compile
def train_step(generator:Sequential, discriminator:Sequential, images:tf.Tensor) -> None:
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_images = generator(noise, training=True)
        disc_real = discriminator(images, training=True)
        disc_fake = discriminator(gen_images, training=True)
        gen_loss = generator_loss(disc_fake)
        disc_loss = discriminator_loss(disc_real, disc_fake)
    gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))

def generate_and_save_images(generator:Sequential, label:str) -> None:
    out_dir = '.generated/'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    display.clear_output(wait=True)
    images = generator(tf.random.normal([16, NOISE_DIM]), training=False)
    _ = plt.figure(figsize=(4, 4)) # TODO: genrate plot dims based on number of samples
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i, :, :, 0] * IMG_SCALE)
        plt.axis('off')
    plt.savefig(f'{out_dir}{label}.png')
    plt.show()

def train(generator:Sequential, discriminator:Sequential, dataset:tf.data.Dataset, epochs:int=EPOCHS) -> None:
    start = time.time()
    checkpoint = make_checkpoint(generator, discriminator)
    checkpoint_dir = '.model_checkpoints/'
    try: checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_existing_objects_matched()
    except Exception as e: print(f'WARN: {e}')
    for epoch in range(epochs):
        msg = f'Epoch {epoch+1}/{epochs}:'
        print(f'{msg} Training...', end='\r')
        _start = time.time()
        for batch in dataset: train_step(generator, discriminator, batch)
        generate_and_save_images(generator, epoch)
        if epoch % (EPOCHS // 10) == 0:
            print(f'{msg} Saving checkpoint...', end='\r')
            checkpoint.save(checkpoint_dir) # save every 10%
        print(f'{msg} Complete in {time.time()-_start:.1f}s')
    generate_and_save_images(generator, 'final')
    print(f'{EPOCHS} epochs completed in {time.time()-start:.1f}s')
