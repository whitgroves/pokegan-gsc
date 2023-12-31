import os
import time
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore bugged CUDA errors; must precede tf import
import tensorflow as tf
keras = tf.keras # autocomplete workaround for lazy-loading
from keras import preprocessing, layers, losses, optimizers
if len(tf.config.list_physical_devices('GPU')) > 0:
    keras.mixed_precision.set_global_policy("mixed_float16") # NVIDIA speed optimization

DATA_ROOT = './.data'
RGB_SCALE = 255
N_CHANNELS = 4 # rgba
BATCH_SIZE = 32
CONV_KERNEL =  (5, 5)
CHKPT_DIR = './.checkpoints'
EPOCHS = 50
NOISE_DIM = 100
N_SAMPLES = 16
GEN_SEED = tf.random.normal([N_SAMPLES, NOISE_DIM])

def load_dataset(image_dir:str|os.PathLike) -> tf.data.Dataset:
    dataset = preprocessing.image_dataset_from_directory(image_dir, label_mode=None, color_mode='rgba',
                                                         image_size=(RGB_SCALE, RGB_SCALE))
    return dataset.map(lambda image : image / RGB_SCALE) # normalize

# wraps tf.data.Dataset since it doesn't seem to work with indexing or next()
def get_sample(dataset:tf.data.Dataset, display=True) -> np.ndarray:
    for image in dataset:
        image = (image.numpy() * RGB_SCALE).astype("uint8")[0] # rescale
        if display:
            plt.axis('off')
            plt.imshow(image)
        yield image

def make_generator() -> keras.Sequential:
    return keras.Sequential([
        layers.Dense(N_SAMPLES*N_SAMPLES*(RGB_SCALE+1), use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((N_SAMPLES, N_SAMPLES, (RGB_SCALE+1))),
        layers.Conv2DTranspose(128*N_CHANNELS, CONV_KERNEL, strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64*N_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(N_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

def make_discriminator() -> keras.Sequential:
    input_shape = (RGB_SCALE, RGB_SCALE, N_CHANNELS) # in v2, this was passed in
    return keras.Sequential([
        layers.Conv2D(64*N_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same', input_shape=input_shape),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128*N_CHANNELS, CONV_KERNEL, strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])

# helper function to compute model loss
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake:tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(fake), fake)

def discriminator_loss(real:tf.Tensor, fake:tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(real), real) + cross_entropy(tf.zeros_like(fake), fake)

# these need to persist
generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

def make_checkpoint(generator:keras.Sequential, discriminator:keras.Sequential) -> tf.train.Checkpoint:
    return tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer,
                               generator=generator, discriminator=discriminator)

@tf.function # auto-compile
def train_step(generator:keras.Sequential, discriminator:keras.Sequential, images:tf.Tensor) -> None:
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

def generate_and_save_images(generator:keras.Sequential, label:str) -> None:
    images = generator(GEN_SEED, training=False)
    _ = plt.figure(figsize=(4, 4)) # TODO: genrate plot dims based on N_SAMPLES
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i, :, :, 0] * RGB_SCALE)
        plt.axis('off')
    plt.savefig(f'{label}.png')
    plt.show()

def train(generator:keras.Sequential, discriminator:keras.Sequential, dataset:tf.data.Dataset) -> None:
    for epoch in range(EPOCHS):
        msg = f'Epoch {epoch+1}/{EPOCHS}:'
        print(f'{msg} Training...', end='\r')
        start = time.time()
        for batch in dataset:
            train_step(generator, discriminator, batch)
        generate_and_save_images(generator, epoch)
        if epoch % 10 == 0: make_checkpoint(generator, discriminator).save()
        print(f'{msg} Complete in {time.time()-start:.1f}s')
    generate_and_save_images(generator, 'final')
