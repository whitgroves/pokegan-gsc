import os
import time
from IPython import display
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore bugged CUDA errors; must precede tf import
import tensorflow as tf
keras = tf.keras # autocomplete workaround for lazy-loading
from keras import utils, layers, losses, optimizers, Sequential
if len(tf.config.list_physical_devices('GPU')) > 0:
    keras.mixed_precision.set_global_policy("mixed_float16") # NVIDIA speed optimization
else: raise SystemError('No GPU detected. Please reload NVIDIA kernel.') # CPU overheats
keras.utils.set_random_seed(1996)

IMG_CHANNELS = 3 # rgb
IMG_SCALE = 128 # 256 causes OOM
IMG_SHAPE = [IMG_SCALE, IMG_SCALE, IMG_CHANNELS]
BATCH_SIZE = 32 # default = 32
NOISE_DIM = 151
NOISE = tf.random.normal([BATCH_SIZE, NOISE_DIM])
GRID_SIZE = int(BATCH_SIZE**0.5)
plt.rc('figure', figsize=(GRID_SIZE, GRID_SIZE), autolayout=True)

def load_dataset(image_dir:str|os.PathLike) -> tf.data.Dataset:
    dataset = utils.image_dataset_from_directory(image_dir, label_mode=None,
                                                 color_mode=('rgba' if IMG_CHANNELS == 4 else 'rgb'),
                                                 image_size=(IMG_SCALE, IMG_SCALE),
                                                 batch_size=BATCH_SIZE)
    dataset = dataset.map(lambda image: image / 255) # [0, 255] -> [0, 1]
    return dataset

# displays a `size` x `size` grid of pictures from `images`
def display_grid(images:tf.Tensor, size:int=GRID_SIZE) -> None:
    for i in range(size**2):
        image = (images[i].numpy() * 255).astype('uint8') # [0, 1] -> [0, 255]
        plt.subplot(size, size, i+1)
        plt.imshow(image) 
        plt.axis('off')

# wraps tf.data.Dataset since it doesn't seem to work with indexing or next()
def get_sample(dataset:tf.data.Dataset, display:bool=True) -> tf.Tensor:
    for batch in dataset:
        if display: display_grid(batch)
        yield batch

def get_optimizer() -> optimizers.Optimizer: # can't re-use same instance
    return optimizers.RMSprop(clipvalue=1, weight_decay=1e-8)

# these need to persist
G_OPTIMIZER = get_optimizer()
D_OPTIMIZER = get_optimizer()

def make_generator() -> Sequential:
    seed_scale = IMG_SCALE // 4 # used for initial dim so final output is correct shape
    seed_channels = IMG_CHANNELS * 256 # not actually channels but upscaled to maintain total array size
    model = Sequential([
        layers.Dense(seed_scale**2 * seed_channels, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((seed_scale, seed_scale, seed_channels)),
        layers.Conv2DTranspose(128 * IMG_CHANNELS, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64 * IMG_CHANNELS, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(IMG_CHANNELS, 5, strides=2, padding='same', use_bias=False, activation='sigmoid')
    ])
    if model.output_shape != (None, *IMG_SHAPE): raise ValueError(f'Wrong output shape: {model.output_shape}')
    # model.compile(get_optimizer(), loss='binary_crossentropy')
    return model

def make_discriminator() -> Sequential:
    model = Sequential([
        layers.Conv2D(64 * IMG_CHANNELS, 5, strides=2, padding='same', input_shape=IMG_SHAPE),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128 * IMG_CHANNELS, 5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(7, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    # model.compile(get_optimizer(), loss='binary_crossentropy')
    return model

# helper function to compute model loss
cross_entropy = losses.BinaryCrossentropy() # add `from_logits=True` if switching back to tanh

def generator_loss(fake:tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(fake), fake)

def discriminator_loss(real:tf.Tensor, fake:tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(real), real) + cross_entropy(tf.zeros_like(fake), fake)

def make_checkpoint(generator:Sequential, discriminator:Sequential) -> tf.train.Checkpoint:
    return tf.train.Checkpoint(generator_optimizer=G_OPTIMIZER,
                               discriminator_optimizer=D_OPTIMIZER,
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
    G_OPTIMIZER.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    D_OPTIMIZER.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))

def generate_and_save_images(generator:Sequential, label:str) -> None:
    out_dir = '.generated/'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    display.clear_output(wait=True)
    images = generator(NOISE, training=False)
    display_grid(images)
    plt.savefig(f'{out_dir}{label}.png')
    plt.show()

def train(generator:Sequential, discriminator:Sequential, dataset:tf.data.Dataset, epochs:int=50) -> None:
    timestamp = time.time()
    checkpoint = make_checkpoint(generator, discriminator)
    checkpoint_dir = '.v5_checkpoints/'
    checkpoint_mgr = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3) # each is >1GB
    try: checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_existing_objects_matched()
    except Exception as e: print(f'WARN: {e}')
    for epoch in range(epochs):
        msg = f'Epoch {epoch+1}/{epochs}:'
        print(f'{msg} Training...', end='\r')
        start = time.time()
        for batch in dataset: train_step(generator, discriminator, batch)
        generate_and_save_images(generator, f'{int(timestamp)}_{epoch}') # timestamp == run ID
        if (epoch+1) % 10 == 0: # save every 10 epochs
            print(f'{msg} Saving...', end='\r')
            checkpoint_mgr.save() 
        print(f'{msg} Complete in {time.time()-start:.1f}s')
    generate_and_save_images(generator, 'final')
    print(f'{epochs} epochs completed in {time.time()-timestamp:.1f}s')
