import os
import time
import imageio
from shutil import rmtree
import tensorflow as tf
keras = tf.keras
from keras import layers

# assert tf.__version__ == '2.5.0'  # sanity check for CUDA compatibility

DATA_ROOT = './.data'
OUTPUT_DIR = os.path.join(DATA_ROOT, 'generated')
NOISE_DIM = 100
BATCH_SIZE = 256
CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)
G_OPTIMIZER = tf.keras.optimizers.Adam(1e-4)
D_OPTIMIZER = tf.keras.optimizers.Adam(1e-4)
CHECKPOINT_DIR = os.path.join(DATA_ROOT, 'dcgan_checkpoints')
GIF_DIR = os.path.join(DATA_ROOT, 'dcgan_progress')
EPOCHS = 2000
SAVE_FREQ = int(EPOCHS / 10)

def load_pokemon_dataset() -> tuple:
    image_scale = 64
    batch_size = 32
    dataset = keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_ROOT, 'pokedex'),
        label_mode=None, 
        color_mode='rgb', 
        image_size=(image_scale, image_scale), 
        batch_size=batch_size
    )
    dataset = dataset.map(lambda x : x / 255.0)  # Normalize images to [0, 1]
    return dataset, (image_scale, image_scale, 3), batch_size

def load_mnist_dataset() -> tuple:
    image_scale = 28
    batch_size = 256
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], image_scale, image_scale, 1).astype('float32')
    train_images = train_images / 255.0  # Normalize the images to [0, 1]
    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
    return dataset, (image_scale, image_scale, 1), batch_size

def create_generator(output_shape:tuple) -> keras.Sequential:
    out_height = output_shape[0]
    out_width = output_shape[1]
    out_channels = output_shape[2]

    model = keras.Sequential()

    tmp_height = int(out_height/4)
    tmp_width = int(out_width/4)
    tmp_channels = out_channels*256

    model.add(layers.Dense(tmp_height*tmp_width*tmp_channels, use_bias=False, input_shape=(NOISE_DIM,)))

    model.add(layers.Reshape((tmp_height, tmp_width, tmp_channels)))
    assert model.output_shape == (None, tmp_height, tmp_width, tmp_channels)

    model.add(layers.Conv2DTranspose(out_channels*128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, tmp_height, tmp_width, out_channels*128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(out_channels*64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(out_height/2), int(out_width/2), out_channels*64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(out_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, out_height, out_width, out_channels)

    return model

def create_discriminator(input_shape:tuple) -> keras.Sequential:
    out_channels = input_shape[2]

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(out_channels*64, (5, 5), strides=(2, 2), padding='same', input_shape=list(input_shape)))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(out_channels*128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def generator_loss(generated_output):
    return CROSS_ENTROPY(tf.ones_like(generated_output), generated_output)

def discriminator_loss(genuine_output, generated_output):
    false_negatives = CROSS_ENTROPY(tf.ones_like(genuine_output), genuine_output)
    false_positives = CROSS_ENTROPY(tf.zeros_like(generated_output), generated_output)
    return false_negatives + false_positives

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated_images = generator(noise, training=True)
        genuine_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)
        g_loss = generator_loss(generated_output)
        d_loss = discriminator_loss(genuine_output, generated_output)
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    G_OPTIMIZER.apply_gradients(zip(g_gradients, generator.trainable_variables))
    D_OPTIMIZER.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

def train(dataset, epochs, checkpoint:tf.train.Checkpoint, restore_from_last_checkpoint:bool=True):
    if restore_from_last_checkpoint:
        checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    for epoch in range(epochs):
        e = epoch + 1  # human-friendly number
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        if (e) % SAVE_FREQ == 0:
            generate_and_save_image(generator, e)
            # checkpoint.save(os.path.join(CHECKPOINT_DIR, 'ckpt'))
        print(f'Time for epoch {e} is {time.time() - start}')
    generate_and_save_image(generator, e)  # generate after final epoch
    checkpoint.save(os.path.join(CHECKPOINT_DIR, 'ckpt'))

def generate_and_save_image(model, epoch):
    g_output = model(tf.random.normal([1, NOISE_DIM]))
    g_output.numpy()
    image = keras.preprocessing.image.array_to_img(g_output[0])  # auto-rescales to [0, 255]
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    image_path = os.path.join(OUTPUT_DIR, f'{epoch:05d}.png')
    image.save(image_path)

def create_progress_gif(gif_dst:str, autoclean:bool=True):
    with imageio.get_writer(gif_dst, mode='I') as writer:
            filenames = os.listdir(OUTPUT_DIR)
            for filename in filenames:
                filepath = os.path.join(OUTPUT_DIR, filename)
                image = imageio.imread(filepath)
                writer.append_data(image)
            # append the final image
            image = imageio.imread(filepath)
            writer.append_data(image)
    if autoclean:
        rmtree(OUTPUT_DIR)  # cleanup after a run

if __name__ == '__main__':
    run_id = time.strftime("%Y%m%d-%H%M%S")

    dataset, image_shape, BATCH_SIZE = load_pokemon_dataset()
    generator = create_generator(image_shape)
    discriminator = create_discriminator(image_shape)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=G_OPTIMIZER,
        discriminator_optimizer=D_OPTIMIZER,
        generator=generator,
        discriminator=discriminator
    )

    try:
        train(dataset, EPOCHS, checkpoint, restore_from_last_checkpoint=False)
    finally:
        create_progress_gif(os.path.join(GIF_DIR, f'{run_id}.gif'))  #, autoclean=False)
    
    # checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    for i in range(150):
        generate_and_save_image(generator, i)
