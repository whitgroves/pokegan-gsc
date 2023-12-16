# from genericpath import isdir
import os
import time
import imageio
from shutil import rmtree
import tensorflow as tf
# # NVIDIA GPU support depends on specific version (in my case CUDA 11.2 + TF 2.5.0)
# # Adjust for your system and update the version as needed. 
# assert tf.__version__ == '2.5.0'
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.python.keras.layers.convolutional import Conv2D
keras = tf.keras
from keras import callbacks, layers
from keras.models import save_model, load_model

def create_discriminator(input_shape:tuple, verbose:bool=True, load_from:str=None) -> keras.Sequential:
    if load_from is not None:
        assert os.path.isdir(load_from)
        d = load_model(load_from)
    else:
        assert len(input_shape) == 3
        d = keras.Sequential(
            [
                layers.InputLayer(input_shape),
                layers.Conv2D(img_scale, kernel_size=4, strides=2, padding='same'),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(img_scale*2, kernel_size=4, strides=2, padding='same'),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(img_scale*2, kernel_size=4, strides=2, padding='same'),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ],
            name='discriminator'
        )
    if verbose:
        print(d.summary())
    return d

def create_generator(latent_dim:int, output_channels:int, verbose:bool=True, load_from:str=None) -> keras.Sequential:
    if load_from is not None:
        assert os.path.isdir(load_from)
        g = load_model(load_from)
    else:
        dim_mult = int(latent_dim/16)  # don't ask me why
        g = keras.Sequential(
            [
                layers.InputLayer(tuple([latent_dim])),
                layers.Dense(dim_mult * dim_mult * latent_dim),
                layers.Reshape((dim_mult, dim_mult, latent_dim)),
                layers.Conv2DTranspose(latent_dim, kernel_size=4, strides=2, padding='same'),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(latent_dim*2, kernel_size=4, strides=2, padding='same'),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(latent_dim*4, kernel_size=4, strides=2, padding='same'),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(output_channels, kernel_size=5, padding='same', activation='sigmoid')
            ],
            name='generator'
        )
    if verbose:
        print(g.summary())
    return g

class GAN(keras.Model):
    def __init__(self, discriminator:keras.Sequential, generator:keras.Sequential, latent_dim:int):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images) -> dict:
        batch_size = tf.shape(real_images)[0]

        # train the discriminator
        d_rand = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(d_rand)
        all_images = tf.concat([fake_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))  # trick to add noise to labels
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_images)
            d_loss = self.loss_fn(labels, predictions)
        d_grad = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_weights))

        # train the generator
        g_rand = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(g_rand))
            g_loss = self.loss_fn(fake_labels, predictions)
        g_grad = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_weights))

        # update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            'd_loss': self.d_loss_metric.result(),
            'g_loss': self.g_loss_metric.result()
        }


class GANMonitor(callbacks.Callback):
    def __init__(self, n_samples:int, latent_dim:int, img_path:str, frequency:int): # d_model:keras.Sequential, g_model:keras.Sequential, model_path:str, 
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.img_path = img_path
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        # self.d_model = d_model
        # self.g_model = g_model
        # self.model_path = model_path
        # if not os.path.isdir(model_path):
        #     os.mkdir(model_path)
        self.frequency = frequency

    def on_epoch_end(self, epoch:int, logs=None):
        if epoch % self.frequency == 0:
            rand = tf.random.normal(shape=(self.n_samples, self.latent_dim))
            g_images = self.model.generator(rand)
            g_images.numpy()
            for i in range(self.n_samples):
                img = keras.preprocessing.image.array_to_img(g_images[i])
                img_path = os.path.join(self.img_path, f'epoch{epoch:04d}sample{i:02d}.png')
                img.save(img_path)
            # self.d_model.save(os.path.join(self.model_path, 'd_model/'))
            # self.g_model.save(os.path.join(self.model_path, 'g_model/'))
        

def create_gan(discriminator:keras.Sequential, generator:keras.Sequential) -> GAN:
    discriminator.trainable = False  # save fix, see below.
    gan = GAN(discriminator, generator, latent_dim)
    gan.compile(
        keras.optimizers.Adam(0.0001),
        keras.optimizers.Adam(0.0001),
        keras.losses.BinaryCrossentropy()
    )
    discriminator.trainable = True  # save fix, full detail here: https://github.com/keras-team/keras/issues/10806#issuecomment-524565808
    return gan

if __name__ == '__main__':
    data_root = './data'
    input_dir = os.path.join(data_root, 'pokedex')
    temp_dir = os.path.join(data_root, 'temp')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    gif_file = os.path.join(data_root, f'progress/{timestamp}.gif')
    
    model_root = os.path.join(data_root, 'model')
    gan_path = os.path.join(model_root, 'gan')
    discriminator_path = os.path.join(model_root, 'discriminator')
    generator_path = os.path.join(model_root, 'generator')

    img_scale = 64
    img_channels = 4
    img_size = (img_scale, img_scale)
    img_shape = img_size + tuple([img_channels])
    latent_dim = img_scale*2

    epochs = 1

    dataset = keras.preprocessing.image_dataset_from_directory(
        input_dir, 
        label_mode=None, 
        color_mode='rgba', 
        image_size=img_size, 
        batch_size=64
    )

    discriminator = create_discriminator(img_shape, verbose=False)  #, load_from=discriminator_path)
    generator = create_generator(latent_dim, img_channels, verbose=False)  #, load_from=generator_path)
    gan = create_gan(discriminator, generator) # gan = load_model(gan_path)

    cb = [
        GANMonitor(1, latent_dim, temp_dir, 10)  # discriminator, generator, model_path, 
    ]
    gan.fit(dataset, epochs=epochs, callbacks=cb)

    # # save progress for the next run
    # discriminator.trainable = False  # see above for details on save fix.
    # gan.build(img_shape)
    # save_model(gan, gan_path)
    # discriminator.trainable = True
    # save_model(discriminator, discriminator_path)
    # save_model(generator, generator_path)
    
    # create gif to review progress
    with imageio.get_writer(gif_file, mode='I') as writer:
        filenames = os.listdir(temp_dir)
        for filename in filenames:
            filepath = os.path.join(temp_dir, filename)
            image = imageio.imread(filepath)
            writer.append_data(image)
        # append the final image
        image = imageio.imread(filepath)
        writer.append_data(image)
    rmtree(temp_dir)  # cleanup after a run


