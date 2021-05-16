import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
from datetime import datetime
from pathlib import Path
import shutil
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from .gen import Generator
from .dis import Discriminator
from .datasource import get_dataset
from .callback import WGANCallback

def image_size_for_stage(stage):
    dim = int(2 ** (stage + 1))
    return (dim, dim)

class WGAN(Model):
    
    epoch = 0
    g_optimizer = Adam(1e-3, 0, 0.99)
    d_optimizer = Adam(1e-3, 0, 0.99)
    
    def __init__(
        self,
        config=None,
        checkpoint_path=None
    ):
        super(WGAN, self).__init__()
        # note that config is shared among the generator and discriminator
        loading_from_save = not config
        if config:
            self.config = config
        else:
            with open(checkpoint_path / 'config.json') as f:
                self.config = json.load(f)
        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config)
        if loading_from_save:
            # prepass to initialize all vars
            self.compile()
            images = tf.random.normal((self.batch_size, *self.image_size, 3))
            self.train_step(images)
            # restore states from checkpoint
            ckpt = self.create_checkpoint()
            ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

    def create_checkpoint(self):
        if self.during_fadein:
                checkpoint = tf.train.Checkpoint(d_optimizer=self.d_optimizer, g_optimizer=self.g_optimizer,
                                                 generator=self.generator_func, discriminator=self.discriminator_func,
                                                 generator_no_fadein=self.generator.model_no_fadein, discriminator_no_fadein=self.discriminator.model_no_fadein)
        else:
            checkpoint = tf.train.Checkpoint(d_optimizer=self.d_optimizer, g_optimizer=self.g_optimizer,
                                             generator=self.generator_func, discriminator=self.discriminator_func)
        return checkpoint
                
    def compile(self):
        super(WGAN, self).compile()
        self.fadein_alpha = None
        self.d_loss_fn = lambda real_logits, fake_logits: - tf.reduce_mean(real_logits) + tf.reduce_mean(fake_logits)
        self.g_loss_fn = lambda gen_img_logits: -tf.reduce_mean(gen_img_logits)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator_func(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2.0)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Get the latent vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, 512)
        )
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator_func(random_latent_vectors, training=True)
            # Get the logits for the fake images
            fake_logits = self.discriminator_func(fake_images, training=True)
            # Get the logits for the real images
            real_logits = self.discriminator_func(real_images, training=True)
            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.d_loss_fn(real_logits, fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(batch_size, real_images, fake_images)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * 10.0

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator_func.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator_func.trainable_variables)
        )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, 512))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator_func(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator_func(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator_func.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator_func.trainable_variables)
        )
        
        return {"d_loss": d_loss, "g_loss": g_loss,
                'mean_real_logits': tf.reduce_mean(real_logits),
                'mean_fake_logits': tf.reduce_mean(fake_logits),
                'mean_gen_logits': tf.reduce_mean(gen_img_logits)}
    
    @property
    def generator_func(self):
        return self.generator.model
    
    @property
    def discriminator_func(self):
        return self.discriminator.model
    
    @property
    def stage(self):
        return self.config['stage']
    
    @property
    def image_size(self):
        return image_size_for_stage(self.stage)
    
    @property
    def batch_size(self):
        batch_sizes = [256, 256, 128, 64, 16, 8, 6, 3, 3]
        return batch_sizes[self.stage-1]
    
    @property
    def n_epochs(self):
        epochs = [35, 45, 60, 75, 75, 75, 75, 75]
        return epochs[self.stage]
    
    @property
    def during_fadein(self):
        return self.generator.model_no_fadein is not None and self.discriminator.model_no_fadein is not None
    
    def fit_n_epochs(self, epochs, initial_epoch=0):
        self.fit(get_dataset(self.image_size, self.batch_size), epochs=epochs, callbacks=[WGANCallback()], initial_epoch=initial_epoch)

    def initial_cycle(self):
        n_epochs = self.n_epochs
        self.fit_n_epochs(n_epochs)
    
    def one_growth_cycle(self):
        n_epochs = self.n_epochs
        self.config['stage'] += 1
        print(f'New stage {self.stage}')
        # grow with fade-in
        self.generator.grow_with_fadein()
        self.discriminator.grow_with_fadein()
        self.compile()
        print('Model growth completed.')
        # stabilise
        print('Stabilising with fade-in.')
        self.fit_n_epochs(n_epochs)
        # remove fade-in
        print('Removed fade-in and re-stabilising.')
        self.generator.remove_fadein()
        self.discriminator.remove_fadein()
        self.compile()
        # stabilise
        self.fit_n_epochs(n_epochs)

    def fit_remaining_epochs(self):
        initial_epoch = self.config['epoch'] + 1
        during_fadein = self.config['during_fadein']
        if during_fadein:
            # first complete the remaining epochs during fadein
            if initial_epoch <= self.n_epochs:
                self.fit_n_epochs(self.n_epochs, initial_epoch=initial_epoch)
            # after fadein, we still have a full self.n_epoch of tuning
            self.generator.remove_fadein()
            self.discriminator.remove_fadein()
            print('Removed fade-in and re-stabilising.')
            self.compile()
            # stabilise
            self.fit_n_epochs(self.n_epochs)
        else:
            self.compile()
            self.fit_n_epochs(self.n_epochs, initial_epoch=initial_epoch)
        
    def set_fadein_alpha(self, alpha):
        self.generator.set_fadein_alpha(alpha)
        self.discriminator.set_fadein_alpha(alpha)
        
    def save(self, checkpoint_path=None):
        # make folder
        if not checkpoint_path:
            time = datetime.now().strftime(f"%d-%m-%Y_%H-%M")
            checkpoint_path = f'saves/save_{self.config["stage"]}_' + time
        folder_path = Path(checkpoint_path)
        if folder_path.is_dir():
            # clear existing folder
            shutil.rmtree(folder_path)
        folder_path.mkdir()
        # save config
        config = self.config.copy()
        config['epoch'] = self.epoch
        config['during_fadein'] = self.during_fadein
        # create 'saves' dir if necessary
        if folder_path.parts[0] == 'saves' and not folder_path.is_dir():
            folder_path.mkdir(parents=True)
        with open(f'{checkpoint_path}/config.json', 'w') as f:
            json.dump(config, f)
        # save model
        ckpt = self.create_checkpoint()
        save_path = ckpt.save(folder_path / 'ckpt')
        print(f'Model saved at {save_path}')

    def generate_images(self, save_path):
        ''' Generate a set of images with the current generator and save them to path. '''
        # generate samples
        latent = tf.random.normal((10 * 10, 512))
        if self.stage < 7:
            images = self.generator_func(latent, training=False)
        else:
            images = []
            for i in range(0, 100, 10):
                images.append(self.generator_func(latent[i:i+10], training=False))
            images = tf.concat(images, axis=0)
        images = tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)
        fig, ax = plt.subplots(10, 10, figsize=(30, 30))
        for i in range(10):
            for j in range(10):
                ax[i][j].imshow(images[i * 10 + j])
        fig.tight_layout()
        # write to image
        fig.savefig(save_path)
        plt.close()