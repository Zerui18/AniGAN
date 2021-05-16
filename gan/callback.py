import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use('Agg')

class WGANCallback(Callback):

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = batch
        # update fadein_alpha every epoch
        if self.model.during_fadein:
            progress = float(self.epoch) / self.model.n_epochs + float(batch) / float(22973 // self.model.batch_size)
            self.model.set_fadein_alpha(progress)

    def on_epoch_begin(self, epoch, logs=None):
        self.model.epoch = self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        # only generate samples sparingly on the first two stages
        if self.model.stage < 3 and epoch % 3 != 0:
            return
        # generate samples
        latent = tf.random.normal((10 * 10, 512))
        if self.model.stage < 7:
            images = self.model.generator_func(latent, training=False)
        else:
            images = []
            for i in range(0, 100, 10):
                images.append(self.model.generator_func(latent[i:i+10], training=False))
            images = tf.concat(images, axis=0)
        images = tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)
        fig, ax = plt.subplots(10, 10, figsize=(30, 30))
        for i in range(10):
            for j in range(10):
                ax[i][j].imshow(images[i * 10 + j])
        fig.tight_layout()
        # write to image
        stage = self.model.stage
        # make dir 'generated' if necessary
        folder_path = Path('generated')
        if not folder_path.is_dir():
            folder_path.mkdir()
        file_path = folder_path / f'stage{stage}_epoch{epoch}.png'
        fig.savefig(file_path)
        plt.close()
        # save model every 10 epochs for stage > 4
        if stage > 4 and epoch > 0:
            freqs = [None, None, None, None, 10, 5, 3, 2, 1]
            n_checkpoint_epochs = freqs[stage - 1]
            if epoch % n_checkpoint_epochs == 0:
                self.model.save()