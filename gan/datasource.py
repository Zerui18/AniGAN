import tensorflow as tf

__DATA_SET_DIR = './images'

def get_dataset(image_size, batch_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(__DATA_SET_DIR, label_mode=None,
                                                             shuffle=True,
                                                             batch_size=batch_size,
                                                             image_size=image_size)
    return ds.map(lambda images: (images-127.5) / 127.5).prefetch(30)