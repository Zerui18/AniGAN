import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import gen_math_ops
import numpy as np

class PixelNormalization(Layer):
    
    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + 1e-8)
    
class Conv2DEqualized(Layer):
    
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1)
    bias_initializer = tf.keras.initializers.Zeros()
    
    def __init__(self, n_kernels, kernel_size, padding, name):
        super(Conv2DEqualized, self).__init__(name=name)
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.c = np.sqrt(2.0 / float(input_shape[-1] * self.kernel_size[0] * self.kernel_size[1]))
        with tf.name_scope(self.name):
            self.kernels = self.add_weight(
                name='kernel',
                shape=(*self.kernel_size, input_shape[-1], self.n_kernels),
                initializer=Conv2DEqualized.kernel_initializer,
                trainable=True)
            self.bias = self.add_weight(
              name='bias',
              shape=(self.n_kernels,),
              initializer=Conv2DEqualized.bias_initializer,
              trainable=True)
        
    def call(self, inputs):
        with tf.name_scope(self.name):
            x = tf.nn.conv2d(inputs, self.kernels * self.c, 1, self.padding.upper(), name='conv_op')
            x = tf.nn.bias_add(x, self.bias, name='bias_op')
            return x
        
    def get_config(self):
        return {
            'n_kernels' : self.n_kernels,
            'kernel_size' : self.kernel_size,
            'padding' : self.padding,
            'name' : self.name
        }

class DenseEqualized(Layer):

    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1)
    bias_initializer = tf.keras.initializers.Zeros()
    
    def __init__(self, n_units, name):
        super(DenseEqualized, self).__init__(name=name)
        self.n_units = n_units

    def build(self, input_shape):
        self.c = np.sqrt(2.0 / float(input_shape[-1]))
        with tf.name_scope(self.name):
            self.kernels = self.add_weight(
                name='weight',
                shape=(input_shape[-1], self.n_units),
                initializer=DenseEqualized.kernel_initializer,
                trainable=True)
            self.bias = self.add_weight(
                name='bias',
                shape=(self.n_units,),
                initializer=DenseEqualized.bias_initializer,
                trainable=True
            )
        
    def call(self, inputs):
        with tf.name_scope(self.name):
            return gen_math_ops.MatMul(a=inputs, b=self.kernels) + self.bias
        
    def get_config(self):
        return {
            'n_units' : self.n_units,
            'name' : self.name
        }
    
class FadeinMerge(Layer):
    
    def __init__(self, name):
        super(FadeinMerge, self).__init__(name=name)
        
    def build(self, input_shapes):
        self.alpha = self.add_weight(name='alpha',
                                     shape=tuple(),
                                     initializer='zero',
                                     trainable=False)
        
    def call(self, inputs):
        return (1 - self.alpha) * inputs[0] + self.alpha * inputs[1]
    
    def get_config(self):
        return {
            'name' : self.name
        }
    
class MiniBatchStdDev(Layer):

    def __init__(self, name):
        super(MiniBatchStdDev, self).__init__(name=name)
        self.group_size = 1

    def call(self, inputs):
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [self.group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [self.group_size, 1, s[2], s[3]])        # [N1HW]  Replicate over group and pixels.
        x = tf.concat([x, y], axis=1)                           # [NCHW]  Append as new fmap.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x
    
    def get_config(self):
        return {
            'name' : self.name
        }
    
custom_objects = {
    'PixelNormalization' : PixelNormalization,
    'Conv2DEqualized' : Conv2DEqualized,
    'DenseEqualized' : DenseEqualized,
    'FadeinMerge' : FadeinMerge,
    'MiniBatchStdDev' : MiniBatchStdDev,
}