from tensorflow.keras.layers import Input, LeakyReLU, AveragePooling2D, Flatten
from tensorflow.keras.models import Model
from .layers import *

def reflow_layers(x_in, layers):
    x = x_in
    for layer in layers:
        x = layer(x)
    return x

class Discriminator:
    
    fadein_alpha = None
    model_no_fadein = None
    
    def __init__(self, config={}):
        # training params
        self.config = config.copy()
        self.initial_build()
            
    def update_input(self):
        stage = self.config['stage']
        image_dim = int(2 ** (stage + 1))
        self.input = Input((image_dim, image_dim, 3), name='dis/input')
    
    def initial_build(self):
        if self.config['during_fadein']:
            self.config['stage'] -= 1
        n_stage = self.config['stage']
        self.update_input()
        x = self.from_rgb(self.input, n_stage, is_identity=False)
        for stage in reversed(range(1, n_stage+1)):
            with tf.name_scope(f'stage_{stage}'):
                if stage > 1:
                    x = self.downscale_block(x, stage)
                else:
                    x = self.last_block(x)
        x = Flatten(name='dis/flatten')(x)
        x = DenseEqualized(1, name='dis/dense')(x)
        self.model = Model(self.input, x)
        if self.config['during_fadein']:
            self.grow_with_fadein()
        
    def grow_with_fadein(self):
        ''' Perform fade-in growth of the discriminator.'''
        self.config['stage'] += 1
        self.update_input()
        stage = self.config["stage"]
        # skip connection
        id_x = AveragePooling2D((2, 2), name='dis/id_ap')(self.input)
        id_x = self.from_rgb(id_x, stage, is_identity=True)
        # main branch
        x = self.from_rgb(self.input, stage, is_identity=False)
        with tf.name_scope(f'stage_{stage}'):
            x = self.downscale_block(x, stage)
        # merge with alpha
        merged_x = FadeinMerge(name='dis/fadein_merge')([id_x, x])
        # build model without fadein to be used later
        # skip: Conv2D(1x1), PixelNorm to get the first non 1x1 Conv2D
        old_model_layers = self.model.layers[2:]
        x = reflow_layers(x, old_model_layers)
        self.model_no_fadein = Model(self.input, x)
        # rebuild model with fadein
        # attach new block
        fadein_x = reflow_layers(merged_x, old_model_layers)
        self.model = Model(self.input, fadein_x)
        self.find_fadein_alpha()
    
    def remove_fadein(self):
        self.model = self.model_no_fadein
        self.model_no_fadein = None
    
    def last_block(self, x):
        prefix = 'dis/stage1/'
        x = MiniBatchStdDev(name=prefix + 'minibatch_std')(x)
        x = Conv2DEqualized(512, (3, 3), padding='same', name=prefix+'conv1')(x)
        x = LeakyReLU(0.2, name=prefix+'lr1')(x)
        x = Conv2DEqualized(512, (4, 4), padding='valid', name=prefix+'conv2')(x)
        x = LeakyReLU(0.2, name=prefix+'lr2')(x)
        return x
    
    def downscale_block(self, x, stage):
        if stage > 4:
            ini_depth = int(512 / (2 ** (stage - 4)))
            final_depth = ini_depth * 2
        else:
            ini_depth = final_depth = 512
        prefix = f'dis/stage{stage}/'
        x = Conv2DEqualized(ini_depth, (3, 3), padding='same', name=prefix+'conv1')(x)
        x = LeakyReLU(0.2, name=prefix+'lr1')(x)
        x = Conv2DEqualized(final_depth , (3, 3), padding='same', name=prefix+'conv2')(x)
        x = LeakyReLU(0.2, name=prefix+'lr2')(x)
        x = AveragePooling2D((2, 2), name=prefix+'ap')(x)
        return x
    
    def from_rgb(self, x, stage, is_identity):
        if stage > 4:
            depth = int(512 / (2 ** (stage - 4)))
            if is_identity:
                depth *= 2
        else:
            depth = 512
        x = Conv2DEqualized(depth, (1, 1), padding='same', name=f'dis/{"id_" if is_identity else ""}from_rgb')(x)
        return x
    
    def set_fadein_alpha(self, alpha):
        self.fadein_alpha.assign(alpha)
        
    def find_fadein_alpha(self):
        layer = self.model.get_layer('dis/fadein_merge')
        if layer:
            self.fadein_alpha = layer.weights[0]