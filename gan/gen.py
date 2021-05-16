from tensorflow.keras.layers import Input, LeakyReLU, Reshape, UpSampling2D
from tensorflow.keras.models import Model
from .layers import *

class Generator:
    
    fadein_alpha = None
    model_no_fadein = None

    def __init__(self, config={}):
        # training params
        self.config = config.copy()
        self.initial_build()
            
    def initial_build(self):
        if self.config['during_fadein']:
            self.config['stage'] -= 1
        n_stage = self.config['stage']
        # normalize inputs
        self.input = Input((512), name='gen/input')
        x = PixelNormalization(name='gen/input_norm')(self.input)
        x = DenseEqualized(512 * 4 * 4, name='gen/input_map')(x)
        x = LeakyReLU(0.2, name='gen/input_map_act')(x)
        x = PixelNormalization(name='gen/input_map_norm')(x)
        x = Reshape((4, 4, 512))(x)
        for stage in range(1, n_stage+1):
            with tf.name_scope(f'stage_{stage}'):
                if stage > 1:
                    x = self.upscale_block(x, stage)
                else:
                    x = self.first_upscale_block(x)
        x = self.to_rgb(x, is_identity=False)
        self.model = Model(self.input, x)
        if self.config['during_fadein']:
            self.grow_with_fadein()
        
    def grow_with_fadein(self):
        self.config['stage'] += 1
        # skip 'to_rgb' layer to get the last Conv2D
        old_model_bottom = self.model.layers[-2]
        # build both residual & normal branches
        with tf.name_scope(f'stage_{self.config["stage"]}'):
            x_merged, x = self.upscale_block(old_model_bottom.output, self.config['stage'], use_fadein=True)
        # build model without fadein
        self.model_no_fadein = Model(self.input, x)
        # rebuild model with fadein
        self.model = Model(self.input, x_merged)
        self.find_fadein_alpha()
    
    def remove_fadein(self):
        del self.model
        self.model = self.model_no_fadein
    
    def first_upscale_block(self, x):
        prefix = 'gen/stage1/'
        x = Conv2DEqualized(512, (4, 4), padding='same', name=prefix+'conv1')(x)
        x = LeakyReLU(0.2, name=prefix+'lr1')(x)
        x = PixelNormalization(name=prefix+'pn1')(x)
        x = Conv2DEqualized(512, (3, 3), padding='same', name=prefix+'conv2')(x)
        x = LeakyReLU(0.2, name=prefix+'lr2')(x)
        x = PixelNormalization(name=prefix+'pn2')(x)
        return x
    
    def upscale_block(self, x, stage, use_fadein=False):
        if stage > 4:
            depth = int(512 / (2 ** (stage - 4)))
        else:
            depth = 512
        prefix = f'gen/stage{stage}/'
        x = UpSampling2D((2, 2), name=prefix+'us1')(x)
        if use_fadein:
            id_x = self.to_rgb(x, is_identity=True)
        x = Conv2DEqualized(depth, (3, 3), padding='same', name=prefix+'conv1')(x)
        x = LeakyReLU(0.2, name=prefix+'lr1')(x)
        x = PixelNormalization(name=prefix+'pn1')(x)
        x = Conv2DEqualized(depth, (3, 3), padding='same', name=prefix+'conv2')(x)
        x = LeakyReLU(0.2, name=prefix+'lr2')(x)
        x = PixelNormalization(name=prefix+'pn2')(x)
        # additional steps if this block is being faded in
        if use_fadein:
            # to_rgb before addition
            x = self.to_rgb(x, is_identity=False)
            # add and return both the merged and original branches
            merged_x = FadeinMerge(name='gen/fadein_merge')([id_x, x])
            return merged_x, x
        # return normally
        return x
    
    def to_rgb(self, x, is_identity=False):
        x = Conv2DEqualized(3, (1, 1), padding='same', name=f'gen/{"id_" if is_identity else ""}to_rgb')(x)
        return x
    
    def set_fadein_alpha(self, alpha):
        self.fadein_alpha.assign(alpha)
        
    def find_fadein_alpha(self):
        layer = self.model.get_layer('gen/fadein_merge')
        if layer:
            self.fadein_alpha = layer.weights[0]