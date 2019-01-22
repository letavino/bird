import numpy as np

from keras import backend as K
from keras.engine import Layer

######################################################
##### Dropout Klassen
######################################################

# Normaler Dropout
#
# Aktiviert während Trainingsphase
# Deaktibiert während Testphase

class GalDropout(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(GalDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)

            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(GalDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
	
	
# Gal Dropout
#
# Aktiviert während Trainingsphase und Testphase
	
class GalDropout2(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(GalDropout2, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
     
            return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(GalDropout2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
