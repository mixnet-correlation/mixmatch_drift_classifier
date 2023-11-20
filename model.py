'''
The model is inspired in the DF model by Sirinam et al
https://github.com/triplet-fingerprinting/tf/blob/master/src/model_training/DF_model.py
'''

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten


class DriftModel(Model):
    '''Our drift model.'''

    # This drift model exclusively computes a score that only regards drift
    def __init__(self,
                 logger,
                 kernel_size,
                 pool_size):

        super().__init__()
        self.preproc = ()
        self.conv_layers_drift = ()
        self.pool_layers_drift = ()
        self.fcc_drift = ()

        logger.info(f"In model.py init: using {kernel_size=}")

        # self.preproc += (Normalization(),)

        self.conv_layers_drift += (Conv1D(4, kernel_size,
                                   activation='relu', padding='same'),)
        self.conv_layers_drift += (Conv1D(8, kernel_size,
                                   activation='relu', padding='same'),)
        self.conv_layers_drift += (Conv1D(16, kernel_size,
                                   activation='relu', padding='same'),)

        self.pool_layers_drift += (
            AveragePooling1D(pool_size=pool_size, strides=2),)
        self.pool_layers_drift += (
            AveragePooling1D(pool_size=pool_size, strides=2),)
        self.pool_layers_drift += (
            AveragePooling1D(pool_size=pool_size, strides=2),)

        self.flatten_drift = Flatten()

        self.fcc_drift += (Dense(16, activation='relu'),)
        self.fcc_drift += (Dense(16, activation='relu'),)
        self.fcc_drift += (Dense(1, activation='sigmoid'),)

    def call(self, inputs):
        '''Conduct model action.'''

        inflow, outflow = inputs
        drift_score = tf.cast(outflow, dtype=np.float32) - \
            tf.cast(inflow, dtype=np.float32)

        for l in self.preproc:
            drift_score = l(drift_score)

        for k in range(len(self.conv_layers_drift)):
            drift_score = self.conv_layers_drift[k](drift_score)
            drift_score = self.pool_layers_drift[k](drift_score)
        drift_score = self.flatten_drift(drift_score)

        for l in self.fcc_drift:
            drift_score = l(drift_score)

        return drift_score
