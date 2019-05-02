__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, Activation

import keras
from keras import backend as K


def colorization_task(inp):
    x = Conv2D(24, (3, 3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(24, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#    x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(96, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(96, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(96, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(96, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(2, (5, 5), padding='same', name='Conv_recon')(x)
    decoded3 = Activation('sigmoid')(x)
    
    return decoded3

    
        
def kullback_leibler_divergence(y_pred, y_true):
    """ Information about probabilistic divergence.
    
       Difference btw KLD and JSD
       
       JS_divergence is the symmetrization of KL_divergence.
       The reason that JS-divergence is not so often used is probably 
       that it is less well-known and does not offer must-have properties.
       
       Alternatives of KLD is Hellinger divergences, Wasserstein metric, \
       and Bregman divergences.
    
    """
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)


def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    return K.mean(_logcosh(y_pred - y_true), axis=-1)


def wasserstein_dist(y_true, y_pred):
    return K.mean(y_true * y_pred)


def identity_block(input_tensor, kernel_size, filters, stage, block, is_add=True):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    if is_add:
        x = keras.layers.add([x, input_tensor])
        x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), is_add=True):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    if is_add:
        x = keras.layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x
    else:
        return x, shortcut