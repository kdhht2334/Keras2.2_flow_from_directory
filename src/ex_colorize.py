__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-

# Argument parsing for # of GPUs
import argparse
app = argparse.ArgumentParser()
#app.add_argument("-o", "--output", required=True,
#                 help="path to output plot")
app.add_argument("-g", "--gpus", type=int, default=1,
                 help="# of GPUs to use for training")
args = vars(app.parse_args())

G = args["gpus"]

# In case of you don't want to use GPU device.
import os
#os.environ['CUDA_VISIBLE_DEVICES'] 

# Set GPU options for loading keras library
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(G)
set_session(tf.Session(config=config))

print()
print("[INFO] training with {} GPUs ...".format(G))
print()

import numpy as np

from six.moves import xrange

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Concatenate, add
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Conv2DTranspose, UpSampling2D
from keras import models, optimizers
from keras import backend as K
from keras.engine import Input
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, LearningRateScheduler

from opt import conv_block, identity_block, colorization_task


def preprocess(x):  # In case of training generative model, 
                    # switch off preprocess_input, -=, and *= operations.
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    x /= 255.
    #x -= 0.5
    #x *= 2.0
    return x


def _add(arg1, arg2):
    x1 = arg1; x2 = arg2
    return x1 + x2


def _multiply(arg1, arg2):
    x1 = arg1; x2 = arg2
    return x1 * x2

# Keras specific
if K.image_dim_ordering() == "th":
    channel_axis = 1
    input_shape = (3, 224, 224)
    bn_axis = 1
else:
    channel_axis = -1
    input_shape = (224, 224, 3)
    bn_axis = 3
    
# Training parameters
batch_size = 32
maxepoches = 100
num_classes = 50
learning_rate = 0.1
lr_decay = 1e-6

ds_val = ds = '../../datasets/ILSVRC2012'

# Build model

input_img = Input(shape=(224, 224, 3))
x = keras.layers.ZeroPadding2D((3, 3))(input_img)
x1 = Conv2D(24, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)  # 112
x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x1)
x = Activation('relu')(x)
x2 = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 56; original pooling kernel size is 3

x = conv_block(x2,    3, [24, 64, 64], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [24, 64, 64], stage=2, block='b')
x = identity_block(x, 3, [24, 64, 64], stage=2, block='c')

x3 = conv_block(x,     3, [64, 64, 64], stage=3, block='a')  # 28
x = identity_block(x3, 3, [64, 64, 64], stage=3, block='b')
x = identity_block(x,  3, [64, 64, 64], stage=3, block='c')
x = identity_block(x,  3, [64, 64, 64], stage=3, block='d')

x4 = conv_block(x,     3, [64, 96, 96], stage=4, block='a')  # 14
x = identity_block(x4, 3, [64, 96, 96], stage=4, block='b')
x = identity_block(x,  3, [64, 96, 96], stage=4, block='c')
x = identity_block(x,  3, [64, 96, 96], stage=4, block='d')
x = identity_block(x,  3, [64, 96, 96], stage=4, block='e')
x = identity_block(x,  3, [64, 96, 96], stage=4, block='f')

x5 = conv_block(x,      3, [96, 96, 96], stage=5, block='a')  # 7
x = identity_block(x5,  3, [96, 96, 96], stage=5, block='b')
x_b = identity_block(x, 3, [96, 96, 96], stage=5, block='c')

decoded3 = colorization_task(x_b)
model_cor = models.Model(input_img, decoded3)

# variables of loss weights
gamma = K.variable(1.0)

# Optimization details
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model_cor.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  loss_weights=[gamma],
                  metrics=['mse'])


# %%

print()
print("Data augmentation processing...")
print()

# data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess)

train_generator2 = train_datagen.flow_from_directory(
        '%s/train/' % ds,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="colorize")

print()
print("Training start!!")
print()

    
for epoch in xrange(maxepoches):
    print()
    print("[INFO] epoch: %d" % (epoch))

    history = model_cor.fit_generator(
                train_generator2, steps_per_epoch=2000,  #train_generator2.samples // batch_size,
                epochs=1, verbose=1, callbacks=[])
    
    if epoch > 50:    
        learning_rate = learning_rate / 10.
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model_cor.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          loss_weights=[gamma],
                          metrics=['mse'])
        print("[INFO] Lr changed.")
    