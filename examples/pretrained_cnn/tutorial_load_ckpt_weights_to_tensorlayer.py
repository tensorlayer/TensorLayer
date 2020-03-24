#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, Flatten, Dense, MaxPool2d)
from tensorlayer.models import Model
from tensorlayer.files import maybe_download_and_extract
import numpy as np
import tensorflow as tf

filename = 'ckpt_parameters.zip'
url_score = 'https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/'

# download weights
down_file = tl.files.maybe_download_and_extract(
    filename=filename, working_directory='model/', url_source=url_score, extract=True
)

model_file = 'model/ckpt_parameters'

# ckpt to npz, rename_key used to match TL naming rule
tl.files.ckpt_to_npz_dict(model_file, rename_key=True)
weights = np.load('model.npz', allow_pickle=True)

# View the parameters and weights shape
for key in weights.keys():
    print(key, weights[key].shape)


# build model
def create_model(inputs_shape):
    W_init = tl.initializers.truncated_normal(stddev=5e-2)
    W_init2 = tl.initializers.truncated_normal(stddev=0.04)
    ni = Input(inputs_shape)
    nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, name='conv1_1')(ni)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1_1')(nn)
    nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv1_2')(nn)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1_2')(nn)

    nn = Conv2d(128, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv2_1')(nn)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2_1')(nn)
    nn = Conv2d(128, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv2_2')(nn)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2_2')(nn)

    nn = Conv2d(256, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv3_1')(nn)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool3_1')(nn)
    nn = Conv2d(256, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv3_2')(nn)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool3_2')(nn)

    nn = Conv2d(512, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv4_1')(nn)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool4_1')(nn)
    nn = Conv2d(512, (3, 3), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv4_2')(nn)
    nn = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool4_2')(nn)

    nn = Flatten(name='flatten')(nn)
    nn = Dense(1000, act=None, W_init=W_init2, name='output')(nn)

    M = Model(inputs=ni, outputs=nn, name='cnn')
    return M


net = create_model([None, 224, 224, 3])
# loaded weights whose name is not found in network's weights will be skipped.
# If ckpt has the same naming rule as TL, We can restore the model with tl.files.load_and_assign_ckpt(model_dir=, network=, skip=True)
tl.files.load_and_assign_npz_dict(network=net, skip=True)

# you can use the following code to view the restore the model parameters.
net_weights_name = [w.name for w in net.all_weights]
for i in range(len(net_weights_name)):
    print(net_weights_name[i], net.all_weights[net_weights_name.index(net_weights_name[i])])
