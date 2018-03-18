#! /usr/bin/python
# -*- coding: utf-8 -*-

import time, os, json
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, ConcatLayer, DropoutLayer, GlobalMeanPool2d


def decode_predictions(preds, top=5): # keras.applications.resnet50
    """Decodes the prediction of an ImageNet model.
    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.
    # Returns§
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    fpath = os.path.join("data", "imagenet_class_index.json")
    if tl.files.file_exists(fpath) is False:
        raise Exception("{} / download imagenet_class_index.json from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data")
    if isinstance(preds, np.ndarray) is False:
        preds = np.asarray(preds)
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    with open(fpath) as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def squeezenet(x, is_train=True, reuse=False):
    # https://github.com/wohlert/keras-squeezenet
    # https://github.com/DT42/squeezenet_demo/blob/master/model.py
    with tf.variable_scope("squeezenet", reuse=reuse):
        with tf.variable_scope("input"):
            n = InputLayer(x)
            # n = Conv2d(n, 96, (7,7),(2,2),tf.nn.relu,'SAME',name='conv1')
            n = Conv2d(n, 64, (3,3),(2,2),tf.nn.relu,'SAME',name='conv1')
            n = MaxPool2d(n, (3,3),(2,2), 'VALID', name='max')

        with tf.variable_scope("fire2"):
            n = Conv2d(n, 16, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 64, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 64, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')

        with tf.variable_scope("fire3"):
            n = Conv2d(n, 16, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 64, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 64, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')
            n = MaxPool2d(n, (3,3), (2,2), 'VALID', name='max')

        with tf.variable_scope("fire4"):
            n = Conv2d(n, 32, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 128, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 128, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')

        with tf.variable_scope("fire5"):
            n = Conv2d(n, 32, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 128, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 128, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')
            n = MaxPool2d(n, (3,3),(2,2), 'VALID', name='max')

        with tf.variable_scope("fire6"):
            n = Conv2d(n, 48, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 192, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 192, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')

        with tf.variable_scope("fire7"):
            n = Conv2d(n, 48, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 192, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 192, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')

        with tf.variable_scope("fire8"):
            n = Conv2d(n, 64, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 256, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 256, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')

        with tf.variable_scope("fire9"):
            n = Conv2d(n, 64, (1,1),(1,1),tf.nn.relu, 'SAME',name='squeeze1x1')
            n1 = Conv2d(n, 256, (1,1),(1,1),tf.nn.relu, 'SAME',name='expand1x1')
            n2 = Conv2d(n, 256, (3,3),(1,1),tf.nn.relu, 'SAME',name='expand3x3')
            n = ConcatLayer([n1,n2], -1, name='concat')

        with tf.variable_scope("output"):
            n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='drop1')
            n = Conv2d(n, 1000, (1,1),(1,1),padding='VALID', name='conv10') # 13, 13, 1000
            n = GlobalMeanPool2d(n)
            # print(n)
            # exit()
        return n

x = tf.placeholder(tf.float32, (None, 224, 224, 3))
n = squeezenet(x, False, False)
softmax = tf.nn.softmax(n.outputs)
n.print_layers()
n.print_params(False)

sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)

# ## model : https://github.com/avoroshilov/tf-squeezenet
# import scipy.io
# v = scipy.io.loadmat('sqz_full.mat')
# print(type(v), v.keys())
# print(len(v['conv1'][0]))
#
# params = [v['conv1'][0][0], v['conv1'][0][1].squeeze(0)]
# for i in range(2, 10):
#     print("fire%d"%i)
#     pp = [v['fire%d/squeeze1x1'%i][0][0], v['fire%d/squeeze1x1'%i][0][1].squeeze(0),
#         v['fire%d/expand1x1'%i][0][0], v['fire%d/expand1x1'%i][0][1].squeeze(0),
#         v['fire%d/expand3x3'%i][0][0], v['fire%d/expand3x3'%i][0][1].squeeze(0)]
#     params.extend(pp)
# params.extend([v['conv10'][0][0], v['conv10'][0][1].squeeze(0)])
# # for p in n.all_params:
# #     for key in v.keys():
# #         if key in p.name:
# #             print(key, p.name)
# #             for param in v[key]:
# #                 p.assign(param)
# #                 exit()
#
# for key in v.keys():
#     if '_' not in key:
#         print(key, v[key][0][0].shape, v[key][0][1].shape)
#
# # print(v['conv1'][0][0].shape, v['conv1'][0][1].shape)
# # tl.files.assign_params(sess, params, n)


import keras.backend as K

from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

# from keras.applications.imagenet_utils import decode_predictions
# from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
# from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'squeezenet_weights.h5'

def _fire(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
    expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    return x

def SqueezeNet(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)
    # print(input_shape)
    # exit()
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="valid")(x)

    x = _fire(x, (16, 64, 64), name="fire2")
    x = _fire(x, (16, 64, 64), name="fire3")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="valid")(x)

    x = _fire(x, (32, 128, 128), name="fire4")
    x = _fire(x, (32, 128, 128), name="fire5")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="valid")(x)

    x = _fire(x, (48, 192, 192), name="fire6")
    x = _fire(x, (48, 192, 192), name="fire7")

    x = _fire(x, (64, 256, 256), name="fire8")
    x = _fire(x, (64, 256, 256), name="fire9")

    if include_top:
        x = Dropout(0.5, name='dropout9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        print(x)
        x = AveragePooling2D(pool_size=(13, 13), name='avgpool10')(x)
        print(x)
        x = Flatten(name='flatten10')(x)
        x = Activation("softmax", name='softmax')(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D(name="avgpool10")(x)
        else:
            x = GlobalMaxPooling2D(name="maxpool10")(x)

    model = Model(img_input, x, name="squeezenet")

    if weights == 'imagenet':
        # weights_path = get_file('squeezenet_weights.h5',
        #                         WEIGHTS_PATH,
        #                         cache_subdir='models')
        weights_path = 'squeezenet_weights.h5'
        model.load_weights(weights_path)

    return model

model = SqueezeNet()
for i in range(len(model.get_weights())):
    print(model.get_weights()[i].shape, n.all_params[i].get_shape().as_list(), n.all_params[i].name)
    if list(model.get_weights()[i].shape) != n.all_params[i].get_shape().as_list():
        print("  [x] check: %s" % n.all_params[i].name)

params = model.get_weights()

tl.files.assign_params(sess, params, n)

img = tl.vis.read_image('data/tiger.jpeg','')  # test data in github
img = tl.prepro.imresize(img, (224, 224))
start_time = time.time()
prob = sess.run(softmax, feed_dict={x: [img]})[0]
print(prob.sum())
print("  End time : %.5ss" % (time.time() - start_time))

print('Predicted:', decode_predictions([prob], top=3)[0])
tl.files.save_npz(n.all_params, name='squeezenet.npz', sess=sess)
tl.files.load_and_assign_npz(sess=sess, name='squeezenet.npz', network=n)
