#! /usr/bin/python
# -*- coding: utf8 -*-
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model

##================== PREPARE DATA ============================================##
X_train, y_train, X_val, y_val, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))


def pad_distort_im_fn(x):
    """ Zero pads an image to 40x40, and distort it.

    Examples
    ---------
    x = pad_distort_im_fn(X_train[0])
    print(x, x.shape, x.max())
    tl.vis.save_image(x, '_xd.png')
    tl.vis.save_image(X_train[0], '_x.png')
    """
    b = np.zeros((40, 40, 1), dtype=np.float32)
    o = int((40 - 28) / 2)
    b[o:o + 28, o:o + 28] = x
    x = b
    x = tl.prepro.rotation(x, rg=30, is_random=True, fill_mode='constant')
    x = tl.prepro.shear(x, 0.05, is_random=True, fill_mode='constant')
    x = tl.prepro.shift(x, wrg=0.25, hrg=0.25, is_random=True, fill_mode='constant')
    x = tl.prepro.zoom(x, zoom_range=(0.95, 1.05))
    return x


def pad_distort_ims_fn(X):
    """ Zero pads images to 40x40, and distort them. """
    X_40 = []
    for X_a, _ in tl.iterate.minibatches(X, X, 50, shuffle=False):
        X_40.extend(tl.prepro.threading_data(X_a, fn=pad_distort_im_fn))
    X_40 = np.asarray(X_40)
    return X_40


# create dataset with size of 40x40 with distortion
X_train_40 = pad_distort_ims_fn(X_train)
X_val_40 = pad_distort_ims_fn(X_val)
X_test_40 = pad_distort_ims_fn(X_test)

tl.vis.save_images(X_test[0:32], [4, 8], '_imgs_original.png')
tl.vis.save_images(X_test_40[0:32], [4, 8], '_imgs_distorted.png')


##================== DEFINE MODEL ============================================##
def get_model(inputs_shape):
    ni = Input(inputs_shape)

    ## 1. Localisation network
    # use MLP as the localisation net
    nn = Flatten()(ni)
    nn = Dense(n_units=20, act=tf.nn.tanh)(nn)
    nn = Dropout(keep=0.8)(nn)
    # you can also use CNN instead for MLP as the localisation net

    ## 2. Spatial transformer module (sampler)
    stn = SpatialTransformer2dAffine(out_size=(40, 40), in_channels=20)
    nn = stn((nn, ni))
    s = nn

    ## 3. Classifier
    nn = Conv2d(16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME')(nn)
    nn = Conv2d(16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME')(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units=1024, act=tf.nn.relu)(nn)
    nn = Dense(n_units=10, act=tf.identity)(nn)

    M = Model(inputs=ni, outputs=[nn, s])
    return M


net = get_model([None, 40, 40, 1])

##================== DEFINE TRAIN OPS ========================================##
n_epoch = 100
learning_rate = 0.0001
print_freq = 10
batch_size = 64
train_weights = net.trainable_weights
optimizer = tf.optimizers.Adam(lr=learning_rate)

##================== TRAINING ================================================##
print("Training ...")
for epoch in range(n_epoch):
    start_time = time.time()

    net.train()  # enable dropout

    for X_train_a, y_train_a in tl.iterate.minibatches(X_train_40, y_train, batch_size, shuffle=True):
        # input_dim must be of length 4
        X_train_a = tf.expand_dims(X_train_a, 3)

        with tf.GradientTape() as tape:
            ## compute outputs
            _logits, _ = net(X_train_a)  # alternatively, you can use MLP(x, is_train=True) and remove MLP.train()
            ## compute loss and update model
            _loss = tl.cost.cross_entropy(_logits, y_train_a, name='train_loss')

        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

    ## use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:

        net.eval()  # disable dropout

        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))

        train_loss, train_acc, n_iter = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train_40, y_train, batch_size, shuffle=False):
            # input_dim must be of length 4
            X_train_a = tf.expand_dims(X_train_a, 3)

            _logits, _ = net(X_train_a)  # alternatively, you can use MLP(x, is_train=False) and remove MLP.eval()
            train_loss += tl.cost.cross_entropy(_logits, y_train_a, name='eval_train_loss')
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_train_a))
            n_iter += 1
        print("   train loss: %f" % (train_loss / n_iter))
        print("   train acc: %f" % (train_acc / n_iter))

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(X_val_40, y_val, batch_size, shuffle=False):
            # input_dim must be of length 4
            X_val_a = tf.expand_dims(X_val_a, 3)

            _logits, _ = net(X_val_a)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_val_a, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_val_a))
            n_iter += 1
        print("   val loss: %f" % (val_loss / n_iter))
        print("   val acc: %f" % (val_acc / n_iter))

        print('save images')
        _, trans_imgs = net(tf.expand_dims(X_test_40[0:64], 3))
        trans_imgs = trans_imgs.numpy()
        tl.vis.save_images(trans_imgs[0:32], [4, 8], '_imgs_distorted_after_stn_%s.png' % epoch)

##================== EVALUATION ==============================================##
print('Evaluation')

net.eval()

test_loss, test_acc, n_iter = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test_40, y_test, batch_size, shuffle=False):
    # input_dim must be of length 4
    X_test_a = tf.expand_dims(X_test_a, 3)

    _logits, _ = net(X_test_a)
    test_loss += tl.cost.cross_entropy(_logits, y_test_a, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_test_a))
    n_iter += 1
print("   test loss: %f" % (test_loss / n_iter))
print("   test acc: %f" % (test_acc / n_iter))
