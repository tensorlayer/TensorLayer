#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)

# set gpu mem fraction or allow growth
# tl.utils.set_gpu_fraction()

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

# define the network
ni = tl.layers.Input([None, 784])
nn = tl.layers.Dropout(keep=0.8)(ni)
nn = tl.layers.Dense(n_units=800, act=tf.nn.relu)(nn)
nn = tl.layers.Dropout(keep=0.5)(nn)
nn = tl.layers.Dense(n_units=800, act=tf.nn.relu)(nn)
nn = tl.layers.Dropout(keep=0.5)(nn)
nn = tl.layers.Dense(n_units=10, act=None)(nn)
network = tl.models.Model(inputs=ni, outputs=nn, name="mlp")


# define metric.
def acc(_logits, y_batch):
    # return np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(_logits, 1), tf.convert_to_tensor(y_batch, tf.int64)), tf.float32), name='accuracy'
    )


# print network information
print(network)

# open tensorboard
# tl.utils.open_tensorboard('./tb_log', port=6006)

# train the network
tl.utils.fit(
    network, train_op=tf.optimizers.Adam(learning_rate=0.0001), cost=tl.cost.cross_entropy, X_train=X_train,
    y_train=y_train, acc=acc, batch_size=256, n_epoch=20, X_val=X_val, y_val=y_val, eval_train=True,
    tensorboard_dir='./tb_log'
)

# test
tl.utils.test(network, acc, X_test, y_test, batch_size=None, cost=tl.cost.cross_entropy)

# evaluation
_logits = tl.utils.predict(network, X_test)
y_pred = np.argmax(_logits, 1)
tl.utils.evaluation(y_test, y_pred, n_classes=10)

# save network weights
network.save_weights('model.h5')

# close tensorboard
# tl.utils.exit_tensorflow(port=6006)
