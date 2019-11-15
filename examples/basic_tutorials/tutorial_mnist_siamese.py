'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.
Get 96.7% accuracy on test data after 20 epochs training.

For more details, see the reference paper.

# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf


'''

import random
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import Dense, Dropout, Flatten, Input
from tensorlayer.models import Model

num_classes = 10
epochs = 20
batch_size = 128
input_shape = (None, 784)


def contrastive_loss(label, feature1, feature2):
    margin = 1.0
    eucd = tf.sqrt(tf.reduce_sum(tf.square(feature1 - feature2), axis=1))
    return tf.reduce_mean(label * tf.square(eucd) + (1 - label) * tf.square(tf.maximum(margin - eucd, 0)))


def compute_accuracy(label, feature1, feature2):
    eucd = tf.sqrt(tf.reduce_sum((feature1 - feature2)**2, axis=1))
    pred = tf.cast(eucd < 0.5, label.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, act=tf.nn.relu)(x)
    x = Dropout(0.9)(x)
    x = Dense(128, act=tf.nn.relu)(x)
    x = Dropout(0.9)(x)
    x = Dense(128, act=tf.nn.relu)(x)
    return Model(input, x)


def get_siamese_network(input_shape):
    """Create siamese network with shared base network as layer
    """
    base_layer = create_base_network(input_shape).as_layer()

    ni_1 = Input(input_shape)
    ni_2 = Input(input_shape)
    nn_1 = base_layer(ni_1)
    nn_2 = base_layer(ni_2)
    return Model(inputs=[ni_1, ni_2], outputs=[nn_1, nn_2])


def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels).astype(np.float32)


# get network
model = get_siamese_network(input_shape)

# create training+val+test positive and negative pairs
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)

digit_indices = [np.where(y_val == i)[0] for i in range(num_classes)]
val_pairs, val_y = create_pairs(X_val, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

# training settings
print_freq = 5
train_weights = model.trainable_weights
optimizer = tf.optimizers.RMSprop()


@tf.function
def train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        _out1, _out2 = model([X_batch[:, 0, :], X_batch[:, 1, :]])
        _loss = contrastive_loss(y_batch, _out1, _out2)

    grad = tape.gradient(_loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))

    _acc = compute_accuracy(y_batch, _out1, _out2)
    return _loss, _acc


# begin training
for epoch in range(epochs):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    model.train()  # enable dropout
    for X_batch, y_batch in tl.iterate.minibatches(tr_pairs, tr_y, batch_size, shuffle=True):
        _loss, _acc = train_step(X_batch, y_batch)
        train_loss += _loss
        train_acc += _acc
        n_iter += 1

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, epochs, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

# evaluate on test data
model.eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in tl.iterate.minibatches(te_pairs, te_y, batch_size, shuffle=False):
    _out1, _out2 = model([X_batch[:, 0, :], X_batch[:, 1, :]])
    test_loss += contrastive_loss(y_batch, _out1, _out2)
    test_acc += compute_accuracy(y_batch, _out1, _out2)
    n_iter += 1
print("   test loss: {}".format(test_loss / n_iter))
print("   test acc:  {}".format(test_acc / n_iter))
