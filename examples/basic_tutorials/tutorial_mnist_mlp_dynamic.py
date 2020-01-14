import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.models import Model

## enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)

## prepare MNIST data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))


## define the network
class CustomModel(Model):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(keep=0.8)  #(self.innet)
        self.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784)  #(self.dropout1)
        self.dropout2 = Dropout(keep=0.8)  #(self.dense1)
        self.dense2 = Dense(n_units=800, act=tf.nn.relu, in_channels=800)  #(self.dropout2)
        self.dropout3 = Dropout(keep=0.8)  #(self.dense2)
        self.dense3 = Dense(n_units=10, act=tf.nn.relu, in_channels=800)  #(self.dropout3)

    def forward(self, x, foo=None):
        z = self.dropout1(x)
        z = self.dense1(z)
        z = self.dropout2(z)
        z = self.dense2(z)
        z = self.dropout3(z)
        out = self.dense3(z)
        if foo is not None:
            out = tf.nn.relu(out)
        return out


MLP = CustomModel()

## start training
n_epoch = 500
batch_size = 500
print_freq = 5
train_weights = MLP.trainable_weights
optimizer = tf.optimizers.Adam(learning_rate=0.0001)

## the following code can help you understand SGD deeply
for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP.train()  # enable dropout
        with tf.GradientTape() as tape:
            ## compute outputs
            _logits = MLP(X_batch, foo=1)
            ## compute loss and update model
            _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

    ## use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        MLP.eval()  # disable dropout
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            _logits = MLP(X_batch, foo=1)
            train_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   train foo=1 loss: {}".format(train_loss / n_iter))
        print("   train foo=1 acc:  {}".format(train_acc / n_iter))
        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _logits = MLP(X_batch, foo=1)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val foo=1 loss: {}".format(val_loss / n_iter))
        print("   val foo=1 acc:  {}".format(val_acc / n_iter))
        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _logits = MLP(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val foo=0 loss: {}".format(val_loss / n_iter))
        print("   val foo=0 acc:  {}".format(val_acc / n_iter))

## use testing data to evaluate the model
MLP.eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
    _logits = MLP(X_batch, foo=1)
    test_loss += tl.cost.cross_entropy(_logits, y_batch, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print("   test foo=1 loss: {}".format(val_loss / n_iter))
print("   test foo=1 acc:  {}".format(val_acc / n_iter))
