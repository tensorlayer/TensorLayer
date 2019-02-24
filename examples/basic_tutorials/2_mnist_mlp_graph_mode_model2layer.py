import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, Dropout
from tensorlayer.models import Model
# import tensorflow.contrib.eager as tfe

## enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# ## enable eager mode
# tf.enable_eager_execution()

## prepare MNIST data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

# print(X_train.shape)
# print(y_train.shape)
# exit()


## define the network
# the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
def get_model(inputs_shape):
    ni = Input(inputs_shape)
    nn = Dropout(keep=0.8)(ni)
    nn = Dense(n_units=800, act=tf.nn.relu)(nn)
    nn = Dropout(keep=0.8)(nn)
    nn = Dense(n_units=800, act=tf.nn.relu)(nn)

    # FIXME: currently assume the inputs and outputs are both Layer. They can be lists.
    M_hidden = Model(inputs=ni, outputs=nn, name="mlp_hidden")

    nn = Dropout(keep=0.8)(M_hidden.as_layer())
    nn = Dense(n_units=10, act=tf.nn.relu)(nn)
    M = Model(inputs=ni, outputs=nn, name="mlp")
    return M


MLP = get_model([None, 784])
# MLP.print_layers()
# MLP.print_weights()

x = tf.placeholder(tf.float32, shape=[None, 784], name='inputs')
y_ = tf.placeholder(tf.int64, shape=[None], name='targets')

## get output tensors for training and testing
# 1) use ``is_train''
y1 = MLP(x, is_train=True)
y2 = MLP(x, is_train=False)
# 2) alternatively, you can use the switching method
# MLP.train()
# y1 = MLP(x)
# MLP.eval()
# y2 = MLP(x)

## cost and optimizer for training
cost = tl.cost.cross_entropy(y1, y_, name='train_loss')
train_weights = MLP.weights  #tl.layers.get_variables_with_name('MLP', True, False)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_weights)

## cost and accuracy for evaluation
cost_test = tl.cost.cross_entropy(y2, y_, name='test_loss')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## initialize all variables in the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## start training
n_epoch = 500
batch_size = 500
print_freq = 5

for epoch in range(n_epoch):
    start_time = time.time()
    for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_batch, y_: y_batch})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_batch, y_: y_batch})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_batch, y_: y_batch})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
        print("   val acc: %f" % (val_acc / n_batch))

print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_batch, y_batch in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
    err, ac = sess.run([cost_test, acc], feed_dict={x: X_batch, y_: y_batch})
    test_loss += err
    test_acc += ac
    n_batch += 1
print("   test loss: %f" % (test_loss / n_batch))
print("   test acc: %f" % (test_acc / n_batch))
