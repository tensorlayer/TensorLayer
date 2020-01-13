"""Sample task script."""

import tensorflow as tf

import tensorlayer as tl

# tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# connect to database
db = tl.db.TensorHub(ip='localhost', port=27017, dbname='temp', project_name='tutorial')

# load dataset from database
X_train, y_train, X_val, y_val, X_test, y_test = db.find_top_dataset('mnist')


# define the network
def mlp():
    ni = tl.layers.Input([None, 784], name='input')
    net = tl.layers.Dropout(keep=0.8, name='drop1')(ni)
    net = tl.layers.Dense(n_units=n_units1, act=tf.nn.relu, name='relu1')(net)
    net = tl.layers.Dropout(keep=0.5, name='drop2')(net)
    net = tl.layers.Dense(n_units=n_units2, act=tf.nn.relu, name='relu2')(net)
    net = tl.layers.Dropout(keep=0.5, name='drop3')(net)
    net = tl.layers.Dense(n_units=10, act=None, name='output')(net)
    M = tl.models.Model(inputs=ni, outputs=net)
    return M


network = mlp()

# cost and accuracy
cost = tl.cost.cross_entropy


def acc(y, y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.convert_to_tensor(y_, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# define the optimizer
train_op = tf.optimizers.Adam(learning_rate=0.0001)

# train the network
# tl.utils.fit(
#     network, train_op, cost, X_train, y_train, acc=acc, batch_size=500, n_epoch=20, print_freq=5,
#     X_val=X_val, y_val=y_val, eval_train=False
# )

tl.utils.fit(
    network,
    train_op=tf.optimizers.Adam(learning_rate=0.0001),
    cost=tl.cost.cross_entropy,
    X_train=X_train,
    y_train=y_train,
    acc=acc,
    batch_size=256,
    n_epoch=20,
    X_val=X_val,
    y_val=y_val,
    eval_train=False,
)

# evaluation and save result that match the result_key
test_accuracy = tl.utils.test(network, acc, X_test, y_test, batch_size=None, cost=cost)
test_accuracy = float(test_accuracy)

# save model into database
db.save_model(network, model_name='mlp', name=str(n_units1) + '-' + str(n_units2), test_accuracy=test_accuracy)
# in other script, you can load the model as follow
# net = db.find_model(sess=sess, model_name=str(n_units1)+'-'+str(n_units2)
