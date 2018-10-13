"""Sample task script."""

import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()

# connect to database
db = tl.db.TensorHub(ip='localhost', port=27017, dbname='temp', project_name='tutorial')

# load dataset from database
X_train, y_train, X_val, y_val, X_test, y_test = db.find_top_dataset('mnist')

# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')


# define the network
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
        net = tl.layers.Input(name='input')(x)
        net = tl.layers.Dropout(keep=0.8, is_fix=True, is_train=is_train, name='drop1')(net)
        net = tl.layers.Dense(n_units=n_units1, act=tf.nn.relu, name='relu1')(net)
        net = tl.layers.Dropout(keep=0.5, is_fix=True, is_train=is_train, name='drop2')(net)
        net = tl.layers.Dense(n_units=n_units2, act=tf.nn.relu, name='relu2')(net)
        net = tl.layers.Dropout(keep=0.5, is_fix=True, is_train=is_train, name='drop3')(net)
        net = tl.layers.Dense(n_units=10, act=None, name='output')(net)
    return net


# define inferences
net_train = mlp(x, is_train=True, reuse=False)
net_test = mlp(x, is_train=False, reuse=True)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the optimizer
train_weights = tl.layers.get_variables_with_name('MLP', True, False)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_weights)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# train the network
tl.utils.fit(sess, net_train, train_op, cost, X_train, y_train, x, y_, acc=acc, \
    batch_size=500, n_epoch=1, print_freq=5, X_val=X_val, y_val=y_val, eval_train=False)

# evaluation and save result that match the result_key
test_accuracy = tl.utils.test(sess, net_test, acc_test, X_test, y_test, x, y_, batch_size=None, cost=cost_test)
test_accuracy = float(test_accuracy)

# save model into database
db.save_model(net_train, model_name='mlp', name=str(n_units1) + '-' + str(n_units2), test_accuracy=test_accuracy)
# in other script, you can load the model as follow
# net = db.find_model(sess=sess, model_name=str(n_units1)+'-'+str(n_units2)
