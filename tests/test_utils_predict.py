import tensorflow as tf
import tensorlayer as tl
import numpy as np

sess = tf.InteractiveSession()

# case 1: No. of examples is not divisible by batch_size but the input placeholder's first dim is None.
x = tf.placeholder(tf.float32, [None, 5, 5, 3])
X = np.ones([127, 5, 5, 3])
net = tl.layers.InputLayer(x)
y = net.outputs
y_op = tf.nn.softmax(y)
result = tl.utils.predict(sess, net, X, x, y_op, batch_size=8)

# case 2: No. of examples > batch_size & not divisible by batch_size
# ValueError: Cannot feed value of shape (7, 5, 5, 3) for Tensor 'Placeholder_1:0', which has shape '(8, 5, 5, 3)'
x = tf.placeholder(tf.float32, [8, 5, 5, 3])
X = np.ones([127, 5, 5, 3])
net = tl.layers.InputLayer(x)
y = net.outputs
y_op = tf.nn.softmax(y)
result = tl.utils.predict(sess, net, X, x, y_op, batch_size=8)

# case 3: No. of examples < batch_size (actually same with the last mini-batch in case 2)
# ValueError: Cannot feed value of shape (7, 5, 5, 3) for Tensor 'Placeholder_2:0', which has shape '(8, 5, 5, 3)'
x = tf.placeholder(tf.float32, [8, 5, 5, 3])
X = np.ones([7, 5, 5, 3])
net = tl.layers.InputLayer(x)
y = net.outputs
y_op = tf.nn.softmax(y)
result = tl.utils.predict(sess, net, X, x, y_op, batch_size=8)