


import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
W = tf.Variable([[1.0, 2.0],[3.0, 4.0]])
y = x * W

W.initializer.run()

out = sess.run(y, feed_dict={x: [[1.0, 2.0]]})
print(out)

out = sess.run(y, feed_dict={x: [[1.0, 2.0], [0.0, 3.0]]})
print(out)
