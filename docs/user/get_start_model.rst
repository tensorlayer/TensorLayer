.. _getstartmodel:

=============
Build a model
=============

Understand layer
================

All TensorLayer layers have a number of properties in common:

 - ``layer.outputs`` : a Tensor, the outputs of current layer.
 - ``layer.all_weights`` : a list of Tensor, all network variables in order.
 - ``layer.all_outputs`` : a list of Tensor, all network outputs in order.
 - ``layer.all_drop`` : a dictionary of {placeholder : float}, all keeping probabilities of noise layers.

All TensorLayer layers have a number of methods in common:

 - ``layer.print_weights()`` : print network variable information in order (after ``sess.run(tf.global_variables_initializer())``). alternatively, print all variables by ``tl.layers.print_all_variables()``.
 - ``layer.print_outputs()`` : print network layer information in order.
 - ``layer.count_all_weights()`` : print the number of parameters in the network.

A network starts with the input layer and is followed by layers stacked in order.
A network is essentially a ``Layer`` class.
The key properties of a network are ``network.all_weights``, ``network.all_outputs`` and ``network.all_drop``.
The ``all_weights`` is a list which store pointers to all network parameters in order. For example,
the following script define a 3 layer network, then:

``all_weights`` = [W1, b1, W2, b2, W_out, b_out]

To get specified variable information, you can use ``network.all_weights[2:3]`` or ``get_variables_with_name()``.
``all_outputs`` is a list which stores the pointers to the outputs of all layers, see the example as follow:

``all_outputs`` = [drop(?,784), relu(?,800), drop(?,800), relu(?,800), drop(?,800)], identity(?,10)]

where ``?`` reflects a given batch size. You can print the layer and parameters information by
using ``network.print_outputs()`` and ``network.print_weights()``.
To count the number of parameters in a network, run ``network.count_params()``.

.. code-block:: python

  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
  y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

  network = tl.layers.Input(name='input')(x)
  network = tl.layers.Dropout(keep=0.8, name='drop1')(network)
  network = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu1')(network)
  network = tl.layers.Dropout(keep=0.5, name='drop2')(network)
  network = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu2')(network)
  network = tl.layers.Dropout(keep=0.5, name='drop3')(network)
  network = tl.layers.Dense(n_units=10, act=None, name='output')(network)


  y = network.outputs
  y_op = tf.argmax(tf.nn.softmax(y), 1)

  cost = tl.cost.cross_entropy(y, y_)

  train_params = network.all_weights

  train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                              epsilon=1e-08, use_locking=False).minimize(cost, var_list = train_params)

  sess.run(tf.global_variables_initializer())

  network.print_weights()
  network.print_outputs()

In addition, ``network.all_drop`` is a dictionary which stores the keeping probabilities of all
noise layers. In the above network, they represent the keeping probabilities of dropout layers.

In case for training, you can enable all dropout layers as follow:

.. code-block:: python

  feed_dict = {x: X_train_a, y_: y_train_a}
  feed_dict.update( network.all_drop )
  loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
  feed_dict.update( network.all_drop )

In case for evaluating and testing, you can disable all dropout layers as follow.

.. code-block:: python

  feed_dict = {x: X_val, y_: y_val}
  feed_dict.update(dp_dict)
  print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
  print("   val acc: %f" % np.mean(y_val ==
                          sess.run(y_op, feed_dict=feed_dict)))

For more details, please read the MNIST examples in the example folder.


Customizing layers
===================

A simple layer
^^^^^^^^^^^^^^

To implement a custom layer in TensorLayer, you will have to write a Python class
that subclasses Layer and implement the ``outputs`` expression.

The following is an example implementation of a layer that multiplies its input by 2:

.. code-block:: python

  class Double(Layer):
      def __init__(
          self,
          layer = None,
          name ='double_layer',
      ):
          # manage layer (fixed)
          super(Double, self).__init__(prev_layer=prev_layer, name=name)

          # the input of this layer is the output of previous layer (fixed)
          self.inputs = layer.outputs

          # operation (customized)
          self.outputs = self.inputs * 2

          # update layer (customized)


Your dense layer
^^^^^^^^^^^^^^^^

Before creating your own TensorLayer layer, let's have a look at the Dense layer.
It creates a weight matrix and a bias vector if not exists, and then implements
the output expression.
At the end, for a layer with parameters, we also append the parameters into ``all_weights``.

.. code-block:: python

  class MyDense(Layer):
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.nn.relu,
        name ='simple_dense',
    ):
        # manage layer (fixed)
        super(MyDense, self).__init__(prev_layer=prev_layer, act=act, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = layer.outputs

        # print out info (customized)
        print("  MyDense %s: %d, %s" % (self.name, n_units, act))

        # operation (customized)
        n_in = int(self.inputs._shape[-1])
        with tf.variable_scope(name) as vs:
            # create new parameters
            W = tf.get_variable(name='W', shape=(n_in, n_units))
            b = tf.get_variable(name='b', shape=(n_units))
            # tensor operation
            self.outputs = self._apply_activation(tf.matmul(self.inputs, W) + b)

        # update layer (customized)

Understand model
================

All TensorLayer models have a number of properties in common:

 - ``xxx`` : xxx

Static model
===============

.. code-block:: python

  import tensorflow as tf
  import tensorlayer as tl

Dynamic model
=======================

.. code-block:: python

  xxx

Reuse weights
=======================

Siamese network

.. code-block:: python

  xxx

Save and restore model
=======================

We provide two ways to save and restore models


Save weights only
------------------

.. code-block:: python

  xxx

Save weights and config
------------------------

.. code-block:: python

  xxx
