.. _getstartmodel:

===============
Define a model
===============

TensorLayer provides two ways to define a model.
Static model
Dynamic model allows you to control the forward process

Static model
===============

.. code-block:: python

  import tensorflow as tf
  from tensorlayer.layers import Input, Dropout, Dense
  from tensorlayer.models import Model

  def get_model(inputs_shape):
      ni = Input(inputs_shape)
      nn = Dropout(keep=0.8)(ni)
      nn = Dense(n_units=800, act=tf.nn.relu)(nn)
      nn = Dropout(keep=0.8)(nn)
      nn = Dense(n_units=800, act=tf.nn.relu)(nn)
      nn = Dropout(keep=0.8)(nn)
      nn = Dense(n_units=10, act=tf.nn.relu)(nn)
      M = Model(inputs=ni, outputs=nn, name="mlp")
      return M

  MLP = get_model([None, 784])
  MLP.eval()
  outputs = MLP(data)

Dynamic model
=======================


In this case, you need to manually input the output shape of the previous layer to the new layer.
For example,

.. code-block:: python

  class CustomModel(Model):

      def __init__(self):
          super(CustomModel, self).__init__()

          self.dropout1 = Dropout(keep=0.8)
          self.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784)
          self.dropout2 = Dropout(keep=0.8)#(self.dense1)
          self.dense2 = Dense(n_units=800, act=tf.nn.relu, in_channels=800)
          self.dropout3 = Dropout(keep=0.8)#(self.dense2)
          self.dense3 = Dense(n_units=10, act=tf.nn.relu, in_channels=800)

      def forward(self, x, foo=False):
          z = self.dropout1(x)
          z = self.dense1(z)
          z = self.dropout2(z)
          z = self.dense2(z)
          z = self.dropout3(z)
          out = self.dense3(z)
          if foo:
              out = tf.nn.relu(out)
          return out

  MLP = CustomModel()
  MLP.eval()
  outputs = MLP(data, foo=True) # controls the forward here
  outputs = MLP(data, foo=False)

Reuse weights
=======================

Siamese network

.. code-block:: python

  xxx

Print model information
=======================

.. code-block:: python

  xxx

Get specific weights
=======================

We can get the specific weights by indexing or naming.

.. code-block:: python

  # indexing
  all_weights = MLP.weights
  some_weights = MLP.weights[1:3]

  # naming
  some_weights = MLP.get_weights('bias')


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

Customizing layer
==================

The fully-connected layer is

z = f(x*W+b)

.. code-block:: python

  class Dense(Layer):
      def __init__(self, n_units, act=None, in_channels=None, name=None):
          super(Dense, self).__init__(name)

          self.n_units = n_units
          self.act = act
          self.in_channels = in_channels

          # for dynamic model, it needs the input shape to get the shape of W
          if self.in_channels is not None:
              self.build(self.in_channels)
              self._built = True

      def build(self, inputs_shape):
          if self.in_channels is None and len(inputs_shape) != 2:
              raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")
          if self.in_channels:
              shape = [self.in_channels, self.n_units]
          else:
              self.in_channels = inputs_shape[1]
              shape = [inputs_shape[1], self.n_units]
          self.W = self._get_weights("weights", shape=tuple(shape))
          if self.b_init:
              self.b = self._get_weights("biases", shape=(self.n_units, ))

      @tf.function
      def forward(self, inputs):
          z = tf.matmul(inputs, self.W)
          if self.b_init:
              z = tf.add(z, self.b)
          if self.act:
              z = self.act(z)
          return z
