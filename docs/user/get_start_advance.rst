.. _getstartadvance:

==================
Advanced features
==================


Customizing layer
==================

Layers with weights
----------------------

The fully-connected layer is `a = f(x*W+b)`, the most simple implementation is as follow, which can only support static model.

.. code-block:: python

  class Dense(Layer):
    """The :class:`Dense` class is a fully connected layer.
    
    Parameters
    ----------
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    name : None or str
        A unique layer name. If None, a unique name will be automatically generated.
    """
    
    def __init__(
            self,
            n_units,   # the number of units/channels of this layer
            act=None,  # None: no activation, tf.nn.relu or 'relu': ReLU ...
            name=None, # the name of this layer (optional)
    ):
        super(Dense, self).__init__(name, act=act) # auto naming, dense_1, dense_2 ...
        self.n_units = n_units
        
    def build(self, inputs_shape): # initialize the model weights here
        shape = [inputs_shape[1], self.n_units]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
        self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)

    def forward(self, inputs): # call function
        z = tf.matmul(inputs, self.W) + self.b
        if self.act: # is not None
            z = self.act(z)
        return z

The full implementation is as follow, which supports both static and dynamic models and allows users to control whether to use the bias, how to initialize the weight values.

.. code-block:: python

  class Dense(Layer):
    """The :class:`Dense` class is a fully connected layer.
    
    Parameters
    ----------
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : None or str
        A unique layer name. If None, a unique name will be automatically generated.
    """
    
    def __init__(
            self,
            n_units,
            act=None,
            W_init=tl.initializers.truncated_normal(stddev=0.1),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,  # the number of units/channels of the previous layer
            name=None,
    ):
        # we feed activation function to the base layer, `None` denotes identity function
        # string (e.g., relu, sigmoid) will be converted into function.
        super(Dense, self).__init__(name, act=act) 

        self.n_units = n_units
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        # in dynamic model, the number of input channel is given, we initialize the weights here
        if self.in_channels is not None: 
            self.build(self.in_channels)
            self._built = True

        logging.info(
            "Dense  %s: %d %s" %
            (self.name, self.n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self): # optional, for printing information
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape): # initialize the model weights here
        if self.in_channels: # if the number of input channel is given, use it
            shape = [self.in_channels, self.n_units]
        else:                # otherwise, get it from static model
            self.in_channels = inputs_shape[1]
            shape = [inputs_shape[1], self.n_units]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
        if self.b_init:      # if b_init is None, no bias is applied
            self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)

    def forward(self, inputs):
        z = tf.matmul(inputs, self.W)
        if self.b_init:
            z = tf.add(z, self.b)
        if self.act:
            z = self.act(z)
        return z


Layers with train/test modes
------------------------------

We use Dropout as an example here:

.. code-block:: python
  
  class Dropout(Layer):
      """
      The :class:`Dropout` class is a noise layer which randomly set some
      activations to zero according to a keeping probability.
      Parameters
      ----------
      keep : float
          The keeping probability.
          The lower the probability it is, the more activations are set to zero.
      name : None or str
          A unique layer name.
      """

      def __init__(self, keep, name=None):
          super(Dropout, self).__init__(name)
          self.keep = keep

          self.build()
          self._built = True

          logging.info("Dropout %s: keep: %f " % (self.name, self.keep))

      def build(self, inputs_shape=None):
          pass   # no weights in dropout layer

      def forward(self, inputs):
          if self.is_train:  # this attribute is changed by Model.train() and Model.eval() described above
              outputs = tf.nn.dropout(inputs, rate=1 - (self.keep), name=self.name)
          else:
              outputs = inputs
          return outputs

Pre-trained CNN
================

Get entire CNN
---------------

.. code-block:: python

  import tensorflow as tf
  import tensorlayer as tl
  import numpy as np
  from tensorlayer.models.imagenet_classes import class_names

  vgg = tl.models.vgg16(pretrained=True)
  img = tl.vis.read_image('data/tiger.jpeg')
  img = tl.prepro.imresize(img, (224, 224)).astype(np.float32) / 255
  output = vgg(img, is_train=False)

Get a part of CNN
------------------

.. code-block:: python

  # get VGG without the last layer
  cnn = tl.models.vgg16(end_with='fc2_relu', mode='static').as_layer()
  # add one more layer and build a new model
  ni = tl.layers.Input([None, 224, 224, 3], name="inputs")
  nn = cnn(ni)
  nn = tl.layers.Dense(n_units=100, name='out')(nn)
  model = tl.models.Model(inputs=ni, outputs=nn)
  # train your own classifier (only update the last layer)
  train_weights = model.get_layer('out').all_weights

Reuse CNN
------------------

.. code-block:: python

  # in dynamic model, we can directly use the same model
  # in static model
  vgg_layer = tl.models.vgg16().as_layer()
  ni_1 = tl.layers.Input([None, 224, 224, 3])
  ni_2 = tl.layers.Input([None, 224, 224, 3])
  a_1 = vgg_layer(ni_1)
  a_2 = vgg_layer(ni_2)
  M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])

