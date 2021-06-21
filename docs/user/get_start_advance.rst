.. _getstartadvance:

==================
Advanced features
==================


Customizing layer
==================

Layers with weights
----------------------

The fully-connected layer is `a = f(x*W+b)`, the most simple implementation is as follow.

.. code-block:: python

  from tensorlayer.layers import Module

  class Dense(Module):
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
            in_channels = None
    ):
        super(Dense, self).__init__(name, act=act) # auto naming, dense_1, dense_2 ...
        self.n_units = n_units
        self.in_channels = in_channels
        self.build()
        self._built = True
        
    def build(self): # initialize the model weights here
        shape = [self.in_channels, self.n_units]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
        self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)

    def forward(self, inputs): # call function
        z = tf.matmul(inputs, self.W) + self.b
        if self.act: # is not None
            z = self.act(z)
        return z

The full implementation is as follow, which supports both automatic inference input and dynamic models and allows users to control whether to use the bias, how to initialize the weight values.

.. code-block:: python


  class Dense(Module):
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

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([100, 50], name='input')
    >>> dense = tl.layers.Dense(n_units=800, act=tl.ReLU, in_channels=50, name='dense_1')
    >>> print(dense)
    Dense(n_units=800, relu, in_channels='50', name='dense_1')
    >>> tensor = tl.layers.Dense(n_units=800, act=tl.ReLU, name='dense_2')(net)
    >>> print(tensor)
    tf.Tensor([...], shape=(100, 800), dtype=float32)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`Flatten`.

    """

    def __init__(
        self,
        n_units,
        act=None,
        W_init=tl.initializers.truncated_normal(stddev=0.05),
        b_init=tl.initializers.constant(value=0.0),
        in_channels=None,
        name=None,  # 'dense',
    ):

        super(Dense, self).__init__(name, act=act)

        self.n_units = n_units
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build(self.in_channels)
            self._built = True

        logging.info(
            "Dense  %s: %d %s" %
            (self.name, self.n_units, self.act.__class__.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None and len(inputs_shape) != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")
        if self.in_channels:
            shape = [self.in_channels, self.n_units]
        else:
            self.in_channels = inputs_shape[1]
            shape = [inputs_shape[1], self.n_units]

        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)
            self.b_init_flag = True
            self.bias_add = tl.ops.BiasAdd()

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

        self.matmul = tl.ops.MatMul()

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        z = self.matmul(inputs, self.W)
        if self.b_init_flag:
            z = self.bias_add(z, self.b)
        if self.act_init_flag:
            z = self.act(z)
        return z


Layers with train/test modes
------------------------------

We use Dropout as an example here:

.. code-block:: python
  
  class Dropout(Module):
    """
    The :class:`Dropout` class is a noise layer which randomly set some
    activations to zero according to a keeping probability.

    Parameters
    ----------
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    seed : int or None
        The seed for random dropout.
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> net = tl.layers.Input([10, 200])
    >>> net = tl.layers.Dropout(keep=0.2)(net)

    """

    def __init__(self, keep, seed=0, name=None):  #"dropout"):
        super(Dropout, self).__init__(name)
        self.keep = keep
        self.seed = seed

        self.build()
        self._built = True

        logging.info("Dropout %s: keep: %f " % (self.name, self.keep))

    def __repr__(self):
        s = ('{classname}(keep={keep}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.dropout = tl.ops.Dropout(keep=self.keep, seed=self.seed)

    def forward(self, inputs):
        if self.is_train:
            outputs = self.dropout(inputs)
        else:
            outputs = inputs
        return outputs

Pre-trained CNN
================

Get entire CNN
---------------

.. code-block:: python


  import tensorlayer as tl
  import numpy as np
  from tensorlayer.models.imagenet_classes import class_names
  from examples.model_zoo import vgg16

  vgg = vgg16(pretrained=True)
  img = tl.vis.read_image('data/tiger.jpeg')
  img = tl.prepro.imresize(img, (224, 224)).astype(tl.float32) / 255
  output = vgg(img, is_train=False)

