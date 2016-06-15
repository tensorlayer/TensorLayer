Creating custom layers
======================


A simple layer
--------------

To implement a custom layer in Lasagne, you will have to write a Python class
that subclasses :class:`Layer` and implement at least one method:
`get_output_for()`. This method computes the output of the layer given its
input. Note that both the output and the input are Theano expressions, so they
are symbolic.

The following is an example implementation of a layer that multiplies its input
by 2:

.. code:: python

    class DoubleLayer(lasagne.layers.Layer):
        def get_output_for(self, input, **kwargs):
            return 2 * input

This is all that's required to implement a functioning custom layer class in
Lasagne.


A layer that changes the shape
------------------------------

If the layer does not change the shape of the data (for example because it
applies an elementwise operation), then implementing only this one method is
sufficient. Lasagne will assume that the output of the layer has the same shape
as its input.

However, if the operation performed by the layer changes the shape of the data,
you also need to implement `get_output_shape_for()`. This method computes the
shape of the layer output given the shape of its input. Note that this shape
computation should result in a tuple of integers, so it is *not* symbolic.

This method exists because Lasagne needs a way to propagate shape information
when a network is defined, so it can determine what sizes the parameter tensors
should be, for example. This mechanism allows each layer to obtain the size of
its input from the previous layer, which means you don't have to specify the
input size manually. This also prevents errors stemming from inconsistencies
between the layers' expected and actual shapes.

We can implement a layer that computes the sum across the trailing axis of its
input as follows:

.. code:: python

    class SumLayer(lasagne.layers.Layer):
        def get_output_for(self, input, **kwargs):
            return input.sum(axis=-1)

        def get_output_shape_for(self, input_shape):
            return input_shape[:-1]


It is important that the shape computation is correct, as this shape
information may be used to initialize other layers in the network.


A layer with parameters
-----------------------

If the layer has parameters, these should be initialized in the constructor.
In Lasagne, parameters are represented by Theano shared variables. A method
is provided to create and register parameter variables:
:meth:`lasagne.layers.Layer.add_param()`.

To show how this can be used, here is a layer that multiplies its input
by a matrix ``W`` (much like a typical fully connected layer in a neural
network would). This matrix is a parameter of the layer. The shape of the
matrix will be ``(num_inputs, num_units)``, where ``num_inputs`` is the
number of input features and ``num_units`` has to be specified when the layer
is created.

.. code:: python

    class DotLayer(lasagne.layers.Layer):
        def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), **kwargs):
            super(DotLayer, self).__init__(incoming, **kwargs)
            num_inputs = self.input_shape[1]
            self.num_units = num_units
            self.W = self.add_param(W, (num_inputs, num_units), name='W')

        def get_output_for(self, input, **kwargs):
            return T.dot(input, self.W)

        def get_output_shape_for(self, input_shape):
            return (input_shape[0], self.num_units)

A few things are worth noting here: when overriding the constructor, we need
to call the superclass constructor on the first line. This is important to
ensure the layer functions properly.
Note that we pass ``**kwargs`` - although this is not strictly necessary, it
enables some other cool Lasagne features, such as making it possible to give
the layer a name:

>>> l_dot = DotLayer(l_in, num_units=50, name='my_dot_layer')

The call to ``self.add_param()`` creates the Theano shared variable
representing the parameter, and registers it so it can later be retrieved using
:meth:`lasagne.layers.Layer.get_params()`. It returns the created variable,
which we tuck away in ``self.W`` for easy access.

Note that we've also made it possible to specify a custom initialization
strategy for ``W`` by adding a constructor argument for it, e.g.:

>>> l_dot = DotLayer(l_in, num_units=50, W=lasagne.init.Constant(0.0))

This 'Lasagne idiom' of tucking away a created parameter variable in an
attribute for easy access and adding a constructor argument with the same name
to specify the initialization strategy is very common throughout the library.

Finally, note that we used ``self.input_shape`` to determine the shape of the
parameter matrix. This property is available in all Lasagne layers, once the
superclass constructor has been called.


A layer with multiple behaviors
-------------------------------

Some layers can have multiple behaviors. For example, a layer implementing
dropout should be able to be switched on or off. During training, we want it
to apply dropout noise to its input and scale up the remaining values, but
during evaluation we don't want it to do anything.

For this purpose, the `get_output_for()` method takes optional keyword
arguments (``kwargs``). When `get_output()` is called to compute an expression
for the output of a network, all specified keyword arguments are passed to the
`get_output_for()` methods of all layers in the network.

For layers that add noise for regularization purposes, such as dropout, the
convention in Lasagne is to use the keyword argument ``deterministic`` to
control its behavior.

Lasagne's :class:`lasagne.layers.DropoutLayer` looks roughly like this
(simplified implementation for illustration purposes):

.. code:: python

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    _srng = RandomStreams()

    class DropoutLayer(Layer):
        def __init__(self, incoming, p=0.5, **kwargs):
            super(DropoutLayer, self).__init__(incoming, **kwargs)
            self.p = p

        def get_output_for(self, input, deterministic=False, **kwargs):
            if deterministic:  # do nothing in the deterministic case
                return input
            else:  # add dropout noise otherwise
                retain_prob = 1 - self.p
                input /= retain_prob
                return input * _srng.binomial(input.shape, p=retain_prob,
                                              dtype=theano.config.floatX)
