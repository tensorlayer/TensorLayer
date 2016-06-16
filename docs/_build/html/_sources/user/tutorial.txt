.. _tutorial:

========
Tutorial
========

This tutorial will walk you through building a handwritten digits classifier
using the MNIST dataset, arguably the "Hello World" of neural networks.
More tutorials and examples can be found in the `Lasagne Recipes`_ repository.


Before we start
===============

The tutorial assumes that you are somewhat familiar with neural networks and
Theano (the library which Lasagne is built on top of). You can try to learn
both at once from the `Deeplearning Tutorial`_.

For a more slow-paced introduction to artificial neural networks, we recommend
`Convolutional Neural Networks for Visual Recognition`_ by Andrej Karpathy et
al., `Neural Networks and Deep Learning`_ by Michael Nielsen or a standard text
book such as "Machine Learning" by Tom Mitchell.

To learn more about Theano, have a look at the `Theano tutorial`_. You will not
need all of it, but a basic understanding of how Theano works is required to be
able to use Lasagne. If you're new to Theano, going through that tutorial up to
(and including) "More Examples" should get you covered! `Graph Structures`_ is
a good extra read if you're curious about its inner workings.


Run the MNIST example
=====================

In this first part of the tutorial, we will just run the MNIST example that's
included in the source distribution of Lasagne.

We assume that you have already run through the :ref:`installation`. If you
haven't done so already, get a copy of the source tree of Lasagne, and navigate
to the folder in a terminal window. Enter the ``examples`` folder and run the
``mnist.py`` example script:

.. code-block:: bash

  cd examples
  python mnist.py

If everything is set up correctly, you will get an output like the following:

.. code-block:: text

  Using gpu device 0: GeForce GT 640
  Loading data...
  Downloading train-images-idx3-ubyte.gz
  Downloading train-labels-idx1-ubyte.gz
  Downloading t10k-images-idx3-ubyte.gz
  Downloading t10k-labels-idx1-ubyte.gz
  Building model and compiling functions...
  Starting training...

  Epoch 1 of 500 took 1.858s
    training loss:                1.233348
    validation loss:              0.405868
    validation accuracy:          88.78 %
  Epoch 2 of 500 took 1.845s
    training loss:                0.571644
    validation loss:              0.310221
    validation accuracy:          91.24 %
  Epoch 3 of 500 took 1.845s
    training loss:                0.471582
    validation loss:              0.265931
    validation accuracy:          92.35 %
  Epoch 4 of 500 took 1.847s
    training loss:                0.412204
    validation loss:              0.238558
    validation accuracy:          93.05 %
  ...

The example script allows you to try three different models, selected via the
first command line argument. Run the script with ``python mnist.py --help`` for
more information and feel free to play around with it some more before we have
a look at the implementation.


Understand the MNIST example
============================

Let's now investigate what's needed to make that happen! To follow along, open
up the source code in your favorite editor (or online: `mnist.py`_).


Preface
-------

The first thing you might notice is that besides Lasagne, we also import numpy
and Theano:

.. code-block:: python

  import numpy as np
  import theano
  import theano.tensor as T
  
  import lasagne

While Lasagne is built on top of Theano, it is meant as a supplement helping
with some tasks, not as a replacement. You will always mix Lasagne with some
vanilla Theano code.


Loading data
------------

The first piece of code defines a function ``load_dataset()``. Its purpose is
to download the MNIST dataset (if it hasn't been downloaded yet) and return it
in the form of regular numpy arrays. There is no Lasagne involved at all, so
for the purpose of this tutorial, we can regard it as:

.. code-block:: python

  def load_dataset():
      ...
      return X_train, y_train, X_val, y_val, X_test, y_test

``X_train.shape`` is ``(50000, 1, 28, 28)``, to be interpreted as: 50,000
images of 1 channel, 28 rows and 28 columns each. Note that the number of
channels is 1 because we have monochrome input. Color images would have 3
channels, spectrograms also would have a single channel.
``y_train.shape`` is simply ``(50000,)``, that is, it is a vector the same
length of ``X_train`` giving an integer class label for each image -- namely,
the digit between 0 and 9 depicted in the image (according to the human
annotator who drew that digit).


Building the model
------------------

This is where Lasagne steps in. It allows you to define an arbitrarily
structured neural network by creating and stacking or merging layers.
Since every layer knows its immediate incoming layers, the output layer (or
output layers) of a network double as a handle to the network as a whole, so
usually this is the only thing we will pass on to the rest of the code.

As mentioned above, ``mnist.py`` supports three types of models, and we
implement that via three easily exchangeable functions of the same interface.
First, we'll define a function that creates a Multi-Layer Perceptron (MLP) of
a fixed architecture, explaining all the steps in detail. We'll then present
a function generating an MLP of a custom architecture. Finally, we'll
show how to create a Convolutional Neural Network (CNN).


Multi-Layer Perceptron (MLP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first function, ``build_mlp()``, creates an MLP of two hidden layers of
800 units each, followed by a softmax output layer of 10 units. It applies 20%
dropout to the input data and 50% dropout to the hidden layers. It is similar,
but not fully equivalent to the smallest MLP in [Hinton2012]_ (that paper uses
different nonlinearities, weight initialization and training).

The foundation of each neural network in Lasagne is an
:class:`InputLayer <lasagne.layers.InputLayer>` instance (or multiple of those)
representing the input data that will subsequently be fed to the network. Note
that the ``InputLayer`` is not tied to any specific data yet, but only holds
the shape of the data that will be passed to the network. In addition, it
creates or can be linked to a `Theano variable
<http://deeplearning.net/software/theano/glossary.html#term-variable>`_ that
will represent the network input in the `Theano graph
<http://deeplearning.net/software/theano/glossary.html#term-expression-graph>`_
we'll build from the network later.
Thus, our function starts like this:

.. code-block:: python

    def build_mlp(input_var=None):
        l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                         input_var=input_var)

The four numbers in the shape tuple represent, in order:
``(batchsize, channels, rows, columns)``.
Here we've set the batchsize to ``None``, which means the network will accept
input data of arbitrary batchsize after compilation. If you know the batchsize
beforehand and do not need this flexibility, you should give the batchsize
here -- especially for convolutional layers, this can allow Theano to apply
some optimizations.
``input_var`` denotes the Theano variable we want to link the network's input
layer to. If it is omitted (or set to ``None``), the layer will just create a
suitable variable itself, but it can be handy to link an existing variable to
the network at construction time -- especially if you're creating networks of
multiple input layers. Here, we link it to a variable given as an argument to
the ``build_mlp()`` function.

Before adding the first hidden layer, we'll apply 20% dropout to the input
data. This is realized via a :class:`DropoutLayer
<lasagne.layers.DropoutLayer>` instance:

.. code-block:: python

    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

Note that the first constructor argument is the incoming layer, such that
``l_in_drop`` is now stacked on top of ``l_in``. All layers work this way,
except for layers that merge multiple inputs: those accept a list of incoming
layers as their first constructor argument instead.

We'll proceed with the first fully-connected hidden layer of 800 units. Note
that when stacking a :class:`DenseLayer <lasagne.layers.DenseLayer>` on
higher-order input tensors, they will be flattened implicitly so we don't need
to care about that. In this case, the input will be flattened from 1x28x28
images to 784-dimensional vectors.

.. code-block:: python

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

Again, the first constructor argument means that we're stacking ``l_hid1`` on
top of ``l_in_drop``.
``num_units`` simply gives the number of units for this fully-connected layer.
``nonlinearity`` takes a nonlinearity function, several of which are defined
in :mod:`lasagne.nonlinearities`. Here we've chosen the linear rectifier, so
we'll obtain ReLUs.
Finally, :class:`lasagne.init.GlorotUniform()` gives the initializer for the
weight matrix ``W``. This particular initializer samples weights from a uniform
distribution of a carefully chosen range. Other initializers are available in
:mod:`lasagne.init`, and alternatively, ``W`` could also have been initialized
from a Theano shared variable or numpy array of the correct shape (784x800 in
this case, as the input to this layer has 1*28*28=784 dimensions).
Note that ``lasagne.init.GlorotUniform()`` is the default, so we'll omit it
from here -- we just wanted to highlight that there is a choice.

We'll now add dropout of 50%, another 800-unit dense layer and 50% dropout
again:

.. code-block:: python

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

Finally, we'll add the fully-connected output layer. The main difference is
that it uses the softmax nonlinearity, as we're planning to solve a 10-class
classification problem with this network.

.. code-block:: python

    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

As mentioned above, each layer is linked to its incoming layer(s), so we only
need the output layer(s) to access a network in Lasagne:

.. code-block:: python

    return l_out


Custom MLP
^^^^^^^^^^

The second function has a slightly more extensive signature:

.. code-block:: python

    def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                         drop_hidden=.5):

By default, it creates the same network as ``build_mlp()`` described above, but
it can be customized with respect to the number and size of hidden layers, as
well as the amount of input and hidden dropout. This demonstrates how creating
a network in Python code can be a lot more flexible than a configuration file.
See for yourself:

.. code-block:: python

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network

With two ``if`` clauses and a ``for`` loop, this network definition allows
varying the architecture in a way that would be impossible for a ``.yaml`` file
in `Pylearn2`_ or a ``.cfg`` file in `cuda-convnet`_.

Note that to make the code easier, all the layers are just called ``network``
here -- there is no need to give them different names if all we return is the
last one we created anyway; we just used different names before for clarity.


Convolutional Neural Network (CNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, the ``build_cnn()`` function creates a CNN of two convolution and
pooling stages, a fully-connected hidden layer and a fully-connected output
layer.
The function begins like the others:

.. code-block:: python

    def build_cnn(input_var=None):
        network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                            input_var=input_var)

We don't apply dropout to the inputs, as this tends to work less well for
convolutional layers. Instead of a :class:`DenseLayer
<lasagne.layers.DenseLayer>`, we now add a :class:`Conv2DLayer
<lasagne.layers.Conv2DLayer>` with 32 filters of size 5x5 on top:

.. code-block:: python

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

The nonlinearity and weight initializer can be given just as for the
``DenseLayer`` (and again, ``GlorotUniform()`` is the default, we'll omit it
from now). Strided and padded convolutions are supported as well; see the
:class:`Conv2DLayer <lasagne.layers.Conv2DLayer>` docstring.

.. note::
    For experts: ``Conv2DLayer`` will create a convolutional layer using
    ``T.nnet.conv2d``, Theano's default convolution. On compilation for GPU,
    Theano replaces this with a `cuDNN`_-based implementation if available,
    otherwise falls back to a gemm-based implementation. For details on this,
    please see the `Theano convolution documentation`_.

    Lasagne also provides convolutional layers directly enforcing a specific
    implementation: :class:`lasagne.layers.dnn.Conv2DDNNLayer` to enforce
    cuDNN, :class:`lasagne.layers.corrmm.Conv2DMMLayer` to enforce the
    gemm-based one, :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` for
    Krizhevsky's `cuda-convnet`_.

We then apply max-pooling of factor 2 in both dimensions, using a
:class:`MaxPool2DLayer <lasagne.layers.MaxPool2DLayer>` instance:

.. code-block:: python

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

We add another convolution and pooling stage like the ones before:

.. code-block:: python

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

Then a fully-connected layer of 256 units with 50% dropout on its inputs
(using the :class:`lasagne.layers.dropout` shortcut directly inline):

.. code-block:: python

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

And finally a 10-unit softmax output layer, again with 50% dropout:

.. code-block:: python

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


Training the model
------------------

The remaining part of the ``mnist.py`` script copes with setting up and running
a training loop over the MNIST dataset.


Dataset iteration
^^^^^^^^^^^^^^^^^

It first defines a short helper function for synchronously iterating over two
numpy arrays of input data and targets, respectively, in mini-batches of a
given number of items. For the purpose of this tutorial, we can shorten it to:

.. code-block:: python

    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        if shuffle:
            ...
        for ...:
            yield inputs[...], targets[...]

All that's relevant is that it is a generator function that serves one batch of
inputs and targets at a time until the given dataset (in ``inputs`` and
``targets``) is exhausted, either in sequence or in random order. Below we will
plug this function into our training loop, validation loop and test loop.


Preparation
^^^^^^^^^^^

Let's now focus on the ``main()`` function. A bit simplified, it begins like
this:

.. code-block:: python

    # Load the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    # Create neural network model
    network = build_mlp(input_var)

The first line loads the inputs and targets of the MNIST dataset as numpy
arrays, split into training, validation and test data.
The next two statements define symbolic Theano variables that will represent
a mini-batch of inputs and targets in all the Theano expressions we will
generate for network training and inference. They are not tied to any data yet,
but their dimensionality and data type is fixed already and matches the actual
inputs and targets we will process later.
Finally, we call one of the three functions for building the Lasagne network,
depending on the first command line argument -- we've just removed command line
handling here for clarity. Note that we hand the symbolic input variable to
``build_mlp()`` so it will be linked to the network's input layer.


Loss and update expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Continuing, we create a loss expression to be minimized in training:

.. code-block:: python

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

The first step generates a Theano expression for the network output given the
input variable linked to the network's input layer(s). The second step defines
a Theano expression for the categorical cross-entropy loss between said network
output and the targets. Finally, as we need a scalar loss, we simply take the
mean over the mini-batch. Depending on the problem you are solving, you will
need different loss functions, see :mod:`lasagne.objectives` for more.

Having the model and the loss function defined, we create update expressions
for training the network. An update expression describes how to change the
trainable parameters of the network at each presented mini-batch. We will use
Stochastic Gradient Descent (SGD) with Nesterov momentum here, but the
:mod:`lasagne.updates` module offers several others you can plug in instead:

.. code-block:: python

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

The first step collects all Theano ``SharedVariable`` instances making up the
trainable parameters of the layer, and the second step generates an update
expression for each parameter.

For monitoring progress during training, after each epoch, we evaluate the
network on the validation set. We need a slightly different loss expression
for that:

.. code-block:: python

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

The crucial difference is that we pass ``deterministic=True`` to the
:func:`get_output <lasagne.layers.get_output>` call. This causes all
nondeterministic layers to switch to a deterministic implementation, so in our
case, it disables the dropout layers.
As an additional monitoring quantity, we create an expression for the
classification accuracy:

.. code-block:: python

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

It also builds on the deterministic ``test_prediction`` expression.


Compilation
^^^^^^^^^^^

Equipped with all the necessary Theano expressions, we're now ready to compile
a function performing a training step:

.. code-block:: python

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

This tells Theano to generate and compile a function taking two inputs -- a
mini-batch of images and a vector of corresponding targets -- and returning a
single output: the training loss. Additionally, each time it is invoked, it
applies all parameter updates in the ``updates`` dictionary, thus performing a
gradient descent step with Nesterov momentum.

For validation, we compile a second function:

.. code-block:: python

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

This one also takes a mini-batch of images and targets, then returns the
(deterministic) loss and classification accuracy, not performing any updates.


Training loop
^^^^^^^^^^^^^

We're finally ready to write the training loop. In essence, we just need to do
the following:

.. code-block:: python

    for epoch in range(num_epochs):
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_fn(inputs, targets)

This uses our dataset iteration helper function to iterate over the training
data in random order, in mini-batches of 500 items each, for ``num_epochs``
epochs, and calls the training function we compiled to perform an update step
of the network parameters.

But to be able to monitor the training progress, we capture the training loss,
compute the validation loss and print some information to the console every
time an epoch finishes:

.. code-block:: python

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

At the very end, we re-use the ``val_fn()`` function to compute the loss and
accuracy on the test set, finishing the script.



Where to go from here
=====================

This finishes our introductory tutorial. For more information on what you can
do with Lasagne's layers, just continue reading through :doc:`layers` and
:doc:`custom_layers`.
More tutorials, examples and code snippets can be found in the `Lasagne
Recipes`_ repository.
Finally, the reference lists and explains all layers (:mod:`lasagne.layers`),
weight initializers (:mod:`lasagne.init`), nonlinearities
(:mod:`lasagne.nonlinearities`), loss expressions (:mod:`lasagne.objectives`),
training methods (:mod:`lasagne.updates`) and regularizers
(:mod:`lasagne.regularization`) included in the library, and should also make
it simple to create your own.



.. _Lasagne Recipes: https://github.com/Lasagne/Recipes
.. _Deeplearning Tutorial: http://deeplearning.net/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _Theano tutorial: http://deeplearning.net/software/theano/tutorial/
.. _Graph Structures: http://deeplearning.net/software/theano/extending/graphstructures.html
.. _mnist.py: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
.. [Hinton2012] Improving neural networks by preventing co-adaptation
   of feature detectors. http://arxiv.org/abs/1207.0580
.. _Pylearn2: http://deeplearning.net/software/pylearn2/
.. _cuda-convnet: https://code.google.com/p/cuda-convnet/
.. _cuDNN: https://developer.nvidia.com/cudnn
.. _Theano convolution documentation: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html
