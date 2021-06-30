.. _getstartmodel:

===============
Define a model
===============

TensorLayer provides two ways to define a model.
Sequential model allows you to build model in a fluent way while dynamic model allows you to fully control the forward process.

Sequential model
===============

.. code-block:: python

  from tensorlayer.layers import SequentialLayer
  from tensorlayer.layers import Dense
  import tensorlayer as tl

  def get_model():
      layer_list = []
      layer_list.append(Dense(n_units=800, act=tl.ReLU, in_channels=784, name='Dense1'))
      layer_list.append(Dense(n_units=800, act=tl.ReLU, in_channels=800, name='Dense2'))
      layer_list.append(Dense(n_units=10, act=tl.ReLU, in_channels=800, name='Dense3'))
      MLP = SequentialLayer(layer_list)
      return MLP



Dynamic model
=======================


In this case, you need to manually input the output shape of the previous layer to the new layer.

.. code-block:: python

  import tensorlayer as tl
  from tensorlayer.layers import Module
  from tensorlayer.layers import Dropout, Dense
  class CustomModel(Module):

      def __init__(self):
          super(CustomModel, self).__init__()

          self.dropout1 = Dropout(keep=0.8)
          self.dense1 = Dense(n_units=800, act=tl.ReLU, in_channels=784)
          self.dropout2 = Dropout(keep=0.8)
          self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
          self.dropout3 = Dropout(keep=0.8)
          self.dense3 = Dense(n_units=10, act=None, in_channels=800)

      def forward(self, x, foo=False):
          z = self.dropout1(x)
          z = self.dense1(z)
          z = self.dropout2(z)
          z = self.dense2(z)
          z = self.dropout3(z)
          out = self.dense3(z)
          if foo:
              out = tf.nn.softmax(out)
          return out

  MLP = CustomModel()
  MLP.set_eval()
  outputs = MLP(data, foo=True) # controls the forward here
  outputs = MLP(data, foo=False)
  
  
Dynamic model do not manually input the output shape
=======================


In this case, you do not manually input the output shape of the previous layer to the new layer.

.. code-block:: python

  import tensorlayer as tl
  from tensorlayer.layers import Module
  from tensorlayer.layers import Dropout, Dense
  class CustomModel(Module):

      def __init__(self):
          super(CustomModel, self).__init__()

          self.dropout1 = Dropout(keep=0.8)
          self.dense1 = Dense(n_units=800, act=tl.ReLU)
          self.dropout2 = Dropout(keep=0.8)
          self.dense2 = Dense(n_units=800, act=tl.ReLU)
          self.dropout3 = Dropout(keep=0.8)
          self.dense3 = Dense(n_units=10, act=None)

      def forward(self, x, foo=False):
          z = self.dropout1(x)
          z = self.dense1(z)
          z = self.dropout2(z)
          z = self.dense2(z)
          z = self.dropout3(z)
          out = self.dense3(z)
          if foo:
              out = tf.nn.softmax(out)
          return out

  MLP = CustomModel()
  MLP.init_build(tl.layers.Input(shape=(1, 784))) # init_build must be called to initialize the weights.
  MLP.set_eval()
  outputs = MLP(data, foo=True) # controls the forward here
  outputs = MLP(data, foo=False)

Switching train/test modes
=============================

.. code-block:: python

  # method 1: switch before forward
  MLP.set_train() # enable dropout, batch norm moving avg ...
  output = MLP(train_data)
  ... # training code here
  Model.set_eval()  # disable dropout, batch norm moving avg ...
  output = MLP(test_data)
  ... # testing code here
  
  # method 2: Using packaged training modules
  model = tl.models.Model(network=MLP, loss_fn=tl.cost.softmax_cross_entropy_with_logits, optimizer=optimizer)
  model.train(n_epoch=n_epoch, train_dataset=train_ds)

Reuse weights
=======================

For dynamic model, call the layer multiple time in forward function

.. code-block:: python

  import tensorlayer as tl
  from tensorlayer.layers import Module, Dense, Concat
  class MyModel(Module):
      def __init__(self):
          super(MyModel, self).__init__()
          self.dense_shared = Dense(n_units=800, act=tl.ReLU, in_channels=784)
          self.dense1 = Dense(n_units=10, act=tl.ReLU, in_channels=800)
          self.dense2 = Dense(n_units=10, act=tl.ReLU, in_channels=800)
          self.cat = Concat()

      def forward(self, x):
          x1 = self.dense_shared(x) # call dense_shared twice
          x2 = self.dense_shared(x)
          x1 = self.dense1(x1)
          x2 = self.dense2(x2)
          out = self.cat([x1, x2])
          return out

  model = MyModel()

Print model information
=======================

.. code-block:: python

  print(MLP) # simply call print function

  # Model(
  #   (_inputlayer): Input(shape=[None, 784], name='_inputlayer')
  #   (dropout): Dropout(keep=0.8, name='dropout')
  #   (dense): Dense(n_units=800, relu, in_channels='784', name='dense')
  #   (dropout_1): Dropout(keep=0.8, name='dropout_1')
  #   (dense_1): Dense(n_units=800, relu, in_channels='800', name='dense_1')
  #   (dropout_2): Dropout(keep=0.8, name='dropout_2')
  #   (dense_2): Dense(n_units=10, None, in_channels='800', name='dense_2')
  # )

Get specific weights
=======================

We can get the specific weights by indexing or naming.

.. code-block:: python

  # indexing
  all_weights = MLP.all_weights
  some_weights = MLP.all_weights[1:3]

Save and restore model
=======================

We provide two ways to save and restore models


Save weights only
------------------

.. code-block:: python

  MLP.save_weights('./model_weights.npz') # by default, file will be in hdf5 format
  MLP.load_weights('./model_weights.npz')

Save model weights (optional)
-----------------------------------------------

.. code-block:: python

  # When using packaged training modules. Saving and loading the model can be done as follows
  model = tl.models.Model(network=MLP, loss_fn=tl.cost.softmax_cross_entropy_with_logits, optimizer=optimizer)
  model.train(n_epoch=n_epoch, train_dataset=train_ds)
  model.save_weights('./model.npz', format='npz_dict')
  model.load_weights('./model.npz', format='npz_dict')

