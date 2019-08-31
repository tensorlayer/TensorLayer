.. _getstartmodel:

===============
Define a model
===============

TensorLayer provides two ways to define a model.
Static model allows you to build model in a fluent way while dynamic model allows you to fully control the forward process.

Static model
===============

.. code-block:: python

  import tensorflow as tf
  from tensorlayer.layers import Input, Dropout, Dense
  from tensorlayer.models import Model

  def get_model(inputs_shape):
      ni = Input(inputs_shape)
      nn = Dropout(keep=0.8)(ni)
      nn = Dense(n_units=800, act=tf.nn.relu, name="dense1")(nn) # “name" is optional
      nn = Dropout(keep=0.8)(nn)
      nn = Dense(n_units=800, act=tf.nn.relu)(nn)
      nn = Dropout(keep=0.8)(nn)
      nn = Dense(n_units=10, act=None)(nn)
      M = Model(inputs=ni, outputs=nn, name="mlp") # “name" is optional
      return M

  MLP = get_model([None, 784])
  MLP.eval()
  outputs = MLP(data)

Dynamic model
=======================


In this case, you need to manually input the output shape of the previous layer to the new layer.

.. code-block:: python

  class CustomModel(Model):

      def __init__(self):
          super(CustomModel, self).__init__()

          self.dropout1 = Dropout(keep=0.8)
          self.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784)
          self.dropout2 = Dropout(keep=0.8)
          self.dense2 = Dense(n_units=800, act=tf.nn.relu, in_channels=800)
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
  MLP.eval()
  outputs = MLP(data, foo=True) # controls the forward here
  outputs = MLP(data, foo=False)
  
  
Switching train/test modes
=============================

.. code-block:: python

  # method 1: switch before forward
  Model.train() # enable dropout, batch norm moving avg ...
  output = Model(train_data) 
  ... # training code here
  Model.eval()  # disable dropout, batch norm moving avg ...
  output = Model(test_data) 
  ... # testing code here
  
  # method 2: switch while forward
  output = Model(train_data, is_train=True)
  output = Model(test_data, is_train=False)

Reuse weights
=======================

For static model, call the layer multiple time in model creation

.. code-block:: python

  # create siamese network

  def create_base_network(input_shape):
        '''Base network to be shared (eq. to feature extraction).
        '''
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(128, act=tf.nn.relu)(x)
        x = Dropout(0.9)(x)
        x = Dense(128, act=tf.nn.relu)(x)
        x = Dropout(0.9)(x)
        x = Dense(128, act=tf.nn.relu)(x)
        return Model(input, x)


  def get_siamese_network(input_shape):
        """Create siamese network with shared base network as layer
        """
        base_layer = create_base_network(input_shape).as_layer() # convert model as layer

        ni_1 = Input(input_shape)
        ni_2 = Input(input_shape)
        nn_1 = base_layer(ni_1) # call base_layer twice
        nn_2 = base_layer(ni_2)
        return Model(inputs=[ni_1, ni_2], outputs=[nn_1, nn_2])

  siamese_net = get_siamese_network([None, 784])

For dynamic model, call the layer multiple time in forward function

.. code-block:: python

  class MyModel(Model):
      def __init__(self):
          super(MyModel, self).__init__()
          self.dense_shared = Dense(n_units=800, act=tf.nn.relu, in_channels=784)
          self.dense1 = Dense(n_units=10, act=tf.nn.relu, in_channels=800)
          self.dense2 = Dense(n_units=10, act=tf.nn.relu, in_channels=800)
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
  
  import pprint
  pprint.pprint(MLP.config) # print the model architecture
  #   {'inputs': '_inputlayer_1_node_0',
  #  'model_architecture': [{'args': {'dtype': tf.float32,
  #                                   'layer_type': 'normal',
  #                                   'name': '_inputlayer_1',
  #                                   'shape': [None, 784]},
  #                          'class': '_InputLayer',
  #                          'prev_layer': None},
  #                         {'args': {'keep': 0.8,
  #                                   'layer_type': 'normal',
  #                                   'name': 'dropout_1'},
  #                          'class': 'Dropout',
  #                          'prev_layer': ['_inputlayer_1_node_0']},
  #                         {'args': {'act': 'relu',
  #                                   'layer_type': 'normal',
  #                                   'n_units': 800,
  #                                   'name': 'dense_1'},
  #                          'class': 'Dense',
  #                          'prev_layer': ['dropout_1_node_0']},
  #                         {'args': {'keep': 0.8,
  #                                   'layer_type': 'normal',
  #                                   'name': 'dropout_2'},
  #                          'class': 'Dropout',
  #                          'prev_layer': ['dense_1_node_0']},
  #                         {'args': {'act': 'relu',
  #                                   'layer_type': 'normal',
  #                                   'n_units': 800,
  #                                   'name': 'dense_2'},
  #                          'class': 'Dense',
  #                          'prev_layer': ['dropout_2_node_0']},
  #                         {'args': {'keep': 0.8,
  #                                   'layer_type': 'normal',
  #                                   'name': 'dropout_3'},
  #                          'class': 'Dropout',
  #                          'prev_layer': ['dense_2_node_0']},
  #                         {'args': {'act': None,
  #                                   'layer_type': 'normal',
  #                                   'n_units': 10,
  #                                   'name': 'dense_3'},
  #                          'class': 'Dense',
  #                          'prev_layer': ['dropout_3_node_0']}],
  #  'name': 'mlp',
  #  'outputs': 'dense_3_node_0',
  #  'version_info': {'backend': 'tensorflow',
  #                   'backend_version': '2.0.0-alpha0',
  #                   'save_date': None,
  #                   'tensorlayer_version': '2.1.0',
  #                   'training_device': 'gpu'}}

Get specific weights
=======================

We can get the specific weights by indexing or naming.

.. code-block:: python

  # indexing
  all_weights = MLP.all_weights
  some_weights = MLP.all_weights[1:3]

  # naming
  some_weights = MLP.get_layer('dense1').all_weights


Save and restore model
=======================

We provide two ways to save and restore models


Save weights only
------------------

.. code-block:: python

  MLP.save_weights('model_weights.h5') # by default, file will be in hdf5 format
  MLP.load_weights('model_weights.h5')

Save model architecture and weights (optional)
-----------------------------------------------

.. code-block:: python

  # When using Model.load(), there is no need to reimplement or declare the architecture of the model explicitly in code
  MLP.save('model.h5', save_weights=True)
  MLP = Model.load('model.h5', load_weights=True)

