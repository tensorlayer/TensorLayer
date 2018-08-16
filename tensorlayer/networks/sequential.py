#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import layers

from tensorlayer import logging

from tensorlayer.networks import BaseNetwork

__all__ = ['Sequential']


class Sequential(BaseNetwork):
    """Linear stack of layers.
    Arguments:
            layers: list of layers to add to the model.
    Example:
    ```python
    # Optionally, the first layer can receive an `input_shape` argument:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    # Afterwards, we do automatic shape inference:
    model.add(Dense(32))
    # This is identical to the following:
    model = Sequential()
    model.add(Dense(32, input_dim=500))
    # And to the following:
    model = Sequential()
    model.add(Dense(32, batch_input_shape=(None, 500)))
    # Note that you can also omit the `input_shape` argument:
    # In that case the model gets built the first time you call `fit` (or other
    # training and evaluation methods).
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.compile(optimizer=optimizer, loss=loss)
    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)
    # Note that when using this delayed-build pattern (no input shape specified),
    # the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.weights    # returns []
    # Whereas if you specify the input shape, the model gets built continuously
    # as you are adding layers:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(32))
    model.weights    # returns list of length 4
    When using the delayed-build pattern (no input shape specified), you can
    choose to manually build your model by calling `build(batch_input_shape)`:
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.build((None, 500))
    model.weights    # returns list of length 4
    ```
    """

    def __init__(self, name):

        super(Sequential, self).__init__(name)

        self.add(layers.InputLayer(name='input_layer'))
        '''
        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)
        '''

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.
        Arguments:
                layer: layer instance.
        Raises:
                TypeError: If `layer` is not a layer instance.
                ValueError: In case the `layer` argument does not
                        know its input shape.
                ValueError: In case the `layer` argument has
                        multiple output tensors, or is already connected
                        somewhere else (forbidden in `Sequential` models).
        """

        if not isinstance(layer, layers.Layer):
            raise TypeError('The added layer must be an instance of class Layer. Found: %s' % type(layer))

        if len(self.all_layers) > 0 and isinstance(layer, layers.InputLayer):
            raise TypeError('No need to add another `InputLayer`, it is automatically added to the network')

        if layer.name in self.all_layers_dict.keys():
            raise ValueError("The layer name `%s` already exists in this network" % layer.name)

        self.all_layers_dict[layer.name] = layer
        self.all_layers.append(layer.name)

        # Reset Network State in case it was previously compiled
        self._net = None
        self.outputs = None
        self.is_compiled = False

    def compile(self, input_plh, reuse=False, is_train=True):

        logging.info(
            "** Compiling %s `%s` - reuse: %s, is_train: %s **" % (self.__class__.__name__, self.name, reuse, is_train)
        )

        # Reset All Layers' Inputs
        for name, layer in self.all_layers_dict.items():
            layer.inputs = None
            layer.outputs = None

        with logging.temp_handler("    [*]"):

            _net = self.all_layers_dict[self.all_layers[0]](input_plh)

            with tf.variable_scope(self.name, reuse=reuse):
                for layer in self.all_layers[1:]:
                    _net = self.all_layers_dict[layer](prev_layer=_net, is_train=is_train)
                    self.all_drop.update(_net._local_drop)

            if not self.is_compiled:
                self._net = _net
                self.outputs = self._net.outputs
                self.is_compiled = True

        return self.outputs

    def count_layers(self):
        return len(self.all_layers_dict)

    def __getitem__(self, layer_name):
        return self.all_layers_dict[layer_name]

    '''
        if not self._layers:
            set_inputs = False
            # First layer in model: check that it is an input layer.
            if not isinstance(layer, InputLayer):
                # Create an input tensor and call `layer` on the input tensor.
                # First, we need to infer the expected input shape and dtype.
                first_layer = layer
                if isinstance(layer, (Model, Sequential)):
                    # We were passed a model as first layer.
                    # This requires a specific way to figure out the
                    # input shape and dtype.
                    if not layer.layers:
                        raise ValueError('Cannot add an empty model '
                                                         'to a `Sequential` model.')
                    # In case of nested models: recover the first layer
                    # of the deepest model to infer input shape and dtype.
                    first_layer = layer.layers[0]
                    while isinstance(first_layer, (Model, Sequential)):
                        first_layer = first_layer.layers[0]
                    batch_shape = first_layer._batch_input_shape
                    dtype = first_layer.dtype

                if hasattr(first_layer, '_batch_input_shape'):
                    batch_shape = first_layer._batch_input_shape
                    dtype = first_layer.dtype
                    # Instantiate the input layer.
                    x = Input(
                            batch_shape=batch_shape,
                            dtype=dtype,
                            name=layer.name + '_input')
                    # This will build the current layer
                    # and create the node connecting the current layer
                    # to the input layer we just created.
                    layer(x)
                    set_inputs = True
                else:
                    # The layer doesn't know about its expected shape. We will have to
                    # build the model lazily on `fit`/etc.
                    batch_shape = None
            else:
                # Corner case where the user passes an InputLayer layer via `add`.
                assert len(layer._inbound_nodes[-1].output_tensors) == 1
                set_inputs = True

            if set_inputs:
                if len(layer._inbound_nodes[-1].output_tensors) != 1:
                    raise ValueError('All layers in a Sequential model '
                                                     'should have a single output tensor. '
                                                     'For multi-output layers, '
                                                     'use the functional API.')

                self.outputs = [layer._inbound_nodes[-1].output_tensors[0]]
                self.inputs = network.get_source_inputs(self.outputs[0])
        elif self.outputs:
            output_tensor = layer(self.outputs[0])
            if isinstance(output_tensor, list):
                raise TypeError('All layers in a Sequential model '
                                                'should have a single output tensor. '
                                                'For multi-output layers, '
                                                'use the functional API.')
            self.outputs = [output_tensor]
        if self.inputs:
            self.build()
        else:
            self._layers.append(layer)
    '''
    '''
    @property
    def layers(self):
        # Historically, `sequential.layers` only returns layers that were added
        # via `add`, and omits the auto-generated `InputLayer` that comes at the
        # bottom of the stack.
        if self._layers and isinstance(self._layers[0], InputLayer):
            return self._layers[1:]
        return self._layers
    
    def pop(self):
        """Removes the last layer in the model.
        Raises:
                TypeError: if there are no layers in the model.
        """
        if not self.layers:
            raise TypeError('There are no layers in the model.')

        self._layers.pop()
        self.built = False
        if not self.layers:
            self.outputs = None
            self.inputs = None
        elif self.outputs:
            self.layers[-1]._outbound_nodes = []
            self.outputs = [self.layers[-1].output]
            self.build()

    def build(self, input_shape=None):
        if input_shape and not self.inputs:
            batch_shape = tuple(input_shape)
            dtype = K.floatx()
            x = Input(
                    batch_shape=batch_shape, dtype=dtype, name=self.name + '_input')
            self.inputs = [x]
            for layer in self._layers:
                x = layer(x)
            self.outputs = [x]

        if self.inputs:
            self._init_graph_network(self.inputs, self.outputs, name=self.name)
            self.built = True
        self._track_layers(self._layers)

    def predict_proba(self, x, batch_size=32, verbose=0):
        """Generates class probability predictions for the input samples.
        The input samples are processed batch by batch.
        Arguments:
                x: input data, as a Numpy array or list of Numpy arrays
                        (if the model has multiple inputs).
                batch_size: integer.
                verbose: verbosity mode, 0 or 1.
        Returns:
                A Numpy array of probability predictions.
        """
        preds = self.predict(x, batch_size, verbose)
        if preds.min() < 0. or preds.max() > 1.:
            logging.warning('Network returning invalid probability values. '
                                            'The last layer might not normalize predictions '
                                            'into probabilities '
                                            '(like softmax or sigmoid would).')
        return preds

    def predict_classes(self, x, batch_size=32, verbose=0):
        """Generate class predictions for the input samples.
        The input samples are processed batch by batch.
        Arguments:
                x: input data, as a Numpy array or list of Numpy arrays
                        (if the model has multiple inputs).
                batch_size: integer.
                verbose: verbosity mode, 0 or 1.
        Returns:
                A numpy array of class predictions.
        """
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def get_config(self):
        config = []
        for layer in self.layers:
            config.append({
                    'class_name': layer.__class__.__name__,
                    'config': layer.get_config()
            })
        return copy.deepcopy(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = cls()
        for conf in config:
            layer = layer_module.deserialize(conf, custom_objects=custom_objects)
            model.add(layer)
        return model
    
    '''
