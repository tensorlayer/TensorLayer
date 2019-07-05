from tensorlayer.layers import Layer
import tensorflow as tf
import tensorlayer as tl

class Wrapper(Layer):
  """Abstract wrapper base class.
  Wrappers take another layer and augment it in various ways.
  Do not use this class as a layer, it is only an abstract base class.
  Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.
  Arguments:
    layer: The layer to be wrapped.
  """

  def __init__(self, layer, in_channels, **kwargs):
    assert isinstance(layer, Layer)
    self.layer = layer
    self.input_shape = in_channels
    super(Wrapper, self).__init__(**kwargs)
    self.build(input_shape=self.input_shape)

  def build(self, input_shape=None):
    if not self.layer._built:
      self.layer.build(self.input_shape)
    self._built = True


  def forward(self, inputs):
    if self.__class__.__name__ in tl.layers.inputs.__all__:
        input_tensors = tf.convert_to_tensor(inputs)
    else:
        input_tensors = inputs

    if not self._built:
        if isinstance(self, LayerList):
            self._input_tensors = input_tensors
        inputs_shape = self._compute_shape(input_tensors)
        self.build(inputs_shape)
        self._built = True

    outputs = self.layer.forward(input_tensors)

    if not self._nodes_fixed:
        self._add_node(input_tensors, outputs)

    return outputs