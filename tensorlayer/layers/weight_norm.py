from weightLightModels.wrapper import Wrapper
from tensorlayer.layers import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope


class WeightNorm(Wrapper):
    """ This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNorm wrapper works for keras and tf layers.

    Arguments:
      layer: a `Layer` instance.
      in_channels: parameter required when building dynamic layer
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    
    Example:
    ```python
    weightnorm_conv_Layer = WeightNorm(DepthwiseConv2d(
                filter_size=filter_size,
                in_channels=30, padding=padding), in_channels=30)
    ```
    """
    def __init__(self, layer, data_init=False, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        self.initialized = True
        if data_init:
            self.initialized = False

        super(WeightNorm, self).__init__(layer, **kwargs)

    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector
         with it's norm """
        with variable_scope.variable_scope('compute_weights'):
            self.layer.W = nn_impl.l2_normalize(
                self.layer.v, axis=self.norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector"""
        from tensorflow.python.ops.linalg_ops import norm
        with variable_scope.variable_scope('init_norm'):
            flat = array_ops.reshape(weights, [-1, self.layer_depth])
            return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

    def _data_dep_init(self, inputs):
        """Data dependent initialization for eager execution"""
        from tensorflow.python.ops.nn import moments
        from tensorflow.python.ops.math_ops import sqrt

        with variable_scope.variable_scope('data_dep_init'):
            # Generate data dependent init values
            activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer.call(inputs)
            m_init, v_init = moments(x_init, self.norm_axes)
            scale_init = 1. / sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.layer.g = self.layer.g * scale_init
        self.layer.bias = (-m_init * scale_init)
        self.layer.activation = activation
        self.initialized = True

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer._built:
            self.layer.build(input_shape)
            self.layer._built = False

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`WeightNorm` must wrap a layer that'
                    ' contains a `kernel` for weights'
                )

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.W.shape[-1])
            self.norm_axes = list(range(self.layer.W.shape.ndims - 1))

            self.layer.v = self.layer.W
            self.layer.g = self.layer.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True)

            with ops.control_dependencies([self.layer.g.assign(
                    self._init_norm(self.layer.v))]):
                self._compute_weights()

            self.layer.built = True

        super(WeightNorm, self).build()
        self._built = True

    def call(self, inputs):
        """Call `Layer`"""
        if context.executing_eagerly():
            if not self.initialized:
                self._data_dep_init(inputs)
            self._compute_weights()  # Recompute weights for each forward pass

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())