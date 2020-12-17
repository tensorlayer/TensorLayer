#! /usr/bin/python
# -*- coding: utf-8 -*-

from .common import str2act
from mindspore.nn import Cell
import os
import tensorlayer as tl
from tensorlayer.files import utils
from tensorlayer.layers.utils import (get_variable_with_initializer)
from tensorlayer import logging

_global_layer_name_dict = {}  # TODO: better implementation?


class Module(Cell):

    def __init__(self, name=None, act=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global _global_layer_name_dict
        if name is None:
            prefix = self.__class__.__name__.lower()

            if _global_layer_name_dict.get(prefix) is not None:
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
            else:
                _global_layer_name_dict[prefix] = 0
                name = prefix
            while True:
                if _global_layer_name_dict.get(name) is None:
                    break
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
        else:
            if _global_layer_name_dict.get(name) is not None:
                pass
            else:
                _global_layer_name_dict[name] = 0

        self.name = name
        if isinstance(act, str):
            self.act = str2act(act)
        else:
            if act:
                self.act = act()
            else:
                self.act = act

        # Layer building state
        self._built = False

        # Layer nodes state
        self._nodes = []
        self._nodes_fixed = False

        # Layer weight state
        self._all_weights = []
        self._trainable_weights = []
        self._nontrainable_weights = []

        # Layer training state
        self.is_train = True

    def forward(self, *inputs, **kwargs):
        raise Exception("The forward method must be implemented by inherited class")

    def construct(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def build(self, inputs_shape):
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    def _get_weights(self, var_name, shape, init=tl.initializers.random_normal(), trainable=True):
        """ Get trainable variables. """
        weight = get_variable_with_initializer(
            scope_name=self.name, var_name=var_name, shape=shape, init=init, trainable=trainable
        )
        self.trainable = trainable
        return weight

    def save_weights(self, file_path, format=None):
        """Input file_path, save model weights into a file of given format.
                    Use self.load_weights() to restore.

                Parameters
                ----------
                file_path : str
                    Filename to which the model weights will be saved.
                format : str or None
                    Saved file format.
                    Value should be None, 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
                    1) If this is set to None, then the postfix of file_path will be used to decide saved format.
                    If the postfix is not in ['h5', 'hdf5', 'npz', 'ckpt'], then file will be saved in hdf5 format by default.
                    2) 'hdf5' will save model weights name in a list and each layer has its weights stored in a group of
                    the hdf5 file.
                    3) 'npz' will save model weights sequentially into a npz file.
                    4) 'npz_dict' will save model weights along with its name as a dict into a npz file.
                    5) 'ckpt' will save model weights into a tensorflow ckpt file.

                    Default None.

                Examples
                --------
                1) Save model weights in hdf5 format by default.
                >>> net = vgg16()
                >>> net.save_weights('./model.h5')
                ...
                >>> net.load_weights('./model.h5')

                2) Save model weights in npz/npz_dict format
                >>> net = vgg16()
                >>> net.save_weights('./model.npz')
                >>> net.save_weights('./model.npz', format='npz_dict')

                """

        # self.all_weights = self.network.all_weights
        if self.all_weights is None or len(self.all_weights) == 0:
            logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
            return

        if format is None:
            postfix = file_path.split('.')[-1]
            if postfix in ['h5', 'hdf5', 'npz', 'ckpt']:
                format = postfix
            else:
                format = 'hdf5'

        if format == 'hdf5' or format == 'h5':
            utils.save_weights_to_hdf5(file_path, self)
        elif format == 'npz':
            utils.save_npz(self.all_weights, file_path)
        elif format == 'npz_dict':
            utils.save_npz_dict(self.all_weights, file_path)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
                "Other format is not supported now."
            )

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights().

        Parameters
        ----------
        file_path : str
            Filename from which the model weights will be loaded.
        format : str or None
            If not specified (None), the postfix of the file_path will be used to decide its format. If specified,
            value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            In addition, it should be the same format when you saved the file using self.save_weights().
            Default is None.
        in_order : bool
            Allow loading weights into model in a sequential way or by name. Only useful when 'format' is 'hdf5'.
            If 'in_order' is True, weights from the file will be loaded into model in a sequential way.
            If 'in_order' is False, weights from the file will be loaded into model by matching the name
            with the weights of the model, particularly useful when trying to restore model in eager(graph) mode from
            a weights file which is saved in graph(eager) mode.
            Default is True.
        skip : bool
            Allow skipping weights whose name is mismatched between the file and model. Only useful when 'format' is
            'hdf5' or 'npz_dict'. If 'skip' is True, 'in_order' argument will be ignored and those loaded weights
            whose name is not found in model weights (self.all_weights) will be skipped. If 'skip' is False, error will
            occur when mismatch is found.
            Default is False.

        Examples
        --------
        1) load model from a hdf5 file.
        >>> net = vgg16()
        >>> net.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
        >>> net.load_weights('./model_eager.h5') # load sequentially

        2) load model from a npz file
        >>> net.load_weights('./model.npz')

        2) load model from a npz file, which is saved as npz_dict previously
        >>> net.load_weights('./model.npz', format='npz_dict')

        Notes
        -------
        1) 'in_order' is only useful when 'format' is 'hdf5'. If you are trying to load a weights file which is
           saved in a different mode, it is recommended to set 'in_order' be True.
        2) 'skip' is useful when 'format' is 'hdf5' or 'npz_dict'. If 'skip' is True,
           'in_order' argument will be ignored.

        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("file {} doesn't exist.".format(file_path))

        if format is None:
            format = file_path.split('.')[-1]

        if format == 'hdf5' or format == 'h5':
            if skip ==True or in_order == False:
                # load by weights name
                utils.load_hdf5_to_weights(file_path, self, skip)
            else:
                # load in order
                utils.load_hdf5_to_weights_in_order(file_path, self)
        elif format == 'npz':
            utils.load_and_assign_npz(file_path, self)
        elif format == 'npz_dict':
            utils.load_and_assign_npz_dict(file_path, self, skip)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
                "Other format is not supported now."
            )

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [tl.get_tensor_shape(t) for t in tensors]
        else:
            shape_mem = tl.get_tensor_shape(tensors)
        return shape_mem

    def _add_node(self, input_tensors, output_tensors):
        """Add a LayerNode for this layer given input_tensors, output_tensors.

        WARINING: This function should not be called from outside, it should only be called
        in layer.__call__ when building static model.

        Parameters
        ----------
        input_tensors : Tensor or a list of tensors
            Input tensors to this layer.
        output_tensors : Tensor or a list of tensors
            Output tensors to this layer.

        """
        raise NotImplementedError

    def set_train(self):
        """
        Sets the cell to training mode.

        The cell itself and all children cells will be set to training mode.

        Args:
            mode (bool): Specifies whether the model is training. Default: True.
        """
        self._phase = 'train'
        self.add_flags_recursive(training=True)
        return self

    def eval(self):
        """Set this network in evaluation mode. After calling this method,
        all layers in network are in evaluation mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> net = tl.models.vgg16()
        >>> net.eval()
        # do evaluation

        """
        self._phase = 'predict'
        self.add_flags_recursive(training=False)
        return self

    @property
    def trainable_weights(self):
        """
        Returns all trainable weights.

        Returns a list of all trainable parmeters.

        Args:
            recurse (bool): Whether contains the trainable weights of sublayers. Default: True.

        Returns:
            List, the list of trainable weights.
        """
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=True)))

    @property
    def nontrainable_weights(self):
        """
        Returns all untrainable weights.

        Returns a list of all untrainable weights.

        Args:
            recurse (bool): Whether contains the untrainable weights of sublayers. Default: True.

        Returns:
            List, the list of untrainable weights.
        """
        return list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=True)))

    @property
    def all_weights(self):
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=True))) \
               + list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=True)))


class LayerNode(object):
    """
    The class :class:`LayerNode` class represents a conceptional node for a layer.

    LayerNode is used for building static model and it is actually a light weighted
    wrapper over Layer. Specifically, it is used for building static computational graph
    (see _construct_graph() in tl.models.Model). In static model, each layer relates to
    one or more LayerNode, and the connection relationship between layers is built upon
    LayerNode. In addition, LayerNode eases layer reuse and weights sharing.

    Parameters
    ----------
    layer : tl.layers.Layer
        A tl layer that wants to create a node.
    node_index : int
        Index of this node in layer._nodes.
    in_nodes ：a list of LayerNode
        Father nodes to this node.
    in_tensors : a list of tensors
        Input tensors to this node.
    out_tensors : a list of tensors
        Output tensors to this node.
    in_tensor_idxes : a list of int
        Indexes of each input tensor in its corresponding node's out_tensors.

    Methods
    ---------
    __init__()
        Initializing the LayerNode.
    __call__()
        (1) Forwarding through the layer. (2) Update its input/output tensors.
    """

    def __init__(self, layer, node_index, in_nodes, in_tensors, out_tensors, in_tensor_idxes):
        """

        Parameters
        ----------
        layer
        node_index
        in_nodes
        in_tensors
        out_tensors
        in_tensor_idxes
        """
        self.layer = layer
        self.node_index = node_index
        self.in_nodes = in_nodes
        self.out_nodes = []
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors
        self.name = layer.name + "_node_{}".format(node_index)

        self.in_tensors_idxes = in_tensor_idxes

        self.visited = False

    def __call__(self, inputs, **kwargs):
        """(1) Forwarding through the layer. (2) Update its input/output tensors."""
        outputs = self.layer.forward(inputs, **kwargs)
        self.in_tensors = tolist(inputs)
        self.out_tensors = tolist(outputs)
        return self.out_tensors


def tolist(tensors):
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        ntensors = list()
        for t in tensors:
            ntensors += tolist(t)
        return ntensors
    else:
        return [tensors]
