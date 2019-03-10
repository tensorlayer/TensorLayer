import numpy as np
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorlayer.layers import Layer, ModelLayer
from tensorlayer import logging
from queue import Queue
from tensorlayer.files import utils
import os

__all__ = [
    'Model',
]


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Model():
    """The :class:`Model` class represents a neural network.

    Parameters
    -----------
    inputs : a Layer or list of Layer
        The input(s) to the model.
    outputs : a Layer or list of Layer
        The output(s) to the model.
    name : None or str
        The name of the model.
    """

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, inputs=None, outputs=None, name=None):
        '''

        :param inputs: Layer or list of Layer
        :param outputs: Layer or list of Layer
        :param name: str
        '''
        # Model properties
        # TODO: model auto naming
        self.name = name

        # Model state: train or test
        self.is_train = None

        # Model weights
        self._weights = None

        # Model inputs and outputs
        # TODO: note that in dynamic network, inputs and outputs are both None, may cause problem, test needed
        self._inputs = inputs
        self._outputs = outputs

        # Model converted into a Layer
        self._model_layer = None

        # Layer Node status
        self._layer_node_fixed = False

        # Model layers
        self._all_layers = None

        if inputs is None and outputs is None:
            pass

        else:
            # check type of inputs and outputs
            check_order = ['inputs', 'outputs']
            for co, check_argu in enumerate([inputs, outputs]):
                # FIXME : make this check to a util function later
                if isinstance(check_argu, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(check_argu):
                    pass
                elif isinstance(check_argu, list):
                    if len(check_argu) == 0:
                        raise ValueError(
                            "The argument `%s` is detected as an empty list. " % check_order[co] +
                            "It should be either Tensor or a list of Tensor."
                        )
                    for idx in range(len(check_argu)):
                        # FIXME : make this check to a util function later
                        if not isinstance(check_argu[idx], tf_ops._TensorLike) or not tf_ops.is_dense_tensor_like(check_argu[idx]):
                            raise TypeError(
                                "The argument `%s` should be either Tensor or a list of Tensor "
                                % (check_order[co]) +
                                "but the %s[%d] is detected as %s"
                                % (check_order[co], idx, type(check_argu[idx]))
                            )
                else:
                    raise TypeError("The argument `%s` should be either Tensor or a list of Tensor but received %s" %
                                    (check_order[co], type(check_argu)))

            # build network graph
            self._node_by_depth, self._all_layers = self._construct_graph()

            self._fix_nodes_for_layers()

    def __call__(self, inputs, is_train=None, **kwargs):
        """

        :param inputs: Tensor or list of Tensor, numpy.ndarray of list of numpy.ndarray (if in eager mode)
        :param is_train: boolean
        :return:
        """

        self._check_mode(is_train)

        # fix LayerNodes when first calling
        if self._layer_node_fixed is False:
            self._fix_nodes_for_layers()

        # set training / inference mode if necessary
        if is_train is not None:
            self._set_mode_for_layers(is_train)

        # if self._input is a list, then it must be a static network
        if isinstance(self._inputs, list):
            if not isinstance(inputs, list):
                raise ValueError("The argument `inputs` should be a list of values but detected as %s." % type(inputs))
            elif len(inputs) != len(self._inputs):
                raise ValueError("The argument `inputs` should be a list with len=%d but detected as len=%d."
                                 % (len(self._inputs), len(inputs)))

        # convert inputs to tensor if it is originally not
        # FIXME: not sure convert_to_tensor here or ask user to do it
        if isinstance(inputs, list):
            for idx in range(len(inputs)):
                inputs[idx] = tf.convert_to_tensor(inputs[idx])
        else:
            inputs = tf.convert_to_tensor(inputs)

        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(self, *inputs):
        # FIXME: currently using self._outputs to judge static network or dynamic network
        if self._outputs is None:
            raise ValueError("Outputs not defined. Please define inputs and outputs when the model is created. Or overwrite forward() function.")

        # results = list()
        # TODO: clear memory when necessary
        memory = dict()

        # get each layer's output by going through the graph in depth order
        for depth, nodes in enumerate(self._node_by_depth):
            if depth == 0:
                if isinstance(self.inputs, list):
                    assert len(inputs[0]) == len(nodes)
                    for idx, node in enumerate(nodes):
                        memory[node.name] = node(inputs[0][idx])
                else:
                    memory[nodes[0].name] = nodes[0](inputs[0])
            else:
                for node in nodes:
                    in_nodes = node.in_nodes
                    if len(in_nodes) == 1:
                        node_input = memory[in_nodes[0].name]
                    else:
                        node_input = [memory[inode.name] for inode in in_nodes]
                    memory[node.name] = node(node_input)

        if not isinstance(self._outputs, list):
            return memory[self._outputs._info[0].name]
        else:
            return [memory[tensor._info[0].name] for tensor in self._outputs]

    @property
    def all_layers(self):
        if self._all_layers is not None:
            return self._all_layers

        if self._inputs is not None and self._outputs is not None:
            # static model
            return self._all_layers
        else:
            # dynamic model
            self._all_layers = list()
            attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
            attr_list.remove("weights")
            attr_list.remove("all_layers")
            for idx, attr in enumerate(attr_list):
                try:
                    if isinstance(getattr(self, attr), Layer):
                        nowlayer = getattr(self, attr)
                        if not nowlayer._built:
                            raise AttributeError(
                                "Layer %s not built yet." % repr(nowlayer)
                            )
                        self._all_layers.append(nowlayer)
                except Exception:
                    pass
            return self._all_layers

    @property
    def weights(self):
        if self._weights is not None and len(self._weights) > 0:
            # self._weights already extracted, so do nothing
            pass
        # FIXME: currently using self._outputs to judge static network or dynamic network
        else:
            self._weights = []
            for layer in self.all_layers:
                if layer.weights is not None:
                    self._weights.extend(layer.weights)

        return self._weights

    def train(self):
        if self.is_train != True:
            self.is_train = True
            self._set_mode_for_layers(True)

    def eval(self):
        if self.is_train != False:
            self.is_train = False
            self._set_mode_for_layers(False)

    def test(self):
        self.eval()

    def infer(self):
        self.eval()

    def as_layer(self):

        if self._outputs is None:
            raise AttributeError(
                "Dynamic network cannot be converted to Layer."
            )

        if self._model_layer is None:
            self._model_layer = ModelLayer(self)

        return self._model_layer

    def _check_mode(self, is_train):
        # contradiction test
        if is_train is None and self.is_train is None:
            raise ValueError("Training / inference mode not defined. Argument `is_train` should be set as True / False. Otherwise please use `Model.train()` / `Model.eval()` to switch the mode.")
        elif is_train is not None and self.is_train is not None:
            if is_train == self.is_train:
                logging.warning("Training / inference mode redefined redundantly. Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode.")
            else:
                raise AttributeError("Training / inference mode mismatch. The argument `is_train` is set as %s, " % is_train +
                                     "but the mode is currently set as %s. " % ('Training by Model.train()' if self.is_train else 'Inference by Model.eval()') +
                                     "Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode.")

    def _set_mode_for_layers(self, is_train):
        # FIXME: currently using self._outputs to judge static network or dynamic network
        for layer in self.all_layers:
            layer._set_mode_for_layers(is_train)
        # if self._outputs is not None:
        #     for depth_layers in self.layer_by_depth:
        #         for layer in depth_layers:
        #             layer._set_mode_for_layers(is_train)
        # else:
        #     attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
        #     attr_list.remove("weights")
        #     for idx, attr in enumerate(attr_list):
        #         try:
        #             if isinstance(getattr(self, attr), Layer):
        #                 getattr(self, attr)._set_mode_for_layers(is_train)
        #         except Exception:
        #             pass

    def _fix_nodes_for_layers(self):
        for layer in self.all_layers:
            layer._nodes_fixed = True
        self._layer_node_fixed = True

    # TODO : this function seems to be useless ?
    def _find_idx_of_inputs(self, target_input):
        """
        Return the index of the target_input in self._inputs.
        Return -1 if not found.

        :param target_input: the input layer needs to be located
        :return:
        """
        if isinstance(self._inputs, list):
            for idx, input in enumerate(self._inputs):
                if input is target_input:
                    return idx
        return -1

    def __repr__(self):
        # TODO : need to support static network
        # TODO : need update by @Ruihai
        tmpstr = self.__class__.__name__ + '(\n'
        attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
        attr_list.remove("weights")
        attr_list.remove("_set_mode_for_layers")
        attr_list.remove("release_memory")
        attr_list.remove("_inputs")
        attr_list.remove("_outputs")
        for idx, attr in enumerate(attr_list):
            try:
                if isinstance(getattr(self, attr), Layer) or isinstance(getattr(self, attr), Model):
                    nowlayer = getattr(self, attr)
                    modstr = nowlayer.__repr__()
                    modstr = _addindent(modstr, 2)
                    tmpstr = tmpstr + '  (' + attr + '): ' + modstr + '\n'
                elif isinstance(getattr(self, attr), list) and (isinstance(getattr(self, attr)[0], Layer) or
                                                                isinstance(getattr(self, attr)[0], Model)):
                    for idx, element in enumerate(getattr(self, attr)):
                        modstr = element.__repr__()
                        modstr = _addindent(modstr, 2)
                        tmpstr = tmpstr + '  (' + attr + '[%d]): ' % idx + modstr + '\n'

            except Exception:
                pass
        tmpstr = tmpstr + ')'
        return tmpstr

    def print_all_layers(self):
        # TODO : need update by @Ruihai
        nowoutputs = self._outputs
        if (isinstance(nowoutputs, list) == False):
            nowoutputs = [nowoutputs]
        for out in nowoutputs:
            stacked_layers = list()
            current = out
            while current is not None:
                print(current.name, current == self._inputs)
                stacked_layers.append(current)
                current = current._input_layer
        pass

    ## raise Exceptions for old version codes
    def count_params(self, **kwargs):
        raise Exception("please change count_params --> count_weights")

    def print_params(self, **kwargs):
        raise Exception("please change print_params --> print_weights")

    @property
    def all_params(self):
        raise Exception("please change all_params --> weights")

    @property
    def all_drop(self):
        raise Exception("all_drop is deprecated")

    def _construct_graph(self):
        all_layers = []
        node_by_depth = []  # [[node0, node1], [node2, node3], ...]

        input_tensors_list = self.inputs if isinstance(self.inputs, list) else [self.inputs]
        # check input tensor comes from tl.layers.Input
        # (has '_info' attribute, which records the LayerNode Information)
        for tensor in input_tensors_list:
            if not hasattr(tensor, '_info'):
                raise ValueError('Input tensors to Model "' + self.name + '" ' +
                                 'must come from `tl.layers.Input`. '
                                 'Received: ' + str(tensor) + '.')

        queue_node = Queue()

        # BFS to visit all nodes that should be involved in the computation graph
        output_tensors_list = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        output_nodes = [tensor._info[0] for tensor in output_tensors_list]

        visited_node_names = []
        for out_node in output_nodes:
            queue_node.put(out_node)

            while not queue_node.empty():
                cur_node = queue_node.get()
                in_nodes = cur_node.in_nodes

                for node in in_nodes:
                    node.out_nodes.append(cur_node)
                    if node.name not in visited_node_names:
                        visited_node_names.append(node.name)
                        queue_node.put(node)

        # construct the computation graph in top-sort order
        cur_depth = [tensor._info[0] for tensor in input_tensors_list]
        next_depth = []
        indegrees = {}

        visited_layer_names = []
        while not len(cur_depth) == 0:
            node_by_depth.append(cur_depth)
            for node in cur_depth:
                if node.layer.name not in visited_layer_names:
                    all_layers.append(node.layer)
                    visited_layer_names.append(node.layer.name)
                for out_node in node.out_nodes:
                    if out_node.name not in indegrees.keys():
                        indegrees[out_node.name] = len(out_node.in_nodes)
                    indegrees[out_node.name] -= 1
                    if indegrees[out_node.name] == 0:
                        next_depth.append(out_node)

            cur_depth = next_depth
            next_depth = []

        return node_by_depth, all_layers

    def release_memory(self):
        '''
        WARNING: This function should be called with great caution.

        Release objects that MAY NOT be necessary such as layer.outputs (if in a tf.GradientTape() scope).
        For each layer in the model, layer.inputs and layer.outputs will be set as None but not deleted.

        A void function.
        '''
        for layer in self.all_layers:
            layer._release_memory()

    # FIXME : Model save part @runhai
    # def save(self, filepath):
    #     if self.outputs is None:
    #         raise AssertionError(
    #             "save_graph not support dynamic mode yet"
    #         )
    #     utils.save_graph(network=self, name=filepath)
    #
    #
    # def load(filepath):
    #     return utils.load_graph(name=filepath)

    def save_weights(self, filepath, sess=None, format='hdf5'):
        # TODO: Documentation pending
        """Input filepath and the session(optional), save model weights into a file of given format.
            Use self.load_weights() to restore.

        Parameters
        ----------
        filepath : str
            Filename to which the model weights will be saved.
        sess : None or a tensorflow session
            In eager mode, this should be left as None. In graph mode, must specify it with a tensorflow session.
        format : Save file format
            Value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            'hdf5' will save model weights name in a list and each layer has its weights stored in a group of
            the hdf5 file.
            'npz' will save model weights sequentially into a npz file.
            'npz_dict' will save model weights along with its name as a dict into a npz file.
            'ckpt' will save model weights into a tensorflow ckpt file.

        Examples
        --------
        1) Save model to hdf5 in eager mode
        >>> net = tl.models.vgg.vgg16()
        >>> net.save_weights('./model.h5')

        2) Save model to npz in graph mode
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> net.save_weights('./model.npz', sess=sess, format='npz')

        Returns
        -------

        """
        if self.weights is None:
            logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
            return

        if format == 'hdf5':
            utils.save_weights_to_hdf5(filepath, self.weights, sess)
        elif format == 'npz':
            utils.save_npz(self.weights, filepath, sess)
        elif format == 'npz_dict':
            utils.save_npz_dict(self.weights, filepath, sess)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError("Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
                             "Other format is not supported now.")

    def load_weights(self, filepath, sess=None, format='hdf5', in_order=True, skip=False):
        # TODO: Documentation pending
        """Load model weights from a given file, which should be previously saved by self.save_weights().

        Parameters
        ----------
        filepath : str
            Filename from which the model weights will be loaded.
        sess : None or a tensorflow session
            In eager mode, this should be left as None. In graph mode, must specify it with a tensorflow session.
            Default is 'None'.
        format : Loaded file format
            Value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            In addition, it should be the same format when you saved the file using self.save_weights().
            Default is 'hdf5'.
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
            whose name is not found in model weights (self.weights) will be skipped. If 'skip' is False, error will
            occur when mismatch is found.
            Default is False.

        Examples
        --------
        1) load model from a hdf5 file in eager mode.
        >>> net = tl.models.vgg.vgg16()
        >>> net.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
        >>> net.load_weights('./model_eager.h5') # load sequentially

        2) load model from a npz file in graph mode
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> net.load_weights('./model.npz', sess=sess, format='npz')

        Notes
        -------
        1) 'in_order' is only useful when 'format' is 'hdf5'. If you are trying to load a weights file which is
           saved in a different mode, it is recommended to set 'in_order' be True.
        2) 'skip' is useful when 'format' is 'hdf5' or 'npz_dict'. If 'skip' is True,
           'in_order' argument will be ignored.

        Returns
        -------

        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("file {} doesn't exist.".format(filepath))

        if format == 'hdf5':
            if skip == True or in_order == False:
                # load by weights name
                utils.load_hdf5_to_weights(filepath, self.weights, sess, skip)
            else:
                # load in order
                utils.load_hdf5_to_weights_in_order(filepath, self.weights, sess)
        elif format == 'npz':
            utils.load_and_assign_npz(sess, filepath, self)
        elif format == 'npz_dict':
            utils.load_and_assign_npz_dict(sess, filepath, self, skip)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError("File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
                             "Other format is not supported now.")

    def save_ckpt(self, sess=None, mode_name='model.ckpt', save_dir='checkpoint', global_step=None, printable=False):
        # TODO: Documentation pending
        """"""
        if not os.path.exists(save_dir):
            raise FileNotFoundError("Save directory {} doesn't exist.".format(save_dir))
        utils.save_ckpt(sess, mode_name, save_dir, self.weights, global_step, printable)

    def load_ckpt(self, sess=None, mode_name='model.ckpt', save_dir='checkpoint', is_latest=True, printable=False):
        # TODO: Documentation pending
        """"""
        utils.load_ckpt(sess, mode_name, save_dir, self.weights, is_latest, printable)