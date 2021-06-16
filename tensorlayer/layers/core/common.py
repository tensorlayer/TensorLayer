#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import tensorlayer as tl
from tensorlayer.files import utils
from tensorlayer import logging

_act_dict = {
    "relu": tl.ops.ReLU,
    "relu6": tl.ops.ReLU6,
    "leaky_relu": tl.ops.LeakyReLU,
    "lrelu": tl.ops.LeakyReLU,
    "softplus": tl.ops.Softplus,
    "tanh": tl.ops.Tanh,
    "sigmoid": tl.ops.Sigmoid,
    "softmax": tl.ops.Softmax
}


def str2act(act):
    if len(act) > 5 and act[0:5] == "lrelu":
        try:
            alpha = float(act[5:])
            return tl.ops.LeakyReLU(alpha=alpha)
        except Exception as e:
            raise Exception("{} can not be parsed as a float".format(act[5:]))

    if len(act) > 10 and act[0:10] == "leaky_relu":
        try:
            alpha = float(act[10:])
            return tl.ops.LeakyReLU(alpha=alpha)
        except Exception as e:
            raise Exception("{} can not be parsed as a float".format(act[10:]))

    if act not in _act_dict.keys():
        raise Exception("Unsupported act: {}".format(act))
    return _act_dict[act]

def _save_weights(net, file_path, format=None):
    """Input file_path, save model weights into a file of given format.
                Use net.load_weights() to restore.

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
    >>> optimizer = tl.optimizers.Adam(learning_rate=0.001)
    >>> metric = tl.metric.Accuracy()
    >>> model = tl.models.Model(network=net, loss_fn=tl.cost.cross_entropy, optimizer=optimizer, metrics=metric)
    >>> model.save_weights('./model.h5')
    ...
    >>> model.load_weights('./model.h5')

    2) Save model weights in npz/npz_dict format
    >>> model.save_weights('./model.npz')
    >>> model.save_weights('./model.npz', format='npz_dict')

    """

    if net.all_weights is None or len(net.all_weights) == 0:
        logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
        return

    if format is None:
        postfix = file_path.split('.')[-1]
        if postfix in ['h5', 'hdf5', 'npz', 'ckpt']:
            format = postfix
        else:
            format = 'hdf5'

    if format == 'hdf5' or format == 'h5':
        utils.save_weights_to_hdf5(file_path, net)
    elif format == 'npz':
        utils.save_npz(net.all_weights, file_path)
    elif format == 'npz_dict':
        utils.save_npz_dict(net.all_weights, file_path)
    elif format == 'ckpt':
        # TODO: enable this when tf save ckpt is enabled
        raise NotImplementedError("ckpt load/save is not supported now.")
    else:
        raise ValueError(
            "Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
            "Other format is not supported now."
        )

def _load_weights(net, file_path, format=None, in_order=True, skip=False):
    """Load model weights from a given file, which should be previously saved by net.save_weights().

    Parameters
    ----------
    file_path : str
        Filename from which the model weights will be loaded.
    format : str or None
        If not specified (None), the postfix of the file_path will be used to decide its format. If specified,
        value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
        In addition, it should be the same format when you saved the file using net.save_weights().
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
        whose name is not found in model weights (net.all_weights) will be skipped. If 'skip' is False, error will
        occur when mismatch is found.
        Default is False.

    Examples
    --------
    1) load model from a hdf5 file.
    >>> net = vgg16()
    >>> optimizer = tl.optimizers.Adam(learning_rate=0.001)
    >>> metric = tl.metric.Accuracy()
    >>> model = tl.models.Model(network=net, loss_fn=tl.cost.cross_entropy, optimizer=optimizer, metrics=metric)
    >>> model.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
    >>> model.load_weights('./model_eager.h5') # load sequentially

    2) load model from a npz file
    >>> model.load_weights('./model.npz')

    3) load model from a npz file, which is saved as npz_dict previously
    >>> model.load_weights('./model.npz', format='npz_dict')

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
            utils.load_hdf5_to_weights(file_path, net, skip)
        else:
            # load in order
            utils.load_hdf5_to_weights_in_order(file_path, net)
    elif format == 'npz':
        utils.load_and_assign_npz(file_path, net)
    elif format == 'npz_dict':
        utils.load_and_assign_npz_dict(file_path, net, skip)
    elif format == 'ckpt':
        # TODO: enable this when tf save ckpt is enabled
        raise NotImplementedError("ckpt load/save is not supported now.")
    else:
        raise ValueError(
            "File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
            "Other format is not supported now."
        )
