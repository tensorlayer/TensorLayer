#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.layers.utils import list_remove_repeat

__all__ = [
    'merge_networks',
]


def merge_networks(layers=None):
    """Merge all parameters, layers and dropout probabilities to a :class:`Layer`.
    The output of return network is the first network in the list.

    Parameters
    ----------
    layers : list of :class:`Layer`
        Merge all parameters, layers and dropout probabilities to the first layer in the list.

    Returns
    --------
    :class:`Layer`
        The network after merging all parameters, layers and dropout probabilities to the first network in the list.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> n1 = ...
    >>> n2 = ...
    >>> n1 = tl.layers.merge_networks([n1, n2])

    """
    if layers is None:
        raise Exception("layers should be a list of TensorLayer's Layers.")
    layer = layers[0]

    all_params = []
    all_layers = []
    all_drop = {}

    for l in layers:
        all_params.extend(l.all_weights)
        all_layers.extend(l.all_layers)
        all_drop.update(l.all_drop)

    layer.all_weights = list(all_params)
    layer.all_layers = list(all_layers)
    layer.all_drop = dict(all_drop)

    layer.all_layers = list_remove_repeat(layer.all_layers)
    layer.all_weights = list_remove_repeat(layer.all_weights)

    return layer
