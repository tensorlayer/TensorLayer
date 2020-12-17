#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl

_act_dict = {
    "relu": tl.ops.ReLU,
    "relu6": tl.ops.ReLU6,
    "leaky_relu": tl.ops.LeakyReLU,
    "lrelu": tl.ops.LeakyReLU,
    "softplus": tl.ops.Softplus,
    "tanh": tl.ops.Tanh,
    "sigmoid": tl.ops.Sigmoid,
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
