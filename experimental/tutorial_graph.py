#!/usr/bin/env python3

import os

import tensorflow as tf
import tensorlayer as tl


def draw_graph(name, net):
    g = tl.graph.build_graph(net)
    with open(name + '.dot', 'w') as f:
        g.gen_dot(f)
        print('saved to %s.dot' % name)
    cmd = 'dot %s.dot -Tpng -O' % name
    os.system(cmd)


def draw_vgg16():
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = tl.models.VGG16(x)
    draw_graph('VGG16', net.net)


def draw_SqueezeNetV1():
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = tl.models.SqueezeNetV1(x)
    draw_graph('SqueezeNetV1', net.net)


def draw_MobileNetV1():
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = tl.models.MobileNetV1(x)
    draw_graph('MobileNetV1', net.net)


draw_vgg16()
draw_SqueezeNetV1()
draw_MobileNetV1()
