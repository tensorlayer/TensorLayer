#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.app.human_pose_estimation.common import DataReader, visualize_3D_pose, flip_data
from tensorlayer.app import computer_vision
import numpy as np

datareader = DataReader()
train_data, test_data = datareader.read_2d(which='scale', mode='gt', read_confidence=False)
train_labels, test_labels = datareader.read_3d(which='scale', mode='gt')
network = computer_vision.human_pose_estimation('3D-pose')
test_data = flip_data(test_data)
result = network(test_data)
result = datareader.denormalize3D(np.asarray(result), which='scale')
test_data = datareader.denormalize2D(test_data, which='scale')
test_labels = datareader.denormalize3D(test_labels, which='scale')
visualize_3D_pose(
    test_data, test_labels, result
)  # We plot 4 examples. You can modify this function according to your own needs.
