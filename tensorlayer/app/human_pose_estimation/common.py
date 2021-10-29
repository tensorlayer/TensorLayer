#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

# Reference:
- [pose_lcn](
    https://github.com/rujiewu/pose_lcn)

- [3d-pose-baseline](
    https://github.com/una-dinosauria/3d-pose-baseline)

"""

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec

H36M_NAMES = [''] * 17
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[4] = 'LHip'
H36M_NAMES[5] = 'LKnee'
H36M_NAMES[6] = 'LFoot'
H36M_NAMES[7] = 'Belly'
H36M_NAMES[8] = 'Neck'
H36M_NAMES[9] = 'Nose'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LHand'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RHand'
IN_F = 2
IN_JOINTS = 17
OUT_JOINTS = 17
neighbour_matrix = np.array(
    [
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0.],
        [1., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0.],
        [1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
        [1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.]
    ]
)

ROOT_PATH = '../../examples/app_tutorials/data/'


def mask_weight(weight):
    weights = tf.clip_by_norm(weight, 1)
    L = neighbour_matrix.T
    mask = tf.constant(L)
    input_size, output_size = weights.get_shape()
    input_size, output_size = int(input_size), int(output_size)
    assert input_size % IN_JOINTS == 0 and output_size % IN_JOINTS == 0
    in_F = int(input_size / IN_JOINTS)
    out_F = int(output_size / IN_JOINTS)
    weights = tf.reshape(weights, [IN_JOINTS, in_F, IN_JOINTS, out_F])
    mask = tf.reshape(mask, [IN_JOINTS, 1, IN_JOINTS, 1])

    weights = tf.cast(weights, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    masked_weights = weights * mask
    masked_weights = tf.reshape(masked_weights, [input_size, output_size])
    return masked_weights


def flip_data(data):
    """
    horizontal flip
        data: [N, 17*k] or [N, 17, k], i.e. [x, y], [x, y, confidence] or [x, y, z]
    Return
        result: [2N, 17*k] or [2N, 17, k]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    flipped_data = data.copy().reshape((len(data), 17, -1))
    flipped_data[:, :, 0] *= -1  # flip x of all joints
    flipped_data[:, left_joints + right_joints] = flipped_data[:, right_joints + left_joints]
    flipped_data = flipped_data.reshape(data.shape)

    result = np.concatenate((data, flipped_data), axis=0)

    return result


def unflip_data(data):
    """
    Average original data and flipped data
        data: [2N, 17*3]
    Return
        result: [N, 17*3]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    data = data.copy().reshape((2, -1, 17, 3))
    data[1, :, :, 0] *= -1  # flip x of all joints
    data[1, :, left_joints + right_joints] = data[1, :, right_joints + left_joints]
    data = np.mean(data, axis=0)
    data = data.reshape((-1, 17 * 3))

    return data


class DataReader(object):

    def __init__(self):
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None

    def real_read(self, subset):
        file_name = 'h36m_%s.pkl' % subset
        print('loading %s' % file_name)
        file_path = os.path.join(ROOT_PATH, file_name)
        with open(file_path, 'rb') as f:
            gt = pickle.load(f)
        return gt

    def read_2d(self, which='scale', mode='dt_ft', read_confidence=True):
        if self.gt_trainset is None:
            self.gt_trainset = self.real_read('train')
        if self.gt_testset is None:
            self.gt_testset = self.real_read('test')

        if mode == 'gt':
            trainset = np.empty((len(self.gt_trainset), 17, 2))  # [N, 17, 2]
            testset = np.empty((len(self.gt_testset), 17, 2))  # [N, 17, 2]
            for idx, item in enumerate(self.gt_trainset):
                trainset[idx] = item['joint_3d_image'][:, :2]
            for idx, item in enumerate(self.gt_testset):
                testset[idx] = item['joint_3d_image'][:, :2]
            if read_confidence:
                train_confidence = np.ones((len(self.gt_trainset), 17, 1))  # [N, 17, 1]
                test_confidence = np.ones((len(self.gt_testset), 17, 1))  # [N, 17, 1]
        elif mode == 'dt_ft':
            file_name = 'h36m_sh_dt_ft.pkl'
            file_path = os.path.join(ROOT_PATH, 'dataset', file_name)
            print('loading %s' % file_name)
            with open(file_path, 'rb') as f:
                self.dt_dataset = pickle.load(f)

            trainset = self.dt_dataset['train']['joint3d_image'][:, :, :2].copy()  # [N, 17, 2]
            testset = self.dt_dataset['test']['joint3d_image'][:, :, :2].copy()  # [N, 17, 2]
            if read_confidence:
                train_confidence = self.dt_dataset['train']['confidence'].copy()  # [N, 17, 1]
                test_confidence = self.dt_dataset['test']['confidence'].copy()  # [N, 17, 1]
        else:
            assert 0, 'not supported type %s' % mode

        # normalize
        if which == 'scale':
            # map to [-1, 1]
            for idx, item in enumerate(self.gt_trainset):
                camera_name = item['camera_param']['name']
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
            for idx, item in enumerate(self.gt_testset):
                camera_name = item['camera_param']['name']
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        else:
            assert 0, 'not support normalize type %s' % which

        if read_confidence:
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]

        # reshape
        trainset, testset = trainset.reshape((len(trainset), -1)).astype(np.float32), testset.reshape(
            (len(testset), -1)
        ).astype(np.float32)

        return trainset, testset

    def read_3d(self, which='scale', mode='dt_ft'):
        if self.gt_trainset is None:
            self.gt_trainset = self.real_read('train')
        if self.gt_testset is None:
            self.gt_testset = self.real_read('test')

        # normalize
        train_labels = np.empty((len(self.gt_trainset), 17, 3))
        test_labels = np.empty((len(self.gt_testset), 17, 3))
        if which == 'scale':
            # map to [-1, 1]
            for idx, item in enumerate(self.gt_trainset):
                camera_name = item['camera_param']['name']
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                train_labels[idx, :, :2] = item['joint_3d_image'][:, :2] / res_w * 2 - [1, res_h / res_w]
                train_labels[idx, :, 2:] = item['joint_3d_image'][:, 2:] / res_w * 2
            for idx, item in enumerate(self.gt_testset):
                camera_name = item['camera_param']['name']
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                test_labels[idx, :, :2] = item['joint_3d_image'][:, :2] / res_w * 2 - [1, res_h / res_w]
                test_labels[idx, :, 2:] = item['joint_3d_image'][:, 2:] / res_w * 2
        else:
            assert 0, 'not support normalize type %s' % which

        # reshape
        train_labels, test_labels = train_labels.reshape((-1, 17 * 3)).astype(np.float32), test_labels.reshape(
            (-1, 17 * 3)
        ).astype(np.float32)

        return train_labels, test_labels

    def denormalize3D(self, data, which='scale'):
        if self.gt_testset is None:
            self.gt_testset = self.real_read('test')

        if which == 'scale':
            data = data.reshape((-1, 17, 3)).copy()
            for idx, item in enumerate(self.gt_testset):
                camera_name = item['camera_param']['name']
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                if idx < len(data):
                    data[idx, :, :2] = (data[idx, :, :2] + [1, res_h / res_w]) * res_w / 2
                    data[idx, :, 2:] = data[idx, :, 2:] * res_w / 2
                else:
                    break
        else:
            assert 0
        return data

    def denormalize2D(self, data, which='scale'):
        if self.gt_testset is None:
            self.gt_testset = self.real_read('test')

        if which == 'scale':
            data = data.reshape((-1, 17, 2)).copy()
            for idx, item in enumerate(self.gt_testset):
                camera_name = item['camera_param']['name']
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                if idx < len(data):
                    data[idx, :, :] = (data[idx, :, :] + [1, res_h / res_w]) * res_w / 2
                else:
                    break
        else:
            assert 0
        return data


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):  # blue, orange
    """
  Visualize a 3d skeleton

  Args
    channels: 54x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

    assert channels.size == len(H36M_NAMES) * 3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape(channels, (len(H36M_NAMES), -1))

    I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 14, 15, 8, 11, 12])  # start points
    J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16, 11, 12, 13])  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    """Visualize a 2d skeleton

  Args
    channels: 34x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

    assert channels.size == len(H36M_NAMES) * 2, "channels should have 64 entries, it has %d instead" % channels.size
    vals = np.reshape(channels, (len(H36M_NAMES), -1))

    I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 14, 15, 8, 11, 12])  # start points
    J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16, 11, 12, 13])  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    RADIUS = 350  # space around the subject
    xroot, yroot = vals[0, 0], vals[0, 1]
    ax.set_xlim([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim([-RADIUS + yroot, RADIUS + yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    ax.set_aspect('equal')


def visualize_3D_pose(test_data, label, result):
    fig = plt.figure(figsize=(19.2, 10.8))
    gs1 = gridspec.GridSpec(2, 6)  # 5 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 1, 1
    nsamples = 4
    for i in np.arange(nsamples):
        # Plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx - 1])
        p2d = test_data[exidx, :]
        show2Dpose(p2d, ax1)
        ax1.invert_yaxis()

        # Plot 3d gt
        ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = label[exidx, :]
        show3Dpose(p3d, ax2)

        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx + 1], projection='3d')
        p3d = result[exidx, :]
        show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

        exidx = exidx + 1
        subplot_idx = subplot_idx + 3

    plt.show()
