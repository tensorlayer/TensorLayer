#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
import numpy as np

import time
import numbers
import random
import os
import re
import sys

import threading
# import Queue  # <-- donot work for py3
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue as queue
else:
    import queue as queue

from six.moves import range
import scipy
from scipy import linalg
import scipy.ndimage as ndi

from skimage import transform
from skimage import exposure
import skimage

# linalg https://docs.scipy.org/doc/scipy/reference/linalg.html
# ndimage https://docs.scipy.org/doc/scipy/reference/ndimage.html

## Threading
def threading_data(data=None, fn=None, **kwargs):
    """Return a batch of result by given data.
    Usually be used for data augmentation.

    Parameters
    -----------
    data : numpy array or zip of numpy array, see Examples below.
    fn : the function for data processing.
    more args : the args for fn, see Examples below.

    Examples
    --------
    - Single array
    >>> X --> [batch_size, row, col, 1] greyscale
    >>> results = threading_data(X, zoom, zoom_range=[0.5, 1], is_random=True)
    ... results --> [batch_size, row, col, channel]
    >>> tl.visualize.images2d(images=np.asarray(results), second=0.01, saveable=True, name='after', dtype=None)
    >>> tl.visualize.images2d(images=np.asarray(X), second=0.01, saveable=True, name='before', dtype=None)

    - List of array (e.g. functions with ``multi``)
    >>> X, Y --> [batch_size, row, col, 1]  greyscale
    >>> data = threading_data([_ for _ in zip(X, Y)], zoom_multi, zoom_range=[0.5, 1], is_random=True)
    ... data --> [batch_size, 2, row, col, 1]
    >>> X_, Y_ = data.transpose((1,0,2,3,4))
    ... X_, Y_ --> [batch_size, row, col, 1]
    >>> tl.visualize.images2d(images=np.asarray(X_), second=0.01, saveable=True, name='after', dtype=None)
    >>> tl.visualize.images2d(images=np.asarray(Y_), second=0.01, saveable=True, name='before', dtype=None)

    - Customized function for image segmentation
    >>> def distort_img(data):
    ...     x, y = data
    ...     x, y = flip_axis_multi([x, y], axis=0, is_random=True)
    ...     x, y = flip_axis_multi([x, y], axis=1, is_random=True)
    ...     x, y = crop_multi([x, y], 100, 100, is_random=True)
    ...     return x, y
    >>> X, Y --> [batch_size, row, col, channel]
    >>> data = threading_data([_ for _ in zip(X, Y)], distort_img)
    >>> X_, Y_ = data.transpose((1,0,2,3,4))

    References
    ----------
    - `python queue <https://pymotw.com/2/Queue/index.html#module-Queue>`_
    - `run with limited queue <http://effbot.org/librarybook/queue.htm>`_
    """
    ## plot function info
    # for name, value in kwargs.items():
    #     print('{0} = {1}'.format(name, value))
    # exit()
    # define function for threading
    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    ## start multi-threaded reading.
    results = [None] * len(data) ## preallocate result list
    threads = []
    for i in range(len(data)):
        t = threading.Thread(
                        name='threading_and_return',
                        target=apply_fn,
                        args=(results, i, data[i], kwargs)
                        )
        t.start()
        threads.append(t)

    ## <Milo> wait for all threads to complete
    for t in threads:
        t.join()

    return np.asarray(results)

    ## old implementation
    # define function for threading
    # def function(q, i, data, kwargs):
    #     result = fn(data, **kwargs)
    #     q.put([i, result])
    # ## start threading
    # q = queue.Queue()
    # threads = []
    # for i in range(len(data)):
    #     t = threading.Thread(
    #                     name='threading_and_return',
    #                     target=function,
    #                     args=(q, i, data[i], kwargs)
    #                     )
    #     t.start()
    #     threads.append(t)
    #
    # ## <Milo> wait for all threads to complete
    # for t in threads:
    #     t.join()
    #
    # ## get results
    # results = []
    # for i in range(len(data)):
    #     result = q.get()
    #     results.append(result)
    # results = sorted(results)
    # for i in range(len(results)):
    #     results[i] = results[i][1]
    # return np.asarray(results)


## Image
def rotation(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2,
                    fill_mode='nearest', cval=0.):
    """Rotate an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    rg : int or float
        Degree to rotate, usually 0 ~ 180.
    is_random : boolean, default False
        If True, randomly rotate.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_

    Examples
    ---------
    >>> x --> [row, col, 1] greyscale
    >>> x = rotation(x, rg=40, is_random=False)
    >>> tl.visualize.frame(x[:,:,0], second=0.01, saveable=True, name='temp',cmap='gray')
    """
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi /180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

def rotation_multi(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2,
                    fill_mode='nearest', cval=0.):
    """Rotate multiple images with the same arguments, randomly or non-randomly.
    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``rotation``.

    Examples
    --------
    >>> x, y --> [row, col, 1]  greyscale
    >>> x, y = rotation_multi([x, y], rg=90, is_random=False)
    >>> tl.visualize.frame(x[:,:,0], second=0.01, saveable=True, name='x',cmap='gray')
    >>> tl.visualize.frame(y[:,:,0], second=0.01, saveable=True, name='y',cmap='gray')
    """
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi /180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x[0].shape[row_index], x[0].shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    results = []
    for data in x:
        results.append( apply_transform(data, transform_matrix, channel_index, fill_mode, cval))
    return np.asarray(results)

# crop
def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1, channel_index=2):
    """Randomly or centrally crop an image.

    Parameters
    ----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    wrg : float
        Size of weight.
    hrg : float
        Size of height.
    is_random : boolean, default False
        If True, randomly crop, else central crop.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    """
    h, w = x.shape[row_index], x.shape[col_index]
    assert (h > hrg) and (w > wrg), "The size of cropping should smaller than the original image"
    if is_random:
        h_offset = int(np.random.uniform(0, h-hrg) -1)
        w_offset = int(np.random.uniform(0, w-wrg) -1)
        # print(h_offset, w_offset, x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset].shape)
        return x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset]
    else:   # central crop
        h_offset = int(np.floor((h - hrg)/2.))
        w_offset = int(np.floor((w - wrg)/2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset: h_end, w_offset: w_end]
        # old implementation
        # h_offset = (h - hrg)/2
        # w_offset = (w - wrg)/2
        # # print(x[h_offset: h-h_offset ,w_offset: w-w_offset].shape)
        # return x[h_offset: h-h_offset ,w_offset: w-w_offset]
        # central crop


def crop_multi(x, wrg, hrg, is_random=False, row_index=0, col_index=1, channel_index=2):
    """Randomly or centrally crop multiple images.

    Parameters
    ----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``crop``.
    """
    h, w = x[0].shape[row_index], x[0].shape[col_index]
    assert (h > hrg) and (w > wrg), "The size of cropping should smaller than the original image"
    if is_random:
        h_offset = int(np.random.uniform(0, h-hrg) -1)
        w_offset = int(np.random.uniform(0, w-wrg) -1)
        results = []
        for data in x:
            results.append( data[h_offset: hrg+h_offset ,w_offset: wrg+w_offset])
        return np.asarray(results)
    else:
        # central crop
        h_offset = (h - hrg)/2
        w_offset = (w - wrg)/2
        results = []
        for data in x:
            results.append( data[h_offset: h-h_offset ,w_offset: w-w_offset] )
        return np.asarray(results)

# flip
def flip_axis(x, axis, is_random=False):
    """Flip the axis of an image, such as flip left and right, up and down, randomly or non-randomly,

    Parameters
    ----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    axis : int
        - 0, flip up and down
        - 1, flip left and right
        - 2, flip channel
    is_random : boolean, default False
        If True, randomly zoom.
    """
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
            return x
        else:
            return x
    else:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

def flip_axis_multi(x, axis, is_random=False):
    """Flip the axises of multiple images together, such as flip left and right, up and down, randomly or non-randomly,

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``flip_axis``.
    """
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            # x = np.asarray(x).swapaxes(axis, 0)
            # x = x[::-1, ...]
            # x = x.swapaxes(0, axis)
            # return x
            results = []
            for data in x:
                data = np.asarray(data).swapaxes(axis, 0)
                data = data[::-1, ...]
                data = data.swapaxes(0, axis)
                results.append( data )
            return np.asarray(results)
        else:
            return np.asarray(x)
    else:
        # x = np.asarray(x).swapaxes(axis, 0)
        # x = x[::-1, ...]
        # x = x.swapaxes(0, axis)
        # return x
        results = []
        for data in x:
            data = np.asarray(data).swapaxes(axis, 0)
            data = data[::-1, ...]
            data = data.swapaxes(0, axis)
            results.append( data )
        return np.asarray(results)

# shift
def shift(x, wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1, channel_index=2,
                 fill_mode='nearest', cval=0.):
    """Shift an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    wrg : float
        Percentage of shift in axis x, usually -0.25 ~ 0.25.
    hrg : float
        Percentage of shift in axis y, usually -0.25 ~ 0.25.
    is_random : boolean, default False
        If True, randomly shift.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    """
    h, w = x.shape[row_index], x.shape[col_index]
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
    else:
        tx, ty = hrg * h, wrg * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

def shift_multi(x, wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1, channel_index=2,
                 fill_mode='nearest', cval=0.):
    """Shift images with the same arguments, randomly or non-randomly.
    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``shift``.
    """
    h, w = x[0].shape[row_index], x[0].shape[col_index]
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
    else:
        tx, ty = hrg * h, wrg * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    results = []
    for data in x:
        results.append( apply_transform(data, transform_matrix, channel_index, fill_mode, cval))
    return np.asarray(results)

# shear
def shear(x, intensity=0.1, is_random=False, row_index=0, col_index=1, channel_index=2,
                 fill_mode='nearest', cval=0.):
    """Shear an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    intensity : float
        Percentage of shear, usually -0.5 ~ 0.5 (is_random==True), 0 ~ 0.5 (is_random==False),
        you can have a quick try by shear(X, 1).
    is_random : boolean, default False
        If True, randomly shear.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    """
    if is_random:
        shear = np.random.uniform(-intensity, intensity)
    else:
        shear = intensity
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

def shear_multi(x, intensity=0.1, is_random=False, row_index=0, col_index=1, channel_index=2,
                 fill_mode='nearest', cval=0.):
    """Shear images with the same arguments, randomly or non-randomly.
    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``shear``.
    """
    if is_random:
        shear = np.random.uniform(-intensity, intensity)
    else:
        shear = intensity
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x[0].shape[row_index], x[0].shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    results = []
    for data in x:
        results.append( apply_transform(data, transform_matrix, channel_index, fill_mode, cval))
    return np.asarray(results)

# swirl
def swirl(x, center=None, strength=1, radius=100, rotation=0, output_shape=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False, is_random=False):
    """Swirl an image randomly or non-randomly, see `scikit-image swirl API <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.swirl>`_
    and `example <http://scikit-image.org/docs/dev/auto_examples/plot_swirl.html>`_.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    center : (row, column) tuple or (2,) ndarray, optional
        Center coordinate of transformation.
    strength : float, optional
        The amount of swirling applied.
    radius : float, optional
        The extent of the swirl in pixels. The effect dies out rapidly beyond radius.
    rotation : float, (degree) optional
        Additional rotation applied to the image, usually [0, 360], relates to center.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input image is preserved.
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to be in the range 0-5. See skimage.transform.warp for detail.
    mode : {‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’}, optional
        Points outside the boundaries of the input are filled according to the given mode, with ‘constant’ used as the default. Modes match the behaviour of numpy.pad.
    cval : float, optional
        Used in conjunction with mode ‘constant’, the value outside the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image. This is enabled by default, since higher order interpolation may produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.
    is_random : boolean, default False
        If True, random swirl.
            - random center = [(0 ~ x.shape[0]), (0 ~ x.shape[1])]
            - random strength = [0, strength]
            - random radius = [1e-10, radius]
            - random rotation = [-rotation, rotation]

    Examples
    ---------
    >>> x --> [row, col, 1] greyscale
    >>> x = swirl(x, strength=4, radius=100)
    """
    assert radius != 0, Exception("Invalid radius value")
    rotation = np.pi / 180 * rotation
    if is_random:
        center_h = int(np.random.uniform(0, x.shape[0]))
        center_w = int(np.random.uniform(0, x.shape[1]))
        center = (center_h, center_w)
        strength = np.random.uniform(0, strength)
        radius = np.random.uniform(1e-10, radius)
        rotation = np.random.uniform(-rotation, rotation)

    max_v = np.max(x)
    if max_v > 1:   # Note: the input of this fn should be [-1, 1], rescale is required.
        x = x / max_v
    swirled = skimage.transform.swirl(x, center=center, strength=strength, radius=radius, rotation=rotation,
        output_shape=output_shape, order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)
    if max_v > 1:
        swirled = swirled * max_v
    return swirled

def swirl_multi(x, center=None, strength=1, radius=100, rotation=0, output_shape=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False, is_random=False):
    """Swirl multiple images with the same arguments, randomly or non-randomly.
    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``swirl``.
    """
    assert radius != 0, Exception("Invalid radius value")
    rotation = np.pi / 180 * rotation
    if is_random:
        center_h = int(np.random.uniform(0, x[0].shape[0]))
        center_w = int(np.random.uniform(0, x[0].shape[1]))
        center = (center_h, center_w)
        strength = np.random.uniform(0, strength)
        radius = np.random.uniform(1e-10, radius)
        rotation = np.random.uniform(-rotation, rotation)

    results = []
    for data in x:
        max_v = np.max(data)
        if max_v > 1:   # Note: the input of this fn should be [-1, 1], rescale is required.
            data = data / max_v
        swirled = skimage.transform.swirl(data, center=center, strength=strength, radius=radius, rotation=rotation,
            output_shape=output_shape, order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)
        if max_v > 1:
            swirled = swirled * max_v
        results.append( swirled )
    return np.asarray(results)

# elastic_transform

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
def elastic_transform(x, alpha, sigma, mode="constant", cval=0, is_random=False):
    """Elastic deformation of images as described in `[Simard2003] <http://deeplearning.cs.cmu.edu/pdfs/Simard.pdf>`_ .

    Parameters
    -----------
    x : numpy array, a greyscale image.
    alpha : scalar factor.
    sigma : scalar or sequence of scalars, the smaller the sigma, the more transformation.
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
    mode : default constant, see `scipy.ndimage.filters.gaussian_filter <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`_.
    cval : float, optional. Used in conjunction with mode ‘constant’, the value outside the image boundaries.
    is_random : boolean, default False

    Examples
    ---------
    >>> x = elastic_transform(x, alpha = x.shape[1] * 3, sigma = x.shape[1] * 0.07)

    References
    ------------
    - `Github <https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a>`_.
    - `Kaggle <https://www.kaggle.com/pscion/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation-0878921a>`_
    """
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))
    #
    is_3d = False
    if len(x.shape) == 3 and x.shape[-1] == 1:
        x = x[:,:,0]
        is_3d = True
    elif len(x.shape) == 3 and x.shape[-1] != 1:
        raise Exception("Only support greyscale image")
    assert len(x.shape)==2

    shape = x.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha

    x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
    if is_3d:
        return map_coordinates(x, indices, order=1).reshape((shape[0], shape[1], 1))
    else:
        return map_coordinates(x, indices, order=1).reshape(shape)

def elastic_transform_multi(x, alpha, sigma, mode="constant", cval=0, is_random=False):
    """Elastic deformation of images as described in `[Simard2003] <http://deeplearning.cs.cmu.edu/pdfs/Simard.pdf>`_.

    Parameters
    -----------
    x : list of numpy array
    others : see ``elastic_transform``.
    """
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))

    shape = x[0].shape
    if len(shape) == 3:
        shape = (shape[0], shape[1])
    new_shape = random_state.rand(*shape)

    results = []
    for data in x:
        is_3d = False
        if len(data.shape) == 3 and data.shape[-1] == 1:
            data = data[:,:,0]
            is_3d = True
        elif len(data.shape) == 3 and data.shape[-1] != 1:
            raise Exception("Only support greyscale image")
        assert len(data.shape)==2

        dx = gaussian_filter((new_shape * 2 - 1), sigma, mode=mode, cval=cval) * alpha
        dy = gaussian_filter((new_shape * 2 - 1), sigma, mode=mode, cval=cval) * alpha

        x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
        # print(data.shape)
        if is_3d:
            results.append( map_coordinates(data, indices, order=1).reshape((shape[0], shape[1], 1)))
        else:
            results.append( map_coordinates(data, indices, order=1).reshape(shape) )
    return np.asarray(results)

# zoom
def zoom(x, zoom_range=(0.9, 1.1), is_random=False, row_index=0, col_index=1, channel_index=2,
                fill_mode='nearest', cval=0.):
    """Zoom in and out of a single image, randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    zoom_range : list or tuple
        - If is_random=False, (h, w) are the fixed zoom factor for row and column axies, factor small than one is zoom in.
        - If is_random=True, (min zoom out, max zoom out) for x and y with different random zoom in/out factor.
        e.g (0.5, 1) zoom in 1~2 times.
    is_random : boolean, default False
        If True, randomly zoom.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    """
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)
    if is_random:
        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
            print(" random_zoom : not zoom in/out")
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    else:
        zx, zy = zoom_range
    # print(zx, zy)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

def zoom_multi(x, zoom_range=(0.9, 1.1), is_random=False,
        row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.):
    """Zoom in and out of images with the same arguments, randomly or non-randomly.
    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``zoom``.
    """
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if is_random:
        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
            print(" random_zoom : not zoom in/out")
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    else:
        zx, zy = zoom_range

    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x[0].shape[row_index], x[0].shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    # x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    # return x
    results = []
    for data in x:
        results.append( apply_transform(data, transform_matrix, channel_index, fill_mode, cval))
    return np.asarray(results)

# image = tf.image.random_brightness(image, max_delta=32. / 255.)
# image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
# image = tf.image.random_hue(image, max_delta=0.032)
# image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

# brightness
def brightness(x, gamma=1, gain=1, is_random=False):
    """Change the brightness of a single image, randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    gamma : float, small than 1 means brighter.
        Non negative real number. Default value is 1.

        - If is_random is True, gamma in a range of (1-gamma, 1+gamma).
    gain : float
        The constant multiplier. Default value is 1.
    is_random : boolean, default False
        - If True, randomly change brightness.

    References
    -----------
    - `skimage.exposure.adjust_gamma <http://scikit-image.org/docs/dev/api/skimage.exposure.html>`_
    - `chinese blog <http://www.cnblogs.com/denny402/p/5124402.html>`_
    """
    if is_random:
        gamma = np.random.uniform(1-gamma, 1+gamma)
    x = exposure.adjust_gamma(x, gamma, gain)
    return x

def brightness_multi(x, gamma=1, gain=1, is_random=False):
    """Change the brightness of multiply images, randomly or non-randomly.
    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``brightness``.
    """
    if is_random:
        gamma = np.random.uniform(1-gamma, 1+gamma)

    results = []
    for data in x:
        results.append( exposure.adjust_gamma(data, gamma, gain) )
    return np.asarray(results)


# contrast
def constant(x, cutoff=0.5, gain=10, inv=False, is_random=False):
    # TODO
    x = exposure.adjust_sigmoid(x, cutoff=cutoff, gain=gain, inv=inv)
    return x

def constant_multi():
    #TODO
    pass

# resize
def imresize(x, size=[100, 100], interp='bilinear', mode=None):
    """Resize an image by given output size and method. Warning, this function
    will rescale the value to [0, 255].

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    size : int, float or tuple (h, w)
        - int, Percentage of current size.
        - float, Fraction of current size.
        - tuple, Size of the output image.
    interp : str, optional
        Interpolation to use for re-sizing (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’).
    mode : str, optional
        The PIL image mode (‘P’, ‘L’, etc.) to convert arr before resizing.

    Returns
    --------
    imresize : ndarray
    The resized array of image.

    References
    ------------
    - `scipy.misc.imresize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html>`_
    """
    if x.shape[-1] == 1:
        # greyscale
        x = scipy.misc.imresize(x[:,:,0], size, interp=interp, mode=mode)
        return x[:, :, np.newaxis]
    elif x.shape[-1] == 3:
        # rgb, bgr ..
        return scipy.misc.imresize(x, size, interp=interp, mode=mode)
    else:
        raise Exception("Unsupported channel %d" % x.shape[-1])

# normailization
def samplewise_norm(x, rescale=None, samplewise_center=False, samplewise_std_normalization=False,
            channel_index=2, epsilon=1e-7):
    """Normalize an image by rescale, samplewise centering and samplewise centering in order.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    rescale : rescaling factor.
            If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation)
    samplewise_center : set each sample mean to 0.
    samplewise_std_normalization : divide each input by its std.
    epsilon : small position value for dividing standard deviation.

    Examples
    --------
    >>> x = samplewise_norm(x, samplewise_center=True, samplewise_std_normalization=True)
    >>> print(x.shape, np.mean(x), np.std(x))
    ... (160, 176, 1), 0.0, 1.0

    Notes
    ------
    When samplewise_center and samplewise_std_normalization are True.

    - For greyscale image, every pixels are subtracted and divided by the mean and std of whole image.
    - For RGB image, every pixels are subtracted and divided by the mean and std of this pixel i.e. the mean and std of a pixel is 0 and 1.
    """
    if rescale:
        x *= rescale

    if x.shape[channel_index] == 1:
        # greyscale
        if samplewise_center:
            x = x - np.mean(x)
        if samplewise_std_normalization:
            x = x / np.std(x)
        return x
    elif x.shape[channel_index] == 3:
        # rgb
        if samplewise_center:
            x = x - np.mean(x, axis=channel_index, keepdims=True)
        if samplewise_std_normalization:
            x = x / (np.std(x, axis=channel_index, keepdims=True) + epsilon)
        return x
    else:
        raise Exception("Unsupported channels %d" % x.shape[channel_index])

def featurewise_norm(x, mean=None, std=None, epsilon=1e-7):
    """Normalize every pixels by the same given mean and std, which are usually
    compute from all examples.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    mean : value for subtraction.
    std : value for division.
    epsilon : small position value for dividing standard deviation.
    """
    if mean:
        x = x - mean
    if std:
        x = x / (std + epsilon)
    return x

# whitening
def get_zca_whitening_principal_components_img(X):
    """Return the ZCA whitening principal components matrix.

    Parameters
    -----------
    x : numpy array
        Batch of image with dimension of [n_example, row, col, channel] (default).
    """
    flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    print("zca : computing sigma ..")
    sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
    print("zca : computing U, S and V ..")
    U, S, V = linalg.svd(sigma)
    print("zca : computing principal components ..")
    principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
    return principal_components

def zca_whitening(x, principal_components):
    """Apply ZCA whitening on an image by given principal components matrix.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    principal_components : matrix from ``get_zca_whitening_principal_components_img``.
    """
    # flatx = np.reshape(x, (x.size))
    print(principal_components.shape, x.shape)  # ((28160, 28160), (160, 176, 1))
    # flatx = np.reshape(x, (x.shape))
    # flatx = np.reshape(x, (x.shape[0], ))
    print(flatx.shape)  # (160, 176, 1)
    whitex = np.dot(flatx, principal_components)
    x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

# developing
# def barrel_transform(x, intensity):
#     # https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
#     # TODO
#     pass
#
# def barrel_transform_multi(x, intensity):
#     # https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
#     # TODO
#     pass

# channel shift
def channel_shift(x, intensity, is_random=False, channel_index=2):
    """Shift the channels of an image, randomly or non-randomly, see `numpy.rollaxis <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html>`_.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    intensity : float
        Intensity of shifting.
    is_random : boolean, default False
        If True, randomly shift.
    channel_index : int
        Index of channel, default 2.
    """
    if is_random:
        factor = np.random.uniform(-intensity, intensity)
    else:
        factor = intensity
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + factor, min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
    # x = np.rollaxis(x, channel_index, 0)
    # min_x, max_x = np.min(x), np.max(x)
    # channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
    #                   for x_channel in x]
    # x = np.stack(channel_images, axis=0)
    # x = np.rollaxis(x, 0, channel_index+1)
    # return x

def channel_shift_multi(x, intensity, channel_index=2):
    """Shift the channels of images with the same arguments, randomly or non-randomly, see `numpy.rollaxis <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html>`_ .
    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.

    Parameters
    -----------
    x : list of numpy array
        List of images with dimension of [n_images, row, col, channel] (default).
    others : see ``channel_shift``.
    """
    if is_random:
        factor = np.random.uniform(-intensity, intensity)
    else:
        factor = intensity

    results = []
    for data in x:
        data = np.rollaxis(data, channel_index, 0)
        min_x, max_x = np.min(data), np.max(data)
        channel_images = [np.clip(x_channel + factor, min_x, max_x)
                          for x_channel in x]
        data = np.stack(channel_images, axis=0)
        data = np.rollaxis(x, 0, channel_index+1)
        results.append( data )
    return np.asarray(results)

# noise
def drop(x, keep=0.5):
    """Randomly set some pixels to zero by a given keeping probability.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] or [row, col].
    keep : float (0, 1)
        The keeping probability, the lower more values will be set to zero.
    """
    if len(x.shape) == 3:
        if x.shape[-1]==3: # color
            img_size = x.shape
            mask = np.random.binomial(n=1, p=keep, size=x.shape[:-1])
            for i in range(3):
                x[:,:,i] = np.multiply(x[:,:,i] , mask)
        elif x.shape[-1]==1: # greyscale image
            img_size = x.shape
            x = np.multiply(x , np.random.binomial(n=1, p=keep, size=img_size))
        else:
            raise Exception("Unsupported shape {}".format(x.shape))
    elif len(x.shape) == 2 or 1: # greyscale matrix (image) or vector
        img_size = x.shape
        x = np.multiply(x , np.random.binomial(n=1, p=keep, size=img_size))
    else:
        raise Exception("Unsupported shape {}".format(x.shape))
    return x

# x = np.asarray([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
# x = np.asarray([x,x,x,x,x,x])
# x.shape = 10, 4, 3
# # print(x)
# # exit()
# print(x.shape)
# # exit()
# print(drop(x, keep=1.))
# exit()

# manual transform
def transform_matrix_offset_center(matrix, x, y):
    """Return transform matrix offset center.

    Parameters
    ----------
    matrix : numpy array
        Transform matrix
    x, y : int
        Size of image.

    Examples
    --------
    - See ``rotation``, ``shear``, ``zoom``.
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0.):
    """Return transformed images by given transform_matrix from ``transform_matrix_offset_center``.

    Parameters
    ----------
    x : numpy array
        Batch of images with dimension of 3, [batch_size, row, col, channel].
    transform_matrix : numpy array
        Transform matrix (offset center), can be generated by ``transform_matrix_offset_center``
    channel_index : int
        Index of channel, default 2.
    fill_mode : string
        Method to fill missing pixel, default ‘nearest’, more options ‘constant’, ‘reflect’ or ‘wrap’

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_
    cval : scalar, optional
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0

        - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_

    Examples
    --------
    - See ``rotation``, ``shift``, ``shear``, ``zoom``.
    """
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def projective_transform_by_points(x, src, dst, map_args={}, output_shape=None, order=1, mode='constant', cval=0.0, clip=True, preserve_range=False):
    """Projective transform by given coordinates, usually 4 coordinates. see `scikit-image <http://scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html>`_.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    src : list or numpy
        The original coordinates, usually 4 coordinates of (x, y).
    dst : list or numpy
        The coordinates after transformation, the number of coordinates is the same with src.
    map_args : dict, optional
        Keyword arguments passed to inverse_map.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input image is preserved. Note that, even for multi-band images, only rows and columns need to be specified.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:

        - 0 Nearest-neighbor
        - 1 Bi-linear (default)
        - 2 Bi-quadratic
        - 3 Bi-cubic
        - 4 Bi-quartic
        - 5 Bi-quintic
    mode : {‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’}, optional
        Points outside the boundaries of the input are filled according to the given mode. Modes match the behaviour of numpy.pad.
    cval : float, optional
        Used in conjunction with mode ‘constant’, the value outside the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image. This is enabled by default, since higher order interpolation may produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.

    Examples
    --------
    >>> Assume X is an image from CIFAR 10, i.e. shape == (32, 32, 3)
    >>> src = [[0,0],[0,32],[32,0],[32,32]]
    >>> dst = [[10,10],[0,32],[32,0],[32,32]]
    >>> x = projective_transform_by_points(X, src, dst)

    References
    -----------
    - `scikit-image : geometric transformations <http://scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html>`_
    - `scikit-image : examples <http://scikit-image.org/docs/dev/auto_examples/index.html>`_
    """
    if type(src) is list:   # convert to numpy
        src = np.array(src)
    if type(dst) is list:
        dst = np.array(dst)
    if np.max(x)>1:         # convert to [0, 1]
        x = x/255

    m = transform.ProjectiveTransform()
    m.estimate(dst, src)
    warped = transform.warp(x, m,  map_args=map_args, output_shape=output_shape, order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)
    return warped

# Numpy and PIL
def array_to_img(x, dim_ordering=(0,1,2), scale=True):
    """Converts a numpy array to PIL image object (uint8 format).

    Parameters
    ----------
    x : numpy array
        A image with dimension of 3 and channels of 1 or 3.
    dim_ordering : list or tuple of 3 int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    scale : boolean, default is True
        If True, converts image to [0, 255] from any range of value like [-1, 2].

    References
    -----------
    - `PIL Image.fromarray <http://pillow.readthedocs.io/en/3.1.x/reference/Image.html?highlight=fromarray>`_
    """
    from PIL import Image
    # if dim_ordering == 'default':
    #     dim_ordering = K.image_dim_ordering()
    # if dim_ordering == 'th':  # theano
    #     x = x.transpose(1, 2, 0)
    x = x.transpose(dim_ordering)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            # print(x_max)
            # x /= x_max
            x = x / x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


## Sequence
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='pre', value=0.):
    """Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).

    Parameters
    ----------
    sequences : list of lists where each element is a sequence
    maxlen : int, maximum length
    dtype : type to cast the resulting sequence.
    padding : 'pre' or 'post', pad either before or after each sequence.
    truncating : 'pre' or 'post', remove values from sequences larger than
        maxlen either in the beginning or in the end of the sequence
    value : float, value to pad the sequences to the desired value.

    Returns
    ----------
    x : numpy array with dimensions (number_of_sequences, maxlen)

    Examples
    ----------
    >>> sequences = [[1,1,1,1,1],[2,2,2],[3,3]]
    >>> sequences = pad_sequences(sequences, maxlen=None, dtype='int32',
    ...                  padding='post', truncating='pre', value=0.)
    ... [[1 1 1 1 1]
    ...  [2 2 2 0 0]
    ...  [3 3 0 0 0]]
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def process_sequences(sequences, end_id=0, pad_val=0, is_shorten=True, remain_end_id=False):
    """Set all tokens(ids) after END token to the padding value, and then shorten (option) it to the maximum sequence length in this batch.

    Parameters
    -----------
    sequences : numpy array or list of list with token IDs.
        e.g. [[4,3,5,3,2,2,2,2], [5,3,9,4,9,2,2,3]]
    end_id : int, the special token for END.
    pad_val : int, replace the end_id and the ids after end_id to this value.
    is_shorten : boolean, default True.
        Shorten the sequences.
    remain_end_id : boolean, default False.
        Keep an end_id in the end.

    Examples
    ---------
    >>> sentences_ids = [[4, 3, 5, 3, 2, 2, 2, 2],  <-- end_id is 2
    ...                  [5, 3, 9, 4, 9, 2, 2, 3]]  <-- end_id is 2
    >>> sentences_ids = precess_sequences(sentences_ids, end_id=vocab.end_id, pad_val=0, is_shorten=True)
    ... [[4, 3, 5, 3, 0], [5, 3, 9, 4, 9]]
    """
    max_length = 0
    for i_s, seq in enumerate(sequences):
        is_end = False
        for i_w, n in enumerate(seq):
            if n == end_id and is_end == False: # 1st time to see end_id
                is_end = True
                if max_length < i_w:
                    max_length = i_w
                if remain_end_id is False:
                    seq[i_w] = pad_val      # set end_id to pad_val
            elif is_end == True:
                seq[i_w] = pad_val

    if remain_end_id is True:
        max_length += 1
    if is_shorten:
        for i, seq in enumerate(sequences):
            sequences[i] = seq[:max_length]
    return sequences

def sequences_add_start_id(sequences, start_id=0, remove_last=False):
    """Add special start token(id) in the beginning of each sequence.

    Examples
    ---------
    >>> sentences_ids = [[4,3,5,3,2,2,2,2], [5,3,9,4,9,2,2,3]]
    >>> sentences_ids = sequences_add_start_id(sentences_ids, start_id=2)
    ... [[2, 4, 3, 5, 3, 2, 2, 2, 2], [2, 5, 3, 9, 4, 9, 2, 2, 3]]
    >>> sentences_ids = sequences_add_start_id(sentences_ids, start_id=2, remove_last=True)
    ... [[2, 4, 3, 5, 3, 2, 2, 2], [2, 5, 3, 9, 4, 9, 2, 2]]

    - For Seq2seq
    >>> input = [a, b, c]
    >>> target = [x, y, z]
    >>> decode_seq = [start_id, a, b] <-- sequences_add_start_id(input, start_id, True)
    """
    sequences_out = [[] for _ in range(len(sequences))]#[[]] * len(sequences)
    for i in range(len(sequences)):
        if remove_last:
            sequences_out[i] = [start_id] + sequences[i][:-1]
        else:
            sequences_out[i] = [start_id] + sequences[i]
    return sequences_out

def sequences_get_mask(sequences, pad_val=0):
    """Return mask for sequences.

    Examples
    ---------
    >>> sentences_ids = [[4, 0, 5, 3, 0, 0],
    ...                  [5, 3, 9, 4, 9, 0]]
    >>> mask = sequences_get_mask(sentences_ids, pad_val=0)
    ... [[1 1 1 1 0 0]
    ...  [1 1 1 1 1 0]]
    """
    mask = np.ones_like(sequences)
    for i, seq in enumerate(sequences):
        for i_w in reversed(range(len(seq))):
            if seq[i_w] == pad_val:
                mask[i, i_w] = 0
            else:
                break   # <-- exit the for loop, prepcess next sequence
    return mask


## Text
# see tensorlayer.nlp


## Tensor Opt
def distorted_images(images=None, height=24, width=24):
    """Distort images for generating more training data.

    Features
    ---------
    They are cropped to height * width pixels randomly.

    They are approximately whitened to make the model insensitive to dynamic range.

    Randomly flip the image from left to right.

    Randomly distort the image brightness.

    Randomly distort the image contrast.

    Whiten (Normalize) the images.

    Parameters
    ----------
    images : 4D Tensor
        The tensor or placeholder of images
    height : int
        The height for random crop.
    width : int
        The width for random crop.

    Returns
    -------
    result : tuple of Tensor
        (Tensor for distorted images, Tensor for while loop index)

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    >>> sess = tf.InteractiveSession()
    >>> batch_size = 128
    >>> x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    >>> distorted_images_op = tl.preprocess.distorted_images(images=x, height=24, width=24)
    >>> sess.run(tf.initialize_all_variables())
    >>> feed_dict={x: X_train[0:batch_size,:,:,:]}
    >>> distorted_images, idx = sess.run(distorted_images_op, feed_dict=feed_dict)
    >>> tl.visualize.images2d(X_train[0:9,:,:,:], second=2, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=20212)
    >>> tl.visualize.images2d(distorted_images[1:10,:,:,:], second=10, saveable=False, name='distorted_images', dtype=None, fig_idx=23012)

    Notes
    ------
    - The first image in 'distorted_images' should be removed.

    References
    -----------
    - `tensorflow.models.image.cifar10.cifar10_input <https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/models/image/cifar10/cifar10_input.py>`_
    """
    print("This function is deprecated, please use tf.map_fn instead, e.g:\n   \
            t_image = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=32. / 255.), t_image)\n \
            t_image = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5), t_image)\n \
            t_image = tf.map_fn(lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5), t_image)\n \
            t_image = tf.map_fn(lambda img: tf.image.random_hue(img, max_delta=0.032), t_image)")
    exit()
    # print(" [Warning] distorted_images will be deprecated due to speed, see TFRecord tutorial for more info...")
    try:
        batch_size = int(images._shape[0])
    except:
        raise Exception('unknow batch_size of images')
    distorted_x = tf.Variable(tf.constant(0.1, shape=[1, height, width, 3]))
    i = tf.Variable(tf.constant(0))

    c = lambda distorted_x, i: tf.less(i, batch_size)

    def body(distorted_x, i):
        # 1. Randomly crop a [height, width] section of the image.
        image = tf.random_crop(tf.gather(images, i), [height, width, 3])
        # 2. Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        # 3. Randomly change brightness.
        image = tf.image.random_brightness(image, max_delta=63)
        # 4. Randomly change contrast.
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_whitening(image)
        # 6. Append the image to a batch.
        image = tf.expand_dims(image, 0)
        return tf.concat(0, [distorted_x, image]), tf.add(i, 1)

    result = tf.while_loop(cond=c, body=body, loop_vars=(distorted_x, i), parallel_iterations=16)
    return result


def crop_central_whiten_images(images=None, height=24, width=24):
    """Crop the central of image, and normailize it for test data.

    They are cropped to central of height * width pixels.

    Whiten (Normalize) the images.

    Parameters
    ----------
    images : 4D Tensor
        The tensor or placeholder of images
    height : int
        The height for central crop.
    width : int
        The width for central crop.

    Returns
    -------
    result : tuple Tensor
        (Tensor for distorted images, Tensor for while loop index)

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    >>> sess = tf.InteractiveSession()
    >>> batch_size = 128
    >>> x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    >>> central_images_op = tl.preprocess.crop_central_whiten_images(images=x, height=24, width=24)
    >>> sess.run(tf.initialize_all_variables())
    >>> feed_dict={x: X_train[0:batch_size,:,:,:]}
    >>> central_images, idx = sess.run(central_images_op, feed_dict=feed_dict)
    >>> tl.visualize.images2d(X_train[0:9,:,:,:], second=2, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=20212)
    >>> tl.visualize.images2d(central_images[1:10,:,:,:], second=10, saveable=False, name='central_images', dtype=None, fig_idx=23012)

    Notes
    ------
    The first image in 'central_images' should be removed.

    Code References
    ----------------
    - ``tensorflow.models.image.cifar10.cifar10_input``
    """
    print("This function is deprecated, please use tf.map_fn instead, e.g:\n   \
            t_image = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=32. / 255.), t_image)\n \
            t_image = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5), t_image)\n \
            t_image = tf.map_fn(lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5), t_image)\n \
            t_image = tf.map_fn(lambda img: tf.image.random_hue(img, max_delta=0.032), t_image)")
    exit()
    # print(" [Warning] crop_central_whiten_images will be deprecated due to speed, see TFRecord tutorial for more info...")
    try:
        batch_size = int(images._shape[0])
    except:
        raise Exception('unknow batch_size of images')
    central_x = tf.Variable(tf.constant(0.1, shape=[1, height, width, 3]))
    i = tf.Variable(tf.constant(0))

    c = lambda central_x, i: tf.less(i, batch_size)

    def body(central_x, i):
        # 1. Crop the central [height, width] of the image.
        image = tf.image.resize_image_with_crop_or_pad(tf.gather(images, i), height, width)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_whitening(image)
        # 5. Append the image to a batch.
        image = tf.expand_dims(image, 0)
        return tf.concat(0, [central_x, image]), tf.add(i, 1)

    result = tf.while_loop(cond=c, body=body, loop_vars=(central_x, i), parallel_iterations=16)
    return result












#
