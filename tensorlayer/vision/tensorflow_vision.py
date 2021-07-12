#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops, random_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.image_ops_impl import _AssertAtLeast3DImage
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.image_ops_impl import convert_image_dtype
import numbers
import PIL
from PIL import Image
import math
import scipy
from scipy import ndimage
__all__ = [
    'central_crop',
    'to_tensor',
    'crop',
    'pad',
    'resize',
    'transpose',
    'hwc_to_chw',
    'chw_to_hwc',
    'rgb_to_hsv',
    'hsv_to_rgb',
    'rgb_to_gray',
    'adjust_brightness',
    'adjust_contrast',
    'adjust_hue',
    'adjust_saturation',
    'normalize',
    'hflip',
    'vflip',
    'padtoboundingbox',
    'standardize',
    'random_brightness',
    'random_contrast',
    'random_saturation',
    'random_hue',
    'random_crop',
    'random_resized_crop',
    'random_vflip',
    'random_hflip',
    'random_rotation',
    'random_shear',
    'random_shift',
    'random_zoom',
    'random_affine',
]


def _is_pil_image(image):
    return isinstance(image, Image.Image)


def _is_numpy_image(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})


def _get_image_size(image):
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
        height, width, channels = image_shape
        return height, width
    elif image_shape.ndims == 4:
        batch, height, width, channels = image_shape
        return height, width


def random_factor(factor, name, center=1, bound=(0, float('inf')), non_negative=True):
    if isinstance(factor, numbers.Number):
        if factor < 0:
            raise ValueError('The input value of {} cannot be negative.'.format(name))
        factor = [center - factor, center + factor]
        if non_negative:
            factor[0] = max(0, factor[0])
    elif isinstance(factor, (tuple, list)) and len(factor) == 2:
        if not bound[0] <= factor[0] <= factor[1] <= bound[1]:
            raise ValueError(
                "Please check your value range of {} is valid and "
                "within the bound {}.".format(name, bound)
            )
    else:
        raise TypeError("Input of {} should be either a single value, or a list/tuple of " "length 2.".format(name))
    factor = np.random.uniform(factor[0], factor[1])
    return factor


def central_crop(image, size=None, central_fraction=None):
    '''

	Parameters
	----------
	image :
		input Either a 3-D float Tensor of shape [height, width, depth],
		or a 4-D Tensor of shape [batch_size, height, width, depth].
	central_fraction :
		float (0, 1], fraction of size to crop
	size:
		size (Union[int, sequence]) – The output size of the cropped image. If size is an integer, a square crop of size (size, size) is returned.
		If size is a sequence of length 2, it should be (height, width).
	Returns :
		3-D / 4-D float Tensor, as per the input.
	-------

	'''
    if size is None and central_fraction is None:
        raise ValueError('central_fraction and size can not be both None')

    if size is not None:
        if not isinstance(size, (int, list, tuple)) or (isinstance(size, (list, tuple)) and len(size) != 2):
            raise ValueError(
                "Size should be a single integer or a list/tuple (h, w) of length 2.But"
                "got {}.".format(type(size))
            )
        if isinstance(size, int):
            target_height = size
            target_width = size
        else:
            target_height = size[0]
            target_width = size[1]
        image = ops.convert_to_tensor(image, name='image')
        rank = image.get_shape().ndims
        if rank != 3 and rank != 4:
            raise ValueError(
                '`image` should either be a Tensor with rank = 3 or '
                'rank = 4. Had rank = {}.'.format(rank)
            )

        def _get_dim(tensor, idx):
            static_shape = tensor.get_shape().dims[idx].value
            if static_shape is not None:
                return static_shape, False
            return array_ops.shape(tensor)[idx], True

        if rank == 3:
            img_h, dynamic_h = _get_dim(image, 0)
            img_w, dynamic_w = _get_dim(image, 1)
            img_d = image.get_shape()[2]
        else:
            img_bs = image.get_shape()[0]
            img_h, dynamic_h = _get_dim(image, 1)
            img_w, dynamic_w = _get_dim(image, 2)
            img_d = image.get_shape()[3]

        bbox_h_size = target_height
        bbox_w_size = target_width

        if dynamic_h:
            img_hd = math_ops.cast(img_h, dtypes.float64)
            target_height = math_ops.cast(target_height, dtypes.float64)
            bbox_h_start = math_ops.cast((img_hd - target_height) / 2, dtypes.int32)
        else:
            img_hd = float(img_h)
            target_height = float(target_height)
            bbox_h_start = int((img_hd - target_height) / 2)

        if dynamic_w:
            img_wd = math_ops.cast(img_w, dtypes.float64)
            target_width = math_ops.cast(target_width, dtypes.float64)
            bbox_w_start = math_ops.cast((img_wd - target_width) / 2, dtypes.int32)
        else:
            img_wd = float(img_w)
            target_width = float(target_width)
            bbox_w_start = int((img_wd - target_width) / 2)

        if rank == 3:
            bbox_begin = array_ops.stack([bbox_h_start, bbox_w_start, 0])
            bbox_size = array_ops.stack([bbox_h_size, bbox_w_size, -1])
        else:
            bbox_begin = array_ops.stack([0, bbox_h_start, bbox_w_start, 0])
            bbox_size = array_ops.stack([-1, bbox_h_size, bbox_w_size, -1])

        image = array_ops.slice(image, bbox_begin, bbox_size)

        if rank == 3:
            image.set_shape([None if dynamic_h else bbox_h_size, None if dynamic_w else bbox_w_size, img_d])
        else:
            image.set_shape([img_bs, None if dynamic_h else bbox_h_size, None if dynamic_w else bbox_w_size, img_d])
        return image

    elif central_fraction is not None:
        return tf.image.central_crop(image, central_fraction)


def to_tensor(img, data_format):
    '''Converts a ``image`` to tf.Tensor.
    
    Parameters
    ----------
    img:
        Image to be converted to tensor.
    data_format:
        Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'HWC'.

    Returns:
        Tensor: Converted image.
    -------

    '''
    if not (_is_pil_image(img) or _is_numpy_image(img)):
        raise TypeError('img should be PIL Image or ndarray. But got {}'.format(type(img)))

    if _is_pil_image(img):
        # PIL Image
        if img.mode == 'I':
            image = tf.convert_to_tensor(np.array(img, np.int32, copy=False))
        elif img.mode == 'I;16':
            # cast and reshape not support int16
            image = tf.convert_to_tensor(np.array(img, np.int32, copy=False))
        elif img.mode == 'F':
            image = tf.convert_to_tensor(np.array(img, np.float32, copy=False))
        elif img.mode == '1':
            image = 255 * tf.convert_to_tensor(np.array(img, np.uint8, copy=False))
        else:
            image = tf.convert_to_tensor(np.array(img, copy=False))

        if img.mode == 'YCbCr':
            nchannel = 3
        elif img.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(img.mode)

        dtype = image.dtype
        if dtype == 'tf.uint8':
            image = tf.cast(image, tf.float32) / 255.

        image = tf.reshape(image, shape=[img.size[1], img.size[0], nchannel])
        if data_format == 'CHW':
            image = tf.transpose(image, perm=[2, 0, 1])
        return image
    else:
        if img.ndim == 2:
            img = img[:, :, None]

        if data_format == 'CHW':
            img = tf.convert_to_tensor(img.transpose((2, 0, 1)))
        else:
            img = tf.convert_to_tensor(img)

        dtype = img.dtype
        if dtype == 'tf.uint8':
            img = tf.cast(img, tf.float32) / 255.
        return img


def crop(image, offset_height, offset_width, target_height, target_width):

    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)


def pad(image, padding, padding_value, mode):
    '''

    Parameters
    ----------
    image:
        A 3-D or 4-D Tensor.
    padding:
        An integer or a list/tuple.  If a single number is provided, pad all borders with this value.
        If a tuple or list of 2 values is provided, pad the left and right with the first value and the top and bottom with the second value.
        If 4 values are provided as a list or tuple, pad the  (left , top, right, bottom)  respectively.
    padding_value:
        In "CONSTANT" mode, the scalar pad value to use. Must be same type as tensor.
    mode:
        One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    Returns:
        A padded Tensor. Has the same type as tensor.
    -------

    '''
    image = ops.convert_to_tensor(image, name='image')
    image_shape = image.get_shape()
    if len(image_shape) == 3:
        batch_size = 0
    elif len(image_shape) == 4:
        batch_size = image_shape[0]
    else:
        raise TypeError('Image must  be a 3-D tensor or 4-D tensor.')

    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding))
    elif isinstance(padding, list) or isinstance(padding, tuple):
        if len(padding) == 2:
            padding = ((padding[1], padding[1]), (padding[0], padding[0]))
        elif len(padding) == 4:
            padding = ((padding[1], padding[3]), (padding[0], padding[2]))
        else:
            raise ValueError('The length of padding should be 2 or 4, but got {}.'.format(len(padding)))
    else:
        raise TypeError('Padding should be an integer or a list/tuple, but got {}.'.format(type(padding)))

    if batch_size == 0:
        padding = (padding[0], padding[1], (0, 0))
    else:
        padding = ((0, 0), padding[0], padding[1], (0, 0))

    return tf.pad(image, padding, mode=mode, constant_values=padding_value)


def resize(image, size, method):
    '''

    Parameters
    ----------
    images:
        Input images to resize
    size:
        The output size of the resized image.
        If size is an integer, smaller edge of the image will be resized to this value with
        the same image aspect ratio.
        If size is a sequence of (height, width), this will be the desired output size.
    method:
        An image.ResizeMethod, or string equivalent shoulid be in
        (bilinear, lanczos3, lanczos5, bicubic, gaussian, nearest, area, mitchellcubic).
        Defaults to bilinear.
    preserve_aspect_ratio:
        Whether to preserve the aspect ratio.
    Returns:
        resized images
    -------

    '''
    if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 2)):
        raise TypeError('Size should be a single number or a list/tuple (h, w) of length 2.' 'Got {}.'.format(size))
    image = ops.convert_to_tensor(image)
    orig_dtype = image.dtype
    if orig_dtype not in [dtypes.float16, dtypes.float32]:
        image = convert_image_dtype(image, dtypes.float32)

    if image.get_shape().ndims == 3:
        h, w, _ = image.get_shape().as_list()
    elif image.get_shape().ndims == 4:
        _, h, w, _ = image.get_shape().as_list()

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            size = (h, w)
        if w < h:
            target_w = size
            target_h = int(size * h / w)
            size = (target_h, target_w)
        else:
            target_h = size
            target_w = int(size * w / h)
            size = (target_h, target_w)
    image = tf.image.resize(image, size, method, preserve_aspect_ratio=False)
    return convert_image_dtype(image, orig_dtype, saturate=True)


def transpose(image, order):
    image = ops.convert_to_tensor(image)
    shape = image.get_shape()
    if shape.ndims == 3 or shape.ndims is None:
        if len(order) != 3:
            raise ValueError('if image is 3-D tensor, order should be a list/tuple with length of 3')
        return array_ops.transpose(image, order)
    elif shape.ndims == 4:
        if len(order) != 4:
            raise ValueError('if image is 4-D tensor, order should be a list/tuple with length of 4')
        return array_ops.transpose(image, order)
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')


def hwc_to_chw(image):

    if (len(image.shape) == 3):
        return transpose(image, (2, 0, 1))
    elif (len(image.shape) == 4):
        return transpose(image, (0, 3, 1, 2))
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')


def chw_to_hwc(image):

    if (len(image.shape) == 3):
        return transpose(image, (1, 2, 0))
    elif (len(image.shape) == 4):
        return transpose(image, (0, 2, 3, 1))
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')


def rgb_to_hsv(image):

    return tf.image.rgb_to_hsv(image)


def hsv_to_rgb(image):

    return tf.image.hsv_to_rgb(image)


def rgb_to_gray(image, num_output_channels):

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    image = ops.convert_to_tensor(image, name='image')
    orig_dtype = image.dtype
    flt_image = convert_image_dtype(image, dtypes.float32)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray_float = math_ops.tensordot(flt_image, rgb_weights, [-1, -1])
    gray_float = array_ops.expand_dims(gray_float, -1)
    if num_output_channels == 3:
        gray_float = array_ops.stack([gray_float, gray_float, gray_float], axis=2)
    return convert_image_dtype(gray_float, orig_dtype)


def adjust_brightness(image, brightness_factor):
    '''
    Parameters
    ----------
    images:
        Input images to adjust brightness
    brightness_factor(float): How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        adjusted images
    -------
    '''
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)

    orig_dtype = image.dtype
    if orig_dtype not in [dtypes.float16, dtypes.float32]:
        image = convert_image_dtype(image, dtypes.float32)

    brightness_factor = math_ops.cast(brightness_factor, image.dtype)
    image_zeros = tf.zeros_like(image)
    adjusted = brightness_factor * image + (1.0 - brightness_factor) * image_zeros
    adjusted = tf.clip_by_value(adjusted, clip_value_min=0, clip_value_max=1.0)
    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


def adjust_contrast(image, contrast_factor):
    '''
    Parameters
    ----------
    images:
        Input images to adjust contrast
    contrast_factor(float): How much to adjust the contrast. Can be
            any non negative number. 0 gives a gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        adjusted images
    -------
    '''
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)

    orig_dtype = image.dtype
    if orig_dtype not in [dtypes.float16, dtypes.float32]:
        image = convert_image_dtype(image, dtypes.float32)

    contrast_factor = math_ops.cast(contrast_factor, image.dtype)
    mean = tf.math.reduce_mean(tf.image.rgb_to_grayscale(image), keepdims=True)
    adjusted = contrast_factor * image + (1 - contrast_factor) * mean
    adjusted = tf.clip_by_value(adjusted, clip_value_min=0, clip_value_max=1.0)
    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


def adjust_hue(image, hue_factor):
    '''
    Parameters
    ----------
    images(Tensor):
        Input images to adjust hue
    hue_factor(float): How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns(Tensor):
        Adjusted images
    -------
    '''
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)

    orig_dtype = image.dtype
    if orig_dtype not in [dtypes.float16, dtypes.float32]:
        image = convert_image_dtype(image, dtypes.float32)

    hue_factor = math_ops.cast(hue_factor, image.dtype)
    image = tf.image.rgb_to_hsv(image)
    h, s, v = tf.split(image, num_or_size_splits=[1, 1, 1], axis=2)
    h = (h + hue_factor) % 1.0
    image = tf.concat((h, s, v), axis=2)
    adjusted = tf.image.hsv_to_rgb(image)

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


def adjust_saturation(image, saturation_factor):
    '''
    Parameters
    ----------
    images(Tensor):
        Input images to adjust saturation
    contrast_factor(float): How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns(Tensor):
        Adjusted images
    -------
    '''
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)

    orig_dtype = image.dtype
    if orig_dtype not in [dtypes.float16, dtypes.float32]:
        image = convert_image_dtype(image, dtypes.float32)

    saturation_factor = math_ops.cast(saturation_factor, image.dtype)
    gray_image = tf.image.rgb_to_grayscale(image)
    adjusted = saturation_factor * image + (1 - saturation_factor) * gray_image
    adjusted = tf.clip_by_value(adjusted, clip_value_min=0, clip_value_max=1.0)
    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


def hflip(image):
    '''

    Parameters
    ----------
    image(Tensor):
        Input images to flip an image horizontally (left to right)

    Returns(Tensor):
        Flipped images
    -------

    '''
    return tf.image.flip_left_right(image)


def vflip(image):
    '''

    Parameters
    ----------
    image(Tensor):
        Input images to flip an image vertically (up to down)

    Returns(Tensor):
        Flipped images
    -------

    '''
    return tf.image.flip_up_down(image)


def padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value):
    '''

    Parameters
    ----------
    image:
        4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    offset_height:
        Number of rows of padding_values to add on top.
    offset_width:
        Number of columns of padding_values to add on the left.
    target_height:
        Height of output image.
    target_width:
        Width of output image.
    padding_value:
        value to pad

    Returns:
        If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`
    -------

    '''
    image = ops.convert_to_tensor(image, name='image')

    if offset_height < 0:
        raise ValueError('offset_height must be >= 0')
    if offset_width < 0:
        raise ValueError('offset_width must be >= 0')

    image_shape = image.get_shape()
    if image_shape.ndims == 3:
        height, width, channels = image.get_shape()
    elif image_shape.ndims == 4:
        batch, height, width, channels = image.get_shape()
    else:
        raise ValueError('\'image\' (shape %s) must have either 3 or 4 dimensions.' % image_shape)

    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    if after_padding_height < 0:
        raise ValueError('image height must be <= target - offset')
    if after_padding_width < 0:
        raise ValueError('image width must be <= target - offset')

    return pad(
        image, padding=(offset_width, offset_height, after_padding_width, after_padding_height),
        padding_value=padding_value, mode='constant'
    )


def normalize(image, mean, std, data_format):
    '''
    Parameters
    ----------
    image:
        An n-D Tensor with at least 3 dimensions, the last 3 of which are the dimensions of each image.
    mean:
        List or tuple of mean values for each channel, with respect to channel order.
    std:
         List or tuple of standard deviations for each channel.
    channel_mode:
        Decide to implement standardization on whole image or each channel of image.
    Returns:
        A Tensor with the same shape and dtype as image.
    -------
    '''
    image = ops.convert_to_tensor(image, name='image')
    image = math_ops.cast(image, dtype=tf.float32)
    image = _AssertAtLeast3DImage(image)

    if data_format == 'CHW':
        num_channels = image.shape[0]
    elif data_format == 'HWC':
        num_channels = image.shape[2]

    if isinstance(mean, numbers.Number):
        mean = (mean, ) * num_channels
    elif isinstance(mean, (list, tuple)):
        if len(mean) != num_channels:
            raise ValueError("Length of mean must be 1 or equal to the number of channels({0}).".format(num_channels))
    if isinstance(std, numbers.Number):
        std = (std, ) * num_channels
    elif isinstance(std, (list, tuple)):
        if len(std) != num_channels:
            raise ValueError("Length of std must be 1 or equal to the number of channels({0}).".format(num_channels))

    if data_format == 'CHW':
        std = np.float32(np.array(std).reshape((-1, 1, 1)))
        mean = np.float32(np.array(mean).reshape((-1, 1, 1)))
    elif data_format == 'HWC':
        mean = np.float32(np.array(mean).reshape((1, 1, -1)))
        std = np.float32(np.array(std).reshape((1, 1, -1)))

    mean = ops.convert_to_tensor(mean)
    mean = math_ops.cast(mean, dtype=tf.float32)
    std = ops.convert_to_tensor(std)
    std = math_ops.cast(std, dtype=tf.float32)
    image -= mean
    image = math_ops.divide(image, std)
    return image

def standardize(image):
    '''
    Reference to tf.image.per_image_standardization().
    Linearly scales each image in image to have mean 0 and variance 1.

    Parameters
    ----------
    image:
        An n-D Tensor with at least 3 dimensions, the last 3 of which are the dimensions of each image.

    Returns:
        A Tensor with the same shape as image and its dtype is float32.
    -------

    '''
    image = ops.convert_to_tensor(image, name='image')
    image = math_ops.cast(image, dtype=tf.float32)
    return tf.image.per_image_standardization(image)


def random_brightness(image, brightness_factor):
    '''
    Perform a random brightness on the input image.
    Parameters
    ----------
    image:
        Input images to adjust random brightness
    brightness_factor:
        Brightness adjustment factor (default=(1, 1)). Cannot be negative.
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
        If it is a sequence, it should be [min, max] for the range.

    Returns:
        Adjusted image.
    -------

    '''
    brightness_factor = random_factor(brightness_factor, name='brightness')

    return adjust_brightness(image, brightness_factor)


def random_contrast(image, contrast_factor):
    '''
    Perform a random contrast on the input image.
    Parameters
    ----------
    image:
        Input images to adjust random contrast
    contrast_factor:
        Contrast adjustment factor (default=(1, 1)). Cannot be negative.
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
        If it is a sequence, it should be [min, max] for the range.

    Returns:
        Adjusted image.
    -------

    '''
    contrast_factor = random_factor(contrast_factor, name='contrast')

    return adjust_contrast(image, contrast_factor)


def random_saturation(image, saturation_factor):
    '''
     Perform a random saturation on the input image.
    Parameters
    ----------
    image:
        Input images to adjust random saturation
    saturation_factor:
        Saturation adjustment factor (default=(1, 1)). Cannot be negative.
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
        If it is a sequence, it should be [min, max] for the range.

    Returns:
        Adjusted image.
    -------

    '''

    saturation_factor = random_factor(saturation_factor, name='saturation')

    return adjust_saturation(image, saturation_factor)


def random_hue(image, hue_factor):
    '''
     Perform a random contrast on the input image.
    Parameters
    ----------
    image:
        Input images to adjust random contrast
    brightness_factor:
        Contrast adjustment factor (default=(1, 1)). Cannot be negative.
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
        If it is a sequence, it should be [min, max] for the range.

    Returns:
        Adjusted image.
    -------

    '''
    hue_factor = random_factor(hue_factor, name='hue', center=0, bound=(-0.5, 0.5), non_negative=False)

    return adjust_hue(image, hue_factor)


def random_crop(image, size, padding, pad_if_needed, fill, padding_mode):
    '''

    Parameters
    ----------
    image:
        Input images to crop and pad if needed.
    size:
        Desired output size of the crop. If size is an int instead of sequence like (h, w),
        a square crop (size, size) is made. If provided a sequence of length 1,
        it will be interpreted as (size[0], size[0]).
    padding:
        Optional, padding on each border of the image. Default is None.
        If a single int is provided this is used to pad all borders.
        If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
        If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
    pad_if_needed:
        It will pad the image if smaller than the desired size to avoid raising an exception.
        Since cropping is done after padding, the padding seems to be done at a random offset.
    fill:
        Pixel fill value for constant fill. Default is 0.
    padding_mode:
        Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

    Returns:
        cropped images.
    -------

    '''
    image = ops.convert_to_tensor(image, name='image')
    _AssertAtLeast3DImage(image)

    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
        raise ValueError('Size should be a int or a list/tuple with length of 2. ' 'But got {}'.format(size))

    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name='size')
    if padding is not None:
        image = pad(image, padding, fill, padding_mode)

    image_shape = image.get_shape()
    if image_shape.ndims == 3:
        height, width, channels = image_shape
    elif image_shape.ndims == 4:
        batch, height, width, channels = image_shape

    if pad_if_needed and height < size[0]:
        image = pad(image, (0, size[0] - height), fill, padding_mode)
    if pad_if_needed and width < size[1]:
        image = pad(image, (size[1] - width, 0), fill, padding_mode)

    image_shape = image.get_shape()
    if image_shape.ndims == 3:
        height, width, channels = image_shape
    elif image_shape.ndims == 4:
        batch, height, width, channels = image_shape

    target_height, target_width = size
    if height < target_height or width < target_width:
        raise ValueError(
            'Crop size {} should be smaller than input image size {}. '.format(
                (target_height, target_width), (height, width)
            )
        )

    if target_height == height and target_width == width:
        return crop(image, 0, 0, target_height, target_width)

    offset_height = random_ops.random_uniform([], minval=0, maxval=height - target_height + 1, dtype=size.dtype)

    offset_width = random_ops.random_uniform([], minval=0, maxval=width - target_width + 1, dtype=size.dtype)

    return crop(image, offset_height, offset_width, target_height, target_width)


def random_resized_crop(image, size, scale, ratio, interpolation):
    '''Crop the given image to random size and aspect ratio.

    Parameters
    ----------
    image:
        4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    size:
        Target size of output image, with (height, width) shape. if size is int, target size will be (size, size).
    scale:
        Range of size of the origin size cropped. Default: (0.08, 1.0)
    ratio:
        Range of aspect ratio of the origin aspect ratio cropped. Default: (0.75, 1.33)
    interpolation:
        Interpolation method. Default: 'bilinear'.

    Returns:
        Randomly cropped and resized image.
    -------

    '''

    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        size = size
    else:
        raise TypeError('Size should be a int or a list/tuple with length of 2.' 'But got {}.'.format(size))
    if not (isinstance(scale, (list, tuple)) and len(scale) == 2):
        raise TypeError('Scale should be a list/tuple with length of 2.' 'But got {}.'.format(scale))
    if not (isinstance(ratio, (list, tuple)) and len(ratio) == 2):
        raise TypeError('Scale should be a list/tuple with length of 2.' 'But got {}.'.format(ratio))

    if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
        raise ValueError("Scale and ratio should be of kind (min, max)")
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)

    def get_param(image, scale, ratio):
        height, width = _get_image_size(image)
        area = math_ops.cast(height * width, dtype=dtypes.float32)
        ratio = ops.convert_to_tensor(ratio, dtype=dtypes.float32)
        log_ratio = math_ops.log(ratio)
        for _ in range(10):
            target_area = area * random_ops.random_uniform([], minval=scale[0], maxval=scale[1], dtype=dtypes.float32)
            aspect_ratio = math_ops.exp(
                random_ops.random_uniform([], minval=log_ratio[0], maxval=log_ratio[1], dtype=dtypes.float32)
            )

            target_width = math_ops.to_int32(math_ops.round(math_ops.sqrt(target_area * aspect_ratio)))

            target_height = math_ops.to_int32(math_ops.round(math_ops.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                offset_height = random_ops.random_uniform(
                    [], minval=0, maxval=height - target_height + 1, dtype=dtypes.int32
                )

                offset_width = random_ops.random_uniform(
                    [], minval=0, maxval=width - target_width + 1, dtype=dtypes.int32
                )

                return offset_height, offset_width, target_height, target_width

        height = ops.convert_to_tensor(height, dtype=dtypes.float32)
        width = ops.convert_to_tensor(width, dtype=dtypes.float32)
        in_ratio = width / height
        if in_ratio < ratio[0]:
            target_width = width
            target_height = math_ops.to_int32(math_ops.round(target_width / ratio[0]))
        elif in_ratio > ratio[1]:
            target_height = height
            target_width = math_ops.to_int32(math_ops.round(target_height / ratio[1]))
        else:
            target_height = height
            target_width = width
        offset_height = (height - target_height) // 2
        offset_width = (width - target_width) // 2
        return offset_height, offset_width, target_height, target_width

    offset_height, offset_width, target_heigth, target_width = get_param(image, scale, ratio)
    image = crop(image, offset_height, offset_width, target_heigth, target_width)
    image = resize(image, size, interpolation)
    return image


def random_vflip(image, prob):
    '''Vertically flip the input image randomly with a given probability.

    Parameters
    ----------
    image:
        4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    prob:
        probability of the image being flipped. Default value is 0.5
    Returns:
        A tensor of the same type and shape as image.
    -------

    '''
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)
    random_prob = random_ops.random_uniform([], minval=0, maxval=1.0, dtype=dtypes.float32)
    flip_flag = math_ops.less(random_prob, prob)
    if flip_flag:
        return vflip(image)
    return image


def random_hflip(image, prob):
    '''horizontally flip the input image randomly with a given probability.

    Parameters
    ----------
    image:
        4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    prob:
        probability of the image being flipped. Default value is 0.5
    Returns:
        A tensor of the same type and shape as image.
    -------

    '''
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)
    random_prob = random_ops.random_uniform([], minval=0, maxval=1.0, dtype=dtypes.float32)
    flip_flag = math_ops.less(random_prob, prob)
    if flip_flag:
        return hflip(image)
    return image


def random_rotation(image, degrees, interpolation, expand, center, fill):
    '''Rotate the image by angle.

    Parameters
    ----------
    image:
        Input tensor. Must be 3D.
    degrees:
        Range of degrees to select from.If degrees is a number instead of sequence like (min, max), the range of degrees
    will be (-degrees, +degrees).
    interpolation:
        Points outside the boundaries of the input are filled according to the given mode
        (one of {'nearest', 'bilinear'}).
    expand:
        Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
    center:
        Optional center of rotation, (x, y). Origin is the upper left corner.
        Default is the center of the image.
    fill:
        Pixel fill value for the area outside the rotated image.
        Default is ``0``. If given a number, the value is used for all bands respectively.

    Returns:
        Rotated image tensor.
    -------

    '''
    if isinstance(image, (tf.Tensor, np.ndarray)) and len(image.shape) == 3:
        image = np.asarray(image)
    else:
        'Image should be a 3d tensor or np.ndarray.'
    h, w, c = image.shape[0], image.shape[1], image.shape[2]

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError('If degrees is a single number, it must be positive.' 'But got {}'.format(degrees))
        degrees = (-degrees, degrees)
    elif not (isinstance(degrees, (list, tuple)) and len(degrees) == 2):
        raise ValueError('If degrees is a list/tuple, it must be length of 2.' 'But got {}'.format(degrees))
    else:
        if degrees[0] > degrees[1]:
            raise ValueError('if degrees is a list/tuple, it should be (min, max).')

    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    if interpolation not in ('nearest', 'bilinear'):
        raise ValueError('Interpolation only support {\'nearest\', \'bilinear\'} .')

    orig_dtype = image.dtype
    image = np.asarray(image, dtype=np.float)
    theta = np.random.uniform(degrees[0], degrees[1])
    angle = -math.radians(theta)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    if center is None:
        rotn_center = (w / 2.0, h / 2.0)
    else:
        rotn_center = center

    matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]

    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = transform(-rotn_center[0] - 0, -rotn_center[1] - 0, matrix)
    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]

    if expand:
        # calculate output size
        xx = []
        yy = []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = transform(x, y, matrix)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))
        matrix[2], matrix[5] = transform(-(nw - w) / 2.0, -(nh - h) / 2.0, matrix)
        w, h = nw, nh

    image = np.rollaxis(image, 2, 0)
    dummy = np.ones((1, image.shape[1], image.shape[2]), dtype=image.dtype)
    image = np.concatenate((image, dummy), axis=0)
    final_offset = np.array([matrix[5], matrix[2]])

    channel_images = [
        ndimage.interpolation.affine_transform(
            x_channel, rotation_matrix, final_offset, output_shape=(h, w), order=3, mode='constant', cval=0
        ) for x_channel in image
    ]
    image = np.stack(channel_images, axis=0)
    image = np.rollaxis(image, 0, 3)
    mask = image[:, :, -1:]
    image = image[:, :, :-1]
    mask = np.tile(mask, (1, 1, image.shape[2]))
    fill = np.tile(fill, (image.shape[0], image.shape[1], 1))
    if interpolation == 'nearest':
        mask = mask < 0.5
        image[mask] = fill[mask]
    else:
        image = image * mask + (1.0 - mask) * fill
    image = np.asarray(image, dtype=orig_dtype)
    image = ops.convert_to_tensor(image)
    return image


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def random_shear(image, degrees, interpolation, fill):

    if isinstance(image, (tf.Tensor, np.ndarray)) and len(image.shape) == 3:
        image = np.asarray(image)
    else:
        'Image should be a 3d tensor or np.ndarray.'
    h, w, c = image.shape[0], image.shape[1], image.shape[2]

    if interpolation not in ('nearest', 'bilinear'):
        raise ValueError('Interpolation only support {\'nearest\', \'bilinear\'} .')

    if isinstance(degrees, numbers.Number):
        degrees = (-degrees, degrees, 0, 0)
    elif isinstance(degrees, (list, tuple)) and (len(degrees) == 2 or len(degrees) == 4):
        if len(degrees) == 2:
            degrees = (degrees[0], degrees[1], 0, 0)
    else:
        raise ValueError(
            'degrees should be a single number or a list/tuple with length in (2 ,4).'
            'But got {}'.format(degrees)
        )

    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    orig_dtype = image.dtype
    image = np.asarray(image, dtype=np.float)
    shear = [np.random.uniform(degrees[0], degrees[1]), np.random.uniform(degrees[2], degrees[3])]
    shear = np.deg2rad(shear)
    shear_matrix = np.array(
        [[math.cos(shear[1]), math.sin(shear[1]), 0], [math.sin(shear[0]), math.cos(shear[0]), 0], [0, 0, 1]]
    )
    transform_matrix = shear_matrix
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

    shear_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    image = np.rollaxis(image, 2, 0)

    dummy = np.ones((1, image.shape[1], image.shape[2]), dtype=image.dtype)
    image = np.concatenate((image, dummy), axis=0)

    channel_images = [
        ndimage.interpolation.affine_transform(x_channel, shear_matrix, offset, order=3, mode='constant', cval=0)
        for x_channel in image
    ]

    image = np.stack(channel_images, axis=0)
    image = np.rollaxis(image, 0, 3)
    mask = image[:, :, -1:]
    image = image[:, :, :-1]
    mask = np.tile(mask, (1, 1, c))
    fill = np.tile(fill, (h, w, 1))
    if interpolation == 'nearest':
        mask = mask < 0.5
        image[mask] = fill[mask]
    else:
        image = image * mask + (1.0 - mask) * fill
    image = np.asarray(image, dtype=orig_dtype)
    image = ops.convert_to_tensor(image)
    return image


def random_shift(image, shift, interpolation, fill):

    if isinstance(image, (tf.Tensor, np.ndarray)) and len(image.shape) == 3:
        image = np.asarray(image)
    else:
        'Image should be a 3d tensor or np.ndarray.'
    h, w, c = image.shape[0], image.shape[1], image.shape[2]

    if interpolation not in ('nearest', 'bilinear'):
        raise ValueError('Interpolation only support {\'nearest\', \'bilinear\'} .')

    if not (isinstance(shift, (tuple, list)) and len(shift) == 2):

        raise ValueError('Shift should be a list/tuple with length of 2.' 'But got {}'.format(shift))

    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    orig_dtype = image.dtype
    image = np.asarray(image, dtype=np.float)
    hrg = shift[0]
    wrg = shift[1]
    tx = -np.random.uniform(-hrg, hrg) * w
    ty = -np.random.uniform(-wrg, wrg) * h

    shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    transform_matrix = transform_matrix_offset_center(shift_matrix, h, w)
    shift_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    image = np.rollaxis(image, 2, 0)

    dummy = np.ones((1, image.shape[1], image.shape[2]), dtype=image.dtype)
    image = np.concatenate((image, dummy), axis=0)

    channel_images = [
        ndimage.interpolation.affine_transform(x_channel, shift_matrix, offset, order=3, mode='constant', cval=0)
        for x_channel in image
    ]

    image = np.stack(channel_images, axis=0)
    image = np.rollaxis(image, 0, 3)
    mask = image[:, :, -1:]
    image = image[:, :, :-1]
    mask = np.tile(mask, (1, 1, c))
    fill = np.tile(fill, (h, w, 1))
    if interpolation == 'nearest':
        mask = mask < 0.5
        image[mask] = fill[mask]
    else:
        image = image * mask + (1.0 - mask) * fill
    image = np.asarray(image, dtype=orig_dtype)
    image = ops.convert_to_tensor(image)
    return image


def random_zoom(image, zoom, interpolation, fill):

    if isinstance(image, (tf.Tensor, np.ndarray)) and len(image.shape) == 3:
        image = np.asarray(image)
    else:
        'Image should be a 3d tensor or np.ndarray.'
    h, w, c = image.shape[0], image.shape[1], image.shape[2]

    if interpolation not in ('nearest', 'bilinear'):
        raise ValueError('Interpolation only support {\'nearest\', \'bilinear\'} .')

    if not (isinstance(zoom, (tuple, list)) and len(zoom) == 2):

        raise ValueError('Zoom should be a list/tuple with length of 2.' 'But got {}'.format(zoom))
    if not (0 <= zoom[0] <= zoom[1]):

        raise ValueError('Zoom values should be positive, and zoom[1] should be greater than zoom[0].')

    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    orig_dtype = image.dtype
    image = np.asarray(image, dtype=np.float)
    zoom_factor = 1 / np.random.uniform(zoom[0], zoom[1])
    zoom_matrix = np.array([[zoom_factor, 0, 0], [0, zoom_factor, 0], [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    zoom_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]

    image = np.rollaxis(image, 2, 0)

    dummy = np.ones((1, image.shape[1], image.shape[2]), dtype=image.dtype)
    image = np.concatenate((image, dummy), axis=0)

    channel_images = [
        ndimage.interpolation.affine_transform(x_channel, zoom_matrix, offset, order=3, mode='constant', cval=0)
        for x_channel in image
    ]

    image = np.stack(channel_images, axis=0)
    image = np.rollaxis(image, 0, 3)
    mask = image[:, :, -1:]
    image = image[:, :, :-1]
    mask = np.tile(mask, (1, 1, c))
    fill = np.tile(fill, (h, w, 1))
    if interpolation == 'nearest':
        mask = mask < 0.5
        image[mask] = fill[mask]
    else:
        image = image * mask + (1.0 - mask) * fill
    image = np.asarray(image, dtype=orig_dtype)
    image = ops.convert_to_tensor(image)
    return image


def random_affine(image, degrees, shift, zoom, shear, interpolation, fill):

    if isinstance(image, (tf.Tensor, np.ndarray)) and len(image.shape) == 3:
        image = np.asarray(image)
    else:
        'Image should be a 3d tensor or np.ndarray.'
    h, w, c = image.shape[0], image.shape[1], image.shape[2]

    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )
    orig_dtype = image.dtype
    image = np.asarray(image, dtype=np.float)
    theta = np.random.uniform(degrees[0], degrees[1])
    theta = np.deg2rad(theta)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    transform_matrix = rotation_matrix

    if shift is not None:
        max_dx = float(shift[0] * w)
        max_dy = float(shift[1] * h)
        tx = -int(round(np.random.uniform(-max_dx, max_dx)))
        ty = -int(round(np.random.uniform(-max_dy, max_dy)))
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear is not None:
        shear_x = shear_y = 0
        shear_x = float(np.random.uniform(shear[0], shear[1]))
        if len(shear) == 4:
            shear_y = float(np.random.uniform(shear[2], shear[3]))
        shear_x = np.deg2rad(shear_x)
        shear_y = np.deg2rad(shear_y)
        shear_matrix = np.array(
            [[math.cos(shear_y), math.sin(shear_y), 0], [math.sin(shear_x), math.cos(shear_x), 0], [0, 0, 1]]
        )
        transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zoom is not None:
        zoom = 1 / float(np.random.uniform(zoom[0], zoom[1]))
        zoom_matrix = np.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, 1]])

        transform_matrix = np.dot(transform_matrix, zoom_matrix)

    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    image = np.rollaxis(image, 2, 0)
    finale_affine_matrix = transform_matrix[:2, :2]
    finale_offset = transform_matrix[:2, 2]
    dummy = np.ones((1, h, w), dtype=image.dtype)
    image = np.concatenate((image, dummy), axis=0)

    channel_images = [
        ndimage.interpolation.affine_transform(
            x_channel, finale_affine_matrix, finale_offset, order=3, mode='constant', cval=0
        ) for x_channel in image
    ]

    image = np.stack(channel_images, axis=0)
    image = np.rollaxis(image, 0, 3)
    mask = image[:, :, -1:]
    image = image[:, :, :-1]
    mask = np.tile(mask, (1, 1, c))
    fill = np.tile(fill, (h, w, 1))
    if interpolation == 'nearest':
        mask = mask < 0.5
        image[mask] = fill[mask]
    else:
        image = image * mask + (1.0 - mask) * fill
    image = np.asarray(image, dtype=orig_dtype)
    image = ops.convert_to_tensor(image)
    return image
