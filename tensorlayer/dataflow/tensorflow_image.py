import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.image_ops_impl import _AssertAtLeast3DImage
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.image_ops_impl import convert_image_dtype
__all__ = [
    'CentralCrop',
    'HsvToRgb',
    'AdjustBrightness',
    'AdjustContrast',
    'AdjustHue',
    'AdjustSaturation',
    'Crop',
    'FlipHorizontal',
    'FlipVertical',
    'GrayToRgb',
    'Standardization',
]


def CentralCrop(image, central_fraction=None, size=None):
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

    if central_fraction is None:
        outshape = np.shape(image)
        if len(outshape) == 3:
            h_axis = 0
            w_axis = 1
        elif len(outshape) == 4:
            h_axis = 1
            w_axis = 2

        if isinstance(size, int):
            target_height = size
            target_width = size
        elif isinstance(size, tuple):
            target_height = size[0]
            target_width = size[1]

        central_fraction = max(target_height // outshape[h_axis], target_width // outshape[w_axis])

    return tf.image.central_crop(image, central_fraction)


def HsvToRgb(image):

    return tf.image.hsv_to_rgb(image)


def AdjustBrightness(image, factor):

    return tf.image.adjust_brightness(image, delta=factor)


def AdjustContrast(image, factor):

    return tf.image.adjust_contrast(image, contrast_factor=factor)


def AdjustHue(image, factor):

    return tf.image.adjust_hue(image, delta=factor)


def AdjustSaturation(image, factor):

    return tf.image.adjust_saturation(image, saturation_factor=factor)


def Crop(image, offset_height, offset_width, target_height, target_width):
    '''

	Parameters
	----------
	image:
		A image or  a batch of images
	offset_height:
		Vertical coordinate of the top-left corner of the result in the input.
	offset_width:
		Horizontal coordinate of the top-left corner of the result in the input.
	target_height:
		Height of the result.
	target_width:
		Width of the result.

	Returns:
		Output [batch, target_height, target_width, channels] or [target_height, target_width, channels]
	-------
	'''

    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)


def FlipHorizontal(image):

    return tf.image.flip_left_right(image)


def FlipVertical(image):

    return tf.image.flip_up_down(image)


def GrayToRgb(image):

    return tf.image.grayscale_to_rgb(image)


def RgbToGray(image):

    return tf.image.rgb_to_grayscale(image)


def PadToBoundingBox(image, offset_height, offset_width, target_height, target_width):

    return tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)


def Standardization(image, mean=None, std=None, channel_mode=False):
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
    with ops.name_scope(None, 'Standardization', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)

    orig_dtype = image.dtype
    if orig_dtype not in [dtypes.float16, dtypes.float32]:
        image = convert_image_dtype(image, dtypes.float32)

    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image -= mean
        image = math_ops.divide(image, std, name=scope)
        return convert_image_dtype(image, orig_dtype, saturate=True)

    elif mean is None and std is None:
        if channel_mode:
            num_pixels = math_ops.reduce_prod(array_ops.shape(image)[-3:-1])
            #`num_pixels` is the number of elements in each channels of 'image'
            image_mean = math_ops.reduce_mean(image, axis=[-2, -3], keepdims=True)
            # `image_mean` is the mean of elements in each channels of 'image'

            stddev = math_ops.reduce_std(image, axis=[-2, -3], keepdims=True)
            min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, image.dtype))
            adjusted_sttdev = math_ops.maximum(stddev, min_stddev)

            image -= image_mean
            image = math_ops.divide(image, adjusted_sttdev, name=scope)
            return convert_image_dtype(image, orig_dtype, saturate=True)

        else:
            num_pixels = math_ops.reduce_prod(array_ops.shape(image)[-3:])
            #`num_pixels` is the number of elements in `image`
            image_mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)

            # Apply a minimum normalization that protects us against uniform images.
            stddev = math_ops.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
            min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, image.dtype))
            adjusted_stddev = math_ops.maximum(stddev, min_stddev)

            image -= image_mean
            image = math_ops.divide(image, adjusted_stddev, name=scope)
            return convert_image_dtype(image, orig_dtype, saturate=True)
    else:
        raise ValueError('std and mean must both be None or not None')
