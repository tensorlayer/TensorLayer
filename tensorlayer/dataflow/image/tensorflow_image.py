import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.image_ops_impl import _AssertAtLeast3DImage
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.image_ops_impl import convert_image_dtype
import numbers

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
    'RgbToGray',
    'PadToBoundingbox',
    'Pad',
    'RandomBrightness',
    'RandomContrast',
    'RandomHue',
    'RandomSaturation',
    'RandomCrop',
    'Resize',
    'CropAndResize',
    'CropOrPad',
    'ResizeAndPad',
    'RgbToHsv',
    'Transpose',
    'RandomRotation',
    'RandomShift',
    'RandomShear',
    'RandomZoom',
    'Rescale',
    'RandomFlipVertical',
    'RandomFlipHorizontal',
    'HWC2CHW',
    'CHW2HWC',
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
	If backend is tensorflow, central_fraction will be used preferentially. if size is used,the height-width ratio will be equivalent to original ratio..
	If backend is mindspore, size will be used preferentially.
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

        if isinstance(size, numbers.Number):
            target_height = size
            target_width = size
        elif isinstance(size, tuple) or isinstance(size, list):
            if len(size) == 2:
                target_height = size[0]
                target_width = size[1]
            else:
                raise ValueError('The length of size must be 2')
        else:
            raise ValueError("Size should be a single integer or a list/tuple (h, w) of length 2.")
        if target_height > outshape[h_axis] or target_width > outshape[w_axis]:
            raise ValueError("Centralcrop image size must < original image size.")
        central_fraction = max(target_height / outshape[h_axis], target_width / outshape[w_axis])
    else:
        if central_fraction > 1 or central_fraction <= 0:
            raise ValueError('central_fraction must be in (0,1].')

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


def Crop(image, offset_height, offset_width, target_height, target_width, is_hwc=True):
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


def PadToBoundingbox(image, offset_height, offset_width, target_height, target_width, padding_value=0, is_hwc=True):

    return tf.image.pad_to_bounding_box(
        image,
        offset_height,
        offset_width,
        target_height,
        target_width,
    )


def Pad(image, padding, padding_value=0, mode='constant'):
    '''

    Parameters
    ----------
    image:
        A 3-D or 4-D Tensor.
    padding:
        An integer or a list/tuple.  If a single number is provided, pad all borders with this value.
        If a tuple or list of 2 values is provided, pad the left and top with the first value and the right and bottom with the second value.
        If 4 values are provided as a list or tuple, pad the  (top, bottom, left, right)  respectively.
    padding_value:
        In "CONSTANT" mode, the scalar pad value to use. Must be same type as tensor.
    mode:
        One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    Returns:
        A padded Tensor. Has the same type as tensor.
    -------

    '''
    image_shape = image.shape
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
            padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        elif len(padding) == 4:
            padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError('The length of padding should be 2 or 4, but got {}.'.format(len(padding)))
    else:
        raise TypeError('Padding should be an integer or a list/tuple, but got {}.'.format(type(padding)))
    if batch_size == 0:
        padding = (padding[0], padding[1], (0, 0))
    else:
        padding = ((0, 0), padding[0], padding[1], (0, 0))

    return tf.pad(image, padding, mode=mode, constant_values=padding_value)


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
    image = tf.cast(image, tf.float32)
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


def RandomBrightness(image, factor):
    '''

    Parameters
    ----------
    image:
        An image or images to adjust
    factor:
        Float, must be non-negative. Factor must be (0,1). Random range will be [-factor, factor).
    Returns:
        The brightness-adjusted image(s).
    -------

    '''

    return tf.image.random_brightness(image, factor)


def RandomContrast(image, lower, upper, seed=None):
    '''

    Parameters
    ----------
    image:
        An image tensor with 3 or more dimensions.
    lower:
        float.  Lower bound for the random contrast factor.
    upper:
        float.  Upper bound for the random contrast factor.
    seed:
        A Python integer. Used to create a random seed.

    Returns:
         The contrast-adjusted image(s).
    -------
    '''

    return tf.image.random_contrast(image, lower, upper, seed)


def RandomHue(image, factor, seed=None):
    '''

    Parameters
    ----------
    image:
        RGB image or images. The size of the last dimension must be 3.
    factor:
        float. The maximum value for the random factor.
    seed:
         An operation-specific seed.

    Returns:
        Adjusted image(s), same shape and DType as `image`.
    -------

    '''

    return tf.image.random_hue(image, factor, seed)


def RandomSaturation(image, lower, upper, seed=None):
    '''
    Parameters
    ----------
    image:
        RGB image or images. The size of the last dimension must be 3.
    lower:
        float.  Lower bound for the random saturation factor.
    upper:
        float.  Upper bound for the random saturation factor.
    seed:
        An operation-specific seed.

    Returns:
        Adjusted image(s), same shape and DType as `image`.
    -------
    '''

    return tf.image.random_saturation(image, lower, upper, seed)


def RandomCrop(image, size):
    '''

    Parameters
    ----------
    image:
        Input an image  to crop.
    size:
        a list or tuple. if size is an integer, shape of cropped image  will be [size, size, 3]. if length of size is 2.
        shape of cropped image  will be [height, width, 3].
    Returns:
        A cropped image of the same rank as image and shape size.
    -------
    '''

    if isinstance(size, int):
        crop_size = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        crop_size = (size[0], size[1])
    else:
        raise ValueError("Size should be a single integer or a list/tuple (h, w) of length 2.")

    if len(image.shape) == 3:
        h, w, c = image.shape
        crop_size = crop_size + (c, )
    elif len(image.shape) == 4:
        b, h, w, c = image.shape
        crop_size = (b, ) + crop_size + (c, )

    return tf.image.random_crop(image, size=crop_size)


def Resize(image, size, method='bilinear', preserve_aspect_ratio=False, antialias=False):
    '''

    Parameters
    ----------
    images:
        Input an image to resize
    size:
        if size is an integer, shape of resized image  will be [size, size, 3]. if length of size is 2.
        shape of resized image  will be [height, width, 3].
    method:
        An image.ResizeMethod, or string equivalent shoulid be in
        (bilinear, lanczos3, lanczos5, bicubic, gaussian, nearest, area, mitchellcubic).
        Defaults to bilinear.
    preserve_aspect_ratio:
        Whether to preserve the aspect ratio.
    antialias:
        Whether to use an anti-aliasing filter when downsampling an image.
    Returns:
        an resized image
    -------

    '''
    if isinstance(size, int):
        size = [size, size]
    elif len(size) != 2:
        raise ValueError('Size should be a single integer or a list/tuple (h, w) of length 2.')

    return tf.image.resize(image, size, method, preserve_aspect_ratio, antialias)


def CropAndResize(image, boxes, box_indices, crop_size, method='bilinear', extrapolation_value=0, is_hwc=True):
    '''

    Parameters
    ----------
    image:
        A 4-D tensor of shape [batch, image_height, image_width, depth]. Both image_height and image_width need to be positive.
    boxes:
        A 2-D tensor of shape [num_boxes, 4].
    box_indices:
        A 1-D tensor of shape [num_boxes] with int32 values in [0,batch).
        The value of box_ind[i] specifies the image that the i-th box refers to.
    crop_size:
        A 1-D tensor of 2 elements, size = [crop_height, crop_width]. All cropped image patches are resized to this size.
        The aspect ratio of the image content is not preserved. Both crop_height and crop_width need to be positive.
    method:
        An optional string specifying the sampling method for resizing.
        It can be either "bilinear" or "nearest" and default to "bilinear".
    extrapolation_value:
        An optional float. Defaults to 0. Value used for extrapolation, when applicable.
    Returns:
        A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].
    -------

    '''
    image_shape = image.shape
    boxes_num = 0
    if isinstance(boxes, tf.Tensor):
        boxes_num = boxes.shape[0]
    elif isinstance(boxes, np.ndarray) or isinstance(boxes, list) or isinstance(boxes, tuple):
        boxes = tf.constant(boxes)
        boxes_num = boxes.shape[0]

    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
        crop_size = tf.constant(crop_size)
    elif isinstance(crop_size, np.ndarray) or isinstance(crop_size, list) or isinstance(crop_size, tuple):
        crop_size = tf.constant(crop_size)

    if isinstance(box_indices, np.ndarray) or isinstance(box_indices, list) or isinstance(box_indices, tuple):
        box_indices = tf.constant(box_indices)
    # if input is an image.
    # a 3-D Tensor of shape [image_height, image_width, depth] should use 'tf.expand_dims(image, axis = 0)'
    # to convert input to a 4-D Tensor of shape [batch_size,image_height, image_width, depth]
    if len(image_shape) == 3:
        image = tf.expand_dims(image, axis=0)
        box_indices = np.zeros((boxes_num), dtype=np.int)
        box_indices = tf.constant(box_indices)

    return tf.image.crop_and_resize(
        image, boxes=boxes, box_indices=box_indices, crop_size=crop_size, method=method,
        extrapolation_value=extrapolation_value
    )


def CropOrPad(image, target_height, target_width, is_hwc=True):
    '''
    Resizes an image to a target width and height by either centrally cropping the image or padding it evenly with zeros.
    Parameters
    ----------
    image:
        4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    target_height:
        Target height.
    target_width:
        Target width.
    Returns:
        Cropped and/or padded image.
    -------
    '''

    return tf.image.resize_with_crop_or_pad(image, target_height, target_width)


def ResizeAndPad(image, target_height, target_width, method='bilinear', antialias=False, is_hwc=True):
    '''

    Parameters
    ----------
    image:
        4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    target_height:
        Target height.
    target_width:
        Target height.
    is_hwc:
         The flag of image shape, (H, W, C) or (N, H, W, C) if True and (C, H, W) or (N, C, H, W) if False (default=True).
    Returns:
        Resized and padded image. If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
        If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels].
    -------

    '''

    return tf.image.resize_with_pad(image, target_height, target_width, method=method, antialias=antialias)


def RgbToHsv(image):

    return tf.image.rgb_to_hsv(image)


def Transpose(image, order):
    image = ops.convert_to_tensor(image)
    image = _AssertAtLeast3DImage(image)
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


def RandomRotation(
    image, degrees, fill_mode='nearest', fill_value=0, center=None, expand=False, is_hwc=True, interpolation_order=1
):
    if isinstance(image, tf.Tensor):
        image = np.asarray(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if is_hwc:
        h, w, c = 0, 1, 2
    else:
        h, w, c = 1, 2, 0
    if fill_mode not in ('constant', 'nearest', 'reflect', 'wrap'):
        raise TypeError('fill_mode must be in (constant, nearest, reflect, wrap)')

    image = tf.keras.preprocessing.image.random_rotation(
        image, degrees, h, w, c, fill_mode, fill_value, interpolation_order
    )
    return tf.convert_to_tensor(image)


def RandomShift(image, shift, fill_mode='nearest', fill_value=0, is_hwc=True, interpolation_order=1):
    '''

    Parameters
    ----------
    image
        Input tensor. Must be 3D.
    shift:
        int or list/tuple, if shift is int, Width shift range will equal to height shift range.
        if shift is list/tuple,  shift range will be [width fraction, height fraction]
    is_hwc:
        The flag of image shape, (H, W, C) or (N, H, W, C) if True and (C, H, W) or (N, C, H, W) if False (default=True).
    fill_mode:
        Points outside the boundaries of the input are filled according to the given mode (one of {'constant', 'nearest', 'reflect', 'wrap'}).
    fill_value:
        Value used for points outside the boundaries of the input if mode='constant'.
    interpolation_order
        int, order of spline interpolation. see ndimage.interpolation.affine_transform
    Returns
        Shifted Numpy image tensor.
    -------

    '''
    if isinstance(image, tf.Tensor):
        image = np.asarray(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if isinstance(shift, numbers.Number):
        width_fraction = shift
        height_fraction = shift
    elif isinstance(shift, list) or isinstance(shift, tuple):
        if len(shift) == 2:
            width_fraction = shift[0]
            height_fraction = shift[1]
    else:
        raise ValueError('shift must be number or list/tuple of length 2')

    if is_hwc:
        h, w, c = 0, 1, 2
    else:
        h, w, c = 1, 2, 0
    if fill_mode not in ('constant', 'nearest', 'reflect', 'wrap'):
        raise TypeError('fill_mode must be in (constant, nearest, reflect, wrap)')

    image = tf.keras.preprocessing.image.random_shift(
        image, wrg=width_fraction, hrg=height_fraction, row_axis=h, col_axis=w, channel_axis=c, fill_mode=fill_mode,
        cval=fill_value, interpolation_order=interpolation_order
    )

    return tf.convert_to_tensor(image)


def RandomShear(image, degree, fill_mode='nearest', fill_value=0, is_hwc=True, interpolation_order=1):
    '''

    Parameters
    ----------
    image
        Input tensor. Must be 3D.
    degree:
        Transformation intensity in degrees.
    is_hwc:
        The flag of image shape, (H, W, C) or (N, H, W, C) if True and (C, H, W) or (N, C, H, W) if False (default=True).
    fill_mode:
        Points outside the boundaries of the input are filled according to the given mode (one of {'constant', 'nearest', 'reflect', 'wrap'}).
    fill_value:
        Value used for points outside the boundaries of the input if mode='constant'.
    interpolation_order
        int, order of spline interpolation. see ndimage.interpolation.affine_transform
    Returns
        Shifted Numpy image tensor.
    -------

    '''
    if isinstance(image, tf.Tensor):
        image = np.asarray(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if is_hwc:
        h, w, c = 0, 1, 2
    else:
        h, w, c = 1, 2, 0

    image = tf.keras.preprocessing.image.random_shear(
        image, intensity=degree, row_axis=h, col_axis=w, channel_axis=c, fill_mode=fill_mode, cval=fill_value,
        interpolation_order=interpolation_order
    )
    return tf.convert_to_tensor(image)


def RandomZoom(image, zoom_range, fill_mode='nearest', fill_value=0, is_hwc=True, interpolation_order=1):
    '''

    Parameters
    ----------
    image:
         Input tensor. Must be 3D.
    zoom_range:
        Tuple of floats; zoom range for width and height.
    is_hwc:
        The flag of image shape, (H, W, C) or (N, H, W, C) if True and (C, H, W) or (N, C, H, W) if False (default=True).
    fill_mode:
        Points outside the boundaries of the input are filled according to the given mode (one of {'constant', 'nearest', 'reflect', 'wrap'}).
    fill_value:
        Value used for points outside the boundaries of the input if mode='constant'.
    interpolation_order:
        int, order of spline interpolation. see ndimage.interpolation.affine_transform

    Returns
        Zoomed Numpy image tensor.
    -------

    '''
    if isinstance(image, tf.Tensor):
        image = np.asarray(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if isinstance(zoom_range, numbers.Number):
        zoom_range = (zoom_range, zoom_range)
    elif isinstance(zoom_range, list) or isinstance(zoom_range, tuple):
        if len(zoom_range) == 2:
            zoom_range = (zoom_range[0], zoom_range[1])
    else:
        raise ValueError('shift must be number or list/tuple of length 2')
    if is_hwc:
        h, w, c = 0, 1, 2
    else:
        h, w, c = 1, 2, 0

    image = tf.keras.preprocessing.image.random_zoom(
        image, zoom_range=zoom_range, row_axis=h, col_axis=w, channel_axis=c, fill_mode=fill_mode, cval=fill_value,
        interpolation_order=interpolation_order
    )
    return tf.convert_to_tensor(image)


def Rescale(image, scale, offset=0):
    '''

    Parameters
    ----------
    image:
        3-D image or 4-D images
    scale:
        Float, the scale to apply to the inputs.
    offset:
        Float, the offset to apply to the inputs.
    Returns:
        rescaled images
    -------
    '''
    image = tf.cast(image, dtype=tf.float32)
    scale = tf.cast(scale, dtype=tf.float32)
    offset = tf.cast(offset, dtype=tf.float32)
    return image * scale + offset


def RandomFlipVertical(image):

    return tf.image.random_flip_up_down(image)


def RandomFlipHorizontal(image):

    return tf.image.random_flip_left_right(image)


def HWC2CHW(image):

    if (len(image.shape) == 3):
        return Transpose(image, (2, 0, 1))
    elif (len(image.shape) == 4):
        return Transpose(image, (0, 3, 1, 2))
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')


def CHW2HWC(image):

    if (len(image.shape) == 3):
        return Transpose(image, (1, 2, 0))
    elif (len(image.shape) == 4):
        return Transpose(image, (0, 2, 3, 1))
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')
