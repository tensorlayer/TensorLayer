#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import sin, cos, tan
import math
import numbers
import importlib


def try_import(module_name):
    """Try importing a module, with an informative error message on failure."""
    install_name = module_name

    if module_name.find('.') > -1:
        install_name = module_name.split('.')[0]

    if module_name == 'cv2':
        install_name = 'opencv-python'

    try:
        mod = importlib.import_module(module_name)
        return mod
    except ImportError:
        err_msg = (
            "Failed importing {}. This likely means that some paddle modules "
            "require additional dependencies that have to be "
            "manually installed (usually with `pip install {}`). "
        ).format(module_name, install_name)
        raise ImportError(err_msg)


def crop(image, offset_height, offset_width, target_height, target_width):
    image_height, image_width = image.shape[0:2]
    if offset_width < 0:
        raise ValueError('offset_width must be >0.')
    if offset_height < 0:
        raise ValueError('offset_height must be >0.')
    if target_height < 0:
        raise ValueError('target_height must be >0.')
    if target_width < 0:
        raise ValueError('target_width must be >0.')
    if offset_width + target_width > image_width:
        raise ValueError('offset_width + target_width must be <= image width.')
    if offset_height + target_height > image_height:
        raise ValueError('offset_height + target_height must be <= image height.')

    return image[offset_height:offset_height + target_height, offset_width:offset_width + target_width]


def center_crop(image, size, central_fraction):

    image_height, image_width = image.shape[0:2]
    if size is not None:
        if not isinstance(size, (int, list, tuple)) or (isinstance(size, (list, tuple)) and len(size) != 2):
            raise TypeError(
                "Size should be a single integer or a list/tuple (h, w) of length 2.But"
                "got {}.".format(size)
            )

        if isinstance(size, int):
            target_height = size
            target_width = size
        else:
            target_height = size[0]
            target_width = size[1]

    elif central_fraction is not None:
        if central_fraction <= 0.0 or central_fraction > 1.0:
            raise ValueError('central_fraction must be within (0, 1]')

        target_height = int(central_fraction * image_height)
        target_width = int(central_fraction * image_width)

    crop_top = int(round((image_height - target_height) / 2.))
    crop_left = int(round((image_width - target_width) / 2.))

    return crop(image, crop_top, crop_left, target_height, target_width)


def pad(image, padding, padding_value, mode):

    if isinstance(padding, int):
        top = bottom = left = right = padding

    elif isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            left = right = padding[0]
            top = bottom = padding[1]
        elif len(padding) == 4:
            left = padding[0]
            top = padding[1]
            right = padding[2]
            bottom = padding[3]
        else:
            raise TypeError("The size of the padding list or tuple should be 2 or 4." "But got {}".format(padding))
    else:
        raise TypeError("Padding can be any of: a number, a tuple or list of size 2 or 4." "But got {}".format(padding))

    if mode not in ['constant', 'edge', 'reflect', 'symmetric']:
        raise ValueError("Padding mode should be 'constant', 'edge', 'reflect', or 'symmetric'.")
    cv2 = try_import('cv2')
    _cv2_pad_from_str = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }

    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.copyMakeBorder(
            image, top=top, bottom=bottom, left=left, right=right, borderType=_cv2_pad_from_str[mode],
            value=padding_value
        )[:, :, np.newaxis]
    else:
        return cv2.copyMakeBorder(
            image, top=top, bottom=bottom, left=left, right=right, borderType=_cv2_pad_from_str[mode],
            value=padding_value
        )


def resize(image, size, method):

    if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 2)):
        raise TypeError('Size should be a single number or a list/tuple (h, w) of length 2.' 'Got {}.'.format(size))
    if method not in ('nearest', 'bilinear', 'area', 'bicubic' 'lanczos'):
        raise ValueError(
            "Unknown resize method! resize method must be in "
            "(\'nearest\',\'bilinear\',\'bicubic\',\'area\',\'lanczos\')"
        )
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    h, w = image.shape[:2]

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return image
        if w < h:
            target_w = size
            target_h = int(size * h / w)
        else:
            target_h = size
            target_w = int(size * w / h)
        size = (target_h, target_w)
    output = cv2.resize(image, dsize=(size[1], size[0]), interpolation=_cv2_interp_from_str[method])
    if len(image.shape) == 3 and image.shape[2] == 1:
        return output[:, :, np.newaxis]
    else:
        return output


def transpose(image, order):

    if not (isinstance(order, (list, tuple)) and len(order) == 3):
        raise TypeError("Order must be a list/tuple of length 3." "But got {}.".format(order))

    image_shape = image.shape
    if len(image_shape) == 2:
        image = image[..., np.newaxis]

    return image.transpose(order)


def hwc_to_chw(image):

    image_shape = image.shape
    if len(image_shape) == 2:
        image = image[..., np.newaxis]

    return image.transpose((2, 0, 1))


def chw_to_hwc(image):

    image_shape = image.shape
    if len(image_shape) == 2:
        image = image[..., np.newaxis]

    return image.transpose((1, 2, 0))


def rgb_to_hsv(image):

    cv2 = try_import('cv2')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


def hsv_to_rgb(image):

    cv2 = try_import('cv2')
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def rgb_to_gray(image, num_output_channels):

    cv2 = try_import('cv2')

    if num_output_channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    elif num_output_channels == 3:
        image = np.broadcast_to(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis], image.shape)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return image


def adjust_brightness(image, brightness_factor):
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))
    cv2 = try_import('cv2')

    table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')

    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.LUT(image, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(image, table)


def adjust_contrast(image, contrast_factor):
    """Adjusts contrast of an image.

    Args:
        img (np.array): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        np.array: Contrast adjusted image.

    """
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))
    cv2 = try_import('cv2')

    table = np.array([(i - 127) * contrast_factor + 127 for i in range(0, 256)]).clip(0, 255).astype('uint8')
    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.LUT(image, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(image, table)


def adjust_hue(image, hue_factor):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        image (PIL.Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL.Image: Hue adjusted image.

    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
    cv2 = try_import('cv2')

    dtype = image.dtype
    image = image.astype(np.uint8)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    h, s, v = cv2.split(hsv_img)

    alpha = np.random.uniform(hue_factor, hue_factor)
    h = h.astype(np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        h += np.uint8(alpha * 255)
    hsv_img = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB_FULL).astype(dtype)


def adjust_saturation(image, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        image (np.array): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        np.array: Saturation adjusted image.

    """
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))
    cv2 = try_import('cv2')

    dtype = image.dtype
    image = image.astype(np.float32)
    alpha = np.random.uniform(saturation_factor, saturation_factor)
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_img = gray_img[..., np.newaxis]
    img = image * alpha + gray_img * (1 - alpha)
    return img.clip(0, 255).astype(dtype)


def hflip(image):
    """Horizontally flips the given image.

    Args:
        image (np.array): Image to be flipped.

    Returns:
        np.array:  Horizontall flipped image.

    """
    cv2 = try_import('cv2')

    return cv2.flip(image, 1)


def vflip(image):
    """Vertically flips the given np.array.

    Args:
        image (np.array): Image to be flipped.

    Returns:
        np.array:  Vertically flipped image.

    """
    cv2 = try_import('cv2')

    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.flip(image, 0)[:, :, np.newaxis]
    else:
        return cv2.flip(image, 0)


def padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value):
    '''

    Parameters
    ----------
    image:
       A np.array image to be padded size of (target_width, target_height)
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
        np.array image: padded image
    -------

    '''
    if offset_height < 0:
        raise ValueError('offset_height must be >= 0')
    if offset_width < 0:
        raise ValueError('offset_width must be >= 0')

    height, width = image.shape[:2]
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


def rotate(img, angle, interpolation, expand, center, fill):
    """Rotates the image by angle.

    Args:
        img (np.array): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        interpolation (int|str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set to cv2.INTER_NEAREST.
            when use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "bicubic": cv2.INTER_CUBIC
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    Returns:
        np.array: Rotated image.

    """
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    h, w, c = img.shape
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    if expand:

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        # calculate output size
        xx = []
        yy = []

        angle = -math.radians(angle)
        expand_matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        post_trans = (0, 0)
        expand_matrix[2], expand_matrix[5] = transform(
            -center[0] - post_trans[0], -center[1] - post_trans[1], expand_matrix
        )
        expand_matrix[2] += center[0]
        expand_matrix[5] += center[1]

        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = transform(x, y, expand_matrix)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))

        M[0, 2] += (nw - w) * 0.5
        M[1, 2] += (nh - h) * 0.5

        w, h = int(nw), int(nh)

    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.warpAffine(img, M, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)[:, :,
                                                                                                           np.newaxis]
    else:
        return cv2.warpAffine(img, M, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)


def get_affine_matrix(center, angle, translate, scale, shear):

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def random_shear(image, degrees, interpolation, fill):

    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    h, w, c = image.shape
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    center = (w / 2.0, h / 2.0)
    shear = [-np.random.uniform(degrees[0], degrees[1]), -np.random.uniform(degrees[2], degrees[3])]

    matrix = get_affine_matrix(center=center, angle=0, translate=(0, 0), scale=1.0, shear=shear)
    matrix = np.asarray(matrix).reshape((2, 3))

    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation],
                              borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)


def random_shift(image, shift, interpolation, fill):

    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    h, w, c = image.shape
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )
    hrg = shift[0]
    wrg = shift[1]
    tx = -np.random.uniform(-hrg, hrg) * w
    ty = -np.random.uniform(-wrg, wrg) * h
    center = (w / 2.0, h / 2.0)

    matrix = get_affine_matrix(center=center, angle=0, translate=(tx, ty), scale=1.0, shear=(0, 0))
    matrix = np.asarray(matrix).reshape((2, 3))

    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation],
                              borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)


def random_zoom(image, zoom, interpolation, fill):
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    h, w, c = image.shape
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    scale = 1 / np.random.uniform(zoom[0], zoom[1])
    center = (w / 2.0, h / 2.0)

    matrix = get_affine_matrix(center=center, angle=0, translate=(0, 0), scale=scale, shear=(0, 0))
    matrix = np.asarray(matrix).reshape((2, 3))

    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation],
                              borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)


def random_affine(image, degrees, shift, zoom, shear, interpolation, fill):
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    h, w, c = image.shape
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )
    center = (w / 2.0, h / 2.0)

    angle = -float(np.random.uniform(degrees[0], degrees[1]))

    if shift is not None:
        max_dx = float(shift[0] * h)
        max_dy = float(shift[1] * w)
        tx = -int(round(np.random.uniform(-max_dx, max_dx)))
        ty = -int(round(np.random.uniform(-max_dy, max_dy)))
        shift = [tx, ty]
    else:
        shift = [0, 0]

    if zoom is not None:
        scale = 1 / np.random.uniform(zoom[0], zoom[1])
    else:
        scale = 1.0

    shear_x = shear_y = 0.0
    if shear is not None:
        shear_x = float(np.random.uniform(shear[0], shear[1]))
        if len(shear) == 4:
            shear_y = float(np.random.uniform(shear[2], shear[3]))
    shear = (-shear_x, -shear_y)

    matrix = get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=shear)
    matrix = np.asarray(matrix).reshape((2, 3))

    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation],
                              borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(image, matrix, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)
