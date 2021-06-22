#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PIL
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import colorsys
import random
import math
from numpy import sin, cos, tan
import numbers

_pil_interp_from_str = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'hamming': Image.HAMMING
}


def crop(image, offset_height, offset_width, target_height, target_width):
    image_width, image_height = image.size
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

    return image.crop((offset_width, offset_height, offset_width + target_width, offset_height + target_height))


def center_crop(image, size, central_fraction):

    image_width, image_height = image.size
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
        raise TypeError("Padding mode should be 'constant', 'edge', 'reflect', or 'symmetric'.")

    if mode == 'constant':
        if image.mode == 'P':
            palette = image.getpalette()
            image = ImageOps.expand(image, border=padding, fill=padding_value)
            image.putpalette(palette)
            return image
        return ImageOps.expand(image, border=padding, fill=padding_value)

    if image.mode == 'P':
        palette = image.getpalette()
        image = np.asarray(image)
        image = np.pad(image, ((top, bottom), (left, right)), mode)
        image = Image.fromarray(image)
        image.putpalette(palette)
        return image

    image = np.asarray(image)
    # RGB image
    if len(image.shape) == 3:
        image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode)
    # Grayscale image
    if len(image.shape) == 2:
        image = np.pad(image, ((top, bottom), (left, right)), mode)

    return Image.fromarray(image)


def resize(image, size, method):

    if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 2)):
        raise TypeError('Size should be a single number or a list/tuple (h, w) of length 2.' 'Got {}.'.format(size))

    if method not in ('nearest', 'bilinear', 'bicubic', 'box', 'lanczos', 'hamming'):
        raise ValueError(
            "Unknown resize method! resize method must be in "
            "(\'nearest\',\'bilinear\',\'bicubic\',\'box\',\'lanczos\',\'hamming\')"
        )
    if isinstance(size, int):
        w, h = image.size
        if (w <= h and w == size) or (h <= w and h == size):
            return image
        if w < h:
            ow = size
            oh = int(size * h / w)
            return image.resize((ow, oh), _pil_interp_from_str[method])
        else:
            oh = size
            ow = int(size * w / h)
            return image.resize((ow, oh), _pil_interp_from_str[method])
    else:
        return image.resize(size[::-1], _pil_interp_from_str[method])


def transpose(image, order):

    image = np.asarray(image)
    if not (isinstance(order, (list, tuple)) and len(order) == 3):
        raise TypeError("Order must be a list/tuple of length 3." "But got {}.".format(order))

    image_shape = image.shape
    if len(image_shape) == 2:
        image = image[..., np.newaxis]

    image = image.transpose(order)
    image = Image.fromarray(image)
    return image


def hwc_to_chw(image):

    image_shape = image.shape
    if len(image_shape) == 2:
        image = image[..., np.newaxis]

    image = image.transpose((2, 0, 1))
    image = Image.fromarray(image)
    return image


def chw_to_hwc(image):

    image_shape = image.shape
    if len(image_shape) == 2:
        image = image[..., np.newaxis]

    image = image.transpose((1, 2, 0))
    image = Image.fromarray(image)
    return image


def rgb_to_hsv(image):

    return image.convert('HSV')


def hsv_to_rgb(image):

    return image.convert('RGB')


def rgb_to_gray(image, num_output_channels):

    if num_output_channels == 1:
        img = image.convert('L')
    elif num_output_channels == 3:
        img = image.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


def adjust_brightness(image, brightness_factor):
    """Adjusts brightness of an Image.

    Args:
        image (PIL.Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL.Image: Brightness adjusted image.

    """
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    return image


def adjust_contrast(image, contrast_factor):
    """Adjusts contrast of an Image.

    Args:
        image (PIL.Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL.Image: Contrast adjusted image.

    """
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    return image


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

    input_mode = image.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return image
    h, s, v = image.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    image = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return image


def adjust_saturation(image, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        image (PIL.Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL.Image: Saturation adjusted image.

    """
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)
    return image


def hflip(image):
    """Horizontally flips the given PIL Image.

    Args:
        img (PIL.Image): Image to be flipped.

    Returns:
        PIL.Image:  Horizontall flipped image.

    """

    return image.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(image):
    """Vertically flips the given PIL Image.

    Args:
        img (PIL.Image): Image to be flipped.

    Returns:
        PIL.Image:  Vertically flipped image.

    """

    return image.transpose(Image.FLIP_TOP_BOTTOM)


def padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value):
    '''

    Parameters
    ----------
    image:
       A PIL image to be padded size of (target_width, target_height)
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
        PIL.Image: padded image
    -------

    '''
    if offset_height < 0:
        raise ValueError('offset_height must be >= 0')
    if offset_width < 0:
        raise ValueError('offset_width must be >= 0')

    width, height = image.size
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


def rotate(image, angle, interpolation, expand, center, fill):
    """Rotates the image by angle.

    Args:
        img (PIL.Image): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set to PIL.Image.NEAREST . when use pil backend,
            support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC
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
        PIL.Image: Rotated image.

    """
    c = 1 if image.mode == 'L' else 3
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    return image.rotate(angle, _pil_interp_from_str[interpolation], expand, center, fillcolor=fill)


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

    c = 1 if image.mode == 'L' else 3
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    w, h = image.size
    center = (w / 2.0, h / 2.0)
    shear = [np.random.uniform(degrees[0], degrees[1]), np.random.uniform(degrees[2], degrees[3])]

    interpolation = _pil_interp_from_str[interpolation]
    matrix = get_affine_matrix(center=center, angle=0, translate=(0, 0), scale=1.0, shear=shear)
    output_size = (w, h)
    kwargs = {"fillcolor": fill}
    return image.transform(output_size, Image.AFFINE, matrix, interpolation, **kwargs)


def random_shift(image, shift, interpolation, fill):

    c = 1 if image.mode == 'L' else 3
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    w, h = image.size
    center = (w / 2.0, h / 2.0)
    hrg = shift[0]
    wrg = shift[1]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    matrix = get_affine_matrix(center=center, angle=0, translate=(tx, ty), scale=1.0, shear=(0, 0))
    print(matrix)

    output_size = (w, h)
    kwargs = {"fillcolor": fill}
    return image.transform(output_size, Image.AFFINE, matrix, interpolation, **kwargs)


def random_zoom(image, zoom, interpolation, fill):

    c = 1 if image.mode == 'L' else 3
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )
    w, h = image.size
    scale = np.random.uniform(zoom[0], zoom[1])
    center = (w / 2.0, h / 2.0)

    matrix = get_affine_matrix(center=center, angle=0, translate=(0, 0), scale=scale, shear=(0, 0))

    output_size = (w, h)
    kwargs = {"fillcolor": fill}
    return image.transform(output_size, Image.AFFINE, matrix, interpolation, **kwargs)


def random_affine(image, degrees, shift, zoom, shear, interpolation, fill):

    c = 1 if image.mode == 'L' else 3
    if isinstance(fill, numbers.Number):
        fill = (fill, ) * c
    elif not (isinstance(fill, (list, tuple)) and len(fill) == c):
        raise ValueError(
            'If fill should be a single number or a list/tuple with length of image channels.'
            'But got {}'.format(fill)
        )

    w, h = image.size
    angle = float(np.random.uniform(float(degrees[0]), float(degrees[1])))
    center = (w / 2.0, h / 2.0)
    if shift is not None:
        max_dx = float(shift[0] * w)
        max_dy = float(shift[1] * h)
        tx = int(round(np.random.uniform(-max_dx, max_dx)))
        ty = int(round(np.random.uniform(-max_dy, max_dy)))
        translations = (tx, ty)
    else:
        translations = (0, 0)

    if zoom is not None:
        scale = float(np.random.uniform(zoom[0], zoom[1]))
    else:
        scale = 1.0

    shear_x = shear_y = 0
    if shear is not None:
        shear_x = float(np.random.uniform(shear[0], shear[1]))
        if len(shear) == 4:
            shear_y = float(np.random.uniform(shear[2], shear[3]))
    shear = (shear_x, shear_y)
    matrix = get_affine_matrix(center=center, angle=angle, translate=translations, scale=scale, shear=shear)

    output_size = (w, h)
    kwargs = {"fillcolor": fill}
    return image.transform(output_size, Image.AFFINE, matrix, interpolation, **kwargs)
