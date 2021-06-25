#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mindspore as ms
from . import functional_cv2 as F_cv2
from . import functional_pil as F_pil
import mindspore.ops as P
from mindspore.numpy import std
from PIL import Image
import PIL
import numpy as np
import numbers
import random
import math

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


def _is_tensor_image(image):
    return isinstance(image, ms.Tensor)


def _is_numpy_image(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})


def _get_image_size(img):
    if _is_pil_image(img):
        return img.size[::-1]
    elif _is_numpy_image(img):
        return img.shape[:2]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


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


def to_tensor(image, data_format='HWC'):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray. Got {}'.format(type(image)))

    image = np.asarray(image).astype('float32')

    if image.ndim == 2:
        image = image[:, :, None]

    if data_format == 'CHW':

        image = np.transpose(image, (2, 0, 1))
        image = image / 255.
    else:
        image = image / 255.

    return image


def central_crop(image, size=None, central_fraction=None):

    if size is None and central_fraction is None:
        raise ValueError('central_fraction and size can not be both None')

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):

        return F_pil.center_crop(image, size, central_fraction)

    else:

        return F_cv2.center_crop(image, size, central_fraction)


def crop(image, offset_height, offset_width, target_height, target_width):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):

        return F_pil.crop(image, offset_height, offset_width, target_height, target_width)

    else:

        return F_cv2.crop(image, offset_height, offset_width, target_height, target_width)


def pad(image, padding, padding_value, mode):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.pad(image, padding, padding_value, mode)
    else:
        return F_cv2.pad(image, padding, padding_value, mode)


def resize(image, size, method):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.resize(image, size, method)
    else:
        return F_cv2.resize(image, size, method)


def transpose(image, order):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.transpose(image, order)
    else:
        return F_cv2.transpose(image, order)


def hwc_to_chw(image):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.hwc_to_chw(image)
    else:
        return F_cv2.hwc_to_chw(image)


def chw_to_hwc(image):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.chw_to_hwc(image)
    else:
        return F_cv2.chw_to_hwc(image)


def rgb_to_hsv(image):
    if not (_is_pil_image(image) or isinstance(image, np.ndarray) and (image.ndim == 3)):
        raise TypeError('image should be PIL Image or ndarray with dim=3. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.rgb_to_hsv(image)
    else:
        return F_cv2.rgb_to_hsv(image)


def hsv_to_rgb(image):
    if not (_is_pil_image(image) or isinstance(image, np.ndarray) and (image.ndim == 3)):
        raise TypeError('image should be PIL Image or ndarray with dim=3. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.hsv_to_rgb(image)
    else:
        return F_cv2.hsv_to_rgb(image)


def rgb_to_gray(image, num_output_channels):
    if not (_is_pil_image(image) or isinstance(image, np.ndarray) and (image.ndim == 3)):
        raise TypeError('image should be PIL Image or ndarray with dim=3. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.rgb_to_gray(image, num_output_channels)
    else:
        return F_cv2.rgb_to_gray(image, num_output_channels)


def adjust_brightness(image, brightness_factor):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.adjust_brightness(image, brightness_factor)
    else:
        return F_cv2.adjust_brightness(image, brightness_factor)


def adjust_contrast(image, contrast_factor):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.adjust_contrast(image, contrast_factor)
    else:
        return F_cv2.adjust_contrast(image, contrast_factor)


def adjust_hue(image, hue_factor):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.adjust_hue(image, hue_factor)
    else:
        return F_cv2.adjust_hue(image, hue_factor)


def adjust_saturation(image, saturation_factor):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.adjust_saturation(image, saturation_factor)
    else:
        return F_cv2.adjust_saturation(image, saturation_factor)


def hflip(image):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.hflip(image)
    else:
        return F_cv2.hflip(image)


def vflip(image):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.vflip(image)
    else:
        return F_cv2.vflip(image)


def padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value)
    else:
        return F_cv2.padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value)


def normalize(image, mean, std, data_format):

    if _is_pil_image(image):
        image = np.asarray(image)

    image = image.astype('float32')

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
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)

    if data_format == 'CHW':
        image = (image - mean[None, None, :]) / std[None, None, :]
    elif data_format == 'HWC':
        image = (image - mean[None, None, :]) / std[None, None, :]

    return image


def standardize(image):
    '''
        Reference to tf.image.per_image_standardization().
        Linearly scales each image in image to have mean 0 and variance 1.
    '''

    if _is_pil_image(image):
        image = np.asarray(image)

    image = image.astype('float32')

    num_pixels = image.size
    image_mean = np.mean(image, keep_dims=False)
    stddev = np.std(image, keep_dims=False)
    min_stddev = 1.0 / np.sqrt(num_pixels)
    adjusted_stddev = np.maximum(stddev, min_stddev)

    return (image - image_mean) / adjusted_stddev


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
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    brightness_factor = random_factor(brightness_factor, name='brightness')

    if _is_pil_image(image):
        return F_pil.adjust_brightness(image, brightness_factor)
    else:
        return F_cv2.adjust_brightness(image, brightness_factor)


def random_contrast(image, contrast_factor):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    contrast_factor = random_factor(contrast_factor, name='contrast')

    if _is_pil_image(image):
        return F_pil.adjust_contrast(image, contrast_factor)
    else:
        return F_cv2.adjust_contrast(image, contrast_factor)


def random_saturation(image, saturation_factor):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    saturation_factor = random_factor(saturation_factor, name='saturation')

    if _is_pil_image(image):
        return F_pil.adjust_saturation(image, saturation_factor)
    else:
        return F_cv2.adjust_saturation(image, saturation_factor)


def random_hue(image, hue_factor):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    hue_factor = random_factor(hue_factor, name='hue', center=0, bound=(-0.5, 0.5), non_negative=False)

    if _is_pil_image(image):
        return F_pil.adjust_hue(image, hue_factor)
    else:
        return F_cv2.adjust_hue(image, hue_factor)


def random_crop(image, size, padding, pad_if_needed, fill, padding_mode):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
        raise ValueError('Size should be a int or a list/tuple with length of 2. ' 'But got {}'.format(size))

    height, width = _get_image_size(image)
    if padding is not None:
        image = pad(image, padding, fill, padding_mode)

    if pad_if_needed and height < size[0]:
        image = pad(image, (0, height - size[0]), fill, padding_mode)

    if pad_if_needed and width < size[1]:
        image = pad(image, (width - size[1], 0), fill, padding_mode)

    height, width = _get_image_size(image)
    target_height, target_width = size

    if height < target_height or width < target_width:
        raise ValueError(
            'Crop size {} should be smaller than input image size {}. '.format(
                (target_height, target_width), (height, width)
            )
        )

    offset_height = random.randint(0, height - target_height)
    offset_width = random.randint(0, width - target_width)

    return crop(image, offset_height, offset_width, target_height, target_width)


def random_resized_crop(image, size, scale, ratio, interpolation):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

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

    def _get_param(image, scale, ratio):
        height, width = _get_image_size(image)
        area = height * width
        log_ratio = tuple(math.log(x) for x in ratio)
        for _ in range(10):
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            # return whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    offset_height, offset_width, target_height, target_width = _get_param(image, scale, ratio)

    image = crop(image, offset_height, offset_width, target_height, target_width)
    image = resize(image, size, interpolation)

    return image


def random_vflip(image, prob):

    if random.random() < prob:
        return vflip(image)
    return image


def random_hflip(image, prob):

    if random.random() < prob:
        return hflip(image)
    return image


def random_rotation(image, degrees, interpolation, expand, center, fill):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError('If degrees is a single number, it must be positive.' 'But got {}'.format(degrees))
        degrees = (-degrees, degrees)
    elif not (isinstance(degrees, (list, tuple)) and len(degrees) == 2):
        raise ValueError('If degrees is a list/tuple, it must be length of 2.' 'But got {}'.format(degrees))
    else:
        if degrees[0] > degrees[1]:
            raise ValueError('if degrees is a list/tuple, it should be (min, max).')

    angle = np.random.uniform(degrees[0], degrees[1])

    if _is_pil_image(image):
        return F_pil.rotate(image, angle, interpolation, expand, center, fill)
    else:
        return F_cv2.rotate(image, angle, interpolation, expand, center, fill)


def random_shear(image, degrees, interpolation, fill):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

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

    if _is_pil_image(image):
        return F_pil.random_shear(image, degrees, interpolation, fill)
    else:
        return F_cv2.random_shear(image, degrees, interpolation, fill)


def random_shift(image, shift, interpolation, fill):

    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if not (isinstance(shift, (tuple, list)) and len(shift) == 2):

        raise ValueError('Shift should be a list/tuple with length of 2.' 'But got {}'.format(shift))

    if _is_pil_image(image):
        return F_pil.random_shift(image, shift, interpolation, fill)
    else:
        return F_cv2.random_shift(image, shift, interpolation, fill)


def random_zoom(image, zoom, interpolation, fill):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if not (isinstance(zoom, (tuple, list)) and len(zoom) == 2):

        raise ValueError('Zoom should be a list/tuple with length of 2.' 'But got {}'.format(zoom))
    if not (0 <= zoom[0] <= zoom[1]):

        raise ValueError('Zoom values should be positive, and zoom[1] should be greater than zoom[0].')

    if _is_pil_image(image):
        return F_pil.random_zoom(image, zoom, interpolation, fill)
    else:
        return F_cv2.random_zoom(image, zoom, interpolation, fill)


def random_affine(image, degrees, shift, zoom, shear, interpolation, fill):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.format(type(image)))

    if _is_pil_image(image):
        return F_pil.random_affine(image, degrees, shift, zoom, shear, interpolation, fill)
    else:
        return F_cv2.random_affine(image, degrees, shift, zoom, shear, interpolation, fill)
