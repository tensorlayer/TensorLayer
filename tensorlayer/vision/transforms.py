#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from . import load_vision_backend as F
import numbers
import numpy as np
__all__ = [
    'Crop',
    'CentralCrop',
    'HsvToRgb',
    'AdjustBrightness',
    'AdjustContrast',
    'AdjustHue',
    'AdjustSaturation',
    'FlipHorizontal',
    'FlipVertical',
    'RgbToGray',
    'PadToBoundingbox',
    'Pad',
    'Normalize',
    'StandardizePerImage',
    'RandomBrightness',
    'RandomContrast',
    'RandomHue',
    'RandomSaturation',
    'RandomCrop',
    'Resize',
    'RgbToHsv',
    'Transpose',
    'RandomRotation',
    'RandomShift',
    'RandomShear',
    'RandomZoom',
    'RandomFlipVertical',
    'RandomFlipHorizontal',
    'HWC2CHW',
    'CHW2HWC',
    'ToTensor',
    'Compose',
    'RandomResizedCrop',
    'RandomAffine',
    'ColorJitter',
]


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Parameters
    ----------
    data_format : str
        Data format of output tensor, should be 'HWC' or 'CHW'. Default: 'HWC'.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.ToTensor(data_format='HWC')
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, data_format='HWC'):

        if not data_format in ['CHW', 'HWC']:
            raise ValueError('data_format should be CHW or HWC. Got {}'.format(data_format))

        self.data_format = data_format

    def __call__(self, image):

        F.to_tensor(image, self.data_format)


class CentralCrop(object):
    """Crops the given image at the center.If the size is given, image will be cropped as size.
    If the central_fraction is given, image will cropped as (H * central_fraction, W * fraction).
    Size has a higher priority.

    Parameters
    ----------
    size : int or sequence of int
        The output size of the cropped image.
        If size is an integer, a square crop of size (size, size) is returned.
        If size is a sequence of length 2, it should be (height, width).
    central_fraction : float
        float (0, 1], fraction of size to crop

    Examples
    ----------
    With TensorLayer


    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.CentralCrop(size = (50, 50))
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (50, 50, 3)


    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.CentralCrop(central_fraction=0.5)
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (112, 112, 3)

    """

    def __init__(self, size=None, central_fraction=None):

        self.central_fraction = central_fraction
        self.size = size

    def __call__(self, image):

        F.central_crop(image, self.size, self.central_fraction)


class Compose(object):
    """Composes several transforms together.

    Parameters
    ----------
    transforms : list of 'transform' objects
        list of transforms to compose.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.Compose([tl.vision.transforms.ToTensor(data_format='HWC'),tl.vision.transforms.CentralCrop(size = 100)])
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (100, 100, 3)

    """

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, data):

        for t in self.transforms:

            data = t(data)

        return data


class Crop(object):
    """Crops an image to a specified bounding box.

    Parameters
    ----------
    offset_height : int
        Vertical coordinate of the top-left corner of the bounding box in image.
    offset_width: int
        Horizontal coordinate of the top-left corner of the bounding box in image.
    target_height: int
        Height of the bounding box.
    target_width: int
        Width of the bounding box.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.Crop(offset_height=10, offset_width=10, target_height=100, target_width=100)
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (100, 100, 3)

    """

    def __init__(self, offset_height, offset_width, target_height, target_width):

        self.offset_height = offset_height
        self.offset_width = offset_width
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image):

        return F.crop(image, self.offset_height, self.offset_width, self.target_height, self.target_width)


class Pad(object):
    """Pad the given image on all sides with the given "pad" value.

    Parameters
    ----------
    padding : int or sequenece
        Padding on each border.
        If a single int is provided， this is used to pad all borders.
        If sequence of length 2 is provided， this is the padding on left/right and top/bottom respectively.
        If a sequence of length 4 is provided， this is the padding for the left, top, right and bottom borders respectively.
    padding_value : number or sequenece
        Pixel fill value for constant fill. Default is 0.
        If a tuple of length 3, it is used to fill R, G, B channels respectively.
        This value is only used when the mode is constant.
    mode : str
        Type of padding. Default is constant.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.Pad(padding=10, padding_value=0, mode='constant')
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (244, 244, 3)

    """

    def __init__(self, padding, padding_value=0, mode='constant'):

        self.padding = padding
        self.padding_value = padding_value
        self.mode = mode

    def __call__(self, image):

        return F.pad(image, self.padding, self.padding_value, self.mode)


class Resize(object):
    """Resize the input image to the given size.

    Parameters
    ----------
    size : int or sequenece
        Desired output size.
        If size is a sequence like (h, w), output size will be matched to this.
        If size is an int, smaller edge of the image will be matched to this number.
        i.e, if height > width, then image will be rescaled to (size * height / width, size).
    interpolation : str
        Interpolation method. Default: 'bilinear'.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.Resize(size = (100,100), interpolation='bilinear')
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (100, 100, 3)

    """

    def __init__(self, size, interpolation='bilinear'):

        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):

        return F.resize(image, self.size, self.interpolation)


class Transpose(object):
    """Transpose image(s) by swapping dimension.

    Parameters
    ----------
    order : sequenece of int
        Desired output dimension order.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.Transpose(order=(2, 0 ,1))
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (3, 224, 224)

    """

    def __init__(self, order):

        self.order = order

    def __call__(self, image):

        return F.transpose(image, self.order)


class HWC2CHW(object):
    """Transpose a image shape (H, W, C) to shape (C, H, W).

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.HWC2CHW()
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (3, 224, 224)

    """

    def __call__(self, image):

        F.hwc_to_chw(image)


class CHW2HWC(object):
    """Transpose a image shape (C, H, W) to shape (H, W, C).

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(3, 224, 224) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.CHW2HWC()
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (224, 224, 3)

    """

    def __call__(self, image):

        F.chw_to_hwc(image)


class RgbToHsv(object):
    """Converts a image from RGB to HSV.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RgbToHsv()
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (224, 224, 3)

    """

    def __call__(self, image):

        F.rgb_to_hsv(image)


class HsvToRgb(object):
    """Converts a image from HSV to RGB.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.HsvToRgb()
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (224, 224, 3)

    """

    def __call__(self, image):

        F.hsv_to_rgb(image)


class RgbToGray(object):
    """Converts a image from RGB to grayscale.

    Parameters
    ----------
    num_output_channels: int
        (1 or 3) number of channels desired for output image. Default is 1.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RgbToGray(num_output_channels=1)
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (224, 224, 1)

    """

    def __init__(self, num_output_channels=1):

        self.num_output_channels = num_output_channels

    def __call__(self, image):

        F.rgb_to_gray(image, self.num_output_channels)


class AdjustBrightness(object):
    """Adjust brightness of the image.

    Parameters
    ----------
    brightness_factor: float
        How much to adjust the brightness. Can be any non negative number. 1 gives the original image.
        Default is 1.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.AdjustBrightness(brightness_factor=1)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, brightness_factor=1):
        self.brightness_factor = brightness_factor

    def __call__(self, image):

        return F.adjust_brightness(image, self.brightness_factor)


class AdjustContrast(object):
    """Adjust contrast of the image.

    Parameters
    ----------
    contrast_factor: float
        How much to adjust the contrast. Can be any non negative number. 1 gives the original image.
        Default is 1.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.AdjustContrast(contrast_factor=1)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, contrast_factor=1):

        self.contrast_factor = contrast_factor

    def __call__(self, image):

        return F.adjust_contrast(image, self.contrast_factor)


class AdjustHue(object):
    """Adjust hue of the image.

    Parameters
    ----------
    hue_factor: float
        How much to shift the hue channel. Should be in [-0.5, 0.5].
        0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative direction respectively.
        0 means no shift. Therefore, both -0.5 and 0.5 will give an image with complementary colors while 0 gives the original image.
        Default is 0.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.AdjustHue(hue_factor=0)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, hue_factor=0):

        self.hue_factor = hue_factor

    def __call__(self, image):

        return F.adjust_hue(image, self.hue_factor)


class AdjustSaturation(object):
    """Adjust saturation of the image.

    Parameters
    ----------
    saturation_factor: float
        How much to adjust the saturation. Can be any non negative number. 1 gives the original image.
        Default is 1.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.AdjustSaturation(saturation_factor=1)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, saturation_factor=1):

        self.saturation_factor = saturation_factor

    def __call__(self, image):

        return F.adjust_saturation(image, self.saturation_factor)


class FlipHorizontal(object):
    """Flip an image horizontally.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.FlipHorizontal()
    >>> image = transform(image)
    >>> print(image)

    """

    def __call__(self, image):

        return F.hflip(image)


class FlipVertical(object):
    """Flip an image vertically.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.FlipVertical()
    >>> image = transform(image)
    >>> print(image)

    """

    def __call__(self, image):

        return F.vflip(image)


class PadToBoundingbox(object):
    """Pad image with the specified height and width to target size.

    Parameters
    ----------
    offset_height: int
        Number of rows to add on top.
    offset_width: int
        Number of columns to add on the left.
    target_height: int
        Height of output image.
    target_width: int
        Width of output image.
    padding_value: int or sequence
        value to pad.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand( 224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.PadToBoundingbox(offset_height=10, offset_width=10, target_height=300, target_width=300, padding_value=0)
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (300, 300, 3)
    """

    def __init__(self, offset_height, offset_width, target_height, target_width, padding_value=0):
        self.offset_height = offset_height
        self.offset_width = offset_width
        self.target_height = target_height
        self.target_width = target_width
        self.padding_value = padding_value

    def __call__(self, image):

        return F.padtoboundingbox(
            image, self.offset_height, self.offset_width, self.target_height, self.target_width, self.padding_value
        )


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Parameters
    ----------
    mean: number or sequence
        If mean is a number, mean will be applied for all channels. Sequence of means for each channel.
    std: number or sequnece
        If std is a number, std will be applied for all channels.Sequence of standard deviations for each channel.
    data_format: str
        Data format of input image, should be 'HWC' or 'CHW'. Default: 'HWC'.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand( 224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.Normalize(mean = (155.0, 155.0, 155.0), std = (75.0, 75.0, 75.0),data_format='HWC')
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, mean, std, data_format='HWC'):

        self.mean = mean
        self.std = std
        self.data_format = data_format

    def __call__(self, image):

        return F.normalize(image, self.mean, self.std, self.data_format)


class StandardizePerImage(object):
    """For each 3-D image x in image, computes (x - mean) / adjusted_stddev, where mean is the average of all values in x.
    adjusted_stddev = max(stddev, 1.0/sqrt(N)) is capped away from 0 to protect against division by 0 when handling uniform images.
    N is the number of elements in x. stddev is the standard deviation of all values in x

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand( 224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.StandardizePerImage()
    >>> image = transform(image)
    >>> print(image)

    """

    def __call__(self, image):

        return F.standardize(image)


class RandomBrightness(object):
    """Random adjust brightness of the image.

    Parameters
    ----------
    brightness_factor: float or sequence
        Brightness adjustment factor (default=(1, 1)).
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness_factor), 1+brightness_factor].
        If it is a sequence, it should be [min, max] for the range.Should be non negative numbers.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomBrightness(brightness_factor=(0.5, 2))
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, brightness_factor=(1, 1)):
        self.brighthness_factor = brightness_factor

    def __call__(self, image):

        return F.random_brightness(image, self.brighthness_factor)


class RandomContrast(object):
    """Random adjust contrast of the image.

    Parameters
    ----------
    contrast_factor: float or sequence
        Contrast adjustment factor (default=(1, 1)).
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast_factor), 1+contrast_factor].
        If it is a sequence, it should be [min, max] for the range.Should be non negative numbers.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomContrast(contrast_factor=(0.5, 2))
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, contrast_factor=(1, 1)):

        self.contrast_factor = contrast_factor

    def __call__(self, image):

        return F.random_contrast(image, self.contrast_factor)


class RandomSaturation(object):
    """Random adjust saturation of the image.

    Parameters
    ----------
    saturation_factor: float or sequence
        Saturation adjustment factor (default=(1, 1)).
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation_factor), 1+saturation_factor].
        If it is a sequence, it should be [min, max] for the range.Should be non negative numbers.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomSaturation(saturation_factor=(0.5, 2))
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, saturation_factor=(1, 1)):

        self.saturation_factor = saturation_factor

    def __call__(self, image):

        return F.random_saturation(image, self.saturation_factor)


class RandomHue(object):
    """Random adjust hue of the image.

    Parameters
    ----------
    hue_factor: float or sequence
        Hue adjustment factor (default=(0, 0)).
        If it is a float, the factor is uniformly chosen from the range [-hue_factor, hue_factor].
        If it is a sequence, it should be [min, max] for the range.Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomHue(hue_factor=(-0.5, 0.5))
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, hue_factor=(0, 0)):

        self.hue_factor = hue_factor

    def __call__(self, image):

        return F.random_hue(image, self.hue_factor)


class RandomCrop(object):
    """Crop the given image at a random location.

    Parameters
    ----------
    size: int or sequence
        Desired output size of the crop.
        If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
        If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    padding: int or sequence, optional
        Optional padding on each border of the image.
        If a single int is provided this is used to pad all borders.
        If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
        If a sequence of length 4 is provided, it is used to pad left, top, right, bottom borders respectively.
        Default: 0.
    pad_if_needed: boolean
        It will pad the image if smaller than the desired size to avoid raising an exception.
        Since cropping is done after padding, the padding seems to be done at a random offset.
    fill: number or sequence
        Pixel fill value for constant fill. Default is 0.
        If a tuple of length 3, it is used to fill R, G, B channels respectively.
    padding_mode: str
        Type of padding. Default is constant.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomCrop(size=50, padding=10, pad_if_needed=False, fill=0, padding_mode='constant')
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (70,70,3)

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image):

        return F.random_crop(
            image,
            size=self.size,
            padding=self.padding,
            pad_if_needed=self.pad_if_needed,
            fill=self.fill,
            padding_mode=self.padding_mode,
        )


class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.

    Parameters
    ----------
    size: int or sequence
        Desired output size of the crop.
        If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
        If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    scale: tuple of float
        scale range of the cropped image before resizing, relatively to the origin image.
    ratio: tuple of float
        aspect ratio range of the cropped image before resizing.
    interpolation: str
        Type of interpolation. Default is bilinear.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomResizedCrop(size = (100, 100), scale = (0.08, 1.0), ratio = (3./4.,4./3.), interpolation = 'bilinear')
    >>> image = transform(image)
    >>> print(image)
    >>> image shape : (100,100,3)

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, image):

        return F.random_resized_crop(image, self.size, self.scale, self.ratio, self.interpolation)


class RandomFlipVertical(object):
    """Vertically flip the given image randomly with a given probability.

    Parameters
    ----------
    prob: float
        probability of the image being flipped. Default value is 0.5

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomFlipVertical(prob = 0.5)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, prob=0.5):

        self.prob = prob

    def __call__(self, image):

        return F.random_vflip(image, self.prob)


class RandomFlipHorizontal(object):
    """Horizontally flip the given image randomly with a given probability.

    Parameters
    ----------
    prob: float
        probability of the image being flipped. Default value is 0.5

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomFlipHorizontal(prob = 0.5)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, prob=0.5):

        self.prob = prob

    def __call__(self, image):

        return F.random_hflip(image, self.prob)


class RandomRotation(object):
    """Rotate the image by random angle.

    Parameters
    ----------
    degrees: number or sequnence
        Range of degrees to select from.
        If degrees is a number, the range of degrees will be (-degrees, +degrees).
        If degrees is a sequence, the range of degrees will (degrees[0], degrees[1]).
    interpolation: str
        Interpolation method. Default is 'bilinear'.
    expand: boolean
        If true, expands the output to make it large enough to hold the entire rotated image.
        If false or omitted, make the output image the same size as the input image.
        Note that the expand flag assumes rotation around the center and no translation.
    center: sequence or None
        Optional center of rotation, (x, y). Origin is the upper left corner.
        Default is the center of the image.
    fill: number or sequence
        Pixel fill value for the area outside the rotated image. Default is 0.


    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomRotation(degrees=30, interpolation='bilinear', expand=False, center=None, fill=0)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, degrees, interpolation='bilinear', expand=False, center=None, fill=0):

        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, image):

        return F.random_rotation(image, self.degrees, self.interpolation, self.expand, self.center, self.fill)


class RandomShear(object):
    """Shear the image by random angle.

    Parameters
    ----------
    degrees: number or sequnence
        Range of degrees to select from.
        If degrees is a number, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
        If shear is a sequence of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied.
        If shear is a sequence of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
    interpolation: str
        Interpolation method. Default is 'bilinear'.
    fill: number or sequence
        Pixel fill value for the area outside the sheared image. Default is 0.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomShear(degrees=30, interpolation='bilinear', fill=0)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, degrees, interpolation='bilinear', fill=0):

        self.degrees = degrees
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, image):

        return F.random_shear(image, self.degrees, self.interpolation, self.fill)


class RandomShift(object):
    """Shift the image by random translations.

    Parameters
    ----------
    shift: list or tuple
        Maximum absolute fraction for horizontal and vertical translations.
        shift=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a.
        vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b.
    interpolation: str
        Interpolation method. Default is 'bilinear'.
    fill: number or sequence
        Pixel fill value for the area outside the sheared image. Default is 0.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomShift(shift=(0.2, 0.2), interpolation='bilinear', fill=0)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, shift, interpolation='bilinear', fill=0):

        self.shift = shift
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, image):

        return F.random_shift(image, self.shift, self.interpolation, self.fill)


class RandomZoom(object):
    """Zoom the image by random scale.

    Parameters
    ----------
    zoom: list or tuple
        Scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b.
    interpolation: str
        Interpolation method. Default is 'bilinear'.
    fill: number or sequence
        Pixel fill value for the area outside the sheared image. Default is 0.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomZoom(zoom=(0.2, 0.5), interpolation='bilinear', fill=0)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, zoom, interpolation='bilinear', fill=0):

        self.zoom = zoom
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, image):

        return F.random_zoom(image, self.zoom, self.interpolation, self.fill)


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant.

    Parameters
    ----------
    degrees: number or sequnence
        Range of degrees to select from.
        If degrees is a number, the range of degrees will be (-degrees, +degrees).
        If degrees is a sequence, the range of degrees will (degrees[0], degrees[1]).
        Set to 0 to deactivate rotations.
    shift: sequence or None
        Maximum absolute fraction for horizontal and vertical translations.
        shift=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a.
        vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b.
        Will not shift by default.
    shear: number or sequnence or None
        Range of degrees to select from.
        If degrees is a number, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
        If shear is a sequence of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied.
        If shear is a sequence of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
        Will not apply shear by default.
    zoom: sequence or None
        Scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b.
        Will not zoom by default.
    interpolation: str
        Interpolation method. Default is 'bilinear'.
    fill: number or sequence
        Pixel fill value for the area outside the sheared image. Default is 0.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.RandomAffine(degrees=30, shift=(0.2,0.2), zoom=(0.2, 0.5), shear=30, interpolation='bilinear', fill=0)
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, degrees, shift=None, zoom=None, shear=None, interpolation='bilinear', fill=0):

        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number, it must be positive.' 'But got {}.'.format(degrees))
            degrees = [-degrees, degrees]
        elif not (isinstance(degrees, (list, tuple)) and len(degrees) == 2):
            raise TypeError('If degrees is a list or tuple, it should be length of 2.' 'But got {}'.format(degrees))

        self.degrees = (float(x) for x in degrees)

        if shift is not None:
            if not (isinstance(shift, (list, tuple)) and len(shift) == 2):
                raise TypeError("shift should be a list or tuple of length 2." "But got {}.".format(shift))

            for s in shift:
                if not (0.0 <= s <= 1.0):
                    raise ValueError('shift values should be between 0 and 1.' 'But got {}.'.format(shift))
            self.shift = shift

        if zoom is not None:
            if not (isinstance(zoom, (list, tuple)) and len(zoom) == 2):
                raise TypeError("zoom should be a list or tuple of length 2." "But got {}.".format(zoom))

            if not (0 <= zoom[0] <= zoom[1]):
                raise ValueError("zoom valuse should be positive, and zoom[1] should be less than zoom[0].")

            self.zoom = zoom

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                shear = [-shear, shear]
            elif not (isinstance(shear, (list, tuple)) and len(shear) in (2, 4)):
                raise TypeError('shear should be a list or tuple of length (2, 4).')

            self.shear = (float(x) for x in shear)

        self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (list, tuple, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    def __call__(self, image):

        return F.random_affine(image, self.degrees, self.shift, self.zoom, self.shear, self.interpolation, self.fill)


class ColorJitter(object):
    """Randomly change the brightness, contrast, saturation and hue of an image.

    Parameters
    ----------
    brightness: float or sequence
        Brightness adjustment factor (default=(1, 1)).
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness_factor), 1+brightness_factor].
        If it is a sequence, it should be [min, max] for the range.Should be non negative numbers.
    contrast: float or sequence
        Contrast adjustment factor (default=(1, 1)).
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast_factor), 1+contrast_factor].
        If it is a sequence, it should be [min, max] for the range.Should be non negative numbers.
    saturation: float or sequence
        Saturation adjustment factor (default=(1, 1)).
        If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation_factor), 1+saturation_factor].
        If it is a sequence, it should be [min, max] for the range.Should be non negative numbers.
    hue: float or sequence
        Hue adjustment factor (default=(0, 0)).
        If it is a float, the factor is uniformly chosen from the range [-hue_factor, hue_factor].
        If it is a sequence, it should be [min, max] for the range.Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

    Examples
    ----------
    With TensorLayer

    >>> image = (np.random.rand(224, 224, 3) * 255.).astype(np.uint8)
    >>> transform = tl.vision.transforms.ColorJitter(brightness=(1,5), contrast=(1,5), saturation=(1,5), hue=(-0.2,0.2))
    >>> image = transform(image)
    >>> print(image)

    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):

        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = np.random.permutation(np.arange(4))

        b = None if brightness is None else float(np.random.uniform(brightness[0], brightness[1]))
        c = None if contrast is None else float(np.random.uniform(contrast[0], contrast[1]))
        s = None if saturation is None else float(np.random.uniform(saturation[0], saturation[1]))
        h = None if hue is None else float(np.random.uniform(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, image):

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                image = F.adjust_brightness(image, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                image = F.adjust_contrast(image, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                image = F.adjust_saturation(image, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                image = F.adjust_hue(image, hue_factor)

        return image
