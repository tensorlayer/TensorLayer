import numpy as np
from PIL import Image, ImageOps, ImageEnhance, __version__
import random
import colorsys
import numbers
import math
import io
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

augment_error_message = 'img should be PIL image. Got {}.'


def ToTensor(image):

    image = np.asarray(image).astype(np.float32)
    return image


def ToPIL(image):
    """
    Convert the input image to PIL format.

    Args:
        img: Image to be converted.

    Returns:
        img (PIL image), Converted image.
    """
    return Image.fromarray(np.array(image).astype(np.uint8))


def Decode(image):
    """
    Decode the input image to PIL image format in RGB mode.

    Args:
        img: Image to be decoded.

    Returns:
        img (PIL image), Decoded image in RGB mode.
    """

    try:
        data = io.BytesIO(image)
        img = Image.open(data)
        return img.convert('RGB')
    except IOError as e:
        raise ValueError("{0}\nWARNING: Failed to decode given image.".format(e))
    except AttributeError as e:
        raise ValueError("{0}\nWARNING: Failed to decode, Image might already be decoded.".format(e))


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
    is_hwc:
        If is_hwc is True, the order of image channels is [B,H,W,C] or [H,W,C]. If is_hwc is False, the order of image channels is [B,C,H,W] or [C,H,W,]
	Returns:
		Output [batch, target_height, target_width, channels] or [target_height, target_width, channels]
	-------

    '''
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape_size = len(image.shape)

    if not shape_size in (3, 4):
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). \
                             Got {}'.format(image.shape)
        )
    if shape_size == 3:
        if is_hwc:
            height, width, channels = image.shape
        else:
            channels, height, width = image.shape
    else:
        if is_hwc:
            batch, height, width, channels = image.shape
        else:
            batch, channels, height, width = image.shape

    if offset_width < 0:
        raise ValueError('offset_width must be >0.')
    if offset_height < 0:
        raise ValueError('offset_height must be >0.')
    if target_height < 0:
        raise ValueError('target_height must be >0.')
    if target_width < 0:
        raise ValueError('target_width must be >0.')
    if offset_width + target_width > width:
        raise ValueError('offset_width + target_width must be <= image width.')
    if offset_height + target_height > height:
        raise ValueError('offset_height + target_height must be <= image height.')

    if shape_size == 3:
        if is_hwc:
            return ToTensor(
                image[offset_height:offset_height + target_height, offset_width:offset_width + target_width, :]
            )
        else:
            return ToTensor(
                image[:, offset_height:offset_height + target_height, offset_width:offset_width + target_width]
            )
    else:
        if is_hwc:
            return ToTensor(
                image[:, offset_height:offset_height + target_height, offset_width:offset_width + target_width, :]
            )
        else:
            return ToTensor(
                image[:, :, offset_height:offset_height + target_height, offset_width:offset_width + target_width]
            )


def CentralCrop(image, central_fraction=None, size=None, is_hwc=True):
    '''

	Parameters
	----------
	image :
		input Either a 3-D float Tensor of shape [height, width, depth] or a 4-D Tensor of shape [batch, height, width, depth],
	central_fraction :
		float (0, 1], fraction of size to crop
	size:
		size (Union[int, sequence]) – The output size of the cropped image. If size is an integer, a square crop of size (size, size) is returned.
		If size is a sequence of length 2, it should be (height, width).
	Returns :
		3-D float Tensor or 4-D float Tensor, as per the input.
	-------
	If backend is tensorflow, central_fraction will be used preferentially. if size is used, the height-width ratio will be equivalent to original ratio..
	If backend is mindspore, size will be used preferentially.
	'''
    if size is None and central_fraction is None:
        raise ValueError('central_fraction and size can not be both None')
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape_size = len(image.shape)
    if not shape_size in (3, 4):
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). \
                             Got {}'.format(image.shape)
        )

    if shape_size == 3:
        if is_hwc:
            height, width, channels = image.shape
        else:
            channels, height, width = image.shape
    else:
        if is_hwc:
            batch, height, width, channels = image.shape
        else:
            batch, channels, height, width = image.shape
    if size is None:
        if central_fraction > 1 or central_fraction <= 0:
            raise ValueError('central_fraction must be in (0,1].')
        target_height = int(round(height * central_fraction))
        target_width = int(round(width * central_fraction))
        size = (target_height, target_width)
    if isinstance(size, int):
        size = (size, size)
    crop_height, crop_width = size
    crop_top = int(round((height - crop_height) / 2.))
    crop_left = int(round((width - crop_width) / 2.))

    return Crop(image, crop_top, crop_left, crop_height, crop_width, is_hwc)


def hsv_to_rgb(np_hsv_img, is_hwc):
    """
    Convert HSV img to RGB img.

    Args:
        np_hsv_img (numpy.ndarray): NumPy HSV image array of shape (H, W, C) or (C, H, W) to be converted.
        is_hwc (Bool): If True, the shape of np_hsv_img is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_rgb_img (numpy.ndarray), NumPy HSV image with same shape of np_hsv_img.
    """
    if is_hwc:
        h, s, v = np_hsv_img[:, :, 0], np_hsv_img[:, :, 1], np_hsv_img[:, :, 2]
    else:
        h, s, v = np_hsv_img[0, :, :], np_hsv_img[1, :, :], np_hsv_img[2, :, :]
    to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = to_rgb(h, s, v)

    if is_hwc:
        axis = 2
    else:
        axis = 0
    np_rgb_img = np.stack((r, g, b), axis=axis)
    return np_rgb_img


def HsvToRgb(image, is_hwc=True):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape_size = len(image.shape)

    if not shape_size in (3, 4):
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). \
                             Got {}'.format(image.shape)
        )
    if shape_size == 3:
        batch_size = 0
        if is_hwc:
            num_channels = image.shape[2]
        else:
            num_channels = image.shape[0]
    else:
        batch_size = image.shape[0]
        if is_hwc:
            num_channels = image.shape[3]
        else:
            num_channels = image.shape[1]

    if num_channels != 3:
        raise TypeError('img should be 3 channels RGB img. Got {} channels'.format(num_channels))
    if batch_size == 0:
        return hsv_to_rgb(image, is_hwc)
    return ToTensor([hsv_to_rgb(img, is_hwc) for img in image])


def AdjustBrightness(image, factor):
    '''

	Parameters
	----------
	image:
		input NumPy image array or PIL image
	factor:
		factor should be in the range (-1,1)
	Returns:
	-------
		np darray image
	'''
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if factor >= 1 or factor <= -1:
        raise ValueError('factor must be in (-1,1).')
    image = image + factor * 255
    image = np.clip(image, 0, 255)

    return ToTensor(image)


def AdjustContrast(image, factor):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    image = ImageEnhance.Contrast(image).enhance(factor)

    return ToTensor(image)


def AdjustHue(image, factor):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    image_hue_factor = factor
    if not -1 <= image_hue_factor <= 1:
        raise ValueError('image_hue_factor {} is not in [-1, 1].'.format(image_hue_factor))

    mode = image.mode
    if mode in {'L', '1', 'I', 'F'}:
        return image

    hue, saturation, value = image.convert('HSV').split()

    np_hue = np.array(hue, dtype=np.uint8)

    with np.errstate(over='ignore'):
        np_hue += np.uint8(image_hue_factor * 255)
    hue = Image.fromarray(np_hue, 'L')

    image = Image.merge('HSV', (hue, saturation, value)).convert(mode)

    return ToTensor(image)


def AdjustSaturation(image, factor):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor)

    return ToTensor(image)


def FlipHorizontal(image):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))

    image = np.fliplr(image)

    return image


def FlipVertical(image):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    image = np.flipud(image)

    return image


def GrayToRgb(image):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape = image.shape
    output_image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if len(shape) == 3:
        for i in range(3):
            output_image[:, :, i] = image[:, :, 1]
    elif len(shape) == 2:
        for i in range(3):
            output_image[:, :, i] = image

    return ToTensor(output_image)


def RgbToGray(image):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))
    '''
	将彩色图像转换为灰度（模式“L”）时，库使用ITU-R 601-2 Luma转换：
	L = R * 299/1000 + G * 587/1000 + B * 114/1000
	'''
    image = image.convert('L')
    return ToTensor(image)


def PadToBoundingbox(image, offset_height, offset_width, target_height, target_width, padding_value=0, is_hwc=True):
    '''

	Parameters
	----------
	image:
		A 3-D numpy ndarray or 4-D numpy ndarray image
	offset_height:
		 Number of rows of zeros to add on top.
	offset_width:
		 Number of columns of zeros to add on the left.
	target_height:
		Height of output image.
	target_width
		 Width of output image.
	Returns
		A numpy ndarray image
	-------
	'''

    if offset_height < 0:
        raise ValueError("offset_height must be >= 0")
    if offset_width < 0:
        raise ValueError("offset_width must be >= 0")
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape_size = len(image.shape)
    if not shape_size in (3, 4):
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). \
                             Got {}'.format(image.shape)
        )
    if shape_size == 3:
        if is_hwc:
            height, width, channels = image.shape
        else:
            channels, height, width = image.shape
    else:
        if is_hwc:
            batch, height, width, channels = image.shape
        else:
            batch, channels, height, width = image.shape
    top = offset_height
    bottom = target_height - height - top
    left = offset_width
    right = target_width - width - left

    if bottom < 0:
        raise ValueError("target_height must be >= offset_height + height")

    if right < 0:
        raise ValueError("target_width must be >= offset_width + width")

    if shape_size == 3:
        if is_hwc:
            return ToTensor(
                np.pad(
                    image, ((top, bottom), (left, right), (0, 0)), mode='constant',
                    constant_values=(padding_value, padding_value)
                )
            )
        else:
            return ToTensor(
                np.pad(
                    image, ((0, 0), (top, bottom), (left, right)), mode='constant',
                    constant_values=(padding_value, padding_value)
                )
            )
    else:
        if is_hwc:
            return ToTensor(
                np.pad(
                    image, ((0, 0), (top, bottom), (left, right), (0, 0)), mode='constant',
                    constant_values=(padding_value, padding_value)
                )
            )
        else:
            return ToTensor(
                np.pad(
                    image, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='constant',
                    constant_values=(padding_value, padding_value)
                )
            )


def Pad(image, padding, padding_value=0, mode='constant', is_hwc=True):
    '''

    Parameters
    ----------
    image:
        A 3-D or 4-D Tensor.
    padding:
        An integer or a list/tuple.  If a single number is provided, pad all borders with this value.
        If a tuple or list of 2 values is provided, pad the left and top with the first value and the right and bottom with the second value.
        If 4 values are provided as a list or tuple, pad the left, top, right and bottom respectively.
    padding_value:
        In "CONSTANT" mode, the scalar pad value to use. Must be same type as tensor.
    mode:
        One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    Returns:
        A padded Tensor. Has the same type as tensor.
    -------

    '''
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape_size = image.shape
    if len(shape_size) == 3:
        batch_size = 0
    elif len(shape_size) == 4:
        batch_size = shape_size[0]
    else:
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). \
                                      Got {}'.format(image.shape)
        )
    if mode not in ('constant', 'edge', 'reflect', 'symmetric'):
        raise TypeError('mode should be one of (constant,edge,reflect,symmetric).')

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
        if is_hwc:
            padding = (padding[0], padding[1], (0, 0))
        else:
            padding = (
                (0, 0),
                padding[0],
                padding[1],
            )
    else:
        if is_hwc:
            padding = ((0, 0), padding[0], padding[1], (0, 0))
        else:
            padding = ((0, 0), (0, 0), padding[0], padding[1])
    if mode == 'constant':
        return ToTensor(np.pad(image, padding, mode=mode, constant_values=(padding_value, padding_value)))
    else:
        return ToTensor(np.pad(image, padding, mode=mode))


def Standardization(image, mean=None, std=None, channel_mode=False, is_hwc=True):
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

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    num_shape = image.shape
    if is_hwc:
        height, width, channels = 0, 1, 2
    else:
        channels, height, width = 0, 1, 2
    if mean is not None and std is not None:
        if len(mean) != len(std):
            raise ValueError("Length of mean and std must be equal")
        if len(mean) == 1:
            mean = [mean[0]] * num_shape[channels]
            std = [std[0]] * num_shape[channels]
        mean = np.array(mean, dtype=image.dtype)
        std = np.array(std, dtype=image.dtype)
        return ToTensor((image - mean[:, None, None]) / std[:, None, None])
    elif mean is None and std is None:
        if channel_mode:
            num_pixels = num_shape[height] * num_shape[width]
            image_mean = np.mean(image, axis=(height, width))
            stddev = np.std(image, axis=(height, width))
            min_sttdev = 1 / np.sqrt(num_pixels)
            min_sttdev = [min_sttdev] * num_shape[channels]
            adjusted_sttdev = np.maximum(stddev, min_sttdev)
            image -= image_mean
            image = np.divide(image, adjusted_sttdev)
            return ToTensor(image)
        else:
            num_pixels = num_shape[height] * num_shape[width] * num_shape[channels]
            image_mean = np.mean(image, axis=(0, 1, 2))
            image_mean = [image_mean] * 3
            stddev = np.std(image, axis=(0, 1, 2))
            min_sttdev = 1 / np.sqrt(num_pixels)
            adjusted_sttdev = np.maximum(stddev, min_sttdev)
            adjusted_sttdev = [adjusted_sttdev] * 3
            image -= image_mean
            image = np.divide(image, adjusted_sttdev)
            return ToTensor(image)
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
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if factor < 0 or factor > 1:
        raise ValueError('factor should be in [0,1].')
    delta = random.uniform(-factor, factor)
    image = image + delta * 255
    image = np.clip(image, 0, 255)

    return image


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
    if upper <= lower:
        raise ValueError('upper must be > lower')
    if lower < 0:
        raise ValueError('lower must be non-negative')
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    factor = random.uniform(lower, upper)
    image = ImageEnhance.Contrast(image).enhance(factor)

    return ToTensor(image)


def RandomHue(image, factor, seed=None):
    '''

    Parameters
    ----------
    image:
        RGB image or images. The size of the last dimension must be 3.
    factor:
        float. The maximum value for the random factor.
    seed:
         An operation-specific seed. I

    Returns:
        Adjusted numpy ndarrry image(s).
    -------

    '''

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    if factor > 0.5 or factor < 0:
        raise ValueError('factor should be in [0,0.5].')

    image_hue_factor = random.uniform(-factor, factor)
    mode = image.mode
    if mode in {'L', '1', 'I', 'F'}:
        return image

    hue, saturation, value = image.convert('HSV').split()

    np_hue = np.array(hue, dtype=np.uint8)

    with np.errstate(over='ignore'):
        np_hue += np.uint8(image_hue_factor * 255)
    hue = Image.fromarray(np_hue, 'L')

    image = Image.merge('HSV', (hue, saturation, value)).convert(mode)

    return ToTensor(image)


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

    Returns；
        Adjusted numpy ndarray image(s).
    -------
    '''
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))
    if upper <= lower:
        raise ValueError('upper must be > lower.')

    if lower < 0:
        raise ValueError('lower must be non-negative.')
    factor = random.uniform(lower, upper)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor)

    return ToTensor(image)


def RandomCrop(image, size, is_hwc=True):
    '''

    Parameters
    ----------
    image:
        Input an image  to crop.
    size:
        if size is an integer, shape of cropped image  will be [size, size, 3]. if length of size is 2.
        shape of cropped image  will be [height, width, 3].
    Returns:
        A cropped image of the same rank as image and shape size.
    -------
    '''
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
        raise ValueError("Size should be a single integer or a list/tuple (h, w) of length 2.")

    def _input_to_factor_(image, size, is_hwc):
        if len(image.shape) == 3:
            if is_hwc:
                height, width, channels = image.shape
            else:
                channels, height, width = image.shape
        else:
            if is_hwc:
                batch, height, width, channels = image.shape
            else:
                batch, channels, height, width = image.shape

        target_height, target_width = size
        if target_height > height or target_width > width:
            raise ValueError("Crop size {} is larger than input image size {}".format(size, (height, width)))
        if target_height == height and target_width == width:
            return 0, 0, height, width

        top = random.randint(0, height - target_height)
        left = random.randint(0, width - target_width)
        return top, left, target_height, target_width

    top, left, height, width = _input_to_factor_(image, size, is_hwc)

    return Crop(image, top, left, height, width, is_hwc)


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
        An image.ResizeMethod, or string equivalent. Defaults to bilinear.
    preserve_aspect_ratio:
        Whether to preserve the aspect ratio.
    antialias:
        Whether to use an anti-aliasing filter when downsampling an image.
    Returns:
        an resized image
    -------
    '''
    DE_PY_INTER_MODE = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'cubic': Image.CUBIC,
        'lanczos': Image.LANCZOS,
        'bicubic': Image.BICUBIC
    }
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) or len(size) == 2:
        target_height, target_width = size
        size = (target_width, target_height)
    else:
        raise ValueError("Size should be a single integer or a list/tuple (h, w) of length 2.")
    if method not in ('nearest', 'bilinear', 'cubic', 'lanczos', 'bicubic'):
        raise TypeError('Unknown resize method! resize method must be in (nearest bilinear cubic lanczos bicubic)')

    if preserve_aspect_ratio:
        width, height = image.size
        target_width, target_height = size
        scale_factor_height = float(target_height / height)
        scale_factor_width = float(target_width / width)
        scale_factor = np.minimum(scale_factor_height, scale_factor_width)
        new_target_height = int(scale_factor * height)
        new_target_width = int(scale_factor * width)
        size = (new_target_width, new_target_height)
    interpolation = DE_PY_INTER_MODE[method]
    image = image.resize(size, interpolation)
    if antialias:
        image = image.resize(size, Image.ANTIALIAS)

    return ToTensor(image)


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
    if method not in ["bilinear", "nearest"]:
        raise ValueError('method must be bilinear or nearest.')
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    boxes = np.asarray(boxes)
    box_indices = np.asarray(box_indices)
    image_shape = image.shape
    if len(image_shape) == 4:
        batch_size = image_shape[0]
    elif len(image_shape) == 3:
        image = np.expand_dims(image, axis=0)
    else:
        raise ValueError('Input must be a 3-D or 4-D  image Tensor.')

    box_num = boxes.shape[0]  # boxes.shape is [n,4]. n is the number of boxes.
    if not is_hwc:  # 判断通道顺序,为了便于后续计算，将通道顺序调整为HWC or BHWC
        image = np.transpose(image, (0, 2, 3, 1))
    batch_size, height, width, channels = image.shape
    return_image = np.zeros((box_num, crop_size[0], crop_size[1], 3))
    for i in range(box_num):
        y1, x1, y2, x2 = boxes[i]  # 首先判断图像是否需要翻转 ， 若y1>y2 需要垂直翻转， 若x1>x2 需要水平翻转
        cur_image = image[box_indices[i]]
        if y1 > y2:
            cur_image = FlipVertical(cur_image)
            y1, y2 = y2, y1
        if x1 > x2:
            cur_image = FlipHorizontal(cur_image)
            x1, x2 = x2, x1
        top_padding = 0 if y1 > 0 else int(round(height * (-y1)))
        left_padding = 0 if x1 > 0 else int(round(width * (-x1)))
        bottom_padding = 0 if y2 < 1 else int(round(height * (y2 - 1)))
        right_padding = 0 if x2 < 1 else int(round(width * (x2 - 1)))
        # 判断是否需要padding
        target_height = top_padding + height + bottom_padding
        target_width = left_padding + width + right_padding
        if target_height != height or target_width != width:
            cur_image = PadToBoundingbox(
                cur_image, offset_height=top_padding, offset_width=left_padding, target_height=target_height,
                target_width=target_width, padding_value=extrapolation_value, is_hwc=is_hwc
            )
        offset_height = 0 if y1 < 0 else int(round(height * y1))
        offset_width = 0 if x1 < 0 else int(round(width * x1))
        target_height = int(round(height * (y2 - y1)))
        target_width = int(round(width * (x2 - x1)))
        crop_image = Crop(cur_image, offset_height, offset_width, target_height, target_width, is_hwc)
        resized_image = Resize(crop_image, crop_size, method=method)
        return_image[i] = resized_image
    if not is_hwc:
        return_image = np.transpose(return_image, (0, 3, 1, 2))
    return ToTensor(return_image)


def CropOrPad(image, target_height, target_width, is_hwc=True):
    '''
    Resizes an image to a target width and height by either centrally cropping the image or padding it evenly with zeros.
    Parameters
    ----------
    image:
        3-D Tensor of shape [height, width, channels].
    target_height:
        Target height.
    target_width:
        Target width.
    Returns:
        Cropped and/or padded image.
    -------
    '''

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape_size = len(image.shape)
    if not shape_size in (3, 4):
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). \
                             Got {}'.format(image.shape)
        )
    if target_height < 0:
        raise ValueError('target_height must be >0.')
    if target_width < 0:
        raise ValueError('target_width must be >0.')
    if shape_size == 3:
        if is_hwc:
            height, width, channels = image.shape
        else:
            channels, height, width = image.shape
    else:
        if is_hwc:
            batch, height, width, channels = image.shape
        else:
            batch, channels, height, width = image.shape
    offset_height = height - target_height
    offset_width = width - target_width
    offset_crop_height = max(offset_height // 2, 0)
    offset_crop_width = max(offset_width // 2, 0)
    offset_pad_height = max(-offset_height // 2, 0)
    offset_pad_width = max(-offset_width // 2, 0)
    cropped = Crop(
        image, offset_crop_height, offset_crop_width, min(height, target_height), min(width, target_width), is_hwc
    )

    padded = PadToBoundingbox(cropped, offset_pad_height, offset_pad_width, target_height, target_width, is_hwc=is_hwc)

    return ToTensor(padded)


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
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    shape_size = len(image.shape)
    if not shape_size in (3, 4):
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). \
                             Got {}'.format(image.shape)
        )
    if target_height < 0:
        raise ValueError('target_height must be >0.')
    if target_width < 0:
        raise ValueError('target_width must be >0.')
    if shape_size == 3:
        if is_hwc:
            height, width, channels = image.shape
        else:
            channels, height, width = image.shape
    else:
        if is_hwc:
            batch, height, width, channels = image.shape
        else:
            batch, channels, height, width = image.shape
    height = float(height)
    width = float(width)
    ratio = max(height / target_height, width / target_width)
    resized_height = int(round(height / ratio))
    resized_width = int(round(width / ratio))
    padding_height = max(0, int(round((target_height - resized_height) / 2)))
    padding_width = max(0, int(round((target_width - resized_width) / 2)))
    resized = Resize(
        image, size=(resized_height, resized_width), method=method, antialias=antialias
    )  #需要解决 batch images的resize
    padded = PadToBoundingbox(resized, padding_height, padding_width, target_height, target_width, is_hwc=is_hwc)
    return ToTensor(padded)


def rgb_to_hsv(np_rgb_img, is_hwc):
    """
    Convert RGB img to HSV img.

    Args:
        np_rgb_img (numpy.ndarray): NumPy RGB image array of shape (H, W, C) or (C, H, W) to be converted.
        is_hwc (Bool): If True, the shape of np_hsv_img is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_hsv_img (numpy.ndarray), NumPy HSV image with same type of np_rgb_img.
    """
    if is_hwc:
        r, g, b = np_rgb_img[:, :, 0], np_rgb_img[:, :, 1], np_rgb_img[:, :, 2]
    else:
        r, g, b = np_rgb_img[0, :, :], np_rgb_img[1, :, :], np_rgb_img[2, :, :]
    to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    h, s, v = to_hsv(r, g, b)
    if is_hwc:
        axis = 2
    else:
        axis = 0
    np_hsv_img = np.stack((h, s, v), axis=axis)
    return np_hsv_img


def RgbToHsv(image, is_hwc=True):

    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))

    shape_size = len(image.shape)

    if not shape_size in (3, 4):
        raise TypeError(
            'img shape should be (H, W, C)/(N, H, W, C)/(C ,H, W)/(N, C, H, W). \
                         Got {}'.format(image.shape)
        )

    if shape_size == 3:
        batch_size = 0
        if is_hwc:
            num_channels = image.shape[2]
        else:
            num_channels = image.shape[0]
    else:
        batch_size = image.shape[0]
        if is_hwc:
            num_channels = image.shape[3]
        else:
            num_channels = image.shape[1]

    if num_channels != 3:
        raise TypeError('img should be 3 channels RGB img. Got {} channels'.format(num_channels))
    if batch_size == 0:
        return ToTensor(rgb_to_hsv(image, is_hwc))
    return ToTensor([rgb_to_hsv(img, is_hwc) for img in image])


def Transpose(image, order):
    """
        Transpose the input image with order
    """
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))
    if len(image.shape) == 3:
        if len(order) != 3:
            raise ValueError('if image is 3-D tensor, order should be a list/tuple with length of 3')
        return ToTensor(np.transpose(image, order))
    elif len(image.shape) == 4:
        if len(order) != 3:
            raise ValueError('if image is 3-D tensor, order should be a list/tuple with length of 3')
        return ToTensor(np.transpose(image, order))
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')


def RandomRotation(
    image, degrees, fill_mode='nearest', fill_value=0, center=None, expand=False, is_hwc=True, interpolation_order=1
):
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it cannot be negative.")
        degrees = (-degrees, degrees)
    elif isinstance(degrees, (list, tuple)):
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, the length must be 2.")
    else:
        raise TypeError("Degrees must be a single non-negative number or a sequence")

    DE_PY_INTER_MODE = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'antialias': Image.ANTIALIAS,
        'bicubic': Image.BICUBIC
    }
    if fill_mode not in ('nearest', 'bilinear', 'antialias', 'bicubic'):
        raise TypeError('Fill_mode must be in (nearest,bilinear, antialias,bicubic)')

    if isinstance(fill_value, int):
        fill_value = tuple([fill_value] * 3)

    angle = random.uniform(degrees[0], degrees[1])
    fill_mode = DE_PY_INTER_MODE[fill_mode]
    return ToTensor(image.rotate(angle, fill_mode, expand, center, fillcolor=fill_value))


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
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    if isinstance(shift, numbers.Number):
        width_fraction = shift
        height_fraction = shift
    elif isinstance(shift, list) or isinstance(shift, tuple):
        if len(shift) == 2:
            width_fraction = shift[0]
            height_fraction = shift[1]
    else:
        raise ValueError('shift must be int or list/tuple of length 2')

    DE_PY_INTER_MODE = {'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    if fill_mode not in ('nearest', 'bilinear', 'bicubic'):
        raise TypeError('Fill_mode must be in (nearest,bilinear,bicubic)')
    fill_mode = DE_PY_INTER_MODE[fill_mode]
    width, height = image.size
    max_dx = width_fraction * width
    max_dy = height_fraction * height
    translations = (np.round(random.uniform(-max_dx, max_dx)), np.round(random.uniform(-max_dy, max_dy)))

    scale = 1.0
    shear = 0.0
    output_size = image.size
    center = (width * 0.5 + 0.5, height * 0.5 + 0.5)

    angle = math.radians(0)
    shear = math.radians(shear)
    shear = [shear, 0]
    scale = 1.0 / scale
    d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
        math.sin(angle + shear[0]) * math.sin(angle + shear[1])
    matrix = [
        math.cos(angle + shear[0]),
        math.sin(angle + shear[0]), 0, -math.sin(angle + shear[1]),
        math.cos(angle + shear[1]), 0
    ]
    matrix = [scale / d * m for m in matrix]
    matrix[2] += matrix[0] * (-center[0] - translations[0]) + matrix[1] * (-center[1] - translations[1])
    matrix[5] += matrix[3] * (-center[0] - translations[0]) + matrix[4] * (-center[1] - translations[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]

    if __version__ >= '5':
        kwargs = {"fillcolor": fill_value}
    else:
        kwargs = {}
    return ToTensor(image.transform(output_size, Image.AFFINE, matrix, fill_mode, **kwargs))


def RandomShear(image, degree, fill_mode='nearest', fill_value=0, is_hwc=True, interpolation_order=1):
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
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))
    DE_PY_INTER_MODE = {'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    if fill_mode not in ('nearest', 'bilinear', 'bicubic'):
        raise TypeError('Fill_mode must be in (nearest,bilinear,bicubic)')
    fill_mode = DE_PY_INTER_MODE[fill_mode]
    width, height = image.size
    translations = (0, 0)
    scale = 1.0
    shear = degree
    output_size = image.size
    center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
    angle = math.radians(0)

    if shear is not None:
        if isinstance(shear, numbers.Number):
            shear = (-1 * shear, shear)
            shear = [random.uniform(shear[0], shear[1]), random.uniform(shear[0], shear[1])]
        elif len(shear) == 2 or len(shear) == 4:
            if len(shear) == 2:
                shear = [shear[0], shear[1], shear[0], shear[1]]
            elif len(shear) == 4:
                shear = [s for s in shear]
            shear = [random.uniform(shear[0], shear[1]), random.uniform(shear[2], shear[3])]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " + "two values. Got {}".format(shear)
            )
        shear = [math.radians(s) for s in shear]
    else:
        shear = [0, 0]


    d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
        math.sin(angle + shear[0]) * math.sin(angle + shear[1])
    matrix = [
        math.cos(angle + shear[0]),
        math.sin(angle + shear[0]), 0, -math.sin(angle + shear[1]),
        math.cos(angle + shear[1]), 0
    ]
    matrix = [scale / d * m for m in matrix]
    matrix[2] += matrix[0] * (-center[0] - translations[0]) + matrix[1] * (-center[1] - translations[1])
    matrix[5] += matrix[3] * (-center[0] - translations[0]) + matrix[4] * (-center[1] - translations[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]

    if __version__ >= '5':
        kwargs = {"fillcolor": fill_value}
    else:
        kwargs = {}
    return ToTensor(image.transform(output_size, Image.AFFINE, matrix, fill_mode, **kwargs))


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
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, np.ndarray):
        image = ToPIL(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))
    DE_PY_INTER_MODE = {'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    if isinstance(zoom_range, list) or isinstance(zoom_range, tuple):
        if len(zoom_range) == 2:
            scale = random.uniform(zoom_range[0], zoom_range[1])
        else:
            raise ValueError('The length of zoom_range must be 2')
    else:
        raise ValueError(
            "Zoom_range should be a single value or a tuple/list containing " + "two values. Got {}".format(zoom_range)
        )
    if fill_mode not in ('nearest', 'bilinear', 'bicubic'):
        raise TypeError('Fill_mode must be in (nearest,bilinear,bicubic)')
    fill_mode = DE_PY_INTER_MODE[fill_mode]
    width, height = image.size
    translations = (0, 0)
    shear = (0, 0)
    output_size = image.size
    center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
    angle = math.radians(0)

    d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
        math.sin(angle + shear[0]) * math.sin(angle + shear[1])
    matrix = [
        math.cos(angle + shear[0]),
        math.sin(angle + shear[0]), 0, -math.sin(angle + shear[1]),
        math.cos(angle + shear[1]), 0
    ]
    matrix = [scale / d * m for m in matrix]
    matrix[2] += matrix[0] * (-center[0] - translations[0]) + matrix[1] * (-center[1] - translations[1])
    matrix[5] += matrix[3] * (-center[0] - translations[0]) + matrix[4] * (-center[1] - translations[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]

    if __version__ >= '5':
        kwargs = {"fillcolor": fill_value}
    else:
        kwargs = {}
    return ToTensor(image.transform(output_size, Image.AFFINE, matrix, fill_mode, **kwargs))


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
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))

    return ToTensor(image * scale + offset)


def RandomFlipVertical(image, prob=0.5):

    if prob > random.random():
        image = FlipVertical(image)
    return image


def RandomFlipHorizontal(image, prob=0.5):

    if prob > random.random():
        image = FlipHorizontal(image)
    return image


def HWC2CHW(image):
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))

    image_shape = image.shape
    if (len(image_shape) == 3):
        return Transpose(image, (2, 0, 1))
    elif (len(image_shape) == 4):
        return Transpose(image, (0, 3, 1, 2))
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')


def CHW2HWC(image):
    if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
        image = Decode(image)
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = ToTensor(image)
    if not isinstance(image, np.ndarray):
        raise TypeError('img should be NumPy image. Got {}'.format(type(image)))

    image_shape = image.shape
    if (len(image_shape) == 3):
        return Transpose(image, (1, 2, 0))
    elif (len(image_shape) == 4):
        return Transpose(image, (0, 2, 3, 1))
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')
