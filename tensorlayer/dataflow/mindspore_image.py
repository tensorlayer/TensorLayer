import mindspore.dataset as ms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.py_transforms_util as py_util
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, __version__

__all__ = [
    'CentralCrop', 'HsvToRgb', 'AdjustBrightness', 'AdjustContrast', 'AdjustHue', 'Crop', 'FlipHorizontal',
    'FlipVertical', 'GrayToRgb', 'RgbToGray', 'PadToBoundingBox'
]

augment_error_message = 'img should be PIL image. Got {}. Use Decode() for encoded data or ToPIL() for decoded data.'


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

    if size is None:
        outshape = np.shape(image)
        if len(outshape) == 3:
            h_axis = 0
            w_axis = 1
        elif len(outshape) == 4:
            h_axis = 1
            w_axis = 2

        height = outshape[h_axis]
        width = outshape[w_axis]

        target_height = height * central_fraction
        target_width = width * central_fraction

        size = (target_height, target_width)

    return py_util.center_crop(image, size)


def HsvToRgb(image, is_hwc=True):

    image = np.asarray(image)

    return py_util.hsv_to_rgbs(image, is_hwc=is_hwc)


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

    image = np.asarray(image)
    image = image / 255
    image = image + factor
    index = np.where(image > 1)
    image[index] = 1
    index = np.where(image < 0)
    image[index] = 0
    image = image * 255

    return image


def AdjustContrast(image, factor):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    image = ImageEnhance.Contrast(image).enhance(factor)

    image = np.array(image)

    return image


def AdjustHue(image, factor):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    image_hue_factor = factor
    if not -1 <= image_hue_factor <= 1:
        raise ValueError('image_hue_factor {} is not in [-1, 1].'.format(image_hue_factor))

    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    mode = image.mode
    if mode in {'L', '1', 'I', 'F'}:
        return image

    hue, saturation, value = image.convert('HSV').split()

    np_hue = np.array(hue, dtype=np.uint8)

    with np.errstate(over='ignore'):
        np_hue += np.uint8(image_hue_factor * 255)
    hue = Image.fromarray(np_hue, 'L')

    image = Image.merge('HSV', (hue, saturation, value)).convert(mode)
    return image


def AdjustSaturation(image, factor):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor)
    return image


def Crop(image, offset_height, offset_width, target_height, target_width):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))
    image = np.array(
        image.crop((offset_width, offset_height, offset_width + target_width, offset_width + target_height))
    )
    return image


def FlipHorizontal(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    image = np.array(image.transpose(Image.FLIP_LEFT_RIGHT))

    return image


def FlipVertical(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))

    image = np.array(image.transpose(Image.FLIP_TOP_BOTTOM))

    return image


def GrayToRgb(image):

    image = np.asarray(image)
    shape = image.shape
    output_image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if len(shape) == 3:
        for i in range(3):
            output_image[:, :, i] = image[:, :, 1]
    elif len(shape) == 2:
        for i in range(3):
            output_image[:, :, i] = image

    return output_image


def RgbToGray(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(augment_error_message.format(type(image)))
    '''
	将彩色图像转换为灰度（模式“L”）时，库使用ITU-R 601-2 Luma转换：
	L = R * 299/1000 + G * 587/1000 + B * 114/1000
	'''
    image = image.convert('L')
    image = np.asarray(image)

    return image


def PadToBoundingBox(image, offset_height, offset_width, target_height, target_width):
    '''

	Parameters
	----------
	image:
		A PIL image
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
    image = np.array(image)
    shape = image.shape
    top = offset_height
    bottom = target_height - shape[0] - top
    left = offset_width
    right = target_width - shape[1] - left

    if bottom < 0:
        raise ValueError("target_height must be >= offset_height + height")

    if right < 0:
        raise ValueError("target_width must be >= offset_width + width")

    return np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='constant')


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
    image = np.array(image, dtype=np.float32)
    num_shape = image.shape
    if mean is not None and std is not None:
        if len(mean) != len(std):
            raise ValueError("Length of mean and std must be equal")
        if len(mean) == 1:
            mean = [mean[0]] * num_shape[2]
            std = [std[0]] * num_shape[2]
        mean = np.array(mean, dtype=image.dtype)
        std = np.array(std, dtype=image.dtype)
        return (image - mean[:, None, None]) / std[:, None, None]
    elif mean is None and std is None:
        if channel_mode:
            num_pixels = num_shape[0] * num_shape[1]
            image_mean = np.mean(image, axis=(0, 1))
            stddev = np.std(image, axis=(0, 1))
            min_sttdev = 1 / np.sqrt(num_pixels)
            min_sttdev = [min_sttdev] * num_shape[2]
            adjusted_sttdev = np.maximum(stddev, min_sttdev)

            image -= image_mean
            image = np.divide(image, adjusted_sttdev)
            return image
        else:
            num_pixels = num_shape[0] * num_shape[1] * num_shape[2]
            image_mean = np.mean(image, axis=(0, 1, 2))
            image_mean = [image_mean] * 3
            stddev = np.std(image, axis=(0, 1, 2))
            min_sttdev = 1 / np.sqrt(num_pixels)
            adjusted_sttdev = np.maximum(stddev, min_sttdev)
            adjusted_sttdev = [adjusted_sttdev] * 3

            image -= image_mean
            image = np.divide(image, adjusted_sttdev)
            return image
    else:
        raise ValueError('std and mean must both be None or not None')
