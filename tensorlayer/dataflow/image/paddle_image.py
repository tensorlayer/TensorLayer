import paddle
import numpy as np
from PIL import Image
from paddle.vision.transforms import functional as F

__all_ = [
    'Standardization',
]


def Standardization(img, mean, std, data_format='HWC'):

    if data_format == 'CHW':
        mean = paddle.to_tensor(mean).reshape([-1, 1, 1])
        std = paddle.to_tensor(std).reshape([-1, 1, 1])
    else:
        mean = paddle.to_tensor(mean)
        std = paddle.to_tensor(std)
    return (img - mean) / std
