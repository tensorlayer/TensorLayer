import numpy as np

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256):
    images = []
    for path in paths:
        image = imread(path, mode='RGB')
        height, width, _ = image.shape

        if height < width:
            new_height = resize_len
            new_width = int(width * new_height / height)
        else:
            new_width = resize_len
            new_height = int(height * new_width / width)

        image = imresize(image, [new_height, new_width], interp='nearest')

        # crop the image
        start_h = np.random.choice(new_height - crop_height + 1)
        start_w = np.random.choice(new_width - crop_width + 1)
        image = image[start_h:(start_h + crop_height), start_w:(start_w + crop_width), :]

        images.append(image)

    images = np.stack(images, axis=0)

    return images
