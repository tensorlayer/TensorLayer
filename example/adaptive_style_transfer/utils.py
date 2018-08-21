from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from os import listdir, remove
from os.path import join
from scipy.misc import imread, imresize

PATH = '/root/even/dataset/wiki_all_images'


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


# Normalizes the `content_features` with scaling and offset from `style_features`.
def AdaIN(content_features, style_features, alpha=1, epsilon=1e-5):

    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keep_dims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keep_dims=True)

    normalized_content_features = tf.nn.batch_normalization(
        content_features, content_mean, content_variance, style_mean, tf.sqrt(style_variance), epsilon
    )
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features


def pre_process_dataset(dir_path):

    paths = tl.files.load_file_list(dir_path, regx='\\.(jpg|jpeg|png)', keep_prefix=True)

    print('\norigin files number: %d\n' % len(paths))

    num_delete = 0

    for path in paths:

        try:
            image = imread(path, mode='RGB')
        except IOError:
            num_delete += 1
            print('Cant read this file, will delete it')
            remove(path)

        if len(image.shape) != 3 or image.shape[2] != 3:
            num_delete += 1
            remove(path)
            print('\nimage.shape:', image.shape, ' Remove image <%s>\n' % path)
        else:
            height, width, _ = image.shape

            if height < width:
                new_height = 512
                new_width = int(width * new_height / height)
            else:
                new_width = 512
                new_height = int(height * new_width / width)

            try:
                image = imresize(image, [new_height, new_width], interp='nearest')
            except Exception():
                print('Cant resize this file, will delete it')
                num_delete += 1
                remove(path)

    print('\n\ndelete %d files! Current number of files: %d\n\n' % (num_delete, len(paths) - num_delete))
