"""
Tutorial of fast affine transformation.
To run this tutorial, install opencv-python using pip.

Comprehensive explanation of this tutorial can be found https://tensorlayer.readthedocs.io/en/stable/modules/prepro.html
"""

import multiprocessing
import time

import cv2
import numpy as np
import tensorflow as tf

import tensorlayer as tl

# tl.logging.set_verbosity(tl.logging.DEBUG)
image = tl.vis.read_image('data/tiger.jpeg')
h, w, _ = image.shape


def create_transformation_matrix():
    # 1. Create required affine transformation matrices
    M_rotate = tl.prepro.affine_rotation_matrix(angle=20)
    M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=1)
    M_shift = tl.prepro.affine_shift_matrix(wrg=0.1, hrg=0, h=h, w=w)
    M_shear = tl.prepro.affine_shear_matrix(x_shear=0.2, y_shear=0)
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=0.8)

    # 2. Combine matrices
    # NOTE: operations are applied in a reversed order (i.e., rotation is performed first)
    M_combined = M_shift.dot(M_zoom).dot(M_shear).dot(M_flip).dot(M_rotate)

    # 3. Convert the matrix from Cartesian coordinates (the origin in the middle of image)
    # to image coordinates (the origin on the top-left of image)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
    return transform_matrix


def example1():
    """ Example 1: Applying transformation one-by-one is very SLOW ! """
    st = time.time()
    for _ in range(100):  # Try 100 times and compute the averaged speed
        xx = tl.prepro.rotation(image, rg=-20, is_random=False)
        xx = tl.prepro.flip_axis(xx, axis=1, is_random=False)
        xx = tl.prepro.shear2(xx, shear=(0., -0.2), is_random=False)
        xx = tl.prepro.zoom(xx, zoom_range=1 / 0.8)
        xx = tl.prepro.shift(xx, wrg=-0.1, hrg=0, is_random=False)
    print("apply transforms one-by-one took %fs for each image" % ((time.time() - st) / 100))
    tl.vis.save_image(xx, '_result_slow.png')


def example2():
    """ Example 2: Applying all transforms in one is very FAST ! """
    st = time.time()
    for _ in range(100):  # Repeat 100 times and compute the averaged speed
        transform_matrix = create_transformation_matrix()
        result = tl.prepro.affine_transform_cv2(image, transform_matrix)  # Transform the image using a single operation
    print("apply all transforms once took %fs for each image" % ((time.time() - st) / 100))  # usually 50x faster
    tl.vis.save_image(result, '_result_fast.png')


def example3():
    """ Example 3: Using TF dataset API to load and process image for training """
    n_data = 100
    imgs_file_list = ['data/tiger.jpeg'] * n_data
    train_targets = [np.ones(1)] * n_data

    def generator():
        if len(imgs_file_list) != len(train_targets):
            raise RuntimeError('len(imgs_file_list) != len(train_targets)')
        for _input, _target in zip(imgs_file_list, train_targets):
            yield _input, _target

    def _data_aug_fn(image):
        transform_matrix = create_transformation_matrix()
        result = tl.prepro.affine_transform_cv2(image, transform_matrix)  # Transform the image using a single operation
        return result

    def _map_fn(image_path, target):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.numpy_function(_data_aug_fn, [image], [tf.float32])[0]
        target = tf.reshape(target, ())
        return image, target

    n_epoch = 10
    batch_size = 5
    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.int64))
    dataset = dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    dataset = dataset.repeat(n_epoch)
    dataset = dataset.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.batch(batch_size)  # TODO: consider using tf.contrib.map_and_batch
    dataset = dataset.prefetch(1)  # prefetch 1 batch

    n_step = 0
    st = time.time()
    for img, target in dataset:
        n_step += 1
        pass
    assert n_step == n_epoch * n_data / batch_size
    print("dataset APIs took %fs for each image" % ((time.time() - st) / batch_size / n_step))  # CPU ~ 100%


def example4():
    """ Example 4: Transforming coordinates using affine matrix. """
    transform_matrix = create_transformation_matrix()
    result = tl.prepro.affine_transform_cv2(image, transform_matrix)  # 76 times faster
    # Transform keypoint coordinates
    coords = [[(50, 100), (100, 100), (100, 50), (200, 200)], [(250, 50), (200, 50), (200, 100)]]
    coords_result = tl.prepro.affine_transform_keypoints(coords, transform_matrix)

    def imwrite(image, coords_list, name):
        coords_list_ = []
        for coords in coords_list:
            coords = np.array(coords, np.int32)
            coords = coords.reshape((-1, 1, 2))
            coords_list_.append(coords)
        image = cv2.polylines(image, coords_list_, True, (0, 255, 255), 3)
        cv2.imwrite(name, image[..., ::-1])

    imwrite(image, coords, '_with_keypoints_origin.png')
    imwrite(result, coords_result, '_with_keypoints_result.png')


if __name__ == '__main__':
    example1()
    example2()
    example3()
    example4()
