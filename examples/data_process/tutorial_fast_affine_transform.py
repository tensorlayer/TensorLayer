"""
Demo of fast affine transfromation

Detailed description in https://tensorlayer.readthedocs.io/en/stable/modules/prepro.html
"""
import tensorlayer as tl
import numpy as np
import time

# tl.logging.set_verbosity(tl.logging.DEBUG)
image = tl.vis.read_image('tiger.jpeg')
h, w, _ = image.shape

def example1():
    """ A. apply transforms one-by-one is very SLOW ! """
    st = time.time()
    for _ in range(10):  # try 100 times and compute the averaged speed
        xx = tl.prepro.rotation(image, rg=20, is_random=False)
        xx = tl.prepro.flip_axis(xx, axis=1, is_random=False)
        xx = tl.prepro.shear(xx, intensity=0.2, is_random=False)
        xx = tl.prepro.zoom(xx, zoom_range=0.8)
        xx = tl.prepro.shift(xx, wrg=0.2, hrg=0.2, is_random=False)
    print("apply transforms one-by-one took %fs for each image" % ((time.time() - st) / 10))
    tl.vis.save_image(xx, '_result_slow.png')

def fast_affine_transform(image):
    # 1. get all affine transform matrices
    M_rotate = tl.prepro.affine_rotation_matrix(angle=20)
    M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=1)
    M_shift = tl.prepro.affine_shift_matrix(wrg=0.2, hrg=0.2, h=h, w=w, is_random=False)
    M_shear = tl.prepro.affine_shear_matrix(intensity=0.2, is_random=False)
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=0.8)
    # 2. combine all affine transform matrices to one matrix, the rotation is the first transformation
    M_combined = M_rotate.dot(M_flip).dot(M_shear).dot(M_zoom).dot(M_shift)
    # M_combined = M_rotate.dot(M_shear).dot(M_zoom).dot(M_shift)
    # 3. transfrom the matrix from Cartesian coordinate (the origin in the middle of image)
    # to Image coordinate (the origin on the top-left of image)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, h, w)
    # 4. then we can transfrom the image once for all transformations
    result = tl.prepro.affine_transform(image, transform_matrix)    # OK
    # print('x',M_combined)
    # print(transform_matrix)
    # transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, w, h)
    # transform_matrix = np.float32([[transform_matrix[1,1], transform_matrix[1,0], transform_matrix[1,2]],
    #                     [transform_matrix[0,1], transform_matrix[0,0], transform_matrix[0,2]]])
    # result = tl.prepro.affine_transform_cv2(image, transform_matrix)  # 76 times faster
    return result

def example2():
    """ B. apply all transforms once is very FAST ! """
    st = time.time()
    for _ in range(100):  # try 100 times and compute the averaged speed
        result = fast_affine_transform(image)
    print("apply all transforms once took %fs for each image" % ((time.time() - st) / 100))  # usually 4x faster
    tl.vis.save_image(result, '_result_fast.png')

def example3():
    """ C. in practice, we use TF dataset API to load and process image for training """
    n_data = 100
    imgs_file_list = ['tiger.jpeg'] * n_data
    train_targets = [np.ones(1)] * n_data

    def generator():
        assert len(imgs_file_list) == len(train_targets)
        for _input, _target in zip(imgs_file_list, train_targets):
            yield _input, _target

    def _map_fn(image_path, target):
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.py_func(fast_affine_transform, [image], [tf.float32])
        # image = tf.reshape(image, (h, w, 3))
        target = tf.reshape(target, ())
        return image, target

    import tensorflow as tf
    import multiprocessing
    n_epoch = 10
    batch_size = 5
    dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.int64))
    dataset = dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    dataset = dataset.repeat(n_epoch)
    dataset = dataset.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.batch(batch_size)  # TODO: consider using tf.contrib.map_and_batch
    dataset = dataset.prefetch(1)  # prefetch 1 batch
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    sess = tf.Session()
    # feed `one_element` into a network, for demo, we simply get the data as follows
    n_step = round(n_epoch * n_data / batch_size)
    st = time.time()
    for _ in range(n_step):
        images, targets = sess.run(one_element)
    print("dataset APIs took %fs for each image" % ((time.time() - st) / batch_size / n_step)) # CPU ~ 100%


if __name__ == '__main__':
    example1()
    example2()
    example3()
