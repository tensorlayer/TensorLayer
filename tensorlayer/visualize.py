#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import imageio

import numpy as np

import tensorlayer as tl

import cv2

__all__ = [
    'read_image',
    'read_images',
    'save_image',
    'save_images',
    'draw_boxes_and_labels_to_image',
    'draw_mpii_people_to_image',
]


def read_image(image, path=''):
    """Read one image.

    Parameters
    -----------
    image : str
        The image file name.
    path : str
        The image folder path.

    Returns
    -------
    numpy.array
        The image.

    """
    return imageio.imread(os.path.join(path, image))


def read_images(img_list, path='', n_threads=10, printable=True):
    """Returns all images in list by given path and name of each image file.

    Parameters
    -------------
    img_list : list of str
        The image file names.
    path : str
        The image folder path.
    n_threads : int
        The number of threads to read image.
    printable : boolean
        Whether to print information when reading images.

    Returns
    -------
    list of numpy.array
        The images.

    """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx:idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=read_image, path=path)
        # tl.logging.info(b_imgs.shape)
        imgs.extend(b_imgs)
        if printable:
            tl.logging.info('read %d from %s' % (len(imgs), path))
    return imgs


def save_image(image, image_path='_temp.png'):
    """Save a image.

    Parameters
    -----------
    image : numpy array
        [w, h, c]
    image_path : str
        path

    """
    try:  # RGB
        imageio.imwrite(image_path, image)
    except Exception:  # Greyscale
        imageio.imwrite(image_path, image[:, :, 0])


def save_images(images, size, image_path='_temp.png'):
    """Save multiple images into one single image.

    Parameters
    -----------
    images : numpy array
        (batch, w, h, c)
    size : list of 2 ints
        row and column number.
        number of images should be equal or less than size[0] * size[1]
    image_path : str
        save path

    Examples
    ---------
    >>> import numpy as np
    >>> import tensorlayer as tl
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> tl.visualize.save_images(images, [8, 8], 'temp.png')

    """
    if len(images.shape) == 3:  # Greyscale [batch, h, w] --> [batch, h, w, 1]
        images = images[:, :, :, np.newaxis]

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3), dtype=images.dtype)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    def imsave(images, size, path):
        if np.max(images) <= 1 and (-1 <= np.min(images) < 0):
            images = ((images + 1) * 127.5).astype(np.uint8)
        elif np.max(images) <= 1 and np.min(images) >= 0:
            images = (images * 255).astype(np.uint8)

        return imageio.imwrite(path, merge(images, size))

    if len(images) > size[0] * size[1]:
        raise AssertionError("number of images should be equal or less than size[0] * size[1] {}".format(len(images)))

    return imsave(images, size, image_path)


def draw_boxes_and_labels_to_image(
        image, classes, coords, scores, classes_list, is_center=True, is_rescale=True, save_name=None
):
    """Draw bboxes and class labels on image. Return or save the image with bboxes, example in the docs of ``tl.prepro``.

    Parameters
    -----------
    image : numpy.array
        The RGB image [height, width, channel].
    classes : list of int
        A list of class ID (int).
    coords : list of int
        A list of list for coordinates.
            - Should be [x, y, x2, y2] (up-left and botton-right format)
            - If [x_center, y_center, w, h] (set is_center to True).
    scores : list of float
        A list of score (float). (Optional)
    classes_list : list of str
        for converting ID to string on image.
    is_center : boolean
        Whether the coordinates is [x_center, y_center, w, h]
            - If coordinates are [x_center, y_center, w, h], set it to True for converting it to [x, y, x2, y2] (up-left and botton-right) internally.
            - If coordinates are [x1, x2, y1, y2], set it to False.
    is_rescale : boolean
        Whether to rescale the coordinates from pixel-unit format to ratio format.
            - If True, the input coordinates are the portion of width and high, this API will scale the coordinates to pixel unit internally.
            - If False, feed the coordinates with pixel unit format.
    save_name : None or str
        The name of image file (i.e. image.png), if None, not to save image.

    Returns
    -------
    numpy.array
        The saved image.

    References
    -----------
    - OpenCV rectangle and putText.
    - `scikit-image <http://scikit-image.org/docs/dev/api/skimage.draw.html#skimage.draw.rectangle>`__.

    """
    if len(coords) != len(classes):
        raise AssertionError("number of coordinates and classes are equal")

    if len(scores) > 0 and len(scores) != len(classes):
        raise AssertionError("number of scores and classes are equal")

    # don't change the original image, and avoid error https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    image = image.copy()

    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 430)

    for i, _v in enumerate(coords):
        if is_center:
            x, y, x2, y2 = tl.prepro.obj_box_coord_centroid_to_upleft_butright(coords[i])
        else:
            x, y, x2, y2 = coords[i]

        if is_rescale:  # scale back to pixel unit if the coords are the portion of width and high
            x, y, x2, y2 = tl.prepro.obj_box_coord_scale_to_pixelunit([x, y, x2, y2], (imh, imw))

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x2), int(y2)),  # up-left and botton-right
            [0, 255, 0],
            thick
        )

        cv2.putText(
            image,
            classes_list[classes[i]] + ((" %.2f" % (scores[i])) if (len(scores) != 0) else " "),
            (int(x), int(y)),  # button left
            0,
            1.5e-3 * imh,  # bigger = larger font
            [0, 0, 256],  # self.meta['colors'][max_indx],
            int(thick / 2) + 1
        )  # bold

    if save_name is not None:
        # cv2.imwrite('_my.png', image)
        save_image(image, save_name)
    # if len(coords) == 0:
    #     tl.logging.info("draw_boxes_and_labels_to_image: no bboxes exist, cannot draw !")
    return image


def draw_mpii_pose_to_image(image, poses, save_name='image.png'):
    """Draw people(s) into image using MPII dataset format as input, return or save the result image.

    This is an experimental API, can be changed in the future.

    Parameters
    -----------
    image : numpy.array
        The RGB image [height, width, channel].
    poses : list of dict
        The people(s) annotation in MPII format, see ``tl.files.load_mpii_pose_dataset``.
    save_name : None or str
        The name of image file (i.e. image.png), if None, not to save image.

    Returns
    --------
    numpy.array
        The saved image.

    Examples
    --------
    >>> import pprint
    >>> import tensorlayer as tl
    >>> img_train_list, ann_train_list, img_test_list, ann_test_list = tl.files.load_mpii_pose_dataset()
    >>> image = tl.vis.read_image(img_train_list[0])
    >>> tl.vis.draw_mpii_pose_to_image(image, ann_train_list[0], 'image.png')
    >>> pprint.pprint(ann_train_list[0])

    References
    -----------
    - `MPII Keyponts and ID <http://human-pose.mpi-inf.mpg.de/#download>`__
    """
    # import skimage
    # don't change the original image, and avoid error https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    image = image.copy()

    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 430)
    # radius = int(image.shape[1] / 500) + 1
    radius = int(thick * 1.5)

    if image.max() < 1:
        image = image * 255

    for people in poses:
        # Pose Keyponts
        joint_pos = people['joint_pos']
        # draw sketch
        # joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
        #           5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,
        #           9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder,
        #           13 - l shoulder, 14 - l elbow, 15 - l wrist)
        #
        #               9
        #               8
        #         12 ** 7 ** 13
        #        *      *      *
        #       11      *       14
        #      *        *         *
        #     10    2 * 6 * 3     15
        #           *       *
        #           1       4
        #           *       *
        #           0       5

        lines = [
            [(0, 1), [100, 255, 100]],
            [(1, 2), [50, 255, 50]],
            [(2, 6), [0, 255, 0]],  # right leg
            [(3, 4), [100, 100, 255]],
            [(4, 5), [50, 50, 255]],
            [(6, 3), [0, 0, 255]],  # left leg
            [(6, 7), [255, 255, 100]],
            [(7, 8), [255, 150, 50]],  # body
            [(8, 9), [255, 200, 100]],  # head
            [(10, 11), [255, 100, 255]],
            [(11, 12), [255, 50, 255]],
            [(12, 8), [255, 0, 255]],  # right hand
            [(8, 13), [0, 255, 255]],
            [(13, 14), [100, 255, 255]],
            [(14, 15), [200, 255, 255]]  # left hand
        ]
        for line in lines:
            start, end = line[0]
            if (start in joint_pos) and (end in joint_pos):
                cv2.line(
                    image,
                    (int(joint_pos[start][0]), int(joint_pos[start][1])),
                    (int(joint_pos[end][0]), int(joint_pos[end][1])),  # up-left and botton-right
                    line[1],
                    thick
                )
                # rr, cc, val = skimage.draw.line_aa(int(joint_pos[start][1]), int(joint_pos[start][0]), int(joint_pos[end][1]), int(joint_pos[end][0]))
                # image[rr, cc] = line[1]
        # draw circles
        for pos in joint_pos.items():
            _, pos_loc = pos  # pos_id, pos_loc
            pos_loc = (int(pos_loc[0]), int(pos_loc[1]))
            cv2.circle(image, center=pos_loc, radius=radius, color=(200, 200, 200), thickness=-1)
            # rr, cc = skimage.draw.circle(int(pos_loc[1]), int(pos_loc[0]), radius)
            # image[rr, cc] = [0, 255, 0]

        # Head
        head_rect = people['head_rect']
        if head_rect:  # if head exists
            cv2.rectangle(
                image,
                (int(head_rect[0]), int(head_rect[1])),
                (int(head_rect[2]), int(head_rect[3])),  # up-left and botton-right
                [0, 180, 0],
                thick
            )

    if save_name is not None:
        # cv2.imwrite(save_name, image)
        save_image(image, save_name)
    return image


draw_mpii_people_to_image = draw_mpii_pose_to_image
