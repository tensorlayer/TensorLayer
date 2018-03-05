# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.misc  # save/read image(s)

from . import _logging as logging
from . import prepro

# Uncomment the following line if you got: _tkinter.TclError: no display name and no $DISPLAY environment variable
# import matplotlib
# matplotlib.use('Agg')


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
    return scipy.misc.imread(os.path.join(path, image))


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
        b_imgs = prepro.threading_data(b_imgs_list, fn=read_image, path=path)
        # logging.info(b_imgs.shape)
        imgs.extend(b_imgs)
        if printable:
            logging.info('read %d from %s' % (len(imgs), path))
    return imgs


def save_image(image, image_path=''):
    """Save a image.

    Parameters
    -----------
    image : numpy array
        [w, h, c]
    image_path : str
        path

    """
    try:  # RGB
        scipy.misc.imsave(image_path, image)
    except Exception:  # Greyscale
        scipy.misc.imsave(image_path, image[:, :, 0])


def save_images(images, size, image_path=''):
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

    Returns
    -------
    numpy.array
        The image.

    Examples
    ---------
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> tl.visualize.save_images(images, [8, 8], 'temp.png')

    """
    if len(images.shape) == 3:  # Greyscale [batch, h, w] --> [batch, h, w, 1]
        images = images[:, :, :, np.newaxis]

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    def imsave(images, size, path):
        return scipy.misc.imsave(path, merge(images, size))

    assert len(images) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(images))
    return imsave(images, size, image_path)


def draw_boxes_and_labels_to_image(image, classes, coords, scores, classes_list, is_center=True, is_rescale=True, save_name=None):
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
    assert len(coords) == len(classes), "number of coordinates and classes are equal"
    if len(scores) > 0:
        assert len(scores) == len(classes), "number of scores and classes are equal"

    import cv2

    # don't change the original image, and avoid error https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    image = image.copy()

    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 430)

    for i, _v in enumerate(coords):
        if is_center:
            x, y, x2, y2 = prepro.obj_box_coord_centroid_to_upleft_butright(coords[i])
        else:
            x, y, x2, y2 = coords[i]

        if is_rescale:  # scale back to pixel unit if the coords are the portion of width and high
            x, y, x2, y2 = prepro.obj_box_coord_scale_to_pixelunit([x, y, x2, y2], (imh, imw))

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x2), int(y2)),  # up-left and botton-right
            [0, 255, 0],
            thick)

        cv2.putText(
            image,
            classes_list[classes[i]] + ((" %.2f" % (scores[i])) if (len(scores) != 0) else " "),
            (int(x), int(y)),  # button left
            0,
            1.5e-3 * imh,  # bigger = larger font
            [0, 0, 256],  # self.meta['colors'][max_indx],
            int(thick / 2) + 1)  # bold

    if save_name is not None:
        # cv2.imwrite('_my.png', image)
        save_image(image, save_name)
    # if len(coords) == 0:
    #     logging.info("draw_boxes_and_labels_to_image: no bboxes exist, cannot draw !")
    return image


def frame(I=None, second=5, saveable=True, name='frame', cmap=None, fig_idx=12836):
    """Display a frame(image). Make sure OpenAI Gym render() is disable before using it.

    Parameters
    ----------
    I : numpy.array
        The image.
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : str
        A name to save the image, if saveable is True.
    cmap : None or str
        'gray' for greyscale, None for default, etc.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> env = gym.make("Pong-v0")
    >>> observation = env.reset()
    >>> tl.visualize.frame(observation)

    """
    import matplotlib.pyplot as plt
    if saveable is False:
        plt.ion()
    plt.figure(fig_idx)  # show all feature images

    if len(I.shape) and I.shape[-1] == 1:  # (10,10,1) --> (10,10)
        I = I[:, :, 0]

    plt.imshow(I, cmap)
    plt.title(name)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if saveable:
        plt.savefig(name + '.pdf', format='pdf')
    else:
        plt.draw()
        plt.pause(second)


def CNN2d(CNN=None, second=10, saveable=True, name='cnn', fig_idx=3119362):
    """Display a group of RGB or Greyscale CNN masks.

    Parameters
    ----------
    CNN : numpy.array
        The image. e.g: 64 5x5 RGB images can be (5, 5, 3, 64).
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : str
        A name to save the image, if saveable is True.
    fig_idx : int
        The matplotlib figure index.

    Examples
    --------
    >>> tl.visualize.CNN2d(network.all_params[0].eval(), second=10, saveable=True, name='cnn1_mnist', fig_idx=2012)

    """
    import matplotlib.pyplot as plt
    # logging.info(CNN.shape)    # (5, 5, 3, 64)
    # exit()
    n_mask = CNN.shape[3]
    n_row = CNN.shape[0]
    n_col = CNN.shape[1]
    n_color = CNN.shape[2]
    row = int(np.sqrt(n_mask))
    col = int(np.ceil(n_mask / row))
    plt.ion()  # active mode
    fig = plt.figure(fig_idx)
    count = 1
    for _ir in range(1, row + 1):
        for _ic in range(1, col + 1):
            if count > n_mask:
                break
            fig.add_subplot(col, row, count)
            # logging.info(CNN[:,:,:,count-1].shape, n_row, n_col)   # (5, 1, 32) 5 5
            # exit()
            # plt.imshow(
            #         np.reshape(CNN[count-1,:,:,:], (n_row, n_col)),
            #         cmap='gray', interpolation="nearest")     # theano
            if n_color == 1:
                plt.imshow(np.reshape(CNN[:, :, :, count - 1], (n_row, n_col)), cmap='gray', interpolation="nearest")
            elif n_color == 3:
                plt.imshow(np.reshape(CNN[:, :, :, count - 1], (n_row, n_col, n_color)), cmap='gray', interpolation="nearest")
            else:
                raise Exception("Unknown n_color")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name + '.pdf', format='pdf')
    else:
        plt.draw()
        plt.pause(second)


def images2d(images=None, second=10, saveable=True, name='images', dtype=None, fig_idx=3119362):
    """Display a group of RGB or Greyscale images.

    Parameters
    ----------
    images : numpy.array
        The images.
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : str
        A name to save the image, if saveable is True.
    dtype : None or numpy data type
        The data type for displaying the images.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    >>> tl.visualize.images2d(X_train[0:100,:,:,:], second=10, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=20212)

    """
    import matplotlib.pyplot as plt
    # logging.info(images.shape)    # (50000, 32, 32, 3)
    # exit()
    if dtype:
        images = np.asarray(images, dtype=dtype)
    n_mask = images.shape[0]
    n_row = images.shape[1]
    n_col = images.shape[2]
    n_color = images.shape[3]
    row = int(np.sqrt(n_mask))
    col = int(np.ceil(n_mask / row))
    plt.ion()  # active mode
    fig = plt.figure(fig_idx)
    count = 1
    for _ir in range(1, row + 1):
        for _ic in range(1, col + 1):
            if count > n_mask:
                break
            fig.add_subplot(col, row, count)
            # logging.info(images[:,:,:,count-1].shape, n_row, n_col)   # (5, 1, 32) 5 5
            # plt.imshow(
            #         np.reshape(images[count-1,:,:,:], (n_row, n_col)),
            #         cmap='gray', interpolation="nearest")     # theano
            if n_color == 1:
                plt.imshow(np.reshape(images[count - 1, :, :], (n_row, n_col)), cmap='gray', interpolation="nearest")
                # plt.title(name)
            elif n_color == 3:
                plt.imshow(images[count - 1, :, :], cmap='gray', interpolation="nearest")
                # plt.title(name)
            else:
                raise Exception("Unknown n_color")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name + '.pdf', format='pdf')
    else:
        plt.draw()
        plt.pause(second)


def tsne_embedding(embeddings, reverse_dictionary, plot_only=500, second=5, saveable=False, name='tsne', fig_idx=9862):
    """Visualize the embeddings by using t-SNE.

    Parameters
    ----------
    embeddings : numpy.array
        The embedding matrix.
    reverse_dictionary : dictionary
        id_to_word, mapping id to unique word.
    plot_only : int
        The number of examples to plot, choice the most common words.
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : str
        A name to save the image, if saveable is True.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> see 'tutorial_word2vec_basic.py'
    >>> final_embeddings = normalized_embeddings.eval()
    >>> tl.visualize.tsne_embedding(final_embeddings, labels, reverse_dictionary,
    ...                   plot_only=500, second=5, saveable=False, name='tsne')

    """
    import matplotlib.pyplot as plt

    def plot_with_labels(low_dim_embs, labels, figsize=(18, 18), second=5, saveable=True, name='tsne', fig_idx=9862):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        if saveable is False:
            plt.ion()
            plt.figure(fig_idx)
        plt.figure(figsize=figsize)  #in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        if saveable:
            plt.savefig(name + '.pdf', format='pdf')
        else:
            plt.draw()
            plt.pause(second)

    try:
        from sklearn.manifold import TSNE
        from six.moves import xrange

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        # plot_only = 500
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, second=second, saveable=saveable, \
                                                    name=name, fig_idx=fig_idx)
    except ImportError:
        logging.info("Please install sklearn and matplotlib to visualize embeddings.")


def draw_weights(W=None, second=10, saveable=True, shape=None, name='mnist', fig_idx=2396512):
    """Visualize every columns of the weight matrix to a group of Greyscale img.

    Parameters
    ----------
    W : numpy.array
        The weight matrix
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    shape : a list with 2 int or None
        The shape of feature image, MNIST is [28, 80].
    name : a string
        A name to save the image, if saveable is True.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> tl.visualize.draw_weights(network.all_params[0].eval(), second=10, saveable=True, name='weight_of_1st_layer', fig_idx=2012)

    """
    if shape is None:
        shape = [28, 28]

    import matplotlib.pyplot as plt
    if saveable is False:
        plt.ion()
    fig = plt.figure(fig_idx)  # show all feature images
    n_units = W.shape[1]

    num_r = int(np.sqrt(n_units))  # 每行显示的个数   若25个hidden unit -> 每行显示5个
    num_c = int(np.ceil(n_units / num_r))
    count = int(1)
    for _row in range(1, num_r + 1):
        for _col in range(1, num_c + 1):
            if count > n_units:
                break
            fig.add_subplot(num_r, num_c, count)
            # ------------------------------------------------------------
            # plt.imshow(np.reshape(W[:,count-1],(28,28)), cmap='gray')
            # ------------------------------------------------------------
            feature = W[:, count - 1] / np.sqrt((W[:, count - 1]**2).sum())
            # feature[feature<0.0001] = 0   # value threshold
            # if count == 1 or count == 2:
            #     print(np.mean(feature))
            # if np.std(feature) < 0.03:      # condition threshold
            #     feature = np.zeros_like(feature)
            # if np.mean(feature) < -0.015:      # condition threshold
            #     feature = np.zeros_like(feature)
            plt.imshow(np.reshape(feature, (shape[0], shape[1])), cmap='gray', interpolation="nearest")  #, vmin=np.min(feature), vmax=np.max(feature))
            # plt.title(name)
            # ------------------------------------------------------------
            # plt.imshow(np.reshape(W[:,count-1] ,(np.sqrt(size),np.sqrt(size))), cmap='gray', interpolation="nearest")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name + '.pdf', format='pdf')
    else:
        plt.draw()
        plt.pause(second)


W = draw_weights
