#! /usr/bin/python
# -*- coding: utf8 -*-



import matplotlib.pyplot as plt
import numpy as np
import os



def W(W, second=10, saveable=True, shape=[28,28], name='mnist', fig_idx=2396512):
    """Visualize every columns of the weight matrix for MNIST dataset.

    Parameters
    ----------
    W : numpy.array
        The weight matrix
    second : int
        The display second(s) for the image(s), if saveable is False.
    shape: list with 2 int
        The shape of feature image, MNIST is [28, 80].
    name : a string
        A name to save the image, if saveable is True.
    fig_idx: int
        matplotlib figure index.

    Examples
    --------
    >>> tl.visualize.W(network.all_params[0].eval(), second=10, saveable=True, name='weight_of_1st_layer', fig_idx=2012)
    """
    if saveable is False:
        plt.ion()
    fig = plt.figure(fig_idx)      # show all feature images
    size = W.shape[0]
    n_units = W.shape[1]

    num_r = int(np.sqrt(n_units))  # 每行显示的个数   若25个hidden unit -> 每行显示5个
    num_c = int(np.ceil(n_units/num_r))
    count = int(1)
    for row in range(1, num_r+1):
        for col in range(1, num_c+1):
            if count > n_units:
                break
            a = fig.add_subplot(num_r, num_c, count)
            # ------------------------------------------------------------
            # plt.imshow(np.reshape(W[:,count-1],(28,28)), cmap='gray')
            # ------------------------------------------------------------
            feature = W[:,count-1] / np.sqrt( (W[:,count-1]**2).sum())
            # feature[feature<0.0001] = 0   # value threshold
            # if count == 1 or count == 2:
            #     print(np.mean(feature))
            # if np.std(feature) < 0.03:      # condition threshold
            #     feature = np.zeros_like(feature)
            # if np.mean(feature) < -0.015:      # condition threshold
            #     feature = np.zeros_like(feature)
            plt.imshow(np.reshape(feature ,(shape[0],shape[1])),
                    cmap='gray', interpolation="nearest")#, vmin=np.min(feature), vmax=np.max(feature))
            # ------------------------------------------------------------
            # plt.imshow(np.reshape(W[:,count-1] ,(np.sqrt(size),np.sqrt(size))), cmap='gray', interpolation="nearest")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

def frame(I, second=5, saveable=True, name='frame', fig_idx=12836):
    """Display a frame(image). Make sure OpenAI Gym render() is disable before using it.

    Parameters
    ----------
    I : numpy.array
        The image
    second : int
        The display second(s) for the image(s), if saveable is False.
    name : a string
        A name to save the image, if saveable is True.
    fig_idx: int
        matplotlib figure index.

    Examples
    --------
    >>> env = gym.make("Pong-v0")
    >>> observation = env.reset()
    >>> tf.visualize.frame(observation)
    """
    if saveable is False:
        plt.ion()
    fig = plt.figure(fig_idx)      # show all feature images

    plt.imshow(I)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

def CNN2d(CNN, second=10, saveable=True, name='cnn', fig_idx=3119362):
    """Display a group of RGB or Greyscale CNN masks.

    Parameters
    ----------
    CNN : numpy.array
        The image
    second : int
        The display second(s) for the image(s), if saveable is False.
    name : a string
        A name to save the image, if saveable is True.
    fig_idx: int
        matplotlib figure index.

    Examples
    --------
    >>> tl.visualize.CNN2d(network.all_params[0].eval(), second=10, saveable=True, name='cnn1_mnist', fig_idx=2012)
    """
    # print(CNN.shape)    # (5, 5, 3, 64)
    # exit()
    n_mask = CNN.shape[3]
    n_row = CNN.shape[0]
    n_col = CNN.shape[1]
    n_color = CNN.shape[2]
    row = int(np.sqrt(n_mask))
    col = int(np.ceil(n_mask/row))
    plt.ion()   # active mode
    fig = plt.figure(fig_idx)
    count = 1
    for ir in range(1, row+1):
        for ic in range(1, col+1):
            if count > n_mask:
                break
            a = fig.add_subplot(col, row, count)
            # print(CNN[:,:,:,count-1].shape, n_row, n_col)   # (5, 1, 32) 5 5
            # exit()
            # plt.imshow(
            #         np.reshape(CNN[count-1,:,:,:], (n_row, n_col)),
            #         cmap='gray', interpolation="nearest")     # theano
            if n_color == 1:
                plt.imshow(
                        np.reshape(CNN[:,:,:,count-1], (n_row, n_col)),
                        cmap='gray', interpolation="nearest")
            elif n_color == 3:
                plt.imshow(
                        np.reshape(CNN[:,:,:,count-1], (n_row, n_col, n_color)),
                        cmap='gray', interpolation="nearest")
            else:
                raise Exception("Unknown n_color")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)


def images2d(images, second=10, saveable=True, name='images', dtype=None, fig_idx=3119362):
    """Display a group of RGB or Greyscale images.

    Parameters
    ----------
    images : numpy.array
        The images.
    second : int
        The display second(s) for the image(s), if saveable is False.
    name : a string
        A name to save the image, if saveable is True.
    dtype : None or numpy data type
        The data type for displaying the images.
    fig_idx: int
        matplotlib figure index.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    >>> tl.visualize.images2d(X_train[0:100,:,:,:], second=10, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=20212)
    """
    # print(images.shape)    # (50000, 32, 32, 3)
    # exit()
    if dtype:
        images = np.asarray(images, dtype=dtype)
    n_mask = images.shape[0]
    n_row = images.shape[1]
    n_col = images.shape[2]
    n_color = images.shape[3]
    row = int(np.sqrt(n_mask))
    col = int(np.ceil(n_mask/row))
    plt.ion()   # active mode
    fig = plt.figure(fig_idx)
    count = 1
    for ir in range(1, row+1):
        for ic in range(1, col+1):
            if count > n_mask:
                break
            a = fig.add_subplot(col, row, count)
            # print(images[:,:,:,count-1].shape, n_row, n_col)   # (5, 1, 32) 5 5
            # plt.imshow(
            #         np.reshape(images[count-1,:,:,:], (n_row, n_col)),
            #         cmap='gray', interpolation="nearest")     # theano
            if n_color == 1:
                plt.imshow(
                        np.reshape(images[count-1,:,:], (n_row, n_col)),
                        cmap='gray', interpolation="nearest")
            elif n_color == 3:
                plt.imshow(images[count-1,:,:],
                        cmap='gray', interpolation="nearest")
            else:
                raise Exception("Unknown n_color")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)



#
