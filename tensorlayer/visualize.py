import matplotlib.pyplot as plt
import numpy as np

def W(W, second=10, saveable=True, name='mnist', fig_idx=2396512):
    """
    The :function:`W` visualizes each column of weight matrix
    for MNIST dataset.

    Parameters
    ----------
    W : numpy.array
        The weight matrix
    second : int
        The display second(s) for the image(s), if saveable is False.
    name : a string
        A name to save the image, if saveable is True.
    fig_idx: int
        matplotlib figure index.

    Examples
    --------
    >>> xxx
    >>> xxx
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
            # plt.imshow(np.reshape(W.get_value()[:,count-1],(28,28)), cmap='gray')
            plt.imshow(np.reshape(W[:,count-1] / np.sqrt( (W[:,count-1]**2).sum()) ,(np.sqrt(size),np.sqrt(size))), cmap='gray', interpolation="nearest")
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
    """
    The :function:`frame` display a frame(image).
    Make sure OpenAI Gym render() is disable before using it.

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
    >>> xxx
    >>> xxx
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

def CNN(CNN, second=10, saveable=True, name='cnn', fig_idx=3119362):
    """
    The :function:`CNN` display or save a group of CNN mask.

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
    >>> xxx
    >>> xxx
    """
    n_mask = CNN.shape[0]
    n_row = CNN.shape[1]
    n_col = CNN.shape[2]
    n_color = CNN.shape[3]
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
            plt.imshow(
                    np.reshape(CNN[count-1,:,:,:], (n_row, n_col)),
                    cmap='gray', interpolation="nearest")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # 不显示刻度(tick)
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)






#
