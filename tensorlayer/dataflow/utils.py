import atexit
import logging
import math
import multiprocessing
import os
import weakref

import psutil
import tarfile
import time
import zipfile
import progressbar
from urllib.request import urlretrieve


def exists_or_mkdir(path, verbose=True):
    """
    Check a folder by given name, if not exist, create the folder and return False,
    if directory exists, return True.

    Parameters
    ----------
    path : str
        A folder path.
    verbose : boolean
        If True (default), prints results.

    Returns
    --------
    boolean
        True if folder already exist, otherwise, returns False and create the folder.

    Examples
    --------
    >>> exists_or_mkdir("checkpoints/train")

    """
    if not os.path.exists(path):
        if verbose:
            logging.info("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            logging.info("[!] %s exists ..." % path)
        return True


def download(filename, working_directory, url_source):
    """
    Download file from url_source to the working_directory with given filename.

    Parameters
    ----------
    filename : str
        The name of the downloaded file.
    working_directory : str
        A folder path download the file to
    url_source : str
        The URL to download the file from

    Examples
    --------
    >>> download(filename='train.gz',
    ...          working_directory='data/',
    ...          url_source='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')

    """
    working_directory = os.path.expanduser(working_directory)

    progress_bar = progressbar.ProgressBar()

    def _dlProgress(count, blockSize, totalSize, pbar=progress_bar):
        if (totalSize != 0):

            if not pbar.max_value:
                totalBlocks = math.ceil(float(totalSize) / float(blockSize))
                pbar.max_value = int(totalBlocks)

            pbar.update(count, force=True)

    filepath = os.path.join(working_directory, filename)

    logging.info('Downloading %s...\n' % filename)

    urlretrieve(url_source, filepath, reporthook=_dlProgress)


def maybe_download_and_extract(filename, working_directory, url_source, extract=False, expected_bytes=None):
    """
    Checks if file exists in working_directory otherwise tries to dowload the file,
    and optionally also tries to extract the file if format is ".zip" or ".tar"

    Parameters
    -----------
    filename : str
        The name of the (to be) dowloaded file.
    working_directory : str
        A folder path to search for the file in and dowload the file to
    url_source : str
        The URL to download the file from
    extract : boolean
        If True, tries to uncompress the dowloaded file is ".tar.gz/.tar.bz2" or ".zip" file, default is False.
    expected_bytes : int or None
        If set tries to verify that the downloaded file is of the specified size, otherwise raises an Exception, defaults is None which corresponds to no check being performed.

    Returns
    ----------
    str
        File path of the dowloaded (uncompressed) file.

    Examples
    --------
    >>> down_file = maybe_download_and_extract(filename='train.gz',
    ...                                            working_directory='data/',
    ...                                            url_source='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    >>> maybe_download_and_extract(filename='ADEChallengeData2016.zip',
    ...                            working_directory='data/',
    ...                            url_source='http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip',
    ...                            extract=True)

    """
    working_directory = os.path.expanduser(working_directory)
    exists_or_mkdir(working_directory, verbose=False)
    filepath = os.path.join(working_directory, filename)

    if not os.path.exists(filepath):
        download(filename, working_directory, url_source)
        statinfo = os.stat(filepath)
        logging.info('Succesfully downloaded %s %s bytes.' % (filename, statinfo.st_size))  # , 'bytes.')
        if not (expected_bytes is None) and (expected_bytes != statinfo.st_size):
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        if extract:
            if tarfile.is_tarfile(filepath):
                logging.info('Trying to extract tar file')
                tarfile.open(filepath, 'r').extractall(working_directory)
                logging.info('... Success!')
            elif zipfile.is_zipfile(filepath):
                logging.info('Trying to extract zip file')
                with zipfile.ZipFile(filepath) as zf:
                    zf.extractall(working_directory)
                logging.info('... Success!')
            else:
                logging.info("Unknown compression_format only .tar.gz/.tar.bz2/.tar and .zip supported")
    return filepath


def get_dataloader_speed(dl, num_steps):
    cnt = 0
    start = time.time()
    end = start
    for _ in dl:
        cnt += 1
        if cnt == num_steps:
            end = time.time()
            break
    return (end - start) / num_steps


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes) + "B"
    elif abs(bytes) < 1e6:
        return str(round(bytes / 1e3, 2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def get_process_memory():
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms, mi.vms


def ensure_proc_terminate(proc):
    """
    Make sure processes terminate when main process exit.

    Args:
        proc (multiprocessing.Process or list)
    """
    if isinstance(proc, list):
        for p in proc:
            ensure_proc_terminate(p)
        return

    def stop_proc_by_weak_ref(ref):
        proc = ref()
        if proc is None:
            return
        if not proc.is_alive():
            return
        proc.terminate()
        proc.join()

    assert isinstance(proc, multiprocessing.Process)
    atexit.register(stop_proc_by_weak_ref, weakref.ref(proc))
