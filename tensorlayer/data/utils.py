import atexit
import logging
import math
import multiprocessing
import os
import platform
import re
import resource
import shutil
import weakref

import psutil
import tarfile
import time
import zipfile
import progressbar
from urllib.request import urlretrieve


def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.

    Parameters
    ----------
    path : str
        A folder path.

    """
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]


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

    urlretrieve(url_source + filename, filepath, reporthook=_dlProgress)


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


def natural_keys(text):
    """Sort list of string with number in human order.

    Examples
    ----------
    >>> l = ['im1.jpg', 'im31.jpg', 'im11.jpg', 'im21.jpg', 'im03.jpg', 'im05.jpg']
    >>> l.sort(key=tl.files.natural_keys)
    ['im1.jpg', 'im03.jpg', 'im05', 'im11.jpg', 'im21.jpg', 'im31.jpg']
    >>> l.sort() # that is what we dont want
    ['im03.jpg', 'im05', 'im1.jpg', 'im11.jpg', 'im21.jpg', 'im31.jpg']

    References
    ----------
    - `link <http://nedbatchelder.com/blog/200712/human_sorting.html>`__

    """

    # - alist.sort(key=natural_keys) sorts in human order
    # http://nedbatchelder.com/blog/200712/human_sorting.html
    # (See Toothy's implementation in the comments)
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split('(\d+)', text)]


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
    return mi.rss, mi.vms, mi.shared


def get_peak_memory_usage():
    # peak memory usage (bytes on OS X, kilobytes on Linux)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    platform_name = platform.system().lower()

    # If we are on linux
    if platform_name== "linux" or platform_name == "linux2":
        return format_bytes(rss * 1024)

    # If we are on Mac OS X
    elif platform_name == "darwin":
        return format_bytes(rss)

    # We don't support Windows
    elif platform_name == "win32":
        raise EnvironmentError("The Windows operating system is not supported")

    # Unrecognized platform
    else:
        raise EnvironmentError("Unrecognized platform")


def shutdown_proc(proc):
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join()


def shutdown_proc_by_weakref(ref):
    proc = ref()
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join()


def ensure_subprocess_terminate(proc):
    """
    Make sure subprocesses terminate when main process exit.

    Args:
        proc (multiprocessing.Process or list)
    """
    if isinstance(proc, list):
        for p in proc:
            ensure_subprocess_terminate(p)
        return

    assert isinstance(proc, multiprocessing.Process)
    atexit.register(shutdown_proc_by_weakref, weakref.ref(proc))


def clean_up_socket_files(pipe_names):
    if isinstance(pipe_names, list):
        for pipe_name in pipe_names:
            clean_up_socket_files(pipe_name)
        return

    def remove_socket_files(pipe_name):
        # remove all ipc socket files
        # the environment variable starts with 'ipc://', so file name starts from 6
        try:
            os.remove(pipe_name[6:])
        except (FileNotFoundError, KeyError):
            pass

    atexit.register(remove_socket_files, pipe_names)


def download_file_from_google_drive(ID, destination):
    """Download file from Google Drive.

    See ``tl.files.load_celebA_dataset`` for example.

    Parameters
    --------------
    ID : str
        The driver ID.
    destination : str
        The destination for save file.

    """
    try:
        from tqdm import tqdm
    except ImportError as e:
        print(e)
        raise ImportError("Module tqdm not found. Please install tqdm via pip or other package managers.")

    try:
        import requests
    except ImportError as e:
        print(e)
        raise ImportError("Module requests not found. Please install requests via pip or other package managers.")

    def save_response_content(response, destination, chunk_size=32 * 1024):

        total_size = int(response.headers.get('content-length', 0))
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size), total=total_size, unit='B', unit_scale=True,
                              desc=destination):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': ID}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': ID, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def load_file_list(path=None, regx='\.jpg', printable=True, keep_prefix=False):
    r"""Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : str or None
        A folder path, if `None`, use the current directory.
    regx : str
        The regx of file name.
    printable : boolean
        Whether to print the files infomation.
    keep_prefix : boolean
        Whether to keep path in the file name.

    Examples
    ----------
    >>> file_list = load_file_list(path=None, regx='w1pre_[0-9]+\.(npz)')

    """
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    # return_list.sort()
    if keep_prefix:
        for i, f in enumerate(return_list):
            return_list[i] = os.path.join(path, f)

    if printable:
        logging.info('Match file list = %s' % return_list)
        logging.info('Number of files = %d' % len(return_list))
    return return_list


def file_exists(filepath):
    """Check whether a file exists by given file path."""
    return os.path.isfile(filepath)


def folder_exists(folderpath):
    """Check whether a folder exists by given folder path."""
    return os.path.isdir(folderpath)


def del_folder(folderpath):
    """Delete a folder by given folder path."""
    shutil.rmtree(folderpath)


def del_file(filepath):
    """Delete a file by given file path."""
    os.remove(filepath)


def read_file(filepath):
    """Read a file and return a string.

    Examples
    ---------
    >>> data = read_file('data.txt')
    """
    with open(filepath, 'r') as afile:
        return afile.read()
