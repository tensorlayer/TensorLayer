#! /usr/bin/python
# -*- coding: utf8 -*-




import tensorflow as tf
import tensorlayer as tl
import os
import subprocess
import sys
from sys import platform as _platform
from sys import exit as _exit


def exit_tf(sess=None, port=6006):
    """Close TensorFlow session, TensorBoard and Nvidia-process if available.

    Parameters
    ----------
    sess : a session instance of TensorFlow
        TensorFlow session
    tb_port : an integer
        TensorBoard port you want to close, 6006 as default.
    """
    text = "[TL] Close tensorboard and nvidia-process if available"
    text2 = "[TL] Close tensorboard and nvidia-process not yet supported by this function (tl.ops.exit_tf) on "
    if sess != None:
        sess.close()
    # import time
    # time.sleep(2)
    if _platform == "linux" or _platform == "linux2":
        print('linux: %s' % text)
        os.system('nvidia-smi')
        os.system('fuser '+ port +'/tcp -k')  # kill tensorboard 6006
        os.system("nvidia-smi | grep python |awk '{print $3}'|xargs kill") # kill all nvidia-smi python process
        _exit()
    elif _platform == "darwin":
        print('OS X: %s' % text)
        subprocess.Popen("lsof -i tcp:"+ str(port) +"  | grep -v PID | awk '{print $2}' | xargs kill", shell=True) # kill tensorboard
    elif _platform == "win32":
        print(text2 + "Windows")
        # TODO
    else:
        print(text2 + _platform)

def open_tb(logdir='/tmp/tensorflow', port=6006):
    """Open Tensorboard.

    Parameters
    ----------
    logdir : a string
        Directory where your tensorboard logs are saved
    port : an integer
        TensorBoard port you want to open, 6006 is tensorboard default
    """
    text = "[TL] Open tensorboard, go to localhost:" + str(port) + " to access"
    text2 = " not yet supported by this function (tl.ops.open_tb)"

    if not tl.files.exists_or_mkdir(logdir, verbose=False):
        print("[TL] Log reportory was created at %s" % logdir)

    if _platform == "linux" or _platform == "linux2":
        print('linux %s' % text2)
        # TODO
    elif _platform == "darwin":
        print('OS X: %s' % text)
        subprocess.Popen(sys.prefix + " | python -m tensorflow.tensorboard --logdir=" + logdir + " --port=" + str(port), shell=True) # open tensorboard in localhost:6006/ or whatever port you chose
    elif _platform == "win32":
        print('Windows%s' % text2)
        # TODO
    else:
        print(_platform + text2)

def clear_all(printable=True):
    """Clears all the placeholder variables of keep prob,
    including keeping probabilities of all dropout, denoising, dropconnect etc.

    Parameters
    ----------
    printable : boolean
        If True, print all deleted variables.
    """
    print('clear all .....................................')
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        if 'class' in str(globals()[var]): continue

        if printable:
            print(" clear_all ------- %s" % str(globals()[var]))

        del globals()[var]

# def clear_all2(vars, printable=True):
#     """
#     The :function:`clear_all()` Clears all the placeholder variables of keep prob,
#     including keeping probabilities of all dropout, denoising, dropconnect
#     Parameters
#     ----------
#     printable : if True, print all deleted variables.
#     """
#     print('clear all .....................................')
#     for var in vars:
#         if var[0] == '_': continue
#         if 'func' in str(var): continue
#         if 'module' in str(var): continue
#         if 'class' in str(var): continue
#
#         if printable:
#             print(" clear_all ------- %s" % str(var))
#
#         del var

def set_gpu_fraction(sess=None, gpu_fraction=0.3):
    """Set the GPU memory fraction for the application.

    Parameters
    ----------
    sess : a session instance of TensorFlow
        TensorFlow session
    gpu_fraction : a float
        Fraction of GPU memory, (0 ~ 1]

    References
    ----------
    - `TensorFlow using GPU <https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html>`_
    """
    print("[TL]: GPU MEM Fraction %f" % gpu_fraction)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    return sess





def disable_print():
    """Disable console output, ``suppress_stdout`` is recommended.

    Examples
    ---------
    >>> print("You can see me")
    >>> tl.ops.disable_print()
    >>> print(" You can't see me")
    >>> tl.ops.enable_print()
    >>> print("You can see me")
    """
    # sys.stdout = os.devnull   # this one kill the process
    sys.stdout = None
    sys.stderr = os.devnull

def enable_print():
    """Enable console output, ``suppress_stdout`` is recommended.

    Examples
    --------
    - see tl.ops.disable_print()
    """
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# class temporary_disable_print:
#     """Temporarily disable console output.
#
#     Examples
#     ---------
#     >>> print("You can see me")
#     >>> with tl.ops.temporary_disable_print() as t:
#     >>>     print("You can't see me")
#     >>> print("You can see me")
#     """
#     def __init__(self):
#         pass
#     def __enter__(self):
#         sys.stdout = None
#         sys.stderr = os.devnull
#     def __exit__(self, type, value, traceback):
#         sys.stdout = sys.__stdout__
#         sys.stderr = sys.__stderr__
#         return isinstance(value, TypeError)


from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    """Temporarily disable console output.

    Examples
    ---------
    >>> print("You can see me")
    >>> with tl.ops.suppress_stdout():
    >>>     print("You can't see me")
    >>> print("You can see me")

    References
    -----------
    - `stackoverflow <http://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python>`_
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout



def get_site_packages_directory():
    """Print and return the site-packages directory.

    Examples
    ---------
    >>> loc = tl.ops.get_site_packages_directory()
    """
    import site
    try:
        loc = site.getsitepackages()
        print("[TL] tl.ops : site-packages in ", loc)
        return loc
    except:
        print("[TL] tl.ops : Cannot find package dir from virtual environment")
        return False



def empty_trash():
    """Empty trash folder.

    """
    text = "[TL] Empty the trash"
    if _platform == "linux" or _platform == "linux2":
        print('linux: %s' % text)
        os.system("rm -rf ~/.local/share/Trash/*")
    elif _platform == "darwin":
        print('OS X: %s' % text)
        os.system("sudo rm -rf ~/.Trash/*")
    elif _platform == "win32":
        print('Windows: %s' % text)
        try:
            os.system("rd /s c:\$Recycle.Bin")  # Windows 7 or Server 2008
        except:
            pass
        try:
            os.system("rd /s c:\recycler")  #  Windows XP, Vista, or Server 2003
        except:
            pass
    else:
        print(_platform)

#
