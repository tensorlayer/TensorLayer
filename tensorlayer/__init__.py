"""
Deep learning and Reinforcement learning library for Researchers and Engineers
"""
from __future__ import absolute_import


try:
    install_instr = "Please make sure you install a recent enough version of TensorFlow."
    import tensorflow
except ImportError:
    raise ImportError("__init__.py : Could not import TensorFlow." + install_instr)

from . import activation
from . import cost
from . import files
from . import iterate
from . import layers
from . import ops
from . import utils
from . import visualize
from . import prepro
from . import nlp
from . import rein

# alias
act = activation
vis = visualize

__version__ = "1.6.3rc"

global_flag = {}
global_dict = {}
