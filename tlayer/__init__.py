"""
Deep learning and Reinforcement learning library for Researcher and Engineer
"""
# from __future__ import absolute_import


try:
    install_instr = "Please make sure you install a recent enough version of TensorFlow."
    import tensorflow
except ImportError:
    raise ImportError("Could not import TensorFlow." + install_instr)

from . import activation
from . import cost
from . import files
# from . import init
from . import iterate
from . import layers
from . import ops
from . import utils
from . import visualize
from . import preprocess
from . import nlp
from . import rein


__version__ = "1.1.dev1"
