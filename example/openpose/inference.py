import time
import config
from models import model
import tensorflow as tf
import tensorlayer as tl

n_pos = config.MODEL.n_pos
h, w = 300, 200  # TODO

# define model
x = tf.placeholder(tf.float32, [None, None, None, 3], "image")

_, _, _, net = model(x, n_pos, None, None, False, False)

net.outputs

# restore model parameters

# get one example image with range 0~1

tl.files.read_image()

# inference
# 1st time need time to compile
st = time.time()

t = time.time() - st
print("took {}s i.e. {} FPS".format(t, 1. / t))

# save result
config.LOG.vis_path
