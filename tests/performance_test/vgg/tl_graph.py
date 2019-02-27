import time
import os
import psutil
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Conv2d, Dense, Flatten, Input, MaxPool2d, LayerList
from tensorlayer.models import Model
from exp_config import random_input_generator, MONITOR_INTERVAL, NUM_ITERS, BATCH_SIZE, LERANING_RATE

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


# get the whole model
vgg = tl.models.vgg.vgg16()

# system monitor
info = psutil.virtual_memory()
monitor_interval = MONITOR_INTERVAL
avg_mem_usage = 0
max_mem_usage = 0
count = 0
total_time = 0

# training setting
num_iter = NUM_ITERS
batch_size = BATCH_SIZE

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
y_ = tf.placeholder(tf.int64, shape=[None], name='targets')
y = vgg(x, is_train=True)
cost = tl.cost.cross_entropy(y, y_, name='train_loss')
train_weights = vgg.weights
train_op = tf.train.AdamOptimizer(learning_rate=LERANING_RATE).minimize(cost, var_list=train_weights)

# forbid tensorflow taking up all the GPU memory
# FIXME: enable this to see the GPU memory it consumes, not sure whether it affects performance
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# data generator
gen = random_input_generator(num_iter, batch_size)

# begin training

for idx, data in enumerate(gen):
    x_batch = data[0]
    y_batch = data[1]
    # x_batch = tf.convert_to_tensor(data[0])
    # y_batch = tf.convert_to_tensor(data[1])

    start_time = time.time()

    # forward + backward
    sess.run(train_op, feed_dict={x: x_batch, y_: y_batch})

    end_time = time.time()
    consume_time = end_time - start_time
    total_time += consume_time

    if idx % monitor_interval == 0:
        cur_usage = psutil.Process(os.getpid()).memory_info().rss
        max_mem_usage = max(cur_usage, max_mem_usage)
        avg_mem_usage += cur_usage
        count += 1
        tl.logging.info("[*] {} iteration: memory usage {:.2f}MB, consume time {:.4f}s".format(
            idx, cur_usage / (1024 * 1024), consume_time))

print('consumed time:', total_time)

avg_mem_usage = avg_mem_usage / count / (1024 * 1024)
max_mem_usage = max_mem_usage / (1024 * 1024)
print('average memory usage: {:.2f}MB'.format(avg_mem_usage))
print('maximum memory usage: {:.2f}MB'.format(max_mem_usage))
