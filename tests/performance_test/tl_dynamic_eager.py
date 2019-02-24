import time
import os
import psutil
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from exp_config import random_input_generator, MONITOR_INTERVAL, NUM_ITERS, BATCH_SIZE, LERANING_RATE

# forbid tensorflow taking up all the GPU memory
# FIXME: enable this to see the GPU memory it consumes, not sure whether it affects performance
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution(config=config)

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# get the whole model
vgg = tl.models.VGG16()

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
train_weights = vgg.weights
optimizer = tf.train.AdamOptimizer(learning_rate=LERANING_RATE)

# data generator
gen = random_input_generator(num_iter, batch_size)

# begin training
vgg.train()

for idx, data in enumerate(gen):
    x_batch = tf.convert_to_tensor(data[0])
    y_batch = tf.convert_to_tensor(data[1])

    start_time = time.time()

    # forward + backward
    with tf.GradientTape() as tape:
        ## compute outputs
        _logits = vgg(x_batch).outputs
        ## compute loss and update model
        _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')

    grad = tape.gradient(_loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))

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
