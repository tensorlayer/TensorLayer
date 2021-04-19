import time
import os
import psutil
import keras
from keras.applications.vgg16 import VGG16
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
import tensorflow as tf
from exp_config import random_input_generator, MONITOR_INTERVAL, NUM_ITERS, BATCH_SIZE, LERANING_RATE

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# get the whole model
vgg = VGG16(weights=None)

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
vgg.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=LERANING_RATE))

# data generator
gen = random_input_generator(num_iter, batch_size)

# begin training

for idx, data in enumerate(gen):
    x_batch = data[0]
    y_batch = to_categorical(data[1], num_classes=1000)

    start_time = time.time()

    # forward + backward
    vgg.train_on_batch(x_batch, y_batch)

    end_time = time.time()
    consume_time = end_time - start_time
    total_time += consume_time

    if idx % monitor_interval == 0:
        cur_usage = psutil.Process(os.getpid()).memory_info().rss
        max_mem_usage = max(cur_usage, max_mem_usage)
        avg_mem_usage += cur_usage
        count += 1
        print(
            "[*] {} iteration: memory usage {:.2f}MB, consume time {:.4f}s".format(
                idx, cur_usage / (1024 * 1024), consume_time
            )
        )

print('consumed time:', total_time)

avg_mem_usage = avg_mem_usage / count / (1024 * 1024)
max_mem_usage = max_mem_usage / (1024 * 1024)
print('average memory usage: {:.2f}MB'.format(avg_mem_usage))
print('maximum memory usage: {:.2f}MB'.format(max_mem_usage))
