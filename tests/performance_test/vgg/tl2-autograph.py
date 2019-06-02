import time
import os
import psutil
import tensorflow as tf
import tensorlayer as tl
from exp_config import random_input_generator, MONITOR_INTERVAL, NUM_ITERS, BATCH_SIZE, LERANING_RATE

tf.config.gpu.set_per_process_memory_growth(True)

tl.logging.set_verbosity(tl.logging.DEBUG)

# get the whole model
vgg = tl.models.vgg16()

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
train_weights = vgg.trainable_weights
optimizer = tf.optimizers.Adam(learning_rate=LERANING_RATE)
loss_object = tl.cost.cross_entropy

# data generator
gen = random_input_generator(num_iter, batch_size)


# training function
@tf.function
def train_step(x_batch, y_batch):
    # forward + backward
    with tf.GradientTape() as tape:
        ## compute outputs
        _logits = vgg(x_batch)
        ## compute loss and update model
        _loss = loss_object(_logits, y_batch)

    grad = tape.gradient(_loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))


# begin training
vgg.train()

for idx, data in enumerate(gen):
    start_time = time.time()

    train_step(data[0], data[1])

    end_time = time.time()
    consume_time = end_time - start_time
    total_time += consume_time

    if idx % monitor_interval == 0:
        cur_usage = psutil.Process(os.getpid()).memory_info().rss
        max_mem_usage = max(cur_usage, max_mem_usage)
        avg_mem_usage += cur_usage
        count += 1
        tl.logging.info(
            "[*] {} iteration: memory usage {:.2f}MB, consume time {:.4f}s".format(
                idx, cur_usage / (1024 * 1024), consume_time
            )
        )

print('consumed time:', total_time)

avg_mem_usage = avg_mem_usage / count / (1024 * 1024)
max_mem_usage = max_mem_usage / (1024 * 1024)
print('average memory usage: {:.2f}MB'.format(avg_mem_usage))
print('maximum memory usage: {:.2f}MB'.format(max_mem_usage))
