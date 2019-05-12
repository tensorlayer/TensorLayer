import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16
import time
import os
import psutil
import numpy as np
from exp_config import random_input_generator, MONITOR_INTERVAL, NUM_ITERS, BATCH_SIZE, LERANING_RATE

# set gpu_id 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# system monitor
info = psutil.virtual_memory()
monitor_interval = MONITOR_INTERVAL
avg_mem_usage = 0
max_mem_usage = 0
count = 0
total_time = 0

# get the whole model
vgg = vgg16()

start_time = time.time()
vgg = vgg.to(device)
total_time += time.time() - start_time

# training setting
num_iter = NUM_ITERS
batch_size = BATCH_SIZE
optimizer = optim.Adam(vgg.parameters(), lr=LERANING_RATE)

# data generator
gen = random_input_generator(num_iter, batch_size, format='NCHW')

# begin training

for idx, data in enumerate(gen):

    start_time = time.time()

    x_batch = torch.Tensor(data[0])
    y_batch = torch.Tensor(data[1]).long()
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    # forward + backward
    outputs = vgg(x_batch)
    loss = F.cross_entropy(outputs, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
