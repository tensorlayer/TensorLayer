import numpy as np


def random_input_generator(num, batchsize=32, format='NHWC'):
    input_shape = (batchsize, 224, 224, 3) if format == 'NHWC' else (batchsize, 3, 224, 224)
    rng = np.random.RandomState(1234)
    for i in range(num):
        x = rng.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        y = rng.randint(0, 1000, size=(batchsize, ))
        yield (x, y)


MONITOR_INTERVAL = 50
NUM_ITERS = 300
BATCH_SIZE = 32
LERANING_RATE = 0.0001
