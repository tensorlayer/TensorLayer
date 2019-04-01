# Performance Test (VGG16)

### Introduction

This test compares performance of the following libraries:

1. TensorLayer v. 2.0
2. TensorFlow ([tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)) v. 2.0.0-alpha
3. Keras v. 2.2.4
4. [PyTorch](https://pytorch.org) v. 1.0.1



### With GPU

__Hardware:__

- CPU: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz  40 core
- GPU: Tesla V100-DGXS-32GB

__Experiment Settings:__
- Model: [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
- Batch size: 32
- Number of iterations: 300

__Results:__

|   Mode    |       Lib       |  Data Format  | Max CPU Memory (MB) | Avg CPU Memory (MB) | Runtime (sec) |
| :-------: | :-------------: | :-----------: | :-----------------: | :-----------------: | :-----------: |
| AutoGraph | TensorFlow 2.0  | channel last  |        3370         |        3346         |      49       |
|           | Tensorlayer 2.0 | channel last  |        3358         |        3367         |      50       |
|   Graph   |      Keras      | channel last  |        3776         |        3775         |      62       |
|   Eager   | TensorFlow 2.0  | channel last  |        3293         |        3284         |      65       |
|           | TensorLayer 2.0 | channel last  |        3296         |        3293         |      65       |
|           |     PyTorch     | channel first |        2562         |        2555         |      43       |

