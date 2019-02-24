# Performance Test (VGG16)

### Introduction

This test compares performance of the following libraries:
1. TensorLayer v. 2.0
2. TensorFlow ([tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)) v. 1.13.0dev
3. [Keras](https://keras.io) v. 2.2.4
4. [PyTorch](https://pytorch.org) v. 1.0.1

Hardware: 
- CPU: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz 12 core
- GPU: NVIDIA Corporation GP102 TITAN X 1 core

Experiment Settings
- Model: [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)


| Batch size | Num of Iter |
| :--------: | :---------: |
| 32 | 300 |


### Results


| Mode | Lib | Max GPU Memory (MB) | Max CPU Memory (MB) | Avg CPU Memory (MB) | Runtime (sec) |
| :----: | :----: | :---------:| :--------: | :------: |  :------: |
| Graph | TensorLayer | 8659 | 2719 | 2711 | 78 |
|       | tf.keras    | 8659 | 2668 | 2665 | 77 |
|       | Keras       | 8661 | 2685 | 2682 | 97 |
| Eager | TensorLayer | | | ||
|       | tf.keras    | 8703 | 2019 | 1983 | 103 |
|       | PyTorch     | 8911 | 2178 | 2175 | 79  |

Without GPU
