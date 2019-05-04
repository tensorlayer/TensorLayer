# Performance Test (VGG16)

### Introduction

This test compares performance of the following libraries:

1. TensorLayer v. 2.0
2. TensorFlow ([tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)) v. 2.0.0-alpha
3. [Keras](https://keras.io/) v. 2.2.4



### With GPU

__Hardware:__

- CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz  40 core
- GPU: TITAN Xp

__Experiment Settings:__
- Model: [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
- Batch size: 32
- Number of iterations: 300

__Results:__

|   Mode    |       Lib       |  Data Format  | Max GPU Memory Usage(MB)  |Max CPU Memory Usage(MB) | Avg CPU Memory Usage(MB) | Runtime (sec) |
| :-------: | :-------------: | :-----------: | :-----------------: | :-----------------: | :-----------------: | :-----------: |
| AutoGraph | TensorFlow 2.0  | channel last  | 11833 |      2161         |        2136         |      74       |
|           | Tensorlayer 2.0 | channel last  | 11833 |      2187         |        2169         |      76       |
|   Graph   |      Keras      | channel last  | 8677 |      2580         |        2576         |      101       |
|   Eager   | TensorFlow 2.0  | channel last  | 8723 |      2052         |        2024         |      97       |
|           | TensorLayer 2.0 | channel last  | 8723 |      2010         |        2007         |      95       |

