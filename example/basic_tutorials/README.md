### Training and testing switch

There are two main ways to switch training and testing: 1) reuse model, 2) control by placeholder.

`tutorial_mlp_dropout1.py` : controls the dropout probabilities by using placeholder inside layers (`all_drops`).

`tutorial_mlp_dropout2.py` : defines two models but share the same network parameters, XXX

### Data augmentation

Data augmentation is essential for training, while if the augmentation is complex, it will slow down the training. XXX

Please use `basic_tutorials/tutorial_cifar10_datasetapi.py`.

The reason is that: The `placeholder` is supported for backwards compatibility. It is suggested to use TensorFlow's DataSet API (`tf.data` and `tf.image`) and TFRecord for the sake of performance and generalibity.


TFRecord needs to first store all data into TFRecord format, while Dataset API is simpler that can directly use data XXXX.

### Float16
- For Float16, some GPUs can speed up but some cannot.

### Others

For distributed training, xxx