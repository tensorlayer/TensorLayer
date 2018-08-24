### Training and testing switch

There are two main ways to switch training and testing: 1) reuse model, 2) control by placeholder.

- `tutorial_mlp_dropout1.py` : controls the dropout probabilities by using placeholders inside layers (`all_drops`).

- `tutorial_mlp_dropout2.py` : defines two models but share the same network parameters.

### Data augmentation

- Data augmentation is essential for training, while if the augmentation is complex, it will slow down the training.
We used CIFAR10 classification as example of data augmentation. 
- For the best performance, please use `tutorial_cifar10_datasetapi.py`.
MNIST examples used `placeholder` to feed in data, however `placeholder` is supported for backwards compatibility, and the `tl.prepro.threading_data` is for quick testing. 
- It is suggested to use TensorFlow's DataSet API (`tf.data` and `tf.image`) and TFRecord for the sake of performance and generalibity.
- For TFRecord and Dataset API,
TFRecord needs to first store all data into TFRecord format, while Dataset API is simpler that can directly use data XXXX.

### Float16
- For Float16, some GPUs can speed up but some cannot.

### Others
- For distributed training, xxx