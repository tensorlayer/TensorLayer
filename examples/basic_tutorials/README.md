Something you need to know:

### 1. Eager and Graph modes

1) `1_mnist_mlp_eager_mode.py`: dynamic mode

2) `2_mnist_mlp_graph_mode.py`: static graph mode

### 2. Switching Training and testing

There are two ways to switch training and testing: 

1 ) use Pytorch-like method, turn on and off the training/evaluation as follow:

```python
model.train() # enable dropout, batch norm decay and etc
y1 = model(x)
model.eval() # disable dropout, fix batch norm weights and etc
y2 = model(x)
```

2) use TensorLayer 1.x method, input `is_train` to the model.

```python
y1 = model(x, is_train=True)
y2 = model(x, is_train=False)
```




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
- For distributed training