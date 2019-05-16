Something you need to know:

### 1. Static and dynamic models

1) `tutorial_mnist_mlp_static.py`: static model

2) `tutorial_mnist_mlp_dynamic.py`: dynamic model

### 2. Switching Training and testing

There are two ways to switch the training and testing mode: 

1 ) use Pytorch-like method, turn on and off the training/evaluation as follow:

```python
model.train() # enable dropout, batch norm decay and etc
y1 = model(x)
model.eval() # disable dropout, fix batch norm weights and etc
y2 = model(x)
```

2) use TensorLayer 1.x method, input `is_train` to the model while inferencing.

```python
y1 = model(x, is_train=True)
y2 = model(x, is_train=False)
```



### Data augmentation

- Data augmentation is essential for training, while if the augmentation is complex, it will slow down the training.
We used CIFAR10 classification as example of data augmentation. 
- For the best performance, please use `tutorial_cifar10_datasetapi.py`.
- It is suggested to use TensorFlow's DataSet API (`tf.data` and `tf.image`) and TFRecord for the sake of performance and generalibity.
- For TFRecord and Dataset API,
TFRecord needs to first store all data into TFRecord format, while Dataset API is simpler that can directly use data XXXX.

### Float16
- For Float16, some GPUs can speed up but some cannot.

### Others
- For distributed training
