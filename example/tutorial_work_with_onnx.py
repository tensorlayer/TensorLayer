#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Work with TensorLayer.

Introduction
----------------
ONNX is an open specification that consists of the following components:
- A definition of an extensible computation graph model.
- Definitions of standard data types.
- Definitions of built-in operators
Caffe2, PyTorch, Microsoft Cognitive Toolkit, Apache MXNet and other tools are developing ONNX support. Enabling interoperability between different frameworks and streamlining the path from research to production will increase the speed of innovation in the AI community.

Pre-work:
----------------------------
- Install ONNX package : 
>>> pip install onnx
Note: When installing in a non-Anaconda environment, make sure to install the Protobuf compiler before running the pip installation of onnx. For example, on Ubuntu:
>>>sudo apt-get install protobuf-compiler libprotoc-dev
>>>pip install onnx
More details please go to ONNX official website: https://github.com/onnx/onnx

- Environment：
Ubuntu:16.04.4 LTS
Python:3.6.5
TensorLayer:1.8.6rc2
TensorFlow-gpu:1.8.0
onnx:1.2.2
onnx-tf:1.1.2

Note
------
- This tutorial is refer to the onnx-tf tutorial:
https://github.com/onnx/tutorials/blob/7b549ae622ff8d74a5f5e0c32e109267f4c9ccae/tutorials/OnnxTensorflowExport.ipynb

1.Training
----------
Firstly, we can initiate the training script by issuing the command on your terminal.
>>>python tutorial_work_with_onnx.py 
 Shortly, we should obtain a trained MNIST model. The training process needs no special instrumentation. However, to successfully convert the trained model, onnx-tensorflow requires three pieces of information, all of which can be obtained after training is complete:
 
- Graph definition: 
You need to obtain information about the graph definition in the form of GraphProto. The easiest way to achieve this is to use the following snippet of code as shown in the example training script:
>>>with open("graph.proto", "wb") as file:
>>> graph = tf.get_default_graph().as_graph_def(add_shapes=True)
>>> file.write(graph.SerializeToString())
This code is under the code where you call your architecture in your function

- Shape information: By default, as_graph_def does not serialize any information about the shapes of the intermediate tensor and such information is required by onnx-tensorflow. Thus we request Tensorflow to serialize the shape information by adding the keyword argument add_shapes=True as demonstrated above.

- Checkpoint: Tensorflow checkpoint files contain information about the obtained weight; thus they are needed to convert the trained model to ONNX format.

2.Graph Freezing
----------------
Secondly, we freeze the graph. Thus here we build the free_graph tool in TensorLayer source folder and execute it with the information about where the GraphProto is, where the checkpoint file is and where to put the freozen graph. 
>>>python3 -m tensorflow.python.tools.freeze_graph \
    --input_graph=/root/graph.proto \
    --input_checkpoint=/root/model/model.ckpt \
    --output_graph=/root/frozen_graph.pb \
    --output_node_names=output/bias_add\
    --input_binary=True
    
note: 
input_graph is the path of your proto file
input_checkpoint is the path of your checkpoint file
output_graph is the path where you want to put
output_node is the output node you want to put into your graph:
you can try this code to print and find the node what you want:
>>>print([n.name for n in tf.get_default_graph().as_graph_def().node])

Note that now we have obtained the frozen_graph.pb with graph definition as well as weight information in one file.

3.Model Conversion
-----------------
Thirdly, we convert the model to ONNX format using onnx-tensorflow. Using tensorflow_graph_to_onnx_model from onnx-tensorflow API (documentation available at https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/doc/API.md).
>>>import tensorflow as tf
>>>from onnx_tf.frontend import tensorflow_graph_to_onnx_model

>>>with tf.gfile.GFile("frozen_graph.pb", "rb") as f:
>>>    graph_def = tf.GraphDef()
>>>    graph_def.ParseFromString(f.read())
>>>    onnx_model = tensorflow_graph_to_onnx_model(graph_def,
>>>                                     "output/bias_add",
>>>                                     opset=6)

>>>    file = open("mnist.onnx", "wb")
>>>    file.write(onnx_model.SerializeToString())
>>>    file.close()

Then you will get thr first node info:
>>>input: "cnn1/kernel"
>>>output: "cnn1/kernel/read"
>>>name: "cnn1/kernel/read"
>>>op_type: "Identity"

Inference using Backend(This part onnx-tf is under implementation!!!)
-------------------------------------------------------------------
In this tutorial, we continue our demonstration by performing inference using this obtained ONNX model. Here, we exported an image representing a handwritten 7 and stored the numpy array as image.npz. Using onnx-tf backend, we will classify this image using the converted ONNX model.
>>>import onnx
>>>import numpy as np
>>>from onnx_tf.backend import prepare

>>>model = onnx.load('mnist.onnx')
>>>tf_rep = prepare(model)
>>>#Image Path
>>>img = np.load("./assets/image.npz") 
>>>output = tf_rep.run(img.reshape([1, 784]))
>>>print "The digit is classified as ", np.argmax(output)

You will get the information in your console:
>>>The digit is classified as  7

"""

import time
import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def main_test_cnn_layer():
    """Reimplementation of the TensorFlow official MNIST CNN tutorials:
    - https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py

    More TensorFlow official CNN tutorials can be found here:
    - tutorial_cifar10.py
    - https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html

    - For simplified CNN layer see "Convolutional layer (Simplified)"
      in read the docs website.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

    sess = tf.InteractiveSession()

    batch_size = 128
    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size])

    net = tl.layers.InputLayer(x, name='input')

    ## Simplified conv API (the same with the above layers)
    net = tl.layers.Conv2d(net, 32, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn1')
    net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    net = tl.layers.Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn2')
    net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    ## end of conv
    net = tl.layers.FlattenLayer(net, name='flatten')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop1')
    net = tl.layers.DenseLayer(net, 256, act=tf.nn.relu, name='relu1')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    net = tl.layers.DenseLayer(net, 10, act=None, name='output')

    y = net.outputs

    print([n.name for n in tf.get_default_graph().as_graph_def().node])

    # To string Graph
    with open("graph.proto", "wb") as file:
        graph = tf.get_default_graph().as_graph_def(add_shapes=True)
        file.write(graph.SerializeToString())

    cost = tl.cost.cross_entropy(y, y_, 'cost')

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 200
    learning_rate = 0.0001
    print_freq = 10

    train_params = net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)
    net.print_params()
    net.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)
        # Save the checkpoint every 10 eopchs
        if epoch % 10 == 0:
            tl.files.save_ckpt(sess, mode_name='model.ckpt', save_dir='model', printable=True)
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))

    # Evaluation
    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))


if __name__ == '__main__':

    # CNN
    main_test_cnn_layer()
