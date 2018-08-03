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


def main_test_layers(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    # Note: the softmax is implemented internally in tl.cost.cross_entropy(y, y_)
    # to speed up computation, so we use identity in the last layer.
    # see tf.nn.sparse_softmax_cross_entropy_with_logits()
    if model == 'relu':
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.8, name='drop1')
        net = tl.layers.DenseLayer(net, n_units=800, act=tf.nn.relu, name='relu1')
        net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
        net = tl.layers.DenseLayer(net, n_units=800, act=tf.nn.relu, name='relu2')
        net = tl.layers.DropoutLayer(net, keep=0.5, name='drop3')
        net = tl.layers.DenseLayer(net, n_units=10, act=None, name='output')
    elif model == 'dropconnect':
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropconnectDenseLayer(net, keep=0.8, n_units=800, act=tf.nn.relu, name='dropconnect1')
        net = tl.layers.DropconnectDenseLayer(net, keep=0.5, n_units=800, act=tf.nn.relu, name='dropconnect2')
        net = tl.layers.DropconnectDenseLayer(net, keep=0.5, n_units=10, act=None, name='output')

    # To print all attributes of a Layer.
    # attrs = vars(net)
    # print(', '.join("%s: %s\n" % item for item in attrs.items()))
    # print(net.all_drop)     # {'drop1': 0.8, 'drop2': 0.5, 'drop3': 0.5}

    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, name='xentropy')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # y_op = tf.argmax(tf.nn.softmax(y), 1)

    # You can add more penalty to the cost function as follow.
    # cost = cost + tl.cost.maxnorm_regularizer(1.0)(net.all_params[0]) + tl.cost.maxnorm_regularizer(1.0)(net.all_params[2])
    # cost = cost + tl.cost.lo_regularizer(0.0001)(net.all_params[0]) + tl.cost.lo_regularizer(0.0001)(net.all_params[2])
    # cost = cost + tl.cost.maxnorm_o_regularizer(0.001)(net.all_params[0]) + tl.cost.maxnorm_o_regularizer(0.001)(net.all_params[2])

    # train
    n_epoch = 100
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 5
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    net.print_params()
    net.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable dropout or dropconnect layers
            sess.run(train_op, feed_dict=feed_dict)

            # The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
            # If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder, the shape of the value will be checked for compatibility with the placeholder.
            # If the key is a SparseTensor, the value should be a SparseTensorValue.

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
            # print("   train acc: %f" % (train_acc/ n_batch))
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
            # try:
            #     # You can visualize the weight of 1st hidden layer as follow.
            #     tl.vis.draw_weights(net.all_params[0].eval(), second=10, saveable=True, shape=[28, 28], name='w1_' + str(epoch + 1), fig_idx=2012)
            #     # You can also save the weight of 1st hidden layer to .npz file.
            #     # tl.files.save_npz([net.all_params[0]] , name='w1'+str(epoch+1)+'.npz')
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.draw_weights(), if you want to save the feature images for different dataset")

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

    # Add ops to save and restore all the variables, including variables for training.
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)

    # You can also save the parameters into .npz file.
    tl.files.save_npz(net.all_params, name='model.npz')
    # You can only save one parameter as follow.
    # tl.files.save_npz([net.all_params[0]] , name='model.npz')
    # Then, restore the parameters as follow.
    # load_params = tl.files.load_npz(path='', name='model.npz')
    # tl.files.assign_params(sess, load_params, net)

    # In the end, close TensorFlow session.
    sess.close()


def main_test_denoise_AE(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')

    print("Build net")
    if model == 'relu':
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.5, name='denoising1')  # if drop some inputs, it is denoise AE
        net = tl.layers.DenseLayer(net, n_units=196, act=tf.nn.relu, name='relu1')
        recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=tf.nn.softplus, name='recon_layer1')
    elif model == 'sigmoid':
        # sigmoid - set keep to 1.0, if you want a vanilla Autoencoder
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.5, name='denoising1')
        net = tl.layers.DenseLayer(net, n_units=196, act=tf.nn.sigmoid, name='sigmoid1')
        recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon_layer1')

    ## ready to train
    tl.layers.initialize_global_variables(sess)

    ## print all params
    print("All net Params")
    net.print_params()

    ## pretrain
    print("Pre-train Layer 1")
    recon_layer1.pretrain(
        sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=200, batch_size=128, print_freq=10,
        save=True, save_name='w1pre_'
    )
    # You can also disable denoisong by setting denoise_name=None.
    # recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
    #                           denoise_name=None, n_epoch=500, batch_size=128,
    #                           print_freq=10, save=True, save_name='w1pre_')

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()


def main_test_stacked_denoise_AE(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    if model == 'relu':
        act = tf.nn.relu
        act_recon = tf.nn.softplus
    elif model == 'sigmoid':
        act = tf.nn.sigmoid
        act_recon = act

    # Define net
    print("\nBuild net")
    net = tl.layers.InputLayer(x, name='input')
    # denoise layer for AE
    net = tl.layers.DropoutLayer(net, keep=0.5, name='denoising1')
    # 1st layer
    net = tl.layers.DropoutLayer(net, keep=0.8, name='drop1')
    net = tl.layers.DenseLayer(net, n_units=800, act=act, name=model + '1')
    x_recon1 = net.outputs
    recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=act_recon, name='recon_layer1')
    # 2nd layer
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    net = tl.layers.DenseLayer(net, n_units=800, act=act, name=model + '2')
    recon_layer2 = tl.layers.ReconLayer(net, x_recon=x_recon1, n_units=800, act=act_recon, name='recon_layer2')
    # 3rd layer
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop3')
    net = tl.layers.DenseLayer(net, 10, act=None, name='output')

    # Define fine-tune process
    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    n_epoch = 200
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10

    train_params = net.all_params

    # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

    # Initialize all variables including weights, biases and the variables in train_op
    tl.layers.initialize_global_variables(sess)

    # Pre-train
    print("\nAll net Params before pre-train")
    net.print_params()
    print("\nPre-train Layer 1")
    recon_layer1.pretrain(
        sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10,
        save=True, save_name='w1pre_'
    )
    print("\nPre-train Layer 2")
    recon_layer2.pretrain(
        sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10,
        save=False
    )
    print("\nAll net Params after pre-train")
    net.print_params()

    # Fine-tune
    print("\nFine-tune net")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            feed_dict[tl.layers.LayersConfig.set_keep['denoising1']] = 1  # disable denoising layer
            sess.run(train_op, feed_dict=feed_dict)

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
            # try:
            #     # visualize the 1st hidden layer during fine-tune
            #     tl.vis.draw_weights(net.all_params[0].eval(), second=10, saveable=True, shape=[28, 28], name='w1_' + str(epoch + 1), fig_idx=2012)
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.draw_weights(), if you want to save the feature images for different dataset")

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
    # print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()


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

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size])

    net = tl.layers.InputLayer(x, name='input')
    ## Professional conv API for tensorflow expert
    # net = tl.layers.Conv2dLayer(net,
    #                     act = tf.nn.relu,
    #                     shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME',
    #                     name ='cnn1')     # output: (?, 28, 28, 32)
    # net = tl.layers.PoolLayer(net,
    #                     ksize=[1, 2, 2, 1],
    #                     strides=[1, 2, 2, 1],
    #                     padding='SAME',
    #                     pool = tf.nn.max_pool,
    #                     name ='pool1',)   # output: (?, 14, 14, 32)
    # net = tl.layers.Conv2dLayer(net,
    #                     act = tf.nn.relu,
    #                     shape = [5, 5, 32, 64], # 64 features for each 5x5 patch
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME',
    #                     name ='cnn2')     # output: (?, 14, 14, 64)
    # net = tl.layers.PoolLayer(net,
    #                     ksize=[1, 2, 2, 1],
    #                     strides=[1, 2, 2, 1],
    #                     padding='SAME',
    #                     pool = tf.nn.max_pool,
    #                     name ='pool2',)   # output: (?, 7, 7, 64)
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
            tl.files.save_ckpt(sess,mode_name='model.ckpt', save_dir='model', printable=True)
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
            # try:
            #     tl.vis.CNN2d(net.all_params[0].eval(), second=10, saveable=True, name='cnn1_' + str(epoch + 1), fig_idx=2012)
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.CNN(), if you want to save the feature images for different dataset")

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

    #sess = tf.InteractiveSession()

    # Dropout and Dropconnect
    # main_test_layers(model='relu')  # model = relu, dropconnect

    # Single Denoising Autoencoder
    # main_test_denoise_AE(model='sigmoid')       # model = relu, sigmoid

    # Stacked Denoising Autoencoder
    # main_test_stacked_denoise_AE(model='relu')  # model = relu, sigmoid

    # CNN
    main_test_cnn_layer()
