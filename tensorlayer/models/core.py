#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable
from tensorlayer.files import utils
from tensorlayer import logging
import tensorlayer as tl
from tensorlayer.layers.core import Module
import numpy as np
import os
import time

if tl.BACKEND == 'tensorflow':
    import tensorflow as tf
if tl.BACKEND == 'mindspore':
    import mindspore as ms
    from mindspore.ops import composite
    from mindspore.ops import operations as P
    from mindspore.ops import functional as F
    # from mindspore.parallel._utils import (_get_device_num, _get_mirror_mean, _get_parallel_mode)
    # from mindspore.train.parallel_utils import ParallelMode
    from mindspore.nn.wrap import DistributedGradReducer
    from mindspore.common import ParameterTuple
if tl.BACKEND == 'paddle':
    import paddle as pd


class Model:
    """
    High-Level API for Training or Testing.

    `Model` groups layers into an object with training and inference features.

    Args:
        network : The training or testing network.
        loss_fn : Objective function, if loss_fn is None, the
                             network should contain the logic of loss and grads calculation, and the logic
                             of parallel if needed. Default: None.
        optimizer : Optimizer for updating the weights. Default: None.
        metrics : Dict or set of metrics to be evaluated by the model during

    Examples:
        >>> import tensorlayer as tl
        >>> class Net(Module):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.conv = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), in_channels=5, name='conv2d')
        >>>         self.bn = tl.layers.BatchNorm2d(num_features=32, act=tl.ReLU)
        >>>         self.flatten = tl.layers.Flatten()
        >>>         self.fc = tl.layers.Dense(n_units=12, in_channels=32*224*224) # padding=0
        >>>
        >>>     def construct(self, x):
        >>>         x = self.conv(x)
        >>>         x = self.bn(x)
        >>>         x = self.flatten(x)
        >>>         out = self.fc(x)
        >>>         return out
        >>>
        >>> net = Net()
        >>> loss = tl.cost.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
        >>> optim = tl.layers.Momentum(params=net.trainable_weights, learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
        >>> dataset = get_dataset()
        >>> model.train(2, dataset)
    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, **kwargs):
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.all_weights = network.all_weights
        self.train_weights = self.network.trainable_weights

    def train(self, n_epoch, train_dataset=None, test_dataset=False, print_train_batch=False, print_freq=5):
        if not isinstance(train_dataset, Iterable):
            raise Exception("Expected type in (train_dataset, Iterable), but got {}.".format(type(train_dataset)))

        if tl.BACKEND == 'tensorflow':
            self.tf_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )
        elif tl.BACKEND == 'mindspore':
            self.ms_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )
        elif tl.BACKEND == 'paddle':
            self.pd_train(
                n_epoch=n_epoch, train_dataset=train_dataset, network=self.network, loss_fn=self.loss_fn,
                train_weights=self.train_weights, optimizer=self.optimizer, metrics=self.metrics,
                print_train_batch=print_train_batch, print_freq=print_freq, test_dataset=test_dataset
            )

    def eval(self, test_dataset):
        self.network.eval()
        test_loss, test_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in test_dataset:
            _logits = self.network(X_batch)
            test_loss += self.loss_fn(_logits, y_batch)
            if self.metrics:
                test_acc += self.metrics(_logits, y_batch)
            else:
                test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   test loss: {}".format(test_loss / n_iter))
        print("   test acc:  {}".format(test_acc / n_iter))

    def save_weights(self, file_path, format=None):
        """Input file_path, save model weights into a file of given format.
            Use self.load_weights() to restore.

        Parameters
        ----------
        file_path : str
            Filename to which the model weights will be saved.
        format : str or None
            Saved file format.
            Value should be None, 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            1) If this is set to None, then the postfix of file_path will be used to decide saved format.
            If the postfix is not in ['h5', 'hdf5', 'npz', 'ckpt'], then file will be saved in hdf5 format by default.
            2) 'hdf5' will save model weights name in a list and each layer has its weights stored in a group of
            the hdf5 file.
            3) 'npz' will save model weights sequentially into a npz file.
            4) 'npz_dict' will save model weights along with its name as a dict into a npz file.
            5) 'ckpt' will save model weights into a tensorflow ckpt file.

            Default None.

        Examples
        --------
        1) Save model weights in hdf5 format by default.
        >>> net = vgg16()
        >>> net.save_weights('./model.h5')
        ...
        >>> net.load_weights('./model.h5')

        2) Save model weights in npz/npz_dict format
        >>> net = vgg16()
        >>> net.save_weights('./model.npz')
        >>> net.save_weights('./model.npz', format='npz_dict')

        """

        # self.all_weights = self.network.all_weights
        if self.all_weights is None or len(self.all_weights) == 0:
            logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
            return

        if format is None:
            postfix = file_path.split('.')[-1]
            if postfix in ['h5', 'hdf5', 'npz', 'ckpt']:
                format = postfix
            else:
                format = 'hdf5'

        if format == 'hdf5' or format == 'h5':
            utils.save_weights_to_hdf5(file_path, self)
        elif format == 'npz':
            utils.save_npz(self.all_weights, file_path)
        elif format == 'npz_dict':
            utils.save_npz_dict(self.all_weights, file_path)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
                "Other format is not supported now."
            )

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights().

        Parameters
        ----------
        file_path : str
            Filename from which the model weights will be loaded.
        format : str or None
            If not specified (None), the postfix of the file_path will be used to decide its format. If specified,
            value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            In addition, it should be the same format when you saved the file using self.save_weights().
            Default is None.
        in_order : bool
            Allow loading weights into model in a sequential way or by name. Only useful when 'format' is 'hdf5'.
            If 'in_order' is True, weights from the file will be loaded into model in a sequential way.
            If 'in_order' is False, weights from the file will be loaded into model by matching the name
            with the weights of the model, particularly useful when trying to restore model in eager(graph) mode from
            a weights file which is saved in graph(eager) mode.
            Default is True.
        skip : bool
            Allow skipping weights whose name is mismatched between the file and model. Only useful when 'format' is
            'hdf5' or 'npz_dict'. If 'skip' is True, 'in_order' argument will be ignored and those loaded weights
            whose name is not found in model weights (self.all_weights) will be skipped. If 'skip' is False, error will
            occur when mismatch is found.
            Default is False.

        Examples
        --------
        1) load model from a hdf5 file.
        >>> net = tl.models.vgg16()
        >>> net.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
        >>> net.load_weights('./model_eager.h5') # load sequentially

        2) load model from a npz file
        >>> net.load_weights('./model.npz')

        2) load model from a npz file, which is saved as npz_dict previously
        >>> net.load_weights('./model.npz', format='npz_dict')

        Notes
        -------
        1) 'in_order' is only useful when 'format' is 'hdf5'. If you are trying to load a weights file which is
           saved in a different mode, it is recommended to set 'in_order' be True.
        2) 'skip' is useful when 'format' is 'hdf5' or 'npz_dict'. If 'skip' is True,
           'in_order' argument will be ignored.

        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("file {} doesn't exist.".format(file_path))

        if format is None:
            format = file_path.split('.')[-1]

        if format == 'hdf5' or format == 'h5':
            if skip ==True or in_order == False:
                # load by weights name
                utils.load_hdf5_to_weights(file_path, self, skip)
            else:
                # load in order
                utils.load_hdf5_to_weights_in_order(file_path, self)
        elif format == 'npz':
            utils.load_and_assign_npz(file_path, self)
        elif format == 'npz_dict':
            utils.load_and_assign_npz_dict(file_path, self, skip)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
                "Other format is not supported now."
            )

    def tf_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(X_batch)
                    # compute loss and update model
                    _loss_ce = loss_fn(_logits, y_batch)

                grad = tape.gradient(_loss_ce, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                train_loss += _loss_ce
                if metrics:
                    metrics.update(_logits, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)  # is_train=False, disable dropout
                        val_loss += loss_fn(_logits, y_batch, name='eval_loss')
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))

    def ms_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        net_with_criterion = WithLoss(network, loss_fn)
        train_network = GradWrap(net_with_criterion)
        train_network.set_train()
        for epoch in range(n_epoch):
            start_time = time.time()
            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                output = network(X_batch)
                loss_output = loss_fn(output, y_batch)
                grads = train_network(X_batch, y_batch)
                success = optimizer.apply_gradients(zip(grads, train_weights))
                loss = loss_output.asnumpy()
                train_loss += loss
                if metrics:
                    metrics.update(output, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean((P.Equal()(P.Argmax(axis=1)(output), y_batch).asnumpy()))
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)
                        val_loss += loss_fn(_logits, y_batch, name='eval_loss')
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean((P.Equal()(P.Argmax(axis=1)(_logits), y_batch).asnumpy()))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))

    def pd_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                output = network(X_batch)
                loss = loss_fn(output, y_batch)
                loss_ce = loss.numpy()
                params_grads = optimizer.gradient(loss, network.trainable_weights)
                optimizer.apply_gradients(params_grads)

                train_loss += loss_ce
                if metrics:
                    metrics.update(output, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += pd.metric.accuracy(output, y_batch)
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)  # is_train=False, disable dropout
                        val_loss += loss_fn(_logits, y_batch, name='eval_loss')
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))


class WithLoss(Module):

    def __init__(self, backbone, loss_fn):
        super(WithLoss, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        return self._backbone


class GradWrap(Module):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_weights)

    def forward(self, x, label):
        return composite.GradOperation(get_by_list=True)(self.network, self.weights)(x, label)
