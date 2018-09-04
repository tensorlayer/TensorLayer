#! /usr/bin/python
# -*- coding: utf-8 -*-
import time

import tensorflow as tf

from tensorlayer import files
from tensorlayer import iterate
from tensorlayer import utils
from tensorlayer import visualize

from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.dense import DenseLayer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'ReconLayer',
]


class ReconLayer(DenseLayer):
    """A reconstruction layer for :class:`DenseLayer` to implement AutoEncoder.

    It is often used to pre-train the previous :class:`DenseLayer`

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    x_recon : placeholder or tensor
        The target for reconstruction.
    n_units : int
        The number of units of the layer. It should equal ``x_recon``.
    act : activation function
        The activation function of this layer.
        Normally, for sigmoid layer, the reconstruction activation is ``sigmoid``;
        for rectifying layer, the reconstruction activation is ``softplus``.
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=(None, 784))
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DenseLayer(net, n_units=196, act=tf.nn.sigmoid, name='dense')
    >>> recon = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon')
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
    >>> recon.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=500, batch_size=128, print_freq=1, save=True, save_name='w1pre_')

    Methods
    -------
    pretrain(sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre')
        Start to pre-train the parameters of the previous DenseLayer.

    Notes
    -----
    The input layer should be `DenseLayer` or a layer that has only one axes.
    You may need to modify this part to define your own cost function.
    By default, the cost is implemented as follow:
    - For sigmoid layer, the implementation can be `UFLDL <http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial>`__
    - For rectifying layer, the implementation can be `Glorot (2011). Deep Sparse Rectifier Neural Networks <http://doi.org/10.1.1.208.6449>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            x_recon=None,
            n_units=784,
            act=tf.nn.softplus,
            name='recon',
    ):
        super(ReconLayer, self).__init__(prev_layer=prev_layer, n_units=n_units, act=act, name=name)

        logging.info("ReconLayer %s" % self.name)

        # y : reconstruction outputs; train_params : parameters to train
        # Note that: train_params = [W_encoder, b_encoder, W_decoder, b_encoder]
        y = self.outputs
        self.train_params = self.all_params[-4:]

        # =====================================================================
        #
        # You need to modify the below cost function and optimizer so as to
        # implement your own pre-train method.
        #
        # =====================================================================
        lambda_l2_w = 0.004
        learning_rate = 0.0001
        logging.info("     lambda_l2_w: %f" % lambda_l2_w)
        logging.info("     learning_rate: %f" % learning_rate)

        # Mean-square-error i.e. quadratic-cost
        mse = tf.reduce_sum(tf.squared_difference(y, x_recon), 1)
        mse = tf.reduce_mean(mse)  # in theano: mse = ((y - x) ** 2 ).sum(axis=1).mean()
        # mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y, x_recon)),  1))
        # mse = tf.reduce_mean(tf.squared_difference(y, x_recon)) # <haodong>: Error
        # mse = tf.sqrt(tf.reduce_mean(tf.square(y - x_recon)))   # <haodong>: Error
        # Cross-entropy
        # ce = cost.cross_entropy(y, x_recon)                                               # <haodong>: list , list , Error (only be used for softmax output)
        # ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x_recon))          # <haodong>: list , list , Error (only be used for softmax output)
        # ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, x_recon))   # <haodong>: list , index , Error (only be used for softmax output)
        L2_w = tf.contrib.layers.l2_regularizer(lambda_l2_w)(
            self.train_params[0]
        ) + tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[2])  # faster than the code below
        # L2_w = lambda_l2_w * tf.reduce_mean(tf.square(self.train_params[0])) + lambda_l2_w * tf.reduce_mean( tf.square(self.train_params[2]))

        # DropNeuro
        # P_o = cost.lo_regularizer(0.03)(
        #     self.train_params[0])  # + cost.lo_regularizer(0.5)(self.train_params[2])    # <haodong>: if add lo on decoder, no neuron will be broken
        # P_i = cost.li_regularizer(0.03)(self.train_params[0])  # + cost.li_regularizer(0.001)(self.train_params[2])

        # L1 of activation outputs
        activation_out = self.all_layers[-2]
        L1_a = 0.001 * tf.reduce_mean(
            activation_out
        )  # <haodong>:  theano: T.mean( self.a[i] )         # some neuron are broken, white and black
        # L1_a = 0.001 * tf.reduce_mean( tf.reduce_sum(activation_out, 0) )         # <haodong>: some neuron are broken, white and black
        # L1_a = 0.001 * 100 * tf.reduce_mean( tf.reduce_sum(activation_out, 1) )   # <haodong>: some neuron are broken, white and black
        # KL Divergence
        beta = 4
        rho = 0.15
        p_hat = tf.reduce_mean(activation_out, 0)  # theano: p_hat = T.mean( self.a[i], axis=0 )

        KLD = beta * tf.reduce_sum(
            rho * tf.log(tf.divide(rho, p_hat)) + (1 - rho) * tf.log((1 - rho) / (tf.subtract(float(1), p_hat)))
        )

        # Total cost
        if act == tf.nn.softplus:
            logging.info('     use: mse, L2_w, L1_a')
            self.cost = mse + L1_a + L2_w
        elif act == tf.nn.sigmoid:
            # ----------------------------------------------------
            # Cross-entropy was used in Denoising AE
            # logging.info('     use: ce, L2_w, KLD')
            # self.cost = ce + L2_w + KLD
            # ----------------------------------------------------
            # Mean-squared-error was used in Vanilla AE
            logging.info('     use: mse, L2_w, KLD')
            self.cost = mse + L2_w + KLD
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_o) can remove neurons of AE
            # logging.info('     use: mse, L2_w, KLD, P_o')
            # self.cost = mse + L2_w + KLD + P_o
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_i) can remove neurons of previous layer
            #   If previous layer is InputLayer, it means remove useless features
            # logging.info('     use: mse, L2_w, KLD, P_i')
            # self.cost = mse + L2_w + KLD + P_i
        else:
            raise Exception("Don't support the given reconstruct activation function")

        self.train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                               use_locking=False).minimize(self.cost, var_list=self.train_params)
        # self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.cost, var_list=self.train_params)

    def pretrain(
            self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True,
            save_name='w1pre_'
    ):
        # ====================================================
        #
        # You need to modify the cost function in __init__() so as to
        # get your own pre-train method.
        #
        # ====================================================
        logging.info("     [*] %s start pretrain" % self.name)
        logging.info("     batch_size: %d" % batch_size)
        if denoise_name:
            logging.info("     denoising layer keep: %f" % self.all_drop[LayersConfig.set_keep[denoise_name]])
            dp_denoise = self.all_drop[LayersConfig.set_keep[denoise_name]]
        else:
            logging.info("     no denoising layer")

        for epoch in range(n_epoch):
            start_time = time.time()
            for X_train_a, _ in iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                dp_dict = utils.dict_to_one(self.all_drop)
                if denoise_name:
                    dp_dict[LayersConfig.set_keep[denoise_name]] = dp_denoise
                feed_dict = {x: X_train_a}
                feed_dict.update(dp_dict)
                sess.run(self.train_op, feed_dict=feed_dict)

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                logging.info("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, n_batch = 0, 0
                for X_train_a, _ in iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                    dp_dict = utils.dict_to_one(self.all_drop)
                    feed_dict = {x: X_train_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    train_loss += err
                    n_batch += 1
                logging.info("   train loss: %f" % (train_loss / n_batch))
                val_loss, n_batch = 0, 0
                for X_val_a, _ in iterate.minibatches(X_val, X_val, batch_size, shuffle=True):
                    dp_dict = utils.dict_to_one(self.all_drop)
                    feed_dict = {x: X_val_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    val_loss += err
                    n_batch += 1
                logging.info("   val loss: %f" % (val_loss / n_batch))
                if save:
                    try:
                        visualize.draw_weights(
                            self.train_params[0].eval(), second=10, saveable=True, shape=[28, 28],
                            name=save_name + str(epoch + 1), fig_idx=2012
                        )
                        files.save_npz([self.all_params[0]], name=save_name + str(epoch + 1) + '.npz')
                    except Exception:
                        raise Exception(
                            "You should change the visualize.W() in ReconLayer.pretrain(), if you want to save the feature images for different dataset"
                        )
