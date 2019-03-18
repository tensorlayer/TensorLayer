

import sys
sys.path.append("/Users/wurundi/PycharmProjects/tensorlayer2")
import numpy as np
import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import Input, Conv2d, Flatten, Dense, Dropout, MaxPool2d, Concat, UnStack
from tensorlayer.layers import RNN, Embedding
from tensorlayer.models import Model
from tensorflow.python.ops.rnn_cell import LSTMCell

tl.logging.set_verbosity(tl.logging.DEBUG)
#
#
# class CustomModel(Model):
#
#     def __init__(self):
#         super(CustomModel, self).__init__()
#
#         self.dropout1 = Dropout(keep=0.8)#(self.innet)
#         self.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784)#(self.dropout1)
#         self.dropout2 = Dropout(keep=0.8)#(self.dense1)
#         self.dense2 = Dense(n_units=800, act=tf.nn.relu, in_channels=800)#(self.dropout2)
#         self.dropout3 = Dropout(keep=0.8)#(self.dense2)
#         self.dense3 = Dense(n_units=10, act=tf.nn.relu, in_channels=800)#(self.dropout3)
#         self.dense4 = Dense(n_units=10, in_channels=800)#(self.dropout3)
#
#     def forward(self, x, foo=0):
#         z = self.dropout1(x)
#         z = self.dense1(z)
#         z = self.dropout2(z)
#         z = self.dense2(z)
#         z = self.dropout3(z)
#         if foo == 0:
#             out = self.dense3(z)
#         else:
#             out = self.dense4(z)
#             out = tf.nn.relu(out)
#         return out
#
#
# net = CustomModel()
#
# print(net.all_layers)
# print([x.name for x in net.weights])

def get_model(input_shape):
    ni = Input(input_shape)

    nii = Conv2d(32, filter_size=(3,3), strides=(1, 1))(ni)
    nn = Dropout(keep=0.9)(nii)

    conv = Conv2d(32, filter_size=(3,3), strides=(1, 1))
    tt = conv(nn)   # conv2d_1_node_0
    nn = conv(nn)   # conv2d_1_node_1

    # a branch
    na = Conv2d(64, filter_size=(3,3), strides=(1, 1))(nn)
    na = MaxPool2d()(na)

    # b branch
    nb = MaxPool2d()(nn)
    nb = conv(nb)   # conv2d_1_node_2

    out = Concat()([na, nb])
    M = Model(inputs=ni, outputs=[out, nn, nb], name='model')

    gg = conv(nii)  # this node will not be added since model fixed

    return M


def get_unstack_model(input_shape):
    ni = Input(input_shape)

    nn = Dropout(keep=0.9)(ni)

    a, b, c = UnStack(axis=-1)(nn)

    b = Flatten()(b)
    b = Dense(10)(b)

    M = Model(inputs=ni, outputs=[a, b, c], name='model')
    return M


def get_rnn(input_shape):
    net_in = Input(input_shape, dtype=tf.int32)
    net = Embedding(1000, 10, name='embedding')(net_in)
    lstm = RNN(
        cell_fn=LSTMCell,
        cell_init_args={
            'forget_bias': 0.0,
            'state_is_tuple': True
        },
        n_hidden=10, n_steps=None,
        return_last=False, return_seq_2d=True, name='lstm1'
    )(net)
    net_out = Dense(1000, act=None, name='output')(lstm)
    rnn_model = tl.models.Model(
        inputs=net_in,
        outputs=[net_out, lstm]
    )
    return rnn_model

def get_word2vec():
    vocabulary_size = 800
    batch_size = 10
    embedding_size = 60
    num_sampled = 25
    inputs = tl.layers.Input([batch_size], dtype=tf.int32)
    labels = tl.layers.Input([batch_size, 1], dtype=tf.int32)

    emb_net = tl.layers.Word2vecEmbedding(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        num_sampled=num_sampled,
        activate_nce_loss=True,  # nce loss is activated
        nce_loss_args={},
        E_init=tl.initializers.random_uniform(minval=-1.0, maxval=1.0),
        nce_W_init=tl.initializers.truncated_normal(stddev=float(1.0 / np.sqrt(embedding_size))),
        nce_b_init=tl.initializers.constant(value=0.0),
        name='word2vec_layer',
    )
    emb, nce = emb_net([inputs, labels])

    model = tl.models.Model(inputs=[inputs, labels], outputs=[emb, nce], name="word2vec_model")
    return model


# net = get_model([None, 24, 24, 3])
# net = get_unstack_model([None, 24, 24, 3])
# net = get_rnn([None, None])
net = get_word2vec()
net.train()

for k, v in enumerate(net._node_by_depth):
    print(k, [x.name for x in v], [x.in_tensors_idxes for x in v])

print("*" * 20)

for layer in net._all_layers:
    print(layer.name)

print("*" * 20)

# x = tf.random.normal(shape=(2, 24, 24, 3))
# x = tf.ones(shape=(10,1), dtype=tf.int32)
# outputs = net(x)
x = tf.ones(shape=(10,), dtype=tf.int32)
y = tf.ones(shape=(10,1), dtype=tf.int32)
outputs = net([x, y])

if isinstance(outputs, list):
    print([out.shape for out in outputs])
else:
    print(outputs.shape)

print("*" * 20)

print([w.name for w in net.weights])

print("*" * 20)