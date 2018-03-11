import tensorflow as tf
import tensorlayer as tl

## RNN encoder ====================================================
batch_size = 32
num_steps = 5
vocab_size = 30
hidden_size = 20
keep_prob = 0.8
is_train = True
input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
net = tl.layers.EmbeddingInputlayer(inputs=input_data, vocabulary_size=vocab_size, embedding_size=hidden_size, name='embed')
net = tl.layers.DropoutLayer(net, keep=keep_prob, is_fix=True, is_train=is_train, name='drop1')
net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=hidden_size, n_steps=num_steps, return_last=False, name='lstm1')
lstm1 = net
net = tl.layers.DropoutLayer(net, keep=keep_prob, is_fix=True, is_train=is_train, name='drop2')
net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=hidden_size, n_steps=num_steps, return_last=True, name='lstm2')
lstm2 = net
net = tl.layers.DropoutLayer(net, keep=keep_prob, is_fix=True, is_train=is_train, name='drop3')
net = tl.layers.DenseLayer(net, n_units=vocab_size, name='output')

net.print_layers()
net.print_params(False)

if len(net.all_layers) != 7:
    raise Exception("layers dont match")

if len(net.all_params) != 7:
    raise Exception("params dont match")

if net.count_params() != 7790:
    raise Exception("params dont match")

## CNN+RNN encoder ====================================================
image_size = 100
batch_size = 10
num_steps = 5
x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
net = tl.layers.InputLayer(x, name='in')
net = tl.layers.Conv2d(net, 32, (5, 5), (2, 2), tf.nn.relu, name='cnn1')
net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), name='pool1')
net = tl.layers.Conv2d(net, 10, (5, 5), (2, 2), tf.nn.relu, name='cnn2')
net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), name='pool2')
net = tl.layers.FlattenLayer(net, name='flatten')
net = tl.layers.ReshapeLayer(net, shape=[-1, num_steps, int(net.outputs._shape[-1])])
rnn = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=200, n_steps=num_steps, return_last=False, return_seq_2d=True, name='rnn')
net = tl.layers.DenseLayer(rnn, 3, name='out')

net.print_layers()
net.print_params(False)

if len(net.all_layers) != 8:
    raise Exception("layers dont match")

if len(net.all_params) != 8:
    raise Exception("params dont match")

if net.count_params() != 562245:
    raise Exception("params dont match")

## Bidirectional Synced input and output
batch_size = 10
num_steps = 5
vocab_size = 30
hidden_size = 20
input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
net = tl.layers.EmbeddingInputlayer(inputs=input_data, vocabulary_size=vocab_size, embedding_size=hidden_size, name='emb')
net = tl.layers.BiRNNLayer(
    net, cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=hidden_size, n_steps=num_steps, return_last=False, return_seq_2d=False, name='birnn')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[1:3] != [num_steps, hidden_size * 2]:
    raise Exception("shape dont match")

if len(net.all_layers) != 2:
    raise Exception("layers dont match")

if len(net.all_params) != 5:
    raise Exception("params dont match")

if net.count_params() != 7160:
    raise Exception("params dont match")

# n_layer=2
net = tl.layers.EmbeddingInputlayer(inputs=input_data, vocabulary_size=vocab_size, embedding_size=hidden_size, name='emb2')
net = tl.layers.BiRNNLayer(
    net, cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=hidden_size, n_steps=num_steps, n_layer=2, return_last=False, return_seq_2d=False, name='birnn2')

# net.print_layers()
# net.print_params(False)
#
# shape = net.outputs.get_shape().as_list()
# if shape[1:3] != [num_steps, hidden_size * 2]:
#     raise Exception("shape dont match")
#
# if len(net.all_layers) != 2:
#     raise Exception("layers dont match")
#
# if len(net.all_params) != 5:
#     raise Exception("params dont match")
#
# if net.count_params() != 7160:
#     raise Exception("params dont match")
#
# exit()

## ConvLSTMLayer TODO
# image_size = 100
# batch_size = 10
# num_steps = 5
# x = tf.placeholder(tf.float32, shape=[batch_size, num_steps, image_size, image_size, 3])
# net = tl.layers.InputLayer(x, name='in2')
# net = tl.layers.ConvLSTMLayer(net,
#             feature_map=1,
#             filter_size=(3, 3),
#             cell_fn=tl.layers.BasicConvLSTMCell,
#             initializer=tf.random_uniform_initializer(-0.1, 0.1),
#             n_steps=num_steps,
#             initial_state=None,
#             return_last=False,
#             return_seq_2d=False,
#             name='convlstm')

## Dynamic Synced input and output
batch_size = 32
num_steps = 5
vocab_size = 30
embedding_size = 20
keep_prob = 0.8
is_train = True
input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input")
nin = tl.layers.EmbeddingInputlayer(inputs=input_seqs, vocabulary_size=vocab_size, embedding_size=embedding_size, name='seq_embedding')
rnn = tl.layers.DynamicRNNLayer(
    nin,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    n_hidden=embedding_size,
    dropout=(keep_prob if is_train else None),
    sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
    return_last=False,
    return_seq_2d=True,
    name='dynamicrnn')
net = tl.layers.DenseLayer(rnn, n_units=vocab_size, name="o")

net.print_layers()
net.print_params(False)

shape = rnn.outputs.get_shape().as_list()
if shape[-1] != embedding_size:
    raise Exception("shape dont match")

shape = net.outputs.get_shape().as_list()
if shape[-1] != vocab_size:
    raise Exception("shape dont match")

if len(net.all_layers) != 3:
    raise Exception("layers dont match")

if len(net.all_params) != 5:
    raise Exception("params dont match")

if net.count_params() != 4510:
    raise Exception("params dont match")

# n_layer=3
nin = tl.layers.EmbeddingInputlayer(inputs=input_seqs, vocabulary_size=vocab_size, embedding_size=embedding_size, name='seq_embedding2')
rnn = tl.layers.DynamicRNNLayer(
    nin,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    n_hidden=embedding_size,
    dropout=(keep_prob if is_train else None),
    sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
    n_layer=3,
    return_last=False,
    return_seq_2d=True,
    name='dynamicrnn2')
net = tl.layers.DenseLayer(rnn, n_units=vocab_size, name="o2")

## BiDynamic Synced input and output
rnn = tl.layers.BiDynamicRNNLayer(
    nin,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    n_hidden=embedding_size,
    dropout=(keep_prob if is_train else None),
    sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
    return_last=False,
    return_seq_2d=True,
    name='bidynamicrnn')
net = tl.layers.DenseLayer(rnn, n_units=vocab_size, name="o3")

net.print_layers()
net.print_params(False)

shape = rnn.outputs.get_shape().as_list()
if shape[-1] != embedding_size * 2:
    raise Exception("shape dont match")

shape = net.outputs.get_shape().as_list()
if shape[-1] != vocab_size:
    raise Exception("shape dont match")

if len(net.all_layers) != 3:
    raise Exception("layers dont match")

if len(net.all_params) != 7:
    raise Exception("params dont match")

if net.count_params() != 8390:
    raise Exception("params dont match")

# n_layer=2
rnn = tl.layers.BiDynamicRNNLayer(
    nin,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    n_hidden=embedding_size,
    dropout=(keep_prob if is_train else None),
    sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
    return_last=False,
    return_seq_2d=True,
    name='bidynamicrnn2')
net = tl.layers.DenseLayer(rnn, n_units=vocab_size, name="o4")

## Seq2Seq
from tensorlayer.layers import EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2, DenseLayer
batch_size = 32
encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask")  # tl.prepro.sequences_get_mask()
with tf.variable_scope("model"):
    # for chatbot, you can use the same embedding layer,
    # for translation, you may want to use 2 seperated embedding layers
    with tf.variable_scope("embedding") as vs:
        net_encode = EmbeddingInputlayer(inputs=encode_seqs, vocabulary_size=10000, embedding_size=200, name='seq_embed')
        vs.reuse_variables()
        # tl.layers.set_name_reuse(True)
        net_decode = EmbeddingInputlayer(inputs=decode_seqs, vocabulary_size=10000, embedding_size=200, name='seq_embed')
    net = Seq2Seq(
        net_encode,
        net_decode,
        cell_fn=tf.contrib.rnn.BasicLSTMCell,
        n_hidden=200,
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        encode_sequence_length=retrieve_seq_length_op2(encode_seqs),
        decode_sequence_length=retrieve_seq_length_op2(decode_seqs),
        initial_state_encode=None,
        dropout=None,
        n_layer=2,
        return_seq_2d=True,
        name='Seq2seq')
net = DenseLayer(net, n_units=10000, act=tf.identity, name='oo')
e_loss = tl.cost.cross_entropy_seq_with_mask(logits=net.outputs, target_seqs=target_seqs, input_mask=target_mask, return_details=False, name='cost')
y = tf.nn.softmax(net.outputs)

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 10000:
    raise Exception("shape dont match")

if len(net.all_layers) != 5:
    raise Exception("layers dont match")

if len(net.all_params) != 11:
    raise Exception("params dont match")

if net.count_params() != 5293200:
    raise Exception("params dont match")
