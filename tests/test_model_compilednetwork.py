import pprint

import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

model = tl.networks.Sequential(name="My_Sequential_1D_Network")

model.add(tl.layers.ExpandDimsLayer(axis=1, name="expand_layer_1"))
model.add(tl.layers.FlattenLayer(name="flatten_layer_1"))

model.add(tl.layers.ExpandDimsLayer(axis=2, name="expand_layer_2"))
model.add(tl.layers.TileLayer(multiples=[1, 1, 3], name="tile_layer_2"))
model.add(tl.layers.TransposeLayer(perm=[0, 2, 1], name='transpose_layer_2'))
model.add(tl.layers.FlattenLayer(name="flatten_layer_2"))

model.add(tl.layers.DenseLayer(n_units=10, act=tf.nn.relu, name="seq_layer_1"))

model.add(tl.layers.DenseLayer(n_units=40, act=tf.nn.relu, name="seq_layer_2"))
model.add(tl.layers.DropoutLayer(keep=0.5, is_fix=True, name="dropout_layer_2"))

model.add(tl.layers.DenseLayer(n_units=50, act=tf.nn.relu, name="seq_layer_3"))
model.add(tl.layers.DropoutLayer(keep=0.5, is_fix=False, name="dropout_layer_3"))

model.add(tl.layers.DenseLayer(n_units=50, act=None, name="seq_layer_4"))
model.add(tl.layers.PTRelu6Layer(channel_shared=False, name="ptrelu6_layer_4"))

plh = tf.placeholder(tf.float16, (100, 32))

train_model = model.build(plh, reuse=False, is_train=True)
test_model = model.build(plh, reuse=True, is_train=False)

print("\n#################### TEST Train Model ######################\n")

print(train_model)
#pprint.pprint(train_model.__dict__)
pprint.pprint(train_model.all_layers)
pprint.pprint(train_model.all_weights)

print("\n################### TEST Test Model #######################\n")

print(test_model)
#pprint.pprint(test_model.__dict__)
pprint.pprint(test_model.all_layers)
pprint.pprint(train_model.all_weights)

print("\n################### TEST Get Layer by Name #######################\n")

print(train_model["seq_layer_1"])
print(test_model["seq_layer_1"])

print("\n################### TEST Get Layer Output Shape #######################\n")

print(train_model["seq_layer_1"])
print(test_model["seq_layer_1"])
