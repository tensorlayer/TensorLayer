from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Dense, Conv2D
from tensorflow.python.keras import Model
from tensorflow.python.training import saver
import tensorflow as tf

# get the whole model
# vgg = VGG16(weights=None)
# print([x.name for x in vgg.weights])


class Nested_VGG(Model):

    def __init__(self):
        super(Nested_VGG, self).__init__()
        self.vgg1 = VGG16(weights=None)
        # print([x.name for x in self.vgg1.weights])
        self.vgg2 = VGG16(weights=None)
        self.dense = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')

    def call(self, inputs, training=None, mask=None):
        pass


class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.inner = Nested_VGG()

    def call(self, inputs, training=None, mask=None):
        pass


model = MyModel()
print([x.name for x in model.layers])
# print([x.name for x in model.inner.weights])
print('vgg1:')
print([x.name for x in model.inner.vgg1.weights])
print([x.name for x in model.inner.vgg1.layers])

print('vgg2')
print(model.inner.vgg2.get_layer('block1_conv1').kernel.name)
print([x.name for x in model.inner.vgg2.weights])
print([x.name for x in model.inner.vgg2.layers])
model.save_weights('./keras_model.h5')
