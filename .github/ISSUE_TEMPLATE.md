### New Issue Checklist

- [ ] I have read the [Contribution Guidelines](https://github.com/tensorlayer/tensorlayer/blob/master/CONTRIBUTING.md)
- [ ] I searched for [existing GitHub issues](https://github.com/tensorlayer/tensorlayer/issues)

### Issue Description

[INSERT DESCRIPTION OF THE PROBLEM]

### Reproducible Code

- Which OS are you using ?
- Please provide a reproducible code of your issue. Without any reproducible code, you will probably not receive any help.

[INSERT CODE HERE]

```python
# ======================================================== #
###### THIS CODE IS AN EXAMPLE, REPLACE WITH YOUR OWN ######
# ======================================================== #

import tensorflow as tf
import tensorlayer as tl

x = tf.placeholder(tf.float32, [None, 64])
net_in = tl.layers.InputLayer(x)

net = tl.layers.DenseLayer(net_in, n_units=25, act=tf.nn.relu, name='relu1')

print("Output Shape:", net.outputs.get_shape().as_list()) ### Output Shape: [None, 25]

# ======================================================== #
###### THIS CODE IS AN EXAMPLE, REPLACE WITH YOUR OWN ######
# ======================================================== #
```



