# Hint from TensorLayer
- This implementation is from `https://github.com/deepsense-ai/roi-pooling`, date: 31 Aug 2017.
- To install this, you have to clone TensorLayer from Github instead of pip install.
- Remember to modify the `CUDA_LIB` in Makefile before running `python setup.py install` in this folder.
- Make sure `roi_pooling_example.py` and `test_roi_layer.py` is runable.


----

 
## RoI pooling in TensorFlow

This repo contains the implementation of **Region of Interest pooling** as a custom TensorFlow operation. The CUDA code responsible for the computations was largely taken from the original [Caffe implementation by Ross Girshick](https://github.com/rbgirshick/fast-rcnn).

For more information about RoI pooling you can check out [Region of interest pooling explained](https://deepsense.io/region-of-interest-pooling-explained/) at our [deepsense.io](https://deepsense.io/) blog.

![Region of Interest Pooling animation](roi_pooling_animation.gif)


## Requirements

To compile and use `roi_pooling` layer you need to have:

* [CUDA](https://developer.nvidia.com/cuda-toolkit) (tested with 8.0)
* [https://www.tensorflow.org/](TensorFlow) (tested with 0.12.0 and 1.0.0)

Only official TensorFlow releases are currently supported. If you're using a custom built TensorFlow compiled with a different GCC version (e.g. 5.X) you may need to modify the makefile to [enable the new ABI version](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).


## Install

Since it uses compilation

```bash
$ git clone git@github.com:deepsense-io/roi-pooling.git
$ cd roi-pooling
$ python setup.py install
```

Right now we provide only GPU implementation (no CPU at this time).


## Usage

After successful installation you can use the operation like this:

```python
from roi_pooling.roi_pooling_ops import roi_pooling

# here obtain feature map and regions of interest
rpooling = roi_pooling(feature_map, rois, 7, 7)
# continue the model
```

Working example in Jupyter Notebook: [examples/roi_pooling_minimal_example.ipynb](https://github.com/deepsense-io/roi-pooling/blob/master/examples/roi_pooling_minimal_example.ipynb)

