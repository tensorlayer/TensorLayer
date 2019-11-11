# Spatial Transformer Networks

[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)  (STN) is a dynamic mechanism that produces transformations of input images (or feature maps)including  scaling, cropping, rotations, as well as non-rigid deformations. This enables the network to not only select regions of an image that are most relevant (attention), but also to transform those regions to simplify recognition in the following layers. 

Video for different transformation [click me](https://drive.google.com/file/d/0B1nQa_sA3W2iN3RQLXVFRkNXN0k/view).

In this repositary, we implemented a STN for [2D Affine Transformation](https://en.wikipedia.org/wiki/Affine_transformation) on MNIST dataset. We generated images with size of 40x40 from the original MNIST dataset, and distorted the images by random rotation, shifting, shearing and zoom in/out. The STN was able to learn to automatically apply transformations on distorted images via classification task.


<div align="center">
    <img src="https://github.com/zsdonghao/Spatial-Transformer-Nets/blob/master/images/transform.jpeg" width="50%" height="50%"/>
    <br>  
    <em align="center">Fig 1：Transformation</em>  
</div>


<div align="center">
    <img src="https://github.com/zsdonghao/Spatial-Transformer-Nets/blob/master/images/network.jpeg" width="50%" height="50%"/>
    <br>  
    <em align="center">Fig 2：Network</em>  
</div>

<div align="center">
    <img src="https://github.com/zsdonghao/Spatial-Transformer-Nets/blob/master/images/formula.jpeg" width="50%" height="50%"/>
    <br>  
    <em align="center">Fig 3：Formula</em>  
</div>

## Result

After classification task, the STN is able to transform the distorted image from Fig 4 back to Fig 5.

<div align="center">
    <img src="https://github.com/zsdonghao/Spatial-Transformer-Nets/blob/master/images/before_stn.png" width="50%" height="50%"/>
    <br>  
    <em align="center">Fig 4: Input</em>  
</div>

<div align="center">
    <img src="https://github.com/zsdonghao/Spatial-Transformer-Nets/blob/master/images/after_stn.png" width="50%" height="50%"/>
    <br>  
    <em align="center">Fig 5: Output</em>  
</div>

