# OpenPose using TensorFlow and TensorLayer

## 1. Motivation

Ok, CMU provides OpenPose for real-time 2D pose estimation. However, the training code is based on Caffe and C++, which is hard to customize.
While in practice, developers need to customize their training set, data augmentation methods according to their projects.
In addition, the speed of original model in the paper can be improved.
Therefore, we xxxxx

## 2. Project files

<<<<<<< HEAD
- `models.py` is to define the model structures, e.g. VGG19, Residual Squeezenet
- `data_process.py` is to 
- `train.py` is to train the model
- ....

## 3. Preparation

To have a fast inferecning, the pose-processing for inferencing is based on OpenPose's C++ implmentation, so before you run the inferencing code, 
you should compile ...
=======
- `vgg_models.py` : to define the model structures, currently only VGG19 Based model included
- `utils.py` : to extract databased from cocodataset and groundtruth calculation
- `train.py` : to train the model
- `visual.py`: draw the training result

## 3. Preparation


1. for data processing, COCOAPIs are used, refer to : https://github.com/cocodataset/cocoapi
2. Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess

$ cd tf_pose/pafprocess

$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
>>>>>>> a3206e6b00901b3a5d1e1954d45cee090c1a0cd8

## 4. Use pre-trained model

Runs `xxx.py`, it will automatically download the default VGG19-based model from [here](https://github.com/tensorlayer/pretrained-models), 
and use it for inferencing.
The performance of pre-trained model is as follow:

|             	| Speed      	| AP      	| xxx |
|-------------	|---------------	|---------------	|---------------	|
| VGG19 	| xx	| xx	| xx 	| 
| Residual Squeeze  	| xx	| xx 	| xx 	| 

- Speed is tested on XXX

## 5. Train a model
<<<<<<< HEAD
=======
For your own training, please put .jpg files into coco_dataset/images/ and put .json into coco_dataset/annotations/
>>>>>>> a3206e6b00901b3a5d1e1954d45cee090c1a0cd8

Runs `train.py`, it will automatically download MSCOCO 2017 dataset into `dataset/coco17`. 
The default model in `models.py` is based on VGG19, which is the same with the original paper. 
If you want to customize the model, simply change it in `models.py`.
And then `train.py` will train the model to the end.

## 6. Evaluate a model

<<<<<<< HEAD
Runs `xxx.py`, the API `xxxxx` is ....
=======
Runs `eval.py` for inference 
>>>>>>> a3206e6b00901b3a5d1e1954d45cee090c1a0cd8

## 7. Speed up and deployment

For TensorRT float16 (half-float) inferencing, xxx

## 8. Customization
- Model : change `models.py`.
- Data augmentation : ....
- Train with your own data: ....  
    - 1) prepare your data following MSCOCO format, you need to ...
    - 2) concatenate the list of your own data JSON into ...
- Evaluate on your own testing set:
    - 1) xx

## 9. Discussion

<<<<<<< HEAD
- [TensorLayer Issues 434](https://github.com/tensorlayer/tensorlayer/issues/434)
=======
- [TensorLayer Issues 434](https://github.com/tensorlayer/tensorlayer/issues/434)
>>>>>>> a3206e6b00901b3a5d1e1954d45cee090c1a0cd8
