# OpenPose using TensorFlow and TensorLayer

## 1. Motivation

Ok, CMU provides OpenPose for real-time 2D pose estimation for ["Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"]() However, the training code is based on Caffe and C++, which is hard to customize.
While in practice, developers need to customize their training set, data augmentation methods according to their projects.
In addition, the speed of original model in the paper can be improved.
Therefore, we xxxxx

## 2. Project files

- `config.py` : to config the directories of dataset, training details and etc.
- `models.py` : to define the model structures, currently only VGG19 Based model included
- `utils.py` : to extract databased from cocodataset and groundtruth calculation
- `train.py` : to train the model
- `visualize.py`: draw the training result
- `inference` folder :

## 3. Preparation


1. for data processing, COCOAPIs are used, download cocoapi repo : https://github.com/cocodataset/cocoapi, go into Python folder and make.

```bash
git clone https://github.com/cocodataset/cocoapi
cd cocoapi-master/PythonAPI
make
```

2. Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess

```python
cd pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

** before recompiling **
rm -rf build
rm *.so
```

## 4. Use pre-trained model

In this project, input images are RGB with 0~1

Runs `xxx.py`, it will automatically download the default VGG19-based model from [here](https://github.com/tensorlayer/pretrained-models), 
and use it for inferencing.
The performance of pre-trained model is as follow:

|             	| Speed      	| AP      	| xxx |
|-------------	|---------------	|---------------	|---------------	|
| VGG19 	| xx	| xx	| xx 	| 
| Residual Squeeze  	| xx	| xx 	| xx 	| 

- Speed is tested on XXX

## 5. Train a model
For your own training, please put .jpg files into coco_dataset/images/ and put .json into coco_dataset/annotations/

Runs `train.py`, it will automatically download MSCOCO 2017 dataset into `dataset/coco17`. 
The default model in `models.py` is based on VGG19, which is the same with the original paper. 
If you want to customize the model, simply change it in `models.py`.
And then `train.py` will train the model to the end.

## 6. Evaluate a model

Runs `eval.py` for inference 

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

- [TensorLayer Issues 434](https://github.com/tensorlayer/tensorlayer/issues/434)
- [TensorLayer Issues 416](https://github.com/tensorlayer/tensorlayer/issues/416)



Paper's Model
--------------
Image : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/tree/master/model/_trained_MPI
MPII  : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_MPI/pose_deploy.prototxt
COCO  : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_COCO/pose_deploy.prototxt  <- same architecture but more key points
Visualize Caffe model : http://ethereon.github.io/netscope/#/editor
