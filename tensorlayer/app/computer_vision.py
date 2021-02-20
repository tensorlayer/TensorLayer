#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.app import YOLOv4
from tensorlayer.app import CGCNN
from tensorlayer import logging
from tensorlayer.app import yolo4_input_processing, yolo4_output_processing, result_to_json


class object_detection(object):
    """Model encapsulation.

    Parameters
    ----------
    model_name : str
        Choose the model to inference.

    Methods
    ---------
    __init__()
        Initializing the model.
    __call__()
        (1)Formatted input and output. (2)Inference model.
    list()
        Abstract method. Return available a list of model_name.

    Examples
    ---------
    Object Detection detection MSCOCO with YOLOv4, see `tutorial_object_detection_yolov4.py
    <https://github.com/tensorlayer/tensorlayer/blob/master/example/app_tutorials/tutorial_object_detection_yolov4.py>`__
    With TensorLayer

    >>> # get the whole model
    >>> net = tl.app.computer_vision.object_detection('yolo4-mscoco')
    >>> # use for inferencing
    >>> output = net(img)
    """

    def __init__(self, model_name='yolo4-mscoco'):
        self.model_name = model_name
        if self.model_name == 'yolo4-mscoco':
            self.model = YOLOv4(NUM_CLASS=80, pretrained=True)
        else:
            raise ("The model does not support.")

    def __call__(self, input_data):
        if self.model_name == 'yolo4-mscoco':
            batch_data = yolo4_input_processing(input_data)
            feature_maps = self.model(batch_data, is_train=False)
            pred_bbox = yolo4_output_processing(feature_maps)
            output = result_to_json(input_data, pred_bbox)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        s = ('(model_name={model_name}, model_structure={model}')
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    @property
    def list(self):
        logging.info("The model name list: 'yolov4-mscoco', 'lcn'")


class human_pose_estimation(object):
    """Model encapsulation.

    Parameters
    ----------
    model_name : str
        Choose the model to inference.

    Methods
    ---------
    __init__()
        Initializing the model.
    __call__()
        (1)Formatted input and output. (2)Inference model.
    list()
        Abstract method. Return available a list of model_name.

    Examples
    ---------
    LCN to estimate 3D human poses from 2D poses, see `tutorial_human_3dpose_estimation_LCN.py
    <https://github.com/tensorlayer/tensorlayer/blob/master/example/app_tutorials/tutorial_human_3dpose_estimation_LCN.py>`__
    With TensorLayer

    >>> # get the whole model
    >>> net = tl.app.computer_vision.human_pose_estimation('3D-pose')
    >>> # use for inferencing
    >>> output = net(img)
    """

    def __init__(self, model_name='3D-pose'):
        self.model_name = model_name
        if self.model_name == '3D-pose':
            self.model = CGCNN(pretrained=True)
        else:
            raise ("The model does not support.")

    def __call__(self, input_data):
        if self.model_name == '3D-pose':
            output = self.model(input_data, is_train=False)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        s = ('(model_name={model_name}, model_structure={model}')
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    @property
    def list(self):
        logging.info("The model name list: '3D-pose'")
