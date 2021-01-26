#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.app import YOLOv4
import numpy as np
import tensorflow as tf


class object_detection(object):

    def __init__(self, model_name='yolo4-mscoco'):
        self.model_name = model_name
        if self.model_name == 'yolo4-mscoco':
            self.model = YOLOv4(NUM_CLASS=80, pretrained=True)
        else:
            raise ("The model does not support.")

    def __call__(self, input_data):
        if self.model_name == 'yolo4-mscoco':
            image_data = input_data / 255.
            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)
            batch_data = tf.constant(images_data)
            output = self.model(batch_data, is_train=False)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        s = ('{classname}(model_name={model_name}, model_structure={model}')
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)
