#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.app import YOLOv4, get_anchors, decode, filter_boxes
import numpy as np
import tensorflow as tf
from tensorlayer import logging
import cv2


class object_detection(object):

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
            output = yolo4_output_processing(feature_maps)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        s = ('(model_name={model_name}, model_structure={model}')
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    @property
    def list(self):
        logging.info("The model name list: yolov4-mscoco")


def yolo4_input_processing(original_image):
    image_data = cv2.resize(original_image, (416, 416))
    image_data = image_data / 255.
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    batch_data = tf.constant(images_data)
    return batch_data


def yolo4_output_processing(feature_maps):
    STRIDES = [8, 16, 32]
    ANCHORS = get_anchors([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401])
    NUM_CLASS = 80
    XYSCALE = [1.2, 1.1, 1.05]
    iou_threshold = 0.45
    score_threshold = 0.25

    bbox_tensors = []
    prob_tensors = []
    score_thres = 0.2
    for i, fm in enumerate(feature_maps):
        if i == 0:
            output_tensors = decode(fm, 416 // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        elif i == 1:
            output_tensors = decode(fm, 416 // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        else:
            output_tensors = decode(fm, 416 // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    boxes, pred_conf = filter_boxes(
        pred_bbox, pred_prob, score_threshold=score_thres, input_shape=tf.constant([416, 416])
    )
    pred = {'concat': tf.concat([boxes, pred_conf], axis=-1)}

    for key, value in pred.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50, max_total_size=50, iou_threshold=iou_threshold, score_threshold=score_threshold
    )
    output = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return output
