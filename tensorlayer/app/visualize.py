#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.app import get_anchors, decode, filter_boxes, draw_bbox
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


def yolo4_visualize(original_image, feature_maps):
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
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = draw_bbox(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('result.png', image)
