#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.app import computer_vision
from tensorlayer import visualize
from tensorlayer.app.computer_vision_object_detection.common import read_class_names
import numpy as np
import cv2
from PIL import Image
INPUT_SIZE = 416
image_path = './data/kite.jpg'

class_names = read_class_names('./model/coco.names')
original_image = cv2.imread(image_path)
image = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)
net = computer_vision.object_detection('yolo4-mscoco')
json_result = net(original_image)
image = visualize.draw_boxes_and_labels_to_image_with_json(image, json_result, class_names)
image = Image.fromarray(image.astype(np.uint8))
image.show()
