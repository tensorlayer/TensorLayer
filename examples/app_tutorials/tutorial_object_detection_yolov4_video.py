#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.app import computer_vision
from tensorlayer import visualize
from tensorlayer.app.computer_vision_object_detection.common import read_class_names
import cv2
INPUT_SIZE = 416
video_path = './data/road.mp4'

class_names = read_class_names('./model/coco.names')
vid = cv2.VideoCapture(video_path)
'''
vid = cv2.VideoCapture(0) # the serial number of camera on you device
'''

if not vid.isOpened():
    raise ValueError("Read Video Failed!")
net = computer_vision.object_detection('yolo4-mscoco')
frame_id = 0
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            print("Video processing complete")
            break
        raise ValueError("No image! Try with another video format")

    json_result = net(frame)
    image = visualize.draw_boxes_and_labels_to_image_with_json(frame, json_result, class_names)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    frame_id += 1
