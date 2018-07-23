#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from lxml import etree

import tensorflow as tf

from tensorlayer import logging

from tensorlayer.files.utils import del_file
from tensorlayer.files.utils import del_folder
from tensorlayer.files.utils import folder_exists
from tensorlayer.files.utils import load_file_list
from tensorlayer.files.utils import maybe_download_and_extract

from tensorlayer import utils

__all__ = ['load_voc_dataset']


def load_voc_dataset(path='data', dataset='2012', contain_classes_in_person=False):
    """Pascal VOC 2007/2012 Dataset.

    It has 20 objects:
    aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
    and additional 3 classes : head, hand, foot for person.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to, defaults is ``data/VOC``.
    dataset : str
        The VOC dataset version, `2012`, `2007`, `2007test` or `2012test`. We usually train model on `2007+2012` and test it on `2007test`.
    contain_classes_in_person : boolean
        Whether include head, hand and foot annotation, default is False.

    Returns
    ---------
    imgs_file_list : list of str
        Full paths of all images.
    imgs_semseg_file_list : list of str
        Full paths of all maps for semantic segmentation. Note that not all images have this map!
    imgs_insseg_file_list : list of str
        Full paths of all maps for instance segmentation. Note that not all images have this map!
    imgs_ann_file_list : list of str
        Full paths of all annotations for bounding box and object class, all images have this annotations.
    classes : list of str
        Classes in order.
    classes_in_person : list of str
        Classes in person.
    classes_dict : dictionary
        Class label to integer.
    n_objs_list : list of int
        Number of objects in all images in ``imgs_file_list`` in order.
    objs_info_list : list of str
        Darknet format for the annotation of all images in ``imgs_file_list`` in order. ``[class_id x_centre y_centre width height]`` in ratio format.
    objs_info_dicts : dictionary
        The annotation of all images in ``imgs_file_list``, ``{imgs_file_list : dictionary for annotation}``,
        format from `TensorFlow/Models/object-detection <https://github.com/tensorflow/models/blob/master/object_detection/create_pascal_tf_record.py>`__.

    Examples
    ----------
    >>> imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list,
    >>>     classes, classes_in_person, classes_dict,
    >>>     n_objs_list, objs_info_list, objs_info_dicts = tl.files.load_voc_dataset(dataset="2012", contain_classes_in_person=False)
    >>> idx = 26
    >>> print(classes)
    ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    >>> print(classes_dict)
    {'sheep': 16, 'horse': 12, 'bicycle': 1, 'bottle': 4, 'cow': 9, 'sofa': 17, 'car': 6, 'dog': 11, 'cat': 7, 'person': 14, 'train': 18, 'diningtable': 10, 'aeroplane': 0, 'bus': 5, 'pottedplant': 15, 'tvmonitor': 19, 'chair': 8, 'bird': 2, 'boat': 3, 'motorbike': 13}
    >>> print(imgs_file_list[idx])
    data/VOC/VOC2012/JPEGImages/2007_000423.jpg
    >>> print(n_objs_list[idx])
    2
    >>> print(imgs_ann_file_list[idx])
    data/VOC/VOC2012/Annotations/2007_000423.xml
    >>> print(objs_info_list[idx])
    14 0.173 0.461333333333 0.142 0.496
    14 0.828 0.542666666667 0.188 0.594666666667
    >>> ann = tl.prepro.parse_darknet_ann_str_to_list(objs_info_list[idx])
    >>> print(ann)
    [[14, 0.173, 0.461333333333, 0.142, 0.496], [14, 0.828, 0.542666666667, 0.188, 0.594666666667]]
    >>> c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
    >>> print(c, b)
    [14, 14] [[0.173, 0.461333333333, 0.142, 0.496], [0.828, 0.542666666667, 0.188, 0.594666666667]]

    References
    -------------
    - `Pascal VOC2012 Website <https://pjreddie.com/projects/pascal-voc-dataset-mirror/>`__.
    - `Pascal VOC2007 Website <https://pjreddie.com/projects/pascal-voc-dataset-mirror/>`__.

    """
    path = os.path.join(path, 'VOC')

    def _recursive_parse_xml_to_dict(xml):
        """Recursively parses XML contents to python dict.

        We assume that `object` tags are the only ones that can appear
        multiple times at the same level of a tree.

        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.

        """
        if xml is not None:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = _recursive_parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    import xml.etree.ElementTree as ET

    if dataset == "2012":
        url = "http://pjreddie.com/media/files/"
        tar_filename = "VOCtrainval_11-May-2012.tar"
        extracted_filename = "VOC2012"  #"VOCdevkit/VOC2012"
        logging.info("    [============= VOC 2012 =============]")
    elif dataset == "2012test":
        extracted_filename = "VOC2012test"  #"VOCdevkit/VOC2012"
        logging.info("    [============= VOC 2012 Test Set =============]")
        logging.info(
            "    \nAuthor: 2012test only have person annotation, so 2007test is highly recommended for testing !\n"
        )
        import time
        time.sleep(3)
        if os.path.isdir(os.path.join(path, extracted_filename)) is False:
            logging.info("For VOC 2012 Test data - online registration required")
            logging.info(
                " Please download VOC2012test.tar from:  \n register: http://host.robots.ox.ac.uk:8080 \n voc2012 : http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/ \ndownload: http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar"
            )
            logging.info(" unzip VOC2012test.tar,rename the folder to VOC2012test and put it into %s" % path)
            exit()
        # # http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar
        # url = "http://host.robots.ox.ac.uk:8080/eval/downloads/"
        # tar_filename = "VOC2012test.tar"
    elif dataset == "2007":
        url = "http://pjreddie.com/media/files/"
        tar_filename = "VOCtrainval_06-Nov-2007.tar"
        extracted_filename = "VOC2007"
        logging.info("    [============= VOC 2007 =============]")
    elif dataset == "2007test":
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#testdata
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        url = "http://pjreddie.com/media/files/"
        tar_filename = "VOCtest_06-Nov-2007.tar"
        extracted_filename = "VOC2007test"
        logging.info("    [============= VOC 2007 Test Set =============]")
    else:
        raise Exception("Please set the dataset aug to 2012, 2012test or 2007.")

    # download dataset
    if dataset != "2012test":
        from sys import platform as _platform
        if folder_exists(os.path.join(path, extracted_filename)) is False:
            logging.info("[VOC] {} is nonexistent in {}".format(extracted_filename, path))
            maybe_download_and_extract(tar_filename, path, url, extract=True)
            del_file(os.path.join(path, tar_filename))
            if dataset == "2012":
                if _platform == "win32":
                    os.system("move {}\VOCdevkit\VOC2012 {}\VOC2012".format(path, path))
                else:
                    os.system("mv {}/VOCdevkit/VOC2012 {}/VOC2012".format(path, path))
            elif dataset == "2007":
                if _platform == "win32":
                    os.system("move {}\VOCdevkit\VOC2007 {}\VOC2007".format(path, path))
                else:
                    os.system("mv {}/VOCdevkit/VOC2007 {}/VOC2007".format(path, path))
            elif dataset == "2007test":
                if _platform == "win32":
                    os.system("move {}\VOCdevkit\VOC2007 {}\VOC2007test".format(path, path))
                else:
                    os.system("mv {}/VOCdevkit/VOC2007 {}/VOC2007test".format(path, path))
            del_folder(os.path.join(path, 'VOCdevkit'))
    # object classes(labels)  NOTE: YOU CAN CUSTOMIZE THIS LIST
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    if contain_classes_in_person:
        classes_in_person = ["head", "hand", "foot"]
    else:
        classes_in_person = []

    classes += classes_in_person  # use extra 3 classes for person

    classes_dict = utils.list_string_to_dict(classes)
    logging.info("[VOC] object classes {}".format(classes_dict))

    # 1. image path list
    # folder_imgs = path+"/"+extracted_filename+"/JPEGImages/"
    folder_imgs = os.path.join(path, extracted_filename, "JPEGImages")
    imgs_file_list = load_file_list(path=folder_imgs, regx='\\.jpg', printable=False)
    logging.info("[VOC] {} images found".format(len(imgs_file_list)))

    imgs_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                       )  # 2007_000027.jpg --> 2007000027

    imgs_file_list = [os.path.join(folder_imgs, s) for s in imgs_file_list]
    # logging.info('IM',imgs_file_list[0::3333], imgs_file_list[-1])
    if dataset != "2012test":
        ##======== 2. semantic segmentation maps path list
        # folder_semseg = path+"/"+extracted_filename+"/SegmentationClass/"
        folder_semseg = os.path.join(path, extracted_filename, "SegmentationClass")
        imgs_semseg_file_list = load_file_list(path=folder_semseg, regx='\\.png', printable=False)
        logging.info("[VOC] {} maps for semantic segmentation found".format(len(imgs_semseg_file_list)))
        imgs_semseg_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                                  )  # 2007_000032.png --> 2007000032
        imgs_semseg_file_list = [os.path.join(folder_semseg, s) for s in imgs_semseg_file_list]
        # logging.info('Semantic Seg IM',imgs_semseg_file_list[0::333], imgs_semseg_file_list[-1])
        ##======== 3. instance segmentation maps path list
        # folder_insseg = path+"/"+extracted_filename+"/SegmentationObject/"
        folder_insseg = os.path.join(path, extracted_filename, "SegmentationObject")
        imgs_insseg_file_list = load_file_list(path=folder_insseg, regx='\\.png', printable=False)
        logging.info("[VOC] {} maps for instance segmentation found".format(len(imgs_semseg_file_list)))
        imgs_insseg_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                                  )  # 2007_000032.png --> 2007000032
        imgs_insseg_file_list = [os.path.join(folder_insseg, s) for s in imgs_insseg_file_list]
        # logging.info('Instance Seg IM',imgs_insseg_file_list[0::333], imgs_insseg_file_list[-1])
    else:
        imgs_semseg_file_list = []
        imgs_insseg_file_list = []
    # 4. annotations for bounding box and object class
    # folder_ann = path+"/"+extracted_filename+"/Annotations/"
    folder_ann = os.path.join(path, extracted_filename, "Annotations")
    imgs_ann_file_list = load_file_list(path=folder_ann, regx='\\.xml', printable=False)
    logging.info(
        "[VOC] {} XML annotation files for bounding box and object class found".format(len(imgs_ann_file_list))
    )
    imgs_ann_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                           )  # 2007_000027.xml --> 2007000027
    imgs_ann_file_list = [os.path.join(folder_ann, s) for s in imgs_ann_file_list]
    # logging.info('ANN',imgs_ann_file_list[0::3333], imgs_ann_file_list[-1])

    if dataset == "2012test":  # remove unused images in JPEG folder
        imgs_file_list_new = []
        for ann in imgs_ann_file_list:
            ann = os.path.split(ann)[-1].split('.')[0]
            for im in imgs_file_list:
                if ann in im:
                    imgs_file_list_new.append(im)
                    break
        imgs_file_list = imgs_file_list_new
        logging.info("[VOC] keep %d images" % len(imgs_file_list_new))

    # parse XML annotations
    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotation(file_name):
        """Given VOC2012 XML Annotations, returns number of objects and info."""
        in_file = open(file_name)
        out_file = ""
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        n_objs = 0

        for obj in root.iter('object'):
            if dataset != "2012test":
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
            else:
                cls = obj.find('name').text
                if cls not in classes:
                    continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )
            bb = convert((w, h), b)

            out_file += str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
            n_objs += 1
            if cls in "person":
                for part in obj.iter('part'):
                    cls = part.find('name').text
                    if cls not in classes_in_person:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = part.find('bndbox')
                    b = (
                        float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
                    )
                    bb = convert((w, h), b)
                    # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                    out_file += str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                    n_objs += 1
        in_file.close()
        return n_objs, out_file

    logging.info("[VOC] Parsing xml annotations files")
    n_objs_list = []
    objs_info_list = []  # Darknet Format list of string
    objs_info_dicts = {}
    for idx, ann_file in enumerate(imgs_ann_file_list):
        n_objs, objs_info = convert_annotation(ann_file)
        n_objs_list.append(n_objs)
        objs_info_list.append(objs_info)
        with tf.gfile.GFile(ann_file, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = _recursive_parse_xml_to_dict(xml)['annotation']
        objs_info_dicts.update({imgs_file_list[idx]: data})

    return imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list, classes, classes_in_person, classes_dict, n_objs_list, objs_info_list, objs_info_dicts
