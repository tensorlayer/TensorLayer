#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayer import logging

from tensorlayer.files.utils import del_file
from tensorlayer.files.utils import folder_exists
from tensorlayer.files.utils import load_file_list
from tensorlayer.files.utils import maybe_download_and_extract

__all__ = ['load_mpii_pose_dataset']


def load_mpii_pose_dataset(path='data', is_16_pos_only=False):
    """Load MPII Human Pose Dataset.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to.
    is_16_pos_only : boolean
        If True, only return the peoples contain 16 pose keypoints. (Usually be used for single person pose estimation)

    Returns
    ----------
    img_train_list : list of str
        The image directories of training data.
    ann_train_list : list of dict
        The annotations of training data.
    img_test_list : list of str
        The image directories of testing data.
    ann_test_list : list of dict
        The annotations of testing data.

    Examples
    --------
    >>> import pprint
    >>> import tensorlayer as tl
    >>> img_train_list, ann_train_list, img_test_list, ann_test_list = tl.files.load_mpii_pose_dataset()
    >>> image = tl.vis.read_image(img_train_list[0])
    >>> tl.vis.draw_mpii_pose_to_image(image, ann_train_list[0], 'image.png')
    >>> pprint.pprint(ann_train_list[0])

    References
    -----------
    - `MPII Human Pose Dataset. CVPR 14 <http://human-pose.mpi-inf.mpg.de>`__
    - `MPII Human Pose Models. CVPR 16 <http://pose.mpi-inf.mpg.de>`__
    - `MPII Human Shape, Poselet Conditioned Pictorial Structures and etc <http://pose.mpi-inf.mpg.de/#related>`__
    - `MPII Keyponts and ID <http://human-pose.mpi-inf.mpg.de/#download>`__
    """
    path = os.path.join(path, 'mpii_human_pose')
    logging.info("Load or Download MPII Human Pose > {}".format(path))

    # annotation
    url = "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/"
    tar_filename = "mpii_human_pose_v1_u12_2.zip"
    extracted_filename = "mpii_human_pose_v1_u12_2"
    if folder_exists(os.path.join(path, extracted_filename)) is False:
        logging.info("[MPII] (annotation) {} is nonexistent in {}".format(extracted_filename, path))
        maybe_download_and_extract(tar_filename, path, url, extract=True)
        del_file(os.path.join(path, tar_filename))

    # images
    url = "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/"
    tar_filename = "mpii_human_pose_v1.tar.gz"
    extracted_filename2 = "images"
    if folder_exists(os.path.join(path, extracted_filename2)) is False:
        logging.info("[MPII] (images) {} is nonexistent in {}".format(extracted_filename, path))
        maybe_download_and_extract(tar_filename, path, url, extract=True)
        del_file(os.path.join(path, tar_filename))

    # parse annotation, format see http://human-pose.mpi-inf.mpg.de/#download
    import scipy.io as sio
    logging.info("reading annotations from mat file ...")
    # mat = sio.loadmat(os.path.join(path, extracted_filename, "mpii_human_pose_v1_u12_1.mat"))

    # def fix_wrong_joints(joint):    # https://github.com/mitmul/deeppose/blob/master/datasets/mpii_dataset.py
    #     if '12' in joint and '13' in joint and '2' in joint and '3' in joint:
    #         if ((joint['12'][0] < joint['13'][0]) and
    #                 (joint['3'][0] < joint['2'][0])):
    #             joint['2'], joint['3'] = joint['3'], joint['2']
    #         if ((joint['12'][0] > joint['13'][0]) and
    #                 (joint['3'][0] > joint['2'][0])):
    #             joint['2'], joint['3'] = joint['3'], joint['2']
    #     return joint

    ann_train_list = []
    ann_test_list = []
    img_train_list = []
    img_test_list = []

    def save_joints():
        # joint_data_fn = os.path.join(path, 'data.json')
        # fp = open(joint_data_fn, 'w')
        mat = sio.loadmat(os.path.join(path, extracted_filename, "mpii_human_pose_v1_u12_1.mat"))

        for _, (anno, train_flag) in enumerate(  # all images
                zip(mat['RELEASE']['annolist'][0, 0][0], mat['RELEASE']['img_train'][0, 0][0])):

            img_fn = anno['image']['name'][0, 0][0]
            train_flag = int(train_flag)

            # print(i, img_fn, train_flag) # DEBUG print all images

            if train_flag:
                img_train_list.append(img_fn)
                ann_train_list.append([])
            else:
                img_test_list.append(img_fn)
                ann_test_list.append([])

            head_rect = []
            if 'x1' in str(anno['annorect'].dtype):
                head_rect = zip(
                    [x1[0, 0] for x1 in anno['annorect']['x1'][0]], [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                    [x2[0, 0] for x2 in anno['annorect']['x2'][0]], [y2[0, 0] for y2 in anno['annorect']['y2'][0]]
                )
            else:
                head_rect = []  # TODO

            if 'annopoints' in str(anno['annorect'].dtype):
                annopoints = anno['annorect']['annopoints'][0]
                head_x1s = anno['annorect']['x1'][0]
                head_y1s = anno['annorect']['y1'][0]
                head_x2s = anno['annorect']['x2'][0]
                head_y2s = anno['annorect']['y2'][0]

                for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(annopoints, head_x1s, head_y1s, head_x2s,
                                                                         head_y2s):
                    # if annopoint != []:
                    # if len(annopoint) != 0:
                    if annopoint.size:
                        head_rect = [
                            float(head_x1[0, 0]),
                            float(head_y1[0, 0]),
                            float(head_x2[0, 0]),
                            float(head_y2[0, 0])
                        ]

                        # joint coordinates
                        annopoint = annopoint['point'][0, 0]
                        j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                        x = [x[0, 0] for x in annopoint['x'][0]]
                        y = [y[0, 0] for y in annopoint['y'][0]]
                        joint_pos = {}
                        for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                            joint_pos[int(_j_id)] = [float(_x), float(_y)]
                        # joint_pos = fix_wrong_joints(joint_pos)

                        # visibility list
                        if 'is_visible' in str(annopoint.dtype):
                            vis = [v[0] if v.size > 0 else [0] for v in annopoint['is_visible'][0]]
                            vis = dict([(k, int(v[0])) if len(v) > 0 else v for k, v in zip(j_id, vis)])
                        else:
                            vis = None

                        # if len(joint_pos) == 16:
                        if ((is_16_pos_only ==True) and (len(joint_pos) == 16)) or (is_16_pos_only == False):
                            # only use image with 16 key points / or use all
                            data = {
                                'filename': img_fn,
                                'train': train_flag,
                                'head_rect': head_rect,
                                'is_visible': vis,
                                'joint_pos': joint_pos
                            }
                            # print(json.dumps(data), file=fp)  # py3
                            if train_flag:
                                ann_train_list[-1].append(data)
                            else:
                                ann_test_list[-1].append(data)

    # def write_line(datum, fp):
    #     joints = sorted([[int(k), v] for k, v in datum['joint_pos'].items()])
    #     joints = np.array([j for i, j in joints]).flatten()
    #
    #     out = [datum['filename']]
    #     out.extend(joints)
    #     out = [str(o) for o in out]
    #     out = ','.join(out)
    #
    #     print(out, file=fp)

    # def split_train_test():
    #     # fp_test = open('data/mpii/test_joints.csv', 'w')
    #     fp_test = open(os.path.join(path, 'test_joints.csv'), 'w')
    #     # fp_train = open('data/mpii/train_joints.csv', 'w')
    #     fp_train = open(os.path.join(path, 'train_joints.csv'), 'w')
    #     # all_data = open('data/mpii/data.json').readlines()
    #     all_data = open(os.path.join(path, 'data.json')).readlines()
    #     N = len(all_data)
    #     N_test = int(N * 0.1)
    #     N_train = N - N_test
    #
    #     print('N:{}'.format(N))
    #     print('N_train:{}'.format(N_train))
    #     print('N_test:{}'.format(N_test))
    #
    #     np.random.seed(1701)
    #     perm = np.random.permutation(N)
    #     test_indices = perm[:N_test]
    #     train_indices = perm[N_test:]
    #
    #     print('train_indices:{}'.format(len(train_indices)))
    #     print('test_indices:{}'.format(len(test_indices)))
    #
    #     for i in train_indices:
    #         datum = json.loads(all_data[i].strip())
    #         write_line(datum, fp_train)
    #
    #     for i in test_indices:
    #         datum = json.loads(all_data[i].strip())
    #         write_line(datum, fp_test)

    save_joints()
    # split_train_test()  #

    ## read images dir
    logging.info("reading images list ...")
    img_dir = os.path.join(path, extracted_filename2)
    _img_list = load_file_list(path=os.path.join(path, extracted_filename2), regx='\\.jpg', printable=False)
    # ann_list = json.load(open(os.path.join(path, 'data.json')))
    for i, im in enumerate(img_train_list):
        if im not in _img_list:
            print('missing training image {} in {} (remove from img(ann)_train_list)'.format(im, img_dir))
            # img_train_list.remove(im)
            del img_train_list[i]
            del ann_train_list[i]
    for i, im in enumerate(img_test_list):
        if im not in _img_list:
            print('missing testing image {} in {} (remove from img(ann)_test_list)'.format(im, img_dir))
            # img_test_list.remove(im)
            del img_train_list[i]
            del ann_train_list[i]

    ## check annotation and images
    n_train_images = len(img_train_list)
    n_test_images = len(img_test_list)
    n_images = n_train_images + n_test_images
    logging.info("n_images: {} n_train_images: {} n_test_images: {}".format(n_images, n_train_images, n_test_images))
    n_train_ann = len(ann_train_list)
    n_test_ann = len(ann_test_list)
    n_ann = n_train_ann + n_test_ann
    logging.info("n_ann: {} n_train_ann: {} n_test_ann: {}".format(n_ann, n_train_ann, n_test_ann))
    n_train_people = len(sum(ann_train_list, []))
    n_test_people = len(sum(ann_test_list, []))
    n_people = n_train_people + n_test_people
    logging.info("n_people: {} n_train_people: {} n_test_people: {}".format(n_people, n_train_people, n_test_people))
    # add path to all image file name
    for i, value in enumerate(img_train_list):
        img_train_list[i] = os.path.join(img_dir, value)
    for i, value in enumerate(img_test_list):
        img_test_list[i] = os.path.join(img_dir, value)
    return img_train_list, ann_train_list, img_test_list, ann_test_list
