from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
from pycocotools.coco import maskUtils


class CocoMeta:
    # limb = list(zip(
    #     [1, 8,  9,  1, 11, 12, 1, 2, 3,  2, 1, 5, 6,  5, 1,  0,  0, 14, 15],
    #     [8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17]
    # ))

    limb = list(
        zip(
            [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        )
    )

    #     limb = [
    #     (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    #     (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
    # ]
    def __init__(self, idx, img_url, img_meta, annotations, masks):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])
        self.masks = masks
        joint_list = []

        for anno in annotations:
            if anno.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(anno['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]
            # if joint is marked
            joint_list.append([(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        # 对原 COCO 数据集的转换 其中第二位之所以不一样是为了计算 Neck 等于左右 shoulder 的中点
        transform = list(
            zip(
                [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4],
                [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
            )
        )
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1 - 1]
                j2 = prev_joint[idx2 - 1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))

            # for background
            new_joint.append((-1000, -1000))
            if len(new_joint) != 19:
                print('The Length of joints list should be 0 or 19 but actually:', len(new_joint))
            self.joint_list.append(new_joint)


class PoseInfo:
    # metas = []
    # metas_test=[]
    def __init__(self, data_dir, data_type, anno_path):
        self.metas = []
        self.data_dir = data_dir
        self.data_type = data_type
        self.image_base_dir = '{}/images/{}2014/'.format(data_dir, data_type)
        self.anno_path = '{}/annotations/person_keypoints_{}2014.json'.format(data_dir, data_type)
        self.coco = COCO(self.anno_path)
        self.get_image_annos()
        self.image_list = os.listdir(self.image_base_dir)

    @staticmethod
    def get_keypoints(annos_info):
        annolist = []
        for anno in annos_info:
            adjust_anno = {'keypoints': anno['keypoints'], 'num_keypoints': anno['num_keypoints']}
            annolist.append(adjust_anno)
        return annolist

    def get_image_annos(self):

        images_ids = self.coco.getImgIds()
        len_imgs = len(images_ids)
        for idx in range(len_imgs):

            images_info = self.coco.loadImgs(images_ids[idx])
            image_path = self.image_base_dir + images_info[0]['file_name']
            # filter that some images might not in the list
            if not os.path.exists(image_path):
                continue

            annos_ids = self.coco.getAnnIds(imgIds=images_ids[idx])
            annos_info = self.coco.loadAnns(annos_ids)
            keypoints = self.get_keypoints(annos_info)

            #############################################################################
            anns = annos_info
            prev_center = []
            masks = []

            # sort from the biggest person to the smallest one
            persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

            for id in list(persons_ids):
                person_meta = anns[id]

                if person_meta["iscrowd"]:
                    masks.append(self.coco.annToRLE(person_meta))
                    continue

                # skip this person if parts number is too low or if
                # segmentation area is too small

                if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
                    masks.append(self.coco.annToRLE(person_meta))
                    continue

                person_center = [
                    person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                    person_meta["bbox"][1] + person_meta["bbox"][3] / 2
                ]

                # skip this person if the distance to existing person is too small

                too_close = False
                for pc in prev_center:
                    a = np.expand_dims(pc[:2], axis=0)
                    b = np.expand_dims(person_center, axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2] * 0.3:
                        too_close = True
                        break

                if too_close:
                    # add mask of this person. we don't want to show the network
                    # unlabeled people
                    masks.append(self.coco.annToRLE(person_meta))
                    continue

            ############################################################################
            total_keypoints = sum([ann.get('num_keypoints', 0) for ann in annos_info])
            if total_keypoints > 0:
                meta = CocoMeta(images_ids[idx], image_path, images_info[0], keypoints, masks)
                self.metas.append(meta)

        print("Overall get {}".format(len(self.metas)))

    def load_images(self):
        pass

    def get_image_list(self):
        img_list = []
        for meta in self.metas:
            img_list.append(meta.img_url)
        return img_list

    def get_joint_list(self):
        joint_list = []
        for meta in self.metas:
            joint_list.append(meta.joint_list)
        return joint_list

    def get_mask(self):
        mask_list = []
        for meta in self.metas:
            mask_list.append(meta.masks)
        return mask_list


def output_figure(index, df_val):

    path = df_val.metas[index].img_url
    print(path)
    # I = io.imread(path)
    # plt.imshow(I)

    # hm = df_val.get_maps(index)
    # hm = hm * 255
    # hm = hm.astype(np.int)
    # tmp = np.amax(hm, axis=2)
    # plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.4)

    vectmap = df_val.get_vectors(index)
    tmp2 = vectmap.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    tmp2_odd = tmp2_odd * 255
    tmp2_odd = tmp2_odd.astype(np.int)
    print(tmp2_odd)
    np.savetxt('test', tmp2_odd)
    plt.imshow(tmp2_odd, alpha=0.3)

    tmp2_even = tmp2_even * 255
    tmp2_even = tmp2_even.astype(np.int)
    plt.imshow(tmp2_even, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    data_dir = '/Users/Joel/Desktop/coco'
    data_type = 'val'
    anno_path = '{}/annotations/person_keypoints_{}2014.json'.format(data_dir, data_type)
    df_val = PoseInfo(data_dir, data_type, anno_path)

    for i in range(50):
        meta = df_val.metas[i]
        mask_sig = meta.masks
        print('shape of np mask is ', np.shape(mask_sig), type(mask_sig))
        if mask_sig is not []:
            mask_miss = np.ones((meta.height, meta.width), dtype=np.uint8)
            for seg in mask_sig:
                bin_mask = maskUtils.decode(seg)
                bin_mask = np.logical_not(bin_mask)
                mask_miss = np.bitwise_and(mask_miss, bin_mask)
