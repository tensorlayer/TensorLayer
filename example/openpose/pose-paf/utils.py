from pycocotools.coco import COCO
import numpy as np
import os
from scipy.spatial.distance import cdist
from pycocotools.coco import maskUtils
from tensorlayer.files.utils import maybe_download_and_extract

## download dataset
def load_mscoco_dataset(path='data', dataset='2014'):  # TODO move to tl.files later
    """Download MSCOCO Dataset.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to, defaults is ``data/mscoco...``.
    dataset : str
        The MSCOCO dataset version, `2014` or `2017`.

    Returns
    ---------
    imgs_file_list : list of str
        Full paths of all images.

    Examples
    ----------
    >>>

    References
    -------------
    - `MSCOCO <http://mscoco.org>`__.

    """
    if dataset == "2014":
        logging.info("    [============= MSCOCO 2014 =============]")
        # wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
        url = 'http://images.cocodataset.org/annotations'
        tar_filename = 'annotations_trainval2014.zip'
        extracted_filename = ''
        # wget http://images.cocodataset.org/zips/train2014.zip
        url2 = 'http://images.cocodataset.org/zips'
        tar_filename2 = 'train2014.zip'
        extracted_filename2 = ''
        path = os.path.join(path, 'mscoco2014')
    elif dataset == "2017":
        raise Exception("Unimplement")
        path = os.path.join(path, 'mscoco2017')
    else:
        raise Exception("dataset can only be 2014 and 2017, see MSCOCO website for more details.")

    logging.info("    downloading annotations")
    maybe_download_and_extract(tar_filename, path, url, extract=True)
    del_file(os.path.join(path, tar_filename))

    logging.info("    downloading images")
    maybe_download_and_extract(tar_filename2, path, url2, extract=True)
    del_file(os.path.join(path, tar_filename2))

    return

## xxx
class CocoMeta:
    limb = list(
        zip(
            [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        )
    )

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

    def __init__(self, image_base_dir, anno_path, with_mask):
        self.metas = []
        # self.data_dir = data_dir
        # self.data_type = data_type
        self.image_base_dir = image_base_dir
        self.anno_path = anno_path
        self.with_mask = with_mask
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
            if self.with_mask:
                persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

                for p_id in list(persons_ids):
                    person_meta = anns[p_id]

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

## xxx
import math
import cv2


def get_heatmap(annos, height, width):
    """

    Parameters
    -----------


    Returns
    --------


    """

    # 19 for coco, 15 for MPII
    num_joints = 19

    # the heatmap for every joints takes the maximum over all people
    joints_heatmap = np.zeros((num_joints, height, width), dtype=np.float32)

    # among all people
    for joint in annos:
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum

        for i, points in enumerate(joint):
            if points[0] < 0 or points[1] < 0:
                continue
            joints_heatmap = put_heatmap(joints_heatmap, i, points, 8.0)

    # 0: joint index, 1:y, 2:x
    joints_heatmap = joints_heatmap.transpose((1, 2, 0))

    # background
    joints_heatmap[:, :, -1] = np.clip(1 - np.amax(joints_heatmap, axis=2), 0.0, 1.0)

    mapholder = []
    for i in range(0, 19):
        a = cv2.resize(np.array(joints_heatmap[:, :, i]), (46, 46))
        mapholder.append(a)
    mapholder = np.array(mapholder)
    joints_heatmap = mapholder.transpose(1, 2, 0)

    return joints_heatmap.astype(np.float16)


def put_heatmap(heatmap, plane_idx, center, sigma):
    """

    Parameters
    -----------


    Returns
    --------


    """
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap


def get_vectormap(annos, height, width):
    """

    Parameters
    -----------


    Returns
    --------


    """
    num_joints = 19

    limb = list(
        zip(
            [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        )
    )

    vectormap = np.zeros((num_joints * 2, height, width), dtype=np.float32)
    counter = np.zeros((num_joints, height, width), dtype=np.int16)

    for joint in annos:
        if len(joint) != 19:
            print('THE LENGTH IS NOT 19 ERROR:', len(joint))
        for i, (a, b) in enumerate(limb):
            a -= 1
            b -= 1

            v_start = joint[a]
            v_end = joint[b]
            # exclude invisible or unmarked point
            if v_start[0] < -100 or v_start[1] < -100 or v_end[0] < -100 or v_end[1] < -100:
                continue
            vectormap = cal_vectormap(vectormap, counter, i, v_start, v_end)

    vectormap = vectormap.transpose((1, 2, 0))
    # normalize the PAF (otherwise longer limb gives stronger absolute strength)
    nonzero_vector = np.nonzero(counter)

    for i, y, x in zip(nonzero_vector[0], nonzero_vector[1], nonzero_vector[2]):

        if counter[i][y][x] <= 0:
            continue
        vectormap[y][x][i * 2 + 0] /= counter[i][y][x]
        vectormap[y][x][i * 2 + 1] /= counter[i][y][x]

    mapholder = []
    for i in range(0, 38):
        a = cv2.resize(np.array(vectormap[:, :, i]), (46, 46), interpolation=cv2.INTER_AREA)
        mapholder.append(a)
    mapholder = np.array(mapholder)
    vectormap = mapholder.transpose(1, 2, 0)

    return vectormap.astype(np.float16)


def cal_vectormap(vectormap, countmap, i, v_start, v_end):
    """

    Parameters
    -----------


    Returns
    --------


    """
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - v_start[0]
            bec_y = y - v_start[1]
            dist = abs(bec_x * norm_y - bec_y * norm_x)

            # orthogonal distance is < then threshold
            if dist > threshold:
                continue
            countmap[i][y][x] += 1
            vectormap[i * 2 + 0][y][x] = norm_x
            vectormap[i * 2 + 1][y][x] = norm_y

    return vectormap


def fast_vectormap(vectormap, countmap, i, v_start, v_end):
    """

    Parameters
    -----------


    Returns
    --------


    """
    _, height, width = vectormap.shape[:3]
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]

    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    x_vec = (np.arange(min_x, max_x) - v_start[0]) * norm_y
    y_vec = (np.arange(min_y, max_y) - v_start[1]) * norm_x

    xv, yv = np.meshgrid(x_vec, y_vec)

    dist_matrix = abs(xv - yv)
    filter_matrix = np.where(dist_matrix > threshold, 0, 1)
    countmap[i, min_y:max_y, min_x:max_x] += filter_matrix
    for y in range(max_y - min_y):
        for x in range(max_x - min_x):
            if filter_matrix[y, x] != 0:
                vectormap[i * 2 + 0, min_y + y, min_x + x] = norm_x
                vectormap[i * 2 + 1, min_y + y, min_x + x] = norm_y
    return vectormap


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
