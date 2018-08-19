from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
from pycocotools.coco import maskUtils

from utils import CocoMeta, PoseInfo



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
