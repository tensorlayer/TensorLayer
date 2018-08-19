import sys
import os
import numpy as np
import logging
import argparse
import json
from tqdm import tqdm
import cv2
from estimator import TfPoseEstimator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_size = 10


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)


def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def round_int(val):
    return int(round(val))


def write_coco_json(human, image_w, image_h):
    keypoints = []
    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for coco_id in coco_ids:
        if coco_id not in human.body_parts.keys():
            keypoints.extend([0, 0, 0])
            continue
        body_part = human.body_parts[coco_id]
        keypoints.extend([round_int(body_part.x * image_w), round_int(body_part.y * image_h), 2])
    return keypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument(
        '--resize', type=str, default='432x368', help=
        'if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 '
    )
    parser.add_argument(
        '--resize-out-ratio', type=float, default=8.0,
        help='if provided, resize heatmaps before they are post-processed. default=8.0'
    )
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--cocoyear', type=str, default='2014')
    parser.add_argument('--coco-dir', type=str, default='/Users/Joel/Desktop/coco/')
    parser.add_argument('--data-idx', type=int, default=-5)
    parser.add_argument('--multi-scale', type=bool, default=True)
    args = parser.parse_args()

    val_path = '/Users/Joel/Desktop/1k_list.txt'
    f = open(val_path)
    line = f.readline()
    keys1k = []
    while line:
        line = line.rstrip('\n')
        info = line.split()
        keys1k.append(int(info[1]))
        line = f.readline()

    cocoyear_list = ['2014', '2017']
    if args.cocoyear not in cocoyear_list:
        logger.error('cocoyear should be one of %s' % str(cocoyear_list))
        sys.exit(-1)

    image_dir = args.coco_dir + 'images/val2014'
    coco_json_file = args.coco_dir + 'annotations/person_keypoints_val%s.json' % args.cocoyear
    cocoGt = COCO(coco_json_file)
    catIds = cocoGt.getCatIds(catNms=['person'])
    keys = cocoGt.getImgIds(catIds=catIds)
    keys = keys1k
    if args.data_idx < 0:
        if eval_size > 0:
            keys = keys[:eval_size]  # only use the first #eval_size elements.

        pass
    else:
        keys = [keys[args.data_idx]]
    logger.info('validation %s set size=%d' % (coco_json_file, len(keys)))
    write_json = 'etcs/%s_%s_%f.json' % (args.model, args.resize, args.resize_out_ratio)

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)

    result = []

    e = TfPoseEstimator('/Users/Joel/Desktop/Log_0808/small_inf_model110000.npz', target_size=(w, h))

    for i, k in enumerate(tqdm(keys)):

        img_meta = cocoGt.loadImgs(k)[0]
        img_idx = img_meta['id']

        img_name = os.path.join(image_dir, img_meta['file_name'])
        image = read_imgfile(img_name, None, None)
        print(img_name)
        if image is None:
            logger.error('image not found, path=%s' % img_name)
            # sys.exit(-1)
            continue
        # inference the image with the specified network
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        scores = 0
        ann_idx = cocoGt.getAnnIds(imgIds=[img_idx], catIds=[1])
        anns = cocoGt.loadAnns(ann_idx)
        for human in humans:
            item = {
                'image_id': img_idx,
                'category_id': 1,
                'keypoints': write_coco_json(human, img_meta['width'], img_meta['height']),
                'score': human.score
            }
            result.append(item)
            scores += item['score']

        avg_score = scores / len(humans) if len(humans) > 0 else 0
        # if args.data_idx >= 0:

        if True:
            # logger.info('score:', k, len(humans), len(anns), avg_score)

            import matplotlib.pyplot as plt
            fig = plt.figure()
            a = fig.add_subplot(2, 3, 1)
            plt.imshow(e.draw_humans(image, humans, True))

            a = fig.add_subplot(2, 3, 2)
            # plt.imshow(cv2.resize(image, (e.heatMat.shape[1], e.heatMat.shape[0])), alpha=0.5)
            tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            tmp2 = e.pafMat.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            a = fig.add_subplot(2, 3, 4)
            a.set_title('Vectormap-x')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()

            a = fig.add_subplot(2, 3, 5)
            a.set_title('Vectormap-y')
            # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
            plt.colorbar()
            plt.savefig('eval_img3/' + 'smamm_inf_110k' + '_' + str(i) + ".png", dpi=300)
            plt.cla
            plt.show()

    fp = open(write_json, 'w')
    json.dump(result, fp)
    fp.close()

    cocoDt = cocoGt.loadRes(write_json)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = keys
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print(''.join(["%11.4f |" % x for x in cocoEval.stats]))

    pred = json.load(open(write_json, 'r'))
