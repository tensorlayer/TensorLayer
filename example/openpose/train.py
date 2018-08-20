import os
import cv2
import time
# import argparse
import numpy as np
import _pickle as cPickle
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import keypoint_random_rotate, keypoint_random_resize_shortestedge, keypoint_random_resize, keypoint_random_flip, keypoint_random_crop
from models import model
from config import config
from utils import PoseInfo, get_heatmap, get_vectormap, load_mscoco_dataset, draw_intermedia_results
from pycocotools.coco import maskUtils

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

tl.files.exists_or_mkdir(config.LOG.vis_path, verbose=False)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## define hyper-parameters for training
batch_size = config.TRAIN.batch_size
n_epoch = config.TRAIN.n_epoch
step_size = config.TRAIN.step_size
save_interval = config.TRAIN.save_interval
weight_decay = config.TRAIN.weight_decay
base_lr = config.TRAIN.base_lr
gamma = config.TRAIN.gamma

## define hyper-parameters for model
model_path = config.MODEL.model_path
n_pos = config.MODEL.n_pos
hin = config.MODEL.hin
win = config.MODEL.win
hout = config.MODEL.hout
wout = config.MODEL.wout

# parser = argparse.ArgumentParser(description='Training code for OpenPose using Tensorflow')
# parser.add_argument('--save_interval', type=int, default=5000)
# parser.add_argument(
#     '--model_path', type=str, default='models/vgg19.npy', help='Path to your pretrained vgg19.npy file'
# )
# parser.add_argument('--log_interval', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=8)
# parser.add_argument('--save_path', type=str, default='logging/model/')
# parser.add_argument('--vis_path', type=str, default='logging/val/')
# args = parser.parse_args()
# '''
# file structure:
# data_dir:
#     image_folder : xxxx.jpg
#     'annotations': xxxx.json
# '''


def _data_aug_fn(image, ground_truth):
    """ Data augmentation function """
    ground_truth = cPickle.loads(ground_truth)
    ground_truth = list(ground_truth)

    annos = ground_truth[0]
    mask = ground_truth[1]
    h_mask, w_mask, _ = np.shape(image)
    # mask
    mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)

    for seg in mask:
        bin_mask = maskUtils.decode(seg)
        bin_mask = np.logical_not(bin_mask)
        mask_miss = np.bitwise_and(mask_miss, bin_mask)

    # image data augmentation
    image, annos, mask_miss = keypoint_random_resize(image, annos, mask_miss, zoom_range=(0.8, 1.2))
    image, annos, mask_miss = keypoint_random_rotate(image, annos, mask_miss, rg=15.0)
    image, annos, mask_miss = keypoint_random_flip(image, annos, mask_miss, prob=0.5)
    image, annos, mask_miss = keypoint_random_resize_shortestedge(image, annos, mask_miss,
                                                                  min_size=(hin, win))  # TODO: give size
    image, annos, mask_miss = keypoint_random_crop(image, annos, mask_miss, size=(hin, win))  # TODO: give size

    # generate result maps including keypoints heatmap, pafs and mask
    h, w, _ = np.shape(image)
    height, width, _ = np.shape(image)
    heatmap = get_heatmap(annos, height, width)
    vectormap = get_vectormap(annos, height, width)
    resultmap = np.concatenate((heatmap, vectormap), axis=2)

    image = np.array(image, dtype=np.float32)

    img_mask = mask_miss.reshape(hin, win, 1)
    image = image * np.repeat(img_mask, 3, 2)

    resultmap = np.array(resultmap, dtype=np.float32)
    mask_miss = cv2.resize(mask_miss, (hout, wout), interpolation=cv2.INTER_AREA)
    mask_miss = np.array(mask_miss, dtype=np.float32)
    return image, resultmap, mask_miss


def _map_fn(img_list, annos):
    """ TF Dataset pipeline. """
    image = tf.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image, resultmap, mask = tf.py_func(_data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32])
    return image, resultmap, mask


if __name__ == '__main__':

    ## download MSCOCO data to "data/mscoco..."" folder
    train_im_path, train_ann_path, val_im_path, val_ann_path, _, _ = \
        load_mscoco_dataset(config.DATA.data_path, config.DATA.coco_version)

    ## read coco training images contains valid people
    train_data = PoseInfo(train_im_path, train_ann_path, False)
    train_imgs_file_list = train_data.get_image_list()
    train_objs_info_list = train_data.get_joint_list()
    train_mask_list = train_data.get_mask()
    # train_targets = list(zip(train_objs_info_list, train_mask_list))
    if len(train_imgs_file_list) != len(train_objs_info_list):
        raise Exception("number of training images and annotations do not match")
    else:
        print("number of training images {}".format(len(train_imgs_file_list)))

    ## read coco validating images contains valid people
    val_data = PoseInfo(val_im_path, val_ann_path, False)
    val_imgs_file_list = val_data.get_image_list()
    val_objs_info_list = val_data.get_joint_list()
    val_mask_list = val_data.get_mask()
    # val_targets = list(zip(val_objs_info_list, val_mask_list))
    if len(val_imgs_file_list) != len(val_objs_info_list):
        raise Exception("number of validating images and annotations do not match")
    else:
        print("number of validating images {}".format(len(val_imgs_file_list)))

    ## read your customized images contains valid people
    your_images_path = config.DATA.your_images_path
    your_annos_path = config.DATA.your_annos_path
    your_data = PoseInfo(your_images_path, your_annos_path, False)
    your_imgs_file_list = your_data.get_image_list()
    your_objs_info_list = your_data.get_joint_list()
    your_mask_list = your_data.get_mask()
    if len(your_imgs_file_list) != len(your_objs_info_list):
        raise Exception("number of customized images and annotations do not match")
    else:
        print("number of customized images {}".format(len(your_imgs_file_list)))

    ## choice dataset for training
    # 1. only coco training set
    # imgs_file_list = train_imgs_file_list
    # train_targets = list(zip(train_objs_info_list, train_mask_list))
    # 2. your customized data from "data/your_data" and coco training set
    imgs_file_list = train_imgs_file_list + your_imgs_file_list
    train_targets = list(zip(train_objs_info_list + your_objs_info_list, \
                    train_mask_list + your_mask_list))

    ## define model architecture
    x = tf.placeholder(tf.float32, [None, hin, win, 3], "image")
    confs = tf.placeholder(tf.float32, [None, hout, wout, n_pos], "confidence_maps")
    pafs = tf.placeholder(tf.float32, [None, hout, wout, n_pos * 2], "pafs")
    # if the people does not have keypoints annotations, ignore the area
    img_mask1 = tf.placeholder(tf.float32, [None, hout, wout, n_pos], 'img_mask1')
    img_mask2 = tf.placeholder(tf.float32, [None, hout, wout, n_pos * 2], 'img_mask2')
    num_images = np.shape(imgs_file_list)[0]

    cnn, b1_list, b2_list, net = model(x, n_pos, img_mask1, img_mask2, False, False)

    ## define data augmentation
    def generator():
        """ TF Dataset generartor """
        assert len(imgs_file_list) == len(train_targets)
        for _input, _target in zip(imgs_file_list, train_targets):
            yield _input.encode('utf-8'), cPickle.dumps(_target)

    dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.string))
    dataset = dataset.map(_map_fn, num_parallel_calls=1)
    dataset = dataset.repeat(n_epoch)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)  # larger prefetach buffer means xxxx
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    ## define loss
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    stage_losses = []
    L2 = 0.0
    for idx, (l1, l2) in enumerate(zip(b1_list, b2_list)):
        loss_l1 = tf.nn.l2_loss((tf.concat(l1.outputs, axis=0) - tf.concat(confs, axis=0)) * img_mask1)
        loss_l2 = tf.nn.l2_loss((tf.concat(l2.outputs, axis=0) - tf.concat(pafs, axis=0)) * img_mask2)
        losses.append(tf.reduce_mean([loss_l1, loss_l2]))
        stage_losses.append(loss_l1 / batch_size)
        stage_losses.append(loss_l2 / batch_size)
    last_losses_l1.append(loss_l1)
    last_losses_l2.append(loss_l2)
    last_conf = b1_list[-1].outputs
    last_paf = b2_list[-1].outputs

    for p in tl.layers.get_variables_with_name('kernel', True, True):
        L2 += tf.contrib.layers.l2_regularizer(0.0005)(p)
    total_loss = tf.reduce_sum(losses) / batch_size + L2

    global_step = tf.Variable(1, trainable=False)
    print('Config:', 'n_epoch: ', n_epoch, 'batch_size: ', batch_size, 'base_lr: ', base_lr, 'step_size: ', step_size)
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(base_lr, trainable=False)

    opt = tf.train.MomentumOptimizer(lr_v, 0.9)
    train_op = opt.minimize(total_loss, global_step=global_step)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    ## start training
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # restore pretrained vgg19  TODO: use tl.models.VGG19
        # npy_file = np.load('models', encoding='latin1').item()
        # params = []
        # for val in sorted(npy_file.items()):
        #     if val[0] == 'conv4_3':
        #         break
        #     W = np.asarray(val[1][0])
        #     b = np.asarray(val[1][1])
        #     print("Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        #     params.extend([W, b])
        # tl.files.assign_params(sess, params, cnn)
        # print("Restoring model from npy file")
        # cnn.restore_params(sess)

        # train until the end
        sess.run(tf.assign(lr_v, base_lr))
        while (True):
            tic = time.time()
            gs_num = sess.run(global_step)
            if gs_num != 0 and (gs_num % step_size == 0):
                new_lr_decay = gamma**(gs_num // step_size)
                sess.run(tf.assign(lr_v, base_lr * new_lr_decay))

            # get a batch of training data. TODO change to direct feed without using placeholder
            tran_batch = sess.run(one_element)
            # image
            x_ = tran_batch[0]
            # conf and paf maps
            map_batch = tran_batch[1]
            confs_ = map_batch[:, :, :, 0:19]  # TODO change to n_pos
            pafs_ = map_batch[:, :, :, 19:57]  # TODO change to n_pos:
            # mask
            mask = tran_batch[2]
            mask = mask.reshape(batch_size, 46, 46, 1)
            mask1 = np.repeat(mask, 19, 3)
            mask2 = np.repeat(mask, 38, 3)

            # TODO save some image examples for checking data augmentation
            # os.path.join(config.LOG.vis_path, 'data_aug_{}.png'.format(i))
            # tl.file.save_image()

            [_, the_loss, loss_ll, L2_reg, conf_result, weight_norm, paf_result] = sess.run(
                [train_op, total_loss, stage_losses, L2, last_conf, L2, last_paf], feed_dict={
                    x: x_,
                    confs: confs_,
                    pafs: pafs_,
                    img_mask1: mask1,
                    img_mask2: mask2
                }
            )

            tstring = time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
            lr = sess.run(lr_v)
            print(
                'Total Loss at iteration {} is: {} Learning rate {:10e} weight_norm {:10e} Time: {}'.format(
                    gs_num, the_loss, lr, weight_norm, tstring
                )
            )
            for ix, ll in enumerate(loss_ll):
                print('Network#', ix, 'For Branch', ix % 2 + 1, 'Loss:', ll)

            # save some intermedian results
            if (gs_num != 0) and (gs_num % 1 == 0):  #save_interval == 0):
                draw_intermedia_results(x_, confs_, conf_result, pafs_, paf_result, mask, 'train')
                # np.save(config.LOG.vis_path + 'image' + str(gs_num) + '.npy', x_)
                # np.save(config.LOG.vis_path + 'heat_ground' + str(gs_num) + '.npy', confs_)
                # np.save(config.LOG.vis_path + 'heat_result' + str(gs_num) + '.npy', conf_result)
                # np.save(config.LOG.vis_path + 'paf_ground' + str(gs_num) + '.npy', pafs_)
                # np.save(config.LOG.vis_path + 'mask' + str(gs_num) + '.npy', mask)
                # np.save(config.LOG.vis_path + 'paf_result' + str(gs_num) + '.npy', paf_result)
                tl.files.save_npz_dict(
                    net.all_params, os.path.join(model_path, 'pose' + str(gs_num) + '.npz'), sess=sess
                )
                tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)
            if gs_num > 3000001:
                break
