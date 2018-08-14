"""
Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

Discussion
-----------
Issue 434 : https://github.com/tensorlayer/tensorlayer/issues/434
Issue 416 : https://github.com/tensorlayer/tensorlayer/issues/416

Paper's Model
--------------
Image : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/tree/master/model/_trained_MPI
MPII  : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_MPI/pose_deploy.prototxt
COCO  : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_COCO/pose_deploy.prototxt  <- same architecture but more key points
Visualize Caffe model : http://ethereon.github.io/netscope/#/editor

"""

import numpy as np
import tensorflow as tf
import argparse
import cv2
import tensorlayer as tl
from vgg_model import model
import time
import _pickle as cPickle
from data_process import PoseInfo
from tf_data_aug import _resize_image,pose_rotation,random_flip,pose_resize_shortestedge_random,pose_crop_random,pose_random_scale
from faster_map_cal import get_vectormap, get_heatmap
from pycocotools.coco import maskUtils
import os


parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
parser.add_argument('--save_interval', type=int, default=5000)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_dir = '/home/hao/Workspace/yuding/coco_dataset'
data_type = 'train'
anno_path = '{}/annotations/person_keypoints_{}2014.json'.format(data_dir, data_type)
df_val = PoseInfo(data_dir, data_type, anno_path)

imgs_file_list= df_val.get_image_list()
objs_info_list=df_val.get_joint_list()
mask_list= df_val.get_mask()
targets=list(zip (objs_info_list,mask_list))

def generator():
    inputs = imgs_file_list
    targets = list(zip (objs_info_list,mask_list))
    assert len(inputs) == len(targets)
    for _input, _target in zip(inputs, targets):
        yield _input.encode('utf-8'), cPickle.dumps(_target)
def _data_aug_fn(image, ground_truth):
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

    # image process

    image, annos, mask_miss = pose_random_scale(image, annos, mask_miss)
    image, annos, mask_miss = pose_rotation(image, annos, mask_miss)
    image, annos, mask_miss = random_flip(image, annos, mask_miss)
    image, annos, mask_miss = pose_resize_shortestedge_random(image, annos, mask_miss)
    image, annos, mask_miss = pose_crop_random(image, annos, mask_miss)

    h,w,_=np.shape(image)
    if h != 368 or w != 368:
        image, annos, mask_miss = _resize_image(image, annos, mask_miss, 368, 368)

    height, width, _ = np.shape(image)
    heatmap = get_heatmap(annos, height, width)
    vectormap = get_vectormap(annos, height, width)
    resultmap = np.concatenate((heatmap, vectormap), axis=2)

    image = image
    image = np.array(image, dtype=np.float32)

    img_mask = mask_miss.reshape(368, 368, 1)
    image = image * np.repeat(img_mask, 3, 2)

    resultmap = np.array(resultmap, dtype=np.float32)
    mask_miss = cv2.resize(mask_miss, (46, 46), interpolation=cv2.INTER_AREA)
    mask_miss = np.array(mask_miss, dtype=np.float32)
    return image, resultmap, mask_miss

def _map_fn(img_list, annos):
    image = tf.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image, resultmap, mask = tf.py_func(_data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32])

    return image, resultmap, mask

# MPII 16  COCO 19
n_pos = 19
n_epoch = 80
batch_size = 10
# learning_rate = 0.00001  # reduce it if your GPU memory is small

x = tf.placeholder(tf.float32, [None, 368, 368, 3], "image")  # for tl.models.MobileNetV1, value [0, 1]; for VGG19 value [0, 255]
confs = tf.placeholder(tf.float32, [None, 46, 46, n_pos], "confidence_maps")
pafs = tf.placeholder(tf.float32, [None, 46, 46, n_pos * 2], "pafs")  # x2 for x and y axises
img_mask1 =tf.placeholder(tf.float32,[None,46,46,19],'img_mask1')
img_mask2 =tf.placeholder(tf.float32,[None,46,46,38],'img_mask2')
init = tf.global_variables_initializer()

num_images=np.shape(imgs_file_list)[0]
# ## define model
cnn, b1_list, b2_list, net = model(x, n_pos,img_mask1,img_mask2, False, False)

dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.string))
dataset = dataset.map(_map_fn,num_parallel_calls=4)
#顺序可能会有尾巴
dataset = dataset.repeat(n_epoch)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(buffer_size=8)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

# define loss
losses = []
last_losses_l1 = []
last_losses_l2 = []
stage_losses=[]
L2=0.0
for idx, (l1, l2) in enumerate(zip(b1_list, b2_list)):
    loss_l1 = tf.nn.l2_loss( (tf.concat(l1.outputs, axis=0) - tf.concat(confs, axis=0)) *img_mask1)
    loss_l2 = tf.nn.l2_loss((tf.concat(l2.outputs, axis=0) - tf.concat(pafs, axis=0))*img_mask2)

    losses.append(tf.reduce_mean([loss_l1, loss_l2]))
    stage_losses.append(loss_l1/batch_size)
    stage_losses.append(loss_l2/batch_size)

last_losses_l1.append(loss_l1)
last_losses_l2.append(loss_l2)
last_conf=b1_list[-1].outputs
last_paf =b2_list[-1].outputs
for p in tl.layers.get_variables_with_name('kernel', True, True):
    L2 += tf.contrib.layers.l2_regularizer(0.0005)(p)
total_loss = tf.reduce_sum(losses) /batch_size + L2

global_step = tf.Variable(1, trainable=False)
stepsize = 136106
weight_decay = 5e-4
base_lr = 4e-5
gamma=0.333

print('Config:','n_epoch: ',n_epoch,'batch_size: ',batch_size,'base_lr: ',base_lr,'stepsize: ',stepsize)
# train_params = tl.layers.get_variables_with_name('cpm', True, True)  # dont update pretraied cnn part
with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(base_lr, trainable=False)

opt = tf.train.MomentumOptimizer(lr_v,0.9)
train_op=opt.minimize(total_loss,global_step=global_step)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    MODEL_PATH = '/home/hao/Workspace/yuding/pretrain_vgg/vgg19.npy'
    npy_file = np.load(MODEL_PATH, encoding='latin1').item()
    params = []
    for val in sorted(npy_file.items()):
        if val[0] == 'conv4_3':
            break
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, cnn)
    print("Restoring model from npy file")

    # graph_path='/home/hao/Workspace/yuding/pose-paf-master/model/inf_model156000.npz'
    # tl.files.load_and_assign_npz_dict(graph_path, sess)
    sess.run(tf.assign(lr_v, base_lr))
    while(True):
        tic=time.time()
        gs_num = sess.run(global_step)
        if gs_num != 0 and (gs_num % stepsize == 0):
            new_lr_decay = gamma ** (gs_num // stepsize)
            sess.run(tf.assign(lr_v, base_lr * new_lr_decay))
        tran_batch = sess.run(one_element)
        #image
        x_ = tran_batch[0]
        #conf and paf maps
        map_batch = tran_batch[1]
        confs_ = map_batch[:, :, :, 0:19]
        pafs_ = map_batch[:, :, :, 19:57]
        #mask
        mask = tran_batch[2]
        mask = mask.reshape(batch_size,46,46,1)
        mask1 =np.repeat(mask,19, 3)
        mask2 = np.repeat(mask, 38, 3)

        [_, the_loss, loss_ll,L2_reg,conf_result,weight_norm,paf_result] = sess.run([train_op, total_loss, stage_losses,L2,last_conf,L2, last_paf],
                                          feed_dict={x: x_, confs: confs_, pafs: pafs_,img_mask1:mask1,img_mask2:mask2})
        lr=sess.run(lr_v)
        tstring=time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
        print('Total Loss at iteration {} is: {} Learning rate {:10e} weight_norm {:10e} Time: {}'.format(gs_num,the_loss,lr,weight_norm,tstring))
        print('Time difference', time.time() - tic)

        for ix, ll in enumerate(loss_ll):
            print('Network#',ix,'For Branch',ix%2+1,'Loss:',ll)
        if gs_num!=0 and gs_num % args.save_interval == 0:
            ticks = time.time()
            print('Saved time:',ticks)
            np.save('/home/hao/Workspace/yuding/pose-paf-master/val/image' + str(gs_num) + '.npy', x_)
            np.save('/home/hao/Workspace/yuding/pose-paf-master/val/heat_ground' + str(gs_num) + '.npy', confs_)
            np.save('/home/hao/Workspace/yuding/pose-paf-master/val/heat_result' + str(gs_num) + '.npy', conf_result)
            np.save('/home/hao/Workspace/yuding/pose-paf-master/val/paf_ground' + str(gs_num) + '.npy', pafs_)
            np.save('/home/hao/Workspace/yuding/pose-paf-master/val/mask' + str(gs_num) + '.npy', mask)
            np.save('/home/hao/Workspace/yuding/pose-paf-master/val/paf_result' + str(gs_num) + '.npy', paf_result)
            tl.files.save_npz_dict(net.all_params, '/home/hao/Workspace/yuding/pose-paf-master/model/inf_model' + str(gs_num) + '.npz', sess=sess)
        if gs_num > 3000001:
            break
