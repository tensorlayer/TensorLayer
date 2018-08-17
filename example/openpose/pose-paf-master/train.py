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
import tensorlayer as tl
import argparse
import cv2
from vgg_model import model
from tensorlayer.prepro import keypoints_rotation,keypoint_resize_shortestedge_random,keypoint_random_scale,keypoint_random_flip,keypoint_crop_random
import time
import _pickle as cPickle
from utils import PoseInfo, get_heatmap, get_vectormap
from pycocotools.coco import maskUtils
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
parser.add_argument('--save_interval', type=int, default=5000)
parser.add_argument('--model_path', type=str, default='/Users/Joel/Desktop/TRAINNING_LOG/vgg19.npy',help='Path to your vgg19.npy file')
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--save_path',type=str,default='/Users/Joel/Desktop/Log_1508/model/')
parser.add_argument('--vis_path',type=str,default='/Users/Joel/Desktop/Log_1508/val/')
args = parser.parse_args()

'''
file structure:
data_dir:
    image_folder : xxxx.jpg
    'annotations': xxxx.json
'''

image_folder='val2014'
data_dir = '/Users/Joel/Desktop/coco/coco_dataset'
image_path= '{}/images/{}/'.format(data_dir,image_folder)
anno_path = '{}/annotations/{}'.format(data_dir,'coco.json')


#concat your data here
'''
imgs_file_list: list of path to xxx.jpg
objs_info_list: annotations list of every image 
mask_list: mask of every image 
'''
df_val = PoseInfo(image_path, anno_path,False)
imgs_file_list= df_val.get_image_list()
objs_info_list=df_val.get_joint_list()
mask_list= df_val.get_mask()
targets=list(zip (objs_info_list,mask_list))


MODEL_PATH = args.model_path

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
    image, annos, mask_miss = keypoint_random_scale(image, annos, mask_miss)
    image, annos, mask_miss = keypoints_rotation(image, annos, mask_miss)
    image, annos, mask_miss = keypoint_random_flip(image, annos, mask_miss)
    image, annos, mask_miss = keypoint_resize_shortestedge_random(image, annos, mask_miss)
    image, annos, mask_miss = keypoint_crop_random(image, annos, mask_miss)

    h,w,_=np.shape(image)
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

#  COCO 19
n_pos = 19
n_epoch = 80
batch_size = args.batch_size
stepsize = 136106
weight_decay = 5e-4
base_lr = 4e-5
gamma=0.333

x = tf.placeholder(tf.float32, [None, 368, 368, 3], "image")
confs = tf.placeholder(tf.float32, [None, 46, 46, n_pos], "confidence_maps")
pafs = tf.placeholder(tf.float32, [None, 46, 46, n_pos * 2], "pafs")
img_mask1 =tf.placeholder(tf.float32,[None,46,46,19],'img_mask1')
img_mask2 =tf.placeholder(tf.float32,[None,46,46,38],'img_mask2')
num_images=np.shape(imgs_file_list)[0]
#model define
cnn, b1_list, b2_list, net = model(x, n_pos,img_mask1,img_mask2, False, False)

#dataset api
dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.string))
dataset = dataset.map(_map_fn,num_parallel_calls=1)
dataset = dataset.repeat(n_epoch)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(buffer_size=2)
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
print('Config:','n_epoch: ',n_epoch,'batch_size: ',batch_size,'base_lr: ',base_lr,'stepsize: ',stepsize)
with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(base_lr, trainable=False)

opt = tf.train.MomentumOptimizer(lr_v,0.9)
train_op=opt.minimize(total_loss,global_step=global_step)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)


#start training
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    #for vgg19 restoring
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

    sess.run(tf.assign(lr_v, base_lr))
    while(True):
        tic=time.time()
        gs_num = sess.run(global_step)
        if gs_num != 0 and (gs_num % stepsize == 0):
            new_lr_decay = gamma ** (gs_num // stepsize)
            sess.run(tf.assign(lr_v, base_lr * new_lr_decay))

        #TODO change to direct feed
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

        # if gs_num%args.log_interval==0 and gs_num!=0:
        lr=sess.run(lr_v)
        tstring=time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
        print('Total Loss at iteration {} is: {} Learning rate {:10e} weight_norm {:10e} Time: {}'.format(gs_num,the_loss,lr,weight_norm,tstring))
        for ix, ll in enumerate(loss_ll):
            print('Network#',ix,'For Branch',ix%2+1,'Loss:',ll)

        if gs_num!=0 and gs_num % args.save_interval == 0:
            np.save(args.vis_path+'image' + str(gs_num) + '.npy', x_)
            np.save(args.vis_path+'heat_ground' + str(gs_num) + '.npy', confs_)
            np.save(args.vis_path+'heat_result' + str(gs_num) + '.npy', conf_result)
            np.save(args.vis_path+'paf_ground' + str(gs_num) + '.npy', pafs_)
            np.save(args.vis_path+'mask' + str(gs_num) + '.npy', mask)
            np.save(args.vis_path+'paf_result' + str(gs_num) + '.npy', paf_result)
            tl.files.save_npz_dict(net.all_params, args.save_path +'openpose_model' + str(gs_num) + '.npz', sess=sess)
        if gs_num > 3000001:
            break
