import os
from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict()
config.TRAIN.batch_size = 8
config.TRAIN.save_interval = 5000
config.TRAIN.log_interval = 1
config.TRAIN.n_epoch = 80
config.TRAIN.step_size = 136106  # evey number of step to decay lr
config.TRAIN.base_lr = 4e-5  # initial learning rate
config.TRAIN.gamma = 0.333  # gamma of Adam
config.TRAIN.weight_decay = 5e-4
config.TRAIN.distributed = False

config.MODEL = edict()
config.MODEL.model_path = 'models'  # save directory
config.MODEL.n_pos = 19  # number of keypoints
config.MODEL.hin = 368  # input size during training
config.MODEL.win = 368
config.MODEL.hout = config.MODEL.hin / 8  # output size during training (default 46)
config.MODEL.wout = config.MODEL.win / 8

if (config.MODEL.hin % 16 != 0) or (config.MODEL.win % 16 != 0):
    raise Exception("image size should be divided by 16")

config.DATA = edict()
config.DATA.coco_version = '2017'  # MSCOCO version 2014 or 2017
config.DATA.data_path = 'data'
config.DATA.your_images_path = os.path.join('data', 'your_data', 'images')
config.DATA.your_annos_path = os.path.join('data', 'your_data', 'coco.json')

config.LOG = edict()
config.LOG.vis_path = 'vis'

# config.VALID = edict()

# import json
# def log_config(filename, cfg):
#     with open(filename, 'w') as f:
#         f.write("================================================\n")
#         f.write(json.dumps(cfg, indent=4))
#         f.write("\n================================================\n")
