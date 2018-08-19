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

config.MODEL = edict()
config.MODEL.n_pos = 19  # number of keypoints
config.MODEL.model_path = 'models'

config.DATA = edict()
config.DATA.coco_version = '2014'  # MSCOCO version 2014 or 2017
config.DATA.data_path = 'data'
config.DATA.your_data_path = 'data/your_data'

config.LOG = edict()
config.LOG.vis_path = 'vis'

# config.VALID = edict()

# import json
# def log_config(filename, cfg):
#     with open(filename, 'w') as f:
#         f.write("================================================\n")
#         f.write(json.dumps(cfg, indent=4))
#         f.write("\n================================================\n")
