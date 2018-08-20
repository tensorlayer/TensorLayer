import time, os
import numpy as np
from config import config
from models import model
import tensorflow as tf
import tensorlayer as tl
from utils import draw_intermedia_results

if __name__ == '__main__':
    n_pos = config.MODEL.n_pos
    model_path = config.MODEL.model_path
    h, w = 368, 432 # image size for inferencing, small size can speed up
    if (h % 16 != 0) or (w % 16 != 0):
        raise Exception("image size should be divided by 16")

    ## define model
    x = tf.placeholder(tf.float32, [None, h, w, 3], "image")
    _, _, _, net = model(x, n_pos, None, None, False, False)

    conf_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch1/conf')[0]
    pafs_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch2/pafs')[0]

    def get_peak(pafs_tensor):
        smoother = Smoother({'data': pafs_tensor}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()
        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        tensor_peaks = tf.where(
            tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat, tf.zeros_like(gaussian_heatMat)
        )
        return tensor_peaks

    peak_tensor = get_peak(pafs_tensor)

    ## restore model parameters
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # tl.files.load_and_assign_npz_dict(os.path.join(model_path, 'pose1.npz'), sess)

    ## get one example image with range 0~1
    im = tl.vis.read_image('data/test.jpeg')
    im = tl.prepro.imresize(im, [h, w])
    im = im / 255.  # input image 0~1

    ## inference
    # 1st time need time to compile
    # _, _ = sess.run([conf_tensor, pafs_tensor], feed_dict={x: [im]})
    st = time.time()
    conf, pafs, peak = sess.run([conf_tensor, pafs_tensor, peak_tensor], feed_dict={x: [im]})
    t = time.time() - st
    print("get maps took {}s i.e. {} FPS".format(t, 1. / t))
    # print(conf.shape, pafs.shape, peak.shape)

    ## get coordinate results from maps using conf and pafs from network output, and peak
    # using OpenPose's official C++ code for this part
    # humans = get_json(peak[0], conf[0], pafs[0])
    # print(humans)

    ## draw maps
    draw_intermedia_results([im], None, conf, None, pafs, None, 'inference')

    ## draw connection
