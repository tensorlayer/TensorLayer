from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave
from models import Decoder,Encoder
import utils

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ENCODER_PATH = 'pretrained_models/pretrained_vgg19_encoder_model.npz'
DECODER_PATH = 'pretrained_models/pretrained_vgg19_decoder_model.npz'
content_path = 'images/content/'
style_path = 'images/style/'
output_path = 'images/output/'

if __name__ == '__main__':

    content_images = os.listdir(content_path)
    style_images = os.listdir(style_path)

    with tf.Graph().as_default(), tf.Session() as sess:

        encoder = Encoder()
        decoder = Decoder()

        content_input = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content_input')
        style_input = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='style_input')

        # switch RGB to BGR
        content = tf.reverse(content_input, axis=[-1])
        style = tf.reverse(style_input, axis=[-1])
        # preprocess image
        content = encoder.preprocess(content)
        style = encoder.preprocess(style)

        # encode image
        # we should initial global variables before restore model
        enc_c_net = encoder.encode(content, 'content/')
        enc_s_net = encoder.encode(style, 'style/')

        # pass the encoded images to AdaIN
        target_features = utils.AdaIN(enc_c_net.outputs, enc_s_net.outputs)

        # decode target features back to image
        dec_net = decoder.decode(target_features, prefix="decoder/")

        generated_img = dec_net.outputs

        # deprocess image
        generated_img = encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        sess.run(tf.global_variables_initializer())

        encoder.restore_model(sess, ENCODER_PATH, enc_c_net)
        encoder.restore_model(sess, ENCODER_PATH, enc_s_net)
        decoder.restore_model(sess, DECODER_PATH, dec_net)

        start_time = datetime.now()
        image_count = 0
        for s in style_images:
            for c in content_images:
                image_count = image_count + 1
                # Load image from path and add one extra diamension to it.
                content_image = imread(os.path.join(content_path, c), mode='RGB')
                style_image = imread(os.path.join(style_path, s), mode='RGB')

                content_tensor = np.expand_dims(content_image, axis=0)
                style_tensor = np.expand_dims(style_image, axis=0)
                result = sess.run(generated_img, feed_dict={content_input: content_tensor, style_input: style_tensor})
                result_name = os.path.join(output_path, s.split('.')[0] + '_' + c.split('.')[0] + '.jpg')
                print(result_name, ' is generated')
                imsave(result_name, result[0])
        elapsed_time = datetime.now() - start_time
        print("total image:", image_count, " total_time ", elapsed_time, " average time:", elapsed_time / image_count)
