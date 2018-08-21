import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from scipy.misc import imresize, imsave

import tensorlayer as tl
import utils
from adain_norm import AdaIN
from vgg_decoder import Decoder
from vgg_encoder import Encoder

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ENCODER_PATH = 'pretrained_models/pretrained_vgg19_encoder_model.npz'
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1')
MODEL_SAVE_PATHS = './models/'

EPOCHS = 16
EPSILON = 1e-5
LEARNING_RATE = 1e-4

BATCH_SIZE = 8
HEIGHT = 256
WIDTH = 256
CHANNEL = 3
# batch_size, height, weight, channel_number
INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNEL)
CONTENT_DATA_PATH = '/root/even/dataset/COCO_train_2014/'
STYLE_DATA_PATH = '/root/even/dataset/wiki_all_images/'

if __name__ == '__main__':

    start_time = datetime.now()

    # Get the path of all valid images
    print('Preprocessing training images \n')
    content_images = utils.list_images(CONTENT_DATA_PATH)
    style_images = utils.list_images(STYLE_DATA_PATH)
    num_imgs = min(len(content_images), len(style_images))
    content_images = content_images[:num_imgs]
    style_images = style_images[:num_imgs]
    mod = num_imgs % BATCH_SIZE
    print('Preprocessing finish, %d images in total \n' % (num_imgs - mod))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        content_images = content_images[:-mod]
        style_images = style_images[:-mod]

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        encoder = Encoder()
        decoder = Decoder()

        content_input = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content_input')
        style_input = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style_input')

        # switch RGB to BGR
        content = tf.reverse(content_input, axis=[-1])
        style = tf.reverse(style_input, axis=[-1])
        # preprocess image
        content = encoder.preprocess(content)
        style = encoder.preprocess(style)

        content_enc_net = encoder.encode(content, 'content/')
        style_enc_net = encoder.encode(style, 'style/')

        adain_features = AdaIN(content_enc_net.outputs, style_enc_net.outputs)

        stylied_dec_net = decoder.decode(adain_features, 'stylized_dec/')

        # add the mean values back
        stylied_image = encoder.deprocess(stylied_dec_net.outputs)

        # switch BGR back to RGB
        stylied_image = tf.reverse(stylied_image, axis=[-1])

        # clip to 0..255
        stylied_image = tf.clip_by_value(stylied_image, 0.0, 255.0)

        # switch RGB to BGR
        stylied_image = tf.reverse(stylied_image, axis=[-1])

        # preprocess image
        stylied_image = encoder.preprocess(stylied_image)

        # pass the generated_img to the encoder, and use the output compute loss
        stylied_enc_net = encoder.encode(stylied_image, 'stylized_enc/')

        # compute the content loss
        content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(stylied_enc_net.outputs - adain_features), axis=[1, 2]))

        # compute the style loss
        style_layer_loss = []
        for layer in STYLE_LAYERS:
            enc_style_feat = tl.layers.get_layers_with_name(style_enc_net, 'style/' + layer, True)[0]
            enc__stylied_feat = tl.layers.get_layers_with_name(stylied_enc_net, 'stylized_enc/' + layer, True)[0]

            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc__stylied_feat, [1, 2])

            sigmaS = tf.sqrt(varS + EPSILON)
            sigmaG = tf.sqrt(varG + EPSILON)

            l2_mean = tf.reduce_sum(tf.square(meanG - meanS))
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        style_weight = 2.0

        # compute the total loss
        loss = content_loss + style_weight * style_loss

        # Training step (Only train the decoder params)
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, var_list=stylied_dec_net.all_params)

        sess.run(tf.global_variables_initializer())

        encoder.restore_model(sess, ENCODER_PATH, content_enc_net)
        encoder.restore_model(sess, ENCODER_PATH, style_enc_net)
        encoder.restore_model(sess, ENCODER_PATH, stylied_enc_net)

        # """Start Training"""
        step = 0
        n_batches = int(num_imgs // BATCH_SIZE)

        elapsed_time = datetime.now() - start_time
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)

        print('Now begin to train the model...\n')
        start_time = datetime.now()

        for epoch in range(EPOCHS):

            np.random.shuffle(content_images)
            np.random.shuffle(style_images)

            for batch in range(n_batches):
                # retrive a batch of content and style images
                content_batch_path = content_images[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                style_batch_path = style_images[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]

                content_batch = utils.get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                style_batch = utils.get_train_images(style_batch_path, crop_height=HEIGHT, crop_width=WIDTH)

                # run the training step
                sess.run(train_op, feed_dict={content_input: content_batch, style_input: style_batch})

                if step % 100 == 0:
                    _content_loss, _style_loss, _loss = sess.run(
                        [content_loss, style_loss, loss], feed_dict={
                            content_input: content_batch,
                            style_input: style_batch
                        }
                    )

                    elapsed_time = datetime.now() - start_time
                    print('step: %d,  total loss: %.3f, elapsed time: %s' % (step, _loss, elapsed_time))
                    print('content loss: %.3f' % (_content_loss))
                    print(
                        'style loss  : %.3f,  weighted style loss: %.3f\n' % (_style_loss, style_weight * _style_loss)
                    )

                if step % 1000 == 0:
                    print('save model now,step:', step)
                    tl.files.save_npz(stylied_dec_net.all_params, name=MODEL_SAVE_PATHS + str(step) + '_model.npz')
                    result_image = sess.run(
                        stylied_image, feed_dict={
                            content_input: content_batch,
                            style_input: style_batch
                        }
                    )
                    print("stylied_image generated", result_image.shape)
                    imsave("output/step_" + str(step) + ".png", result_image[0])

                step += 1

            print('One Epoch finished\n!')

        # """Done Training & Save the model"""
        tl.files.save_npz(stylied_dec_net.all_params, name=MODEL_SAVE_PATHS + str(step) + '_model.npz')
