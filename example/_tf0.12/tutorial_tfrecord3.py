#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time
from PIL import Image
import os
import io
import json

"""
You will learn:
1. How to save time-series data (e.g. sentence) into TFRecord format file.
2. How to read time-series data from TFRecord format file.
3. How to create inputs, targets and mask.

Reference
----------
1. Google's im2txt - MSCOCO Image Captioning example
2. TFRecord in http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
3. Batching and Padding data in http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

"""




def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto,
  e.g, An integer label.
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto,
  e.g, an image in byte
  """
  # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
  e.g, sentence in list of ints
  """
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
  e.g, sentence in list of bytes
  """
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


## 1. Save data into TFRecord =====================================================
cwd = os.getcwd()
IMG_DIR = cwd + '/data/cat/'
SEQ_FIR = cwd + '/data/cat_caption.json'
VOC_FIR = cwd + '/vocab.txt'
# read image captions from JSON
with tf.gfile.FastGFile(SEQ_FIR, "r") as f:
    caption_data = json.loads(str(f.read(), encoding = "utf-8"))

processed_capts, img_capts = [], []
for idx in range(len(caption_data['images'])):
    img_capt = caption_data['annotations'][idx]['caption']
    img_capts.append(img_capt)
    processed_capts.append(tl.nlp.process_sentence(img_capt, start_word="<S>", end_word="</S>"))
print("Original Captions: %s" % img_capts)
print("Processed Captions: %s\n" % processed_capts)
# build vocab
_ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

# save
writer = tf.python_io.TFRecordWriter("train.cat_caption")
for idx in range(len(caption_data['images'])):
    # get data
    img_name = caption_data['images'][idx]['file_name']
    img_capt = '<S> ' + caption_data['annotations'][idx]['caption'] + ' </S>'
    img_capt_ids = [vocab.word_to_id(word) for word in img_capt.split(' ')]
    print("%s : %s : %s" % (img_name, img_capt, img_capt_ids))
    img = Image.open(IMG_DIR+img_name)
    img = img.resize((299, 299))
    # tl.visualize.frame(I=img, second=0.2, saveable=False, name=img_name, fig_idx=12234)
    img_raw = img.tobytes()
    img_capt_b = [v.encode() for v in img_capt.split(' ')]
    context = tf.train.Features(feature={   #  Non-serial data uses Feature
      "image/img_raw": _bytes_feature(img_raw),
    })
    feature_lists = tf.train.FeatureLists(feature_list={   # Serial data uses FeatureLists
      "image/caption": _bytes_feature_list(img_capt_b),
      "image/caption_ids": _int64_feature_list(img_capt_ids)
    })
    sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)
    writer.write(sequence_example.SerializeToString())  # Serialize To String
writer.close()

## 2. Simple read one image =======================================================
filename_queue = tf.train.string_input_producer(["train.cat_caption"])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)     # return the file and the name of file
# features, sequence_features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
features, sequence_features = tf.parse_single_sequence_example(serialized_example,
                        context_features={
                        'image/img_raw' : tf.FixedLenFeature([], tf.string),
                        },
                        sequence_features={
                        "image/caption": tf.FixedLenSequenceFeature([], dtype=tf.string),
                        "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                        }
                    )
c = tf.contrib.learn.run_n(features, n=1, feed_dict=None)
from PIL import Image
im = Image.frombytes('RGB', (299, 299), c[0]['image/img_raw'])
tl.visualize.frame(np.asarray(im), second=1, saveable=False, name='frame', fig_idx=1236)
c = tf.contrib.learn.run_n(sequence_features, n=1, feed_dict=None)
print(c[0])


## 3. Prefetch serialized SequenceExample protos ==================================
def distort_image(image, thread_id):
  """Perform random distortions on an image.
  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
  Returns:````
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal"):#, values=[image]): # DH MOdify
  # with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)
  # Randomly distort the colors based on thread id.
  color_ordering = thread_id % 2
  with tf.name_scope("distort_color"):#, values=[image]): # DH MOdify
  # with tf.name_scope("distort_color", values=[image]): # DH MOdify
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image

# def process_image(encoded_image,
#                   is_training,
#                   height,
#                   width,
#                   resize_height=346,
#                   resize_width=346,
#                   thread_id=0,
#                   image_format="jpeg"):
#   """Decode an image, resize and apply random distortions.
#   In training, images are distorted slightly differently depending on thread_id.
#   Args:
#     encoded_image: String Tensor containing the image.
#     is_training: Boolean; whether preprocessing for training or eval.
#     height: Height of the output image.
#     width: Width of the output image.
#     resize_height: If > 0, resize height before crop to final dimensions.
#     resize_width: If > 0, resize width before crop to final dimensions.
#     thread_id: Preprocessing thread id used to select the ordering of color
#       distortions. There should be a multiple of 2 preprocessing threads.
#     image_format: "jpeg" or "png".
#   Returns:
#     A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
#   Raises:
#     ValueError: If image_format is invalid.
#   """
#   # Helper function to log an image summary to the visualizer. Summaries are
#   # only logged in thread 0.
#   def image_summary(name, image):
#     if not thread_id:
#       tf.image_summary(name, tf.expand_dims(image, 0))
#
#   # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
#   with tf.name_scope("decode"):#, values=[encoded_image]):   # DH modify
#   # with tf.name_scope("decode", values=[encoded_image]):   # DH modify
#     if image_format == "jpeg":
#       image = tf.image.decode_jpeg(encoded_image, channels=3)
#     elif image_format == "png":
#       image = tf.image.decode_png(encoded_image, channels=3)
#     else:
#       raise ValueError("Invalid image format: %s" % image_format)
#   image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#   image_summary("original_image", image)
#
#   # Resize image.
#   assert (resize_height > 0) == (resize_width > 0)
#   if resize_height:
#     # image = tf.image.resize_images(image,
#     #                                size=[resize_height, resize_width],
#     #                                method=tf.image.ResizeMethod.BILINEAR)
#
#     image = tf.image.resize_images(image,       # DH Modify
#                                    new_height=resize_height,
#                                    new_width=resize_width,
#                                    method=tf.image.ResizeMethod.BILINEAR)
#
#   # Crop to final dimensions.
#   if is_training:
#     image = tf.random_crop(image, [height, width, 3])
#   else:
#     # Central crop, assuming resize_height > height, resize_width > width.
#     image = tf.image.resize_image_with_crop_or_pad(image, height, width)
#
#   image_summary("resized_image", image)
#
#   # Randomly distort the image.
#   if is_training:
#     image = distort_image(image, thread_id)
#
#   image_summary("final_image", image)
#
#   # Rescale to [-1,1] instead of [0, 1]
#   image = tf.sub(image, 0.5)
#   image = tf.mul(image, 2.0)
#   return image

def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    print("   is_training == True : RandomShuffleQueue")
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    print("   is_training == False : FIFOQueue")
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.scalar_summary(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue

is_training = True
resize_height = resize_width = 346
height = width = 299
# start to read
reader = tf.TFRecordReader()
input_queue = prefetch_input_data(
        reader,
        file_pattern = "train.cat_caption", # sets train.???_caption to read many files
        is_training = is_training,          # if training, shuffle and random choice
        batch_size = 4,
        values_per_shard = 2300,            # mixing between shards in training.
        input_queue_capacity_factor = 2,    # minimum number of shards to keep in the input queue.
        num_reader_threads = 1              # number of threads for prefetching SequenceExample protos.
        )
serialized_sequence_example = input_queue.dequeue()
    # serialized_sequence_example = tf.train.string_input_producer(["train.cat_caption"])   # don't work
context, sequence = tf.parse_single_sequence_example(
        serialized=serialized_sequence_example,
        context_features={
        "image/img_raw": tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
        "image/caption": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        )

img = tf.decode_raw(context["image/img_raw"], tf.uint8)
img = tf.reshape(img, [height, width, 3])
img = tf.image.convert_image_dtype(img, dtype=tf.float32)

try:
    # for TensorFlow 0.11
    img = tf.image.resize_images(img,
                           size=(resize_height, resize_width),
                           method=tf.image.ResizeMethod.BILINEAR)
except:
    # for TensorFlow 0.10
    img = tf.image.resize_images(img,
                               new_height=resize_height,
                               new_width=resize_width,
                               method=tf.image.ResizeMethod.BILINEAR)
# Crop to final dimensions.
if is_training:
    img = tf.random_crop(img, [height, width, 3])
else:
    # Central crop, assuming resize_height > height, resize_width > width.
    img = tf.image.resize_image_with_crop_or_pad(img, height, width)
# Randomly distort the image.
if is_training:
    img = distort_image(img, thread_id=0)
# Rescale to [-1, 1] instead of [0, 1]
img = tf.sub(img, 0.5)
img = tf.mul(img, 2.0)
img_cap = sequence["image/caption"]
img_cap_ids = sequence["image/caption_ids"]
img_batch, img_cap_batch, img_cap_ids_batch = tf.train.batch([img, img_cap, img_cap_ids],   # Note: shuffle_batch doesn't support dynamic_pad
                                                    batch_size=4,
                                                    capacity=50000,
                                                    dynamic_pad=True,   # string list pad with '', int list pad with 0
                                                    num_threads=4)
sess = tf.Session()
# sess.run(tf.initialize_all_variables())
tl.layers.initialize_global_variables(sess)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for _ in range(3):
    print("Step %s" % _)
    # print(sess.run([img, img_cap, img_cap_ids]))                                 # one example only
    imgs, caps, caps_id = sess.run([img_batch, img_cap_batch, img_cap_ids_batch])  # batch of examples with dynamic_pad
    print(caps)
    print(caps_id)
    tl.visualize.images2d((imgs+1)/2, second=1, saveable=False, name='batch', dtype=None, fig_idx=202025)
coord.request_stop()
coord.join(threads)
sess.close()




## 4. Prefetch serialized SequenceExample protos. Create MASK and TARGET =======
def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  enqueue_list = []
  for image, caption in images_and_captions:
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.sub(caption_length, 1), 0)

    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    enqueue_list.append([image, input_seq, target_seq, indicator])

  images, input_seqs, target_seqs, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.scalar_summary("caption_length/batch_min", tf.reduce_min(lengths))
    tf.scalar_summary("caption_length/batch_max", tf.reduce_max(lengths))
    tf.scalar_summary("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, target_seqs, mask


images, input_seqs, target_seqs, input_mask = (
      batch_with_dynamic_pad(images_and_captions=[[img, img_cap]],
                                       batch_size=4,
                                       queue_capacity=50000)
                                       )
sess = tf.Session()
sess.run(tf.initialize_all_variables())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for _ in range(3):
    print("Step %s" % _)
    imgs, inputs, targets, masks = sess.run([images, input_seqs, target_seqs, input_mask])
    print(inputs)
    print(targets)
    print(masks)
    tl.visualize.images2d((imgs+1)/2, second=1, saveable=False, name='batch', dtype=None, fig_idx=202025)
coord.request_stop()
coord.join(threads)
sess.close()














#
