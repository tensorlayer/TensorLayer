#! /usr/bin/python
# -*- coding: utf-8 -*-

# Example of training an Inception V3 model with ImageNet. The parameters are set as in the
# best results of the paper: https://arxiv.org/abs/1512.00567
# The dataset can be downloaded from http://www.image-net.org/ or from the Kaggle competition:
# https://www.kaggle.com/c/imagenet-object-localization-challenge/data

import argparse
import logging
import multiprocessing
import os
import random
import sys
import time
from xml.etree import ElementTree

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import (inception_v3,
                                                                   inception_v3_arg_scope)
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook
from tensorflow.python.training.monitored_session import \
    SingularMonitoredSession

########## VARIABLES ##########

# get the dataset: https://www.kaggle.com/c/imagenet-object-localization-challenge/data
# get the synset dictionary: http://www.image-net.org/archive/words.txt

BASE_DIR = './'
ILSVRC_DIR = os.path.join(BASE_DIR, 'ILSVRC')
SYNSET_DICT = os.path.join(BASE_DIR, 'words.txt')
TRAIN_FILE = os.path.join(BASE_DIR, 'train.csv')
VAL_FILE = os.path.join(BASE_DIR, 'val.csv')
CLASSES_FILE = os.path.join(BASE_DIR, 'classes.csv')
CLASSES_VAL_FILE = os.path.join(BASE_DIR, 'classes_val.csv')
CHECKPOINTS_PATH = './checkpoints'


########## DATASETS ##########

def get_data_sample(annotation_file, annotations_dir, data_dir):
    labels = []
    image_file = annotation_file.replace(annotations_dir, data_dir).replace('.xml', '.JPEG')
    if tf.gfile.Exists(annotation_file) and tf.gfile.Exists(image_file):
        xmltree = ElementTree.parse(annotation_file)
        objects = xmltree.findall("object")
        for object_iter in objects:
            labels.append(object_iter.find("name").text)
    else:
        image_file = None
    return image_file, labels


def might_create_dataset(prefix, file, shuffle=False, suffix='**/*.xml'):
    # load data
    data = []
    labels = set()
    annotations_dir = os.path.join(ILSVRC_DIR, 'Annotations', 'CLS-LOC', prefix)
    data_dir = os.path.join(ILSVRC_DIR, 'Data', 'CLS-LOC', prefix)
    for filename in tf.gfile.Glob(os.path.join(annotations_dir, suffix)):
        image_path, image_labels = get_data_sample(filename, annotations_dir, data_dir)
        if image_path is not None and len(image_labels) > 0:
            data.append([image_path] + image_labels)
            for label in image_labels:
                labels.add(label)
    if shuffle:
        random.shuffle(data)
    # write data
    with tf.gfile.Open(file, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))
    return sorted(labels)


def might_create_training_set():
    if not tf.gfile.Exists(TRAIN_FILE):
        labels = might_create_dataset('train', TRAIN_FILE,
                                      shuffle=True)
        with tf.gfile.Open(CLASSES_FILE, 'w') as f:
            for l in labels:
                f.write('{}\n'.format(l))


def might_create_validation_set():
    if not tf.gfile.Exists(VAL_FILE):
        labels = might_create_dataset('val', VAL_FILE, suffix='*.xml')
        with tf.gfile.Open(CLASSES_VAL_FILE, 'w') as f:
            for l in labels:
                f.write('{}\n'.format(l))


def load_data(file, task_spec=None, batch_size=16, epochs=1, shuffle_size=0):
    # load classes dict:
    with tf.gfile.Open(CLASSES_FILE) as f:
        labels = dict()
        for i, line in enumerate(f.readlines()):
            label = line.strip()
            labels[label] = i
    num_classes = len(labels)
    # count file examples
    with tf.gfile.Open(file) as f:
        size = len(f.readlines())

    image_size = inception_v3.default_image_size
    dataset = tf.data.TextLineDataset([file])
    dataset = dataset.repeat(epochs)
    # split the dataset in shards
    if task_spec is not None and task_spec.num_workers > 1 and not task_spec.is_evaluator():
        dataset = dataset.shard(num_shards=task_spec.num_workers, index=task_spec.shard_index)
    if shuffle_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_size)

    def _parse_example_fn(line):
        line_split = line.split(',')
        filename = line_split[0]
        labels_names = line_split[1:]
        # labels
        one_hot_labels = np.zeros(num_classes, dtype=np.float32)
        for l in labels_names:
            one_hot_labels[labels[l]] = 1.0
        # image
        image_bytes = tf.gfile.FastGFile(filename, 'rb').read()
        return image_bytes, one_hot_labels

    def _map_fn(example_serialized):
        image_bytes, one_hot_labels = tf.py_func(_parse_example_fn, [example_serialized],
                                                 [tf.string, tf.float32], stateful=False)

        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize_images(image, size=[image_size, image_size])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        one_hot_labels = tf.reshape(one_hot_labels, [num_classes])
        return image, one_hot_labels

    max_cpus = multiprocessing.cpu_count()
    dataset = dataset.map(_map_fn, num_parallel_calls=max_cpus)
    dataset = dataset.prefetch(batch_size * max_cpus + 100)
    dataset = dataset.batch(batch_size)
    images, one_hot_classes = dataset.make_one_shot_iterator().get_next()

    images = tf.reshape(images, [batch_size, image_size, image_size, 3])
    one_hot_classes = tf.reshape(one_hot_classes, [batch_size, num_classes])

    return images, one_hot_classes, num_classes, size


########## NETWORK ##########

def build_network(image_input, num_classes=1001, is_training=False):
    net_in = tl.layers.InputLayer(image_input, name='input_layer')
    with slim.arg_scope(inception_v3_arg_scope()):
        network = tl.layers.SlimNetsLayer(layer=net_in,
                                          slim_layer=inception_v3,
                                          slim_args={
                                              'num_classes': num_classes,
                                              'is_training': is_training
                                              },
                                          name='InceptionV3')

    predictions = tf.nn.sigmoid(network.outputs, name='Predictions')
    return network, predictions


########## EVALUATOR ##########

class EvaluatorStops(Exception):
    def __init__(self, message):
        super(EvaluatorStops, self).__init__(message)


class EvaluatorHook(session_run_hook.SessionRunHook):

    def __init__(self, checkpoints_path, saver):
        self.checkpoints_path = checkpoints_path
        self.summary_writer = tf.summary.FileWriter(os.path.join(checkpoints_path, 'validation'))
        self.lastest_checkpoint = None
        self.saver = saver
        self.summary = None

    def after_create_session(self, session, coord):
        checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
        # wait until a new check point is available
        total_waited_secs = 0
        while self.lastest_checkpoint == checkpoint:
            time.sleep(30)  # sleep 30 seconds waiting for a new checkpoint
            checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
            total_waited_secs += 30
            if total_waited_secs > 30 * 60 * 60:
                raise EvaluatorStops('Waited more than half an hour to load a new checkpoint')

        # restore the checkpoint
        self.saver.restore(session, checkpoint)
        self.lastest_checkpoint = checkpoint
        self.eval_step = int(self.lastest_checkpoint.split('-')[-1])

    def end(self, session):
        super(EvaluatorHook, self).end(session)
        # save summaries
        self.summary_writer.add_summary(self.summary, self.eval_step)


########## METRICS ##########

def calculate_metrics(predicted_batch, real_batch, threshold=0.5, is_training=False, ema_decay=0.9):
    with tf.variable_scope('metric'):
        threshold_graph = tf.constant(threshold, name='threshold')
        zero_point_five = tf.constant(0.5)
        predicted_bool = tf.greater_equal(predicted_batch, threshold_graph)
        real_bool = tf.greater_equal(real_batch, zero_point_five)
        predicted_bool_neg = tf.logical_not(predicted_bool)
        real_bool_neg = tf.logical_not(real_bool)
        differences_bool = tf.logical_xor(predicted_bool, real_bool)
        tp = tf.logical_and(predicted_bool, real_bool)
        tn = tf.logical_and(predicted_bool_neg, real_bool_neg)
        fn = tf.logical_and(differences_bool, real_bool)
        fp = tf.logical_and(differences_bool, predicted_bool)
        tp = tf.reduce_sum(tf.cast(tp, tf.float32))
        tn = tf.reduce_sum(tf.cast(tn, tf.float32))
        fn = tf.reduce_sum(tf.cast(fn, tf.float32))
        fp = tf.reduce_sum(tf.cast(fp, tf.float32))

        average_ops = None
        init_op = None
        if is_training:
            ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            average_ops = ema.apply([tp, tn, fp, fn])
            tp = ema.average(tp)
            tn = ema.average(tn)
            fp = ema.average(fp)
            fn = ema.average(fn)
        else:
            tp_v = tf.Variable(0, dtype=tf.float32, name='true_positive', trainable=False)
            tn_v = tf.Variable(0, dtype=tf.float32, name='true_negative', trainable=False)
            fp_v = tf.Variable(0, dtype=tf.float32, name='false_positive', trainable=False)
            fn_v = tf.Variable(0, dtype=tf.float32, name='false_negative', trainable=False)
            init_op = [tf.assign(tp_v, 0), tf.assign(tn_v, 0), tf.assign(fp_v, 0),
                       tf.assign(fn_v, 0)]
            tp = tf.assign_add(tp_v, tp)
            tn = tf.assign_add(tn_v, tn)
            fp = tf.assign_add(fp_v, fp)
            fn = tf.assign_add(fn_v, fn)

        # calculate metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fall_out = fp / (tn + fp)
        f1_score = tp * 2 / (tp * 2 + fp + fn)

        # remove NaNs and set them to 0
        zero = tf.constant(0, dtype=tf.float32)
        precision = tf.cond(tf.equal(tp, 0.0), lambda: zero, lambda: precision)
        recall = tf.cond(tf.equal(tp, 0.0), lambda: zero, lambda: recall)
        accuracy = tf.cond(tf.equal(tp + tn, 0.0), lambda: zero, lambda: accuracy)
        fall_out = tf.cond(tf.equal(fp, 0.0), lambda: zero, lambda: fall_out)
        f1_score = tf.cond(tf.equal(tp, 0.0), lambda: zero, lambda: f1_score)

        # add to tensorboard
        # tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('fall-out', fall_out)
        tf.summary.scalar('f1-score', f1_score)
        tf.summary.scalar('true_positive', tp)
        tf.summary.scalar('true_negative', tn)
        tf.summary.scalar('false_positive', fp)
        tf.summary.scalar('false_negative', fn)

    metrics_ops = {
        # 'accuracy' : accuracy,
        'precision'     : precision,
        'recall'        : recall,
        'fall-out'      : fall_out,
        'f1-score'      : f1_score,
        'true positive' : tp,
        'true negative' : tn,
        'false positive': fp,
        'false negative': fn,
        }
    return init_op, average_ops, metrics_ops


def run_evaluator(task_spec, checkpoints_path, batch_size=32):
    with tf.Graph().as_default():
        # load dataset
        images_input, one_hot_classes, num_classes, dataset_size = \
            load_data(file=VAL_FILE,
                      task_spec=task_spec,
                      batch_size=batch_size,
                      epochs=1)
        network, predictions = build_network(images_input,
                                             num_classes=num_classes,
                                             is_training=False)
        saver = tf.train.Saver()
        # metrics
        metrics_init_ops, _, metrics_ops = \
            calculate_metrics(predicted_batch=predictions,
                              real_batch=one_hot_classes,
                              is_training=False)
        # tensorboard summary
        summary_op = tf.summary.merge_all()
        # session hook
        evaluator_hook = EvaluatorHook(checkpoints_path=checkpoints_path, saver=saver)

        try:
            # infinite loop
            while True:
                with SingularMonitoredSession(hooks=[evaluator_hook]) as sess:
                    sess.run(metrics_init_ops)
                    try:
                        while not sess.should_stop():
                            metrics, summary = sess.run([metrics_ops, summary_op])
                            evaluator_hook.summary = summary
                    except OutOfRangeError:
                        pass
                    logging.info('step: {}  {}'.format(evaluator_hook.eval_step, metrics))
        except EvaluatorStops:
            # the evaluator has waited too long for a new checkpoint
            pass


########## TRAINING ##########

def run_worker(task_spec, checkpoints_path, batch_size=32, epochs=10):
    device_fn = task_spec.device_fn() if task_spec is not None else None
    # create graph
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.device(device_fn):
            # load dataset
            images_input, one_hot_classes, num_classes, dataset_size = \
                load_data(file=TRAIN_FILE,
                          task_spec=task_spec,
                          batch_size=batch_size,
                          epochs=epochs,
                          shuffle_size=10000)
            # network
            network, predictions = build_network(images_input,
                                                 num_classes=num_classes,
                                                 is_training=True)
            # training operations
            loss = tl.cost.sigmoid_cross_entropy(output=network.outputs,
                                                 target=one_hot_classes,
                                                 name='loss')
            steps_per_epoch = dataset_size / batch_size
            learning_rate = tf.train.exponential_decay(learning_rate=0.045,
                                                       global_step=global_step,
                                                       decay_steps=steps_per_epoch * 2,  # 2 epochs
                                                       decay_rate=0.94,
                                                       staircase=True,
                                                       name='learning_rate')
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  decay=0.9,
                                                  epsilon=1.0)
            # clip and apply gradients
            gvs = optimizer.compute_gradients(loss=loss,
                                              var_list=network.all_params)
            capped_gvs = []
            for grad, var in gvs:
                if grad is not None:
                    grad = tf.clip_by_value(grad, -2., 2.)
                capped_gvs.append((grad, var))
            train_op = optimizer.apply_gradients(grads_and_vars=capped_gvs,
                                                 global_step=global_step)
            # metrics
            tf.summary.scalar('learning_rate/value', learning_rate)
            tf.summary.scalar('loss/logits', loss)
            _, metrics_average_ops, metrics_ops = calculate_metrics(predicted_batch=predictions,
                                                                    real_batch=one_hot_classes,
                                                                    is_training=True)
            with tf.control_dependencies([train_op]):
                train_op = tf.group(metrics_average_ops)

        # start training
        hooks = [StopAtStepHook(last_step=steps_per_epoch * epochs)]
        with tl.distributed.DistributedSession(task_spec=task_spec,
                                               hooks=hooks,
                                               checkpoint_dir=checkpoints_path,
                                               save_summaries_secs=None,
                                               save_summaries_steps=300,
                                               save_checkpoint_secs=60 * 60) as sess:
            # print network information
            if task_spec is None or task_spec.is_master():
                network.print_params(False, session=sess)
                network.print_layers()
                sys.stdout.flush()
            # run training
            try:
                last_log_time = time.time()
                next_log_time = last_log_time + 60
                while not sess.should_stop():
                    step, loss_val, learning_rate_val, _, metrics = \
                        sess.run([global_step, loss, learning_rate, train_op, metrics_ops])
                    if task_spec is None or task_spec.is_master():
                        now = time.time()
                        if now > next_log_time:
                            last_log_time = now
                            next_log_time = last_log_time + 60
                            current_epoch = '{:.3f}'.format(float(step) / steps_per_epoch)
                            max_steps = epochs * steps_per_epoch
                            m = 'Epoch: {}/{} Steps: {}/{} Loss: {} Learning rate: {} Metrics: {}'
                            logging.info(m.format(current_epoch, epochs, step, max_steps,
                                                  loss_val, learning_rate_val, metrics))
            except OutOfRangeError:
                pass


########## MAIN ##########

if __name__ == '__main__':
    # print output logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

    if not tf.gfile.Exists(ILSVRC_DIR):
        logging.error('We couldn\'t find the directory "{}"'.format(ILSVRC_DIR))
        logging.error('You need to modify the variable BASE_DIR with the path where the dataset is.')
        logging.error('The dataset can be downloaded from http://www.image-net.org/ or from the Kaggle competition: https://www.kaggle.com/c/imagenet-object-localization-challenge/data')
        exit(-1)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_evaluator', dest='with_evaluator', action='store_true')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.set_defaults(with_evaluator=False)
    args = parser.parse_args()
    logging.info('Batch size: {}'.format(args.batch_size))
    logging.info('Epochs: {}'.format(args.epochs))

    # check the dataset and create them if necessary
    might_create_training_set()
    might_create_validation_set()

    # load environment for distributed training using last worker as evaluator
    task_spec = tl.distributed.TaskSpec()

    if task_spec is None:
        logging.info('Run in single node')
        run_worker(task_spec, CHECKPOINTS_PATH, batch_size=args.batch_size, epochs=args.epochs)
    else:
        if args.with_evaluator:
            # run with evaluator
            logging.info('Last worker is the evaluator')
            task_spec = task_spec.user_last_worker_as_evaluator()

        if task_spec.is_evaluator():
            run_evaluator(task_spec, CHECKPOINTS_PATH, batch_size=args.batch_size)
        else:
            task_spec.create_server()
            run_worker(task_spec, CHECKPOINTS_PATH, batch_size=args.batch_size, epochs=args.epochs)
