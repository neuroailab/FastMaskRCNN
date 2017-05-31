#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.nets.nets_factory as network 

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1

from train.train_utils import _configure_learning_rate, _configure_optimizer, \
    _get_variables_to_train, _get_init_fn, get_var_list_to_restore

from tfutils import base, data, optimizer

import json
import copy
import argparse

resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS

# Build the path dictionary for both train and test
FOLDERs = { 'train': '/mnt/fs0/datasets/mscoco/train_tfrecords',
            'val':  '/mnt/fs0/datasets/mscoco/val_tfrecords'}
KEY_LIST = ['bboxes', 'height', 'images', 'labels', 'num_objects', \
        'segmentation_masks', 'width']
BYTES_KEYs = ['images', 'labels', 'segmentation_masks', 'bboxes']
DATA_PATH = {}
for key_group in FOLDERs:
    for key_feature in KEY_LIST:
        DATA_PATH[ '%s/%s' % (key_group, key_feature) ] = os.path.join(FOLDERs[key_group], key_feature)

def restore(sess):

    if FLAGS.pretrained_model:
        if tf.gfile.IsDirectory(FLAGS.pretrained_model):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
        else:
            checkpoint_path = FLAGS.pretrained_model

        if FLAGS.checkpoint_exclude_scopes is None:
            FLAGS.checkpoint_exclude_scopes='pyramid'
        if FLAGS.checkpoint_include_scopes is None:
            FLAGS.checkpoint_include_scopes='resnet_v1_50'

        vars_to_restore = get_var_list_to_restore()
        for var in vars_to_restore:
            print ('restoring ', var.name)

        try:
           restorer = tf.train.Saver(vars_to_restore)
           restorer.restore(sess, checkpoint_path)
           print ('Restored %d(%d) vars from %s' %(
               len(vars_to_restore), len(tf.global_variables()),
               checkpoint_path ))
        except:
           print ('Checking your params %s' %(checkpoint_path))
           raise

# Build data provider for COCO dataset
class COCO(data.TFRecordsParallelByFileProvider):

    def __init__(self,
                 data_path,
                 key_list,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 *args,
                 **kwargs):
        self.group = group
        self.batch_size = batch_size

        source_dirs = [data_path['%s/%s' % (self.group, v)] for v in key_list]
        meta_dicts = [{v : {'dtype': tf.string, 'shape': []}} if v in BYTES_KEYs else {v : {'dtype': tf.int64, 'shape': []}} for v in key_list]

        super(COCO, self).__init__(
            source_dirs = source_dirs,
            meta_dicts = meta_dicts,
            batch_size=batch_size,
            n_threads=n_threads,
            shuffle = True,
            *args, **kwargs)

    def set_data_shapes(self, data):
        for i in range(len(data)):
            for k in data[i]:
                # set shape[0] to batch size for all entries
                shape = data[i][k].get_shape().as_list()
                shape[0] = self.batch_size
                data[i][k].set_shape(shape)
        return data

    def prep_data(self, data):
        for i in range(len(data)):
            inputs = data[i]

            image = inputs['images']
            image = tf.decode_raw(image, tf.uint8)
            ih = inputs['height']
            iw = inputs['width']
            ih = tf.cast(ih, tf.int32)
            iw = tf.cast(iw, tf.int32)
            inputs['height'] = ih
            inputs['width'] = iw

            imsize = tf.size(image)

            #image = tf.Print(image, [imsize, ih, iw], message = 'Imsize')

            image = tf.cond(tf.equal(imsize, ih * iw), \
                  lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
                  lambda: tf.reshape(image, (ih, iw, 3)))

            image_height = ih
            image_width = iw
            num_instances = inputs['num_objects']
            num_instances = tf.cast(num_instances, tf.int32)
            inputs['num_objects'] = num_instances
            gt_boxes = tf.decode_raw(inputs['bboxes'], tf.float64)
            gt_boxes = tf.reshape(gt_boxes, [num_instances, 4])

            labels = tf.decode_raw(inputs['labels'], tf.int32)
            labels = tf.reshape(labels, [num_instances, 1])
            inputs['labels'] = labels

            gt_boxes = tf.concat([gt_boxes, tf.cast(labels, tf.float64)], 1)
            gt_boxes = tf.cast(gt_boxes, tf.float32)

            gt_masks = tf.decode_raw(inputs['segmentation_masks'], tf.uint8)
            gt_masks = tf.cast(gt_masks, tf.int32)
            gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw])
            #gt_masks = tf.Print(gt_masks, [tf.shape(gt_masks)], message = 'Mask shape before', summarize = 4)

            image, gt_boxes, gt_masks = coco_preprocess.preprocess_image(image, gt_boxes, gt_masks, True)
            #image = tf.Print(image, [tf.shape(image)], message = 'Imsize', summarize = 4)
            #gt_masks = tf.Print(gt_masks, [tf.shape(gt_masks)], message = 'Mask shape', summarize = 4)

            inputs['segmentation_masks'] = gt_masks
            inputs['bboxes'] = gt_boxes
            inputs['images'] = image

        return data

    def set_data_shapes_none(self, data):
        for i in range(len(data)):
            for k in data[i]:
                # set shape[0] to batch size for all entries
                #print('coco before', k, data[i][k].get_shape().as_list())
                #data[i][k].set_shape(None)
                #data[i][k].set_shape(None)
                #print('coco before', k, data[i][k].get_shape().as_list())
                data[i][k] = tf.squeeze(data[i][k], axis=[-1])
                #print('coco', k, data[i][k].get_shape().as_list())
                pass
        return data

    def init_ops(self):
        self.input_ops = super(COCO, self).init_ops()

        # make sure batch size shapes of tensors are set
        #self.input_ops = self.set_data_shapes(self.input_ops)
        self.input_ops = self.set_data_shapes_none(self.input_ops)
        self.input_ops = self.prep_data(self.input_ops)
        #self.input_ops = self.set_data_shapes_none(self.input_ops)

        return self.input_ops


def pack_model(inputs, train = True, back_network = 'resnet50',
            #num_classes=81,
            num_classes=91,
            base_anchors=9,
            weight_decay=0.00005,
            **kwargs
        ):

    # Reshape the input image, batch size 1 supported
    image = inputs['images']
    ih = inputs['height']
    iw = inputs['width']
    im_shape = tf.shape(image)
    #image = tf.Print(image, [im_shape], message = 'shape', summarize = 4)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))
    image = tf.cast(image, tf.float32)

    image_height = ih
    image_width = iw
    num_instances = inputs['num_objects']
    gt_boxes = inputs['bboxes']
    #gt_boxes = tf.reshape(gt_boxes, [num_instances, 4])
    #labels = inputs['labels']
    #labels = tf.reshape(labels, [num_instances, 1])
    #gt_boxes = tf.concat([gt_boxes, tf.cast(labels, tf.float64)], 1)
    #gt_boxes = tf.Print(gt_boxes, [tf.shape(gt_boxes)], message = 'Box shape', summarize = 4)

    # Build the basic network
    logits, end_points, pyramid_map = network.get_network(back_network, image,
            weight_decay=weight_decay)

    # Build the pyramid
    pyramid = pyramid_network.build_pyramid(pyramid_map, end_points)

    # Build the heads
    outputs = \
        pyramid_network.build_heads(pyramid, image_height, image_width, num_classes, base_anchors, 
                    is_training=train, gt_boxes=gt_boxes)

    return {'outputs': outputs, 'pyramid': pyramid}, {'network': back_network}

def pack_loss(labels, logits, 
            #num_classes=81,
            num_classes=91,
            base_anchors=9,
            loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0],
            **kwargs
        ):
    #'targets': ['height', 'width', 'num_objects', 'labels', 'segmentation_masks', 'bboxes'],
    #print(labels)
    ih = labels[0]
    iw = labels[1]
    num_instances = labels[2]
    gt_boxes = labels[5]
    #labels_s = labels[3]
    #gt_boxes = tf.concat([gt_boxes, tf.cast(labels_s, tf.float64)], 1)
    #gt_boxes = tf.Print(gt_boxes, [tf.shape(gt_boxes)], message = 'Box shape loss', summarize = 4)
    gt_masks = labels[4]
    #gt_masks = tf.Print(gt_masks, [tf.shape(gt_masks)], message = 'Mask shape loss', summarize = 4)

    loss, losses, batch_info = pyramid_network.build_losses(logits['pyramid'], logits['outputs'], 
                    gt_boxes, gt_masks,
                    num_classes=num_classes, base_anchors=base_anchors,
                    rpn_box_lw=loss_weights[0], rpn_cls_lw=loss_weights[1],
                    refined_box_lw=loss_weights[2], refined_cls_lw=loss_weights[3],
                    mask_lw=loss_weights[4])

    #outputs['losses'] = losses
    #outputs['total_loss'] = loss
    #outputs['batch_info'] = batch_info
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regular_loss = tf.add_n(regular_losses)
    out_loss = tf.add_n(losses)
    total_loss = tf.add_n(losses + regular_losses)

    return total_loss

# Actual function to train the network
def main():
    parser = argparse.ArgumentParser(description='The script to train the mask R-CNN')
    # System setting
    parser.add_argument('--gpu', default = '0', type = str, action = 'store', help = 'Index of gpu, currently only one gpu is allowed')

    # General setting
    parser.add_argument('--nport', default = 27017, type = int, action = 'store', help = 'Port number of mongodb')
    parser.add_argument('--expId', default = "maskrcnn", type = str, action = 'store', help = 'Name of experiment id')
    parser.add_argument('--cacheDirPrefix', default = "/mnt/fs0/chengxuz/", type = str, action = 'store', help = 'Prefix of cache directory')
    parser.add_argument('--batchsize', default = 1, type = int, action = 'store', help = 'Batch size, only 1 is supported now')
    parser.add_argument('--initlr', default = 0.002, type = float, action = 'store', help = 'Initial learning rate')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp_id  = args.expId
    dbname = 'normalnet-test'
    colname = 'maskrcnn'
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), dbname, colname, exp_id)
    BATCH_SIZE = args.batchsize
    n_threads = 4

    # Define all params
    train_data_param = {
                'func': COCO,
                'data_path': DATA_PATH,
                'group': 'train',
                'n_threads': n_threads,
                'batch_size': 1,
                'key_list': KEY_LIST,
            }
    train_queue_params = {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE,
            'seed': 0,
            'capacity': 10
        }
    NUM_BATCHES_PER_EPOCH = 82783//BATCH_SIZE
    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': args.initlr,
            'decay_rate': 0.94,
            'decay_steps': NUM_BATCHES_PER_EPOCH*2,  # exponential decay each epoch
            'staircase': True
        }
    model_params = {
            'func': pack_model
        }
    optimizer_class = tf.train.MomentumOptimizer
    optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': optimizer_class,
            'clip': True,
            'momentum': .99
        }
    save_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': dbname,
            'collname': colname,
            'exp_id': exp_id,

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 2500,  # keeps loss from every SAVE_LOSS_FREQ steps.
            'save_valid_freq': 5000,
            'save_filters_freq': 5000,
            'cache_filters_freq': 5000,
            'cache_dir': cache_dir,
        }

    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': train_queue_params,
            'thres_loss': np.finfo(np.float32).max,
            'num_steps': 20 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        }
    load_query = None
    load_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': dbname,
            'collname': colname,
            'exp_id': exp_id,
            'do_restore': True,
            'query': load_query 
    }
    loss_func = pack_loss
    loss_params = {
            'targets': ['height', 'width', 'num_objects', 'labels', 'segmentation_masks', 'bboxes'],
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_func,
        }
    postsess_params = {
            'func': restore,
            }
    params = {
        'save_params': save_params,

        'load_params': load_params,

        'model_params': model_params,

        'train_params': train_params,

        'loss_params': loss_params,

        'learning_rate_params': learning_rate_params,

        'optimizer_params': optimizer_params,

        'postsess_params': postsess_params,

        'log_device_placement': False,  # if variable placement has to be logged
        'validation_params': {},
    }

    # Run the training
    base.train_from_params(**params)


if __name__ == '__main__':
    main()


'''
    val_data_param = {
                'func': COCO,
                'data_path': DATA_PATH,
                'group': 'val',
                'n_threads': n_threads,
                'batch_size': 12
            }
    val_queue_params = {
                'queue_type': 'fifo',
                'batch_size': 1,
                'seed': 0,
                'capacity': 10
            }
'''
