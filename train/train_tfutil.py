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

# Build the path dictionary for both train and test
FOLDERs = { 'train': '/mnt/fs0/datasets/mscoco/train_tfrecords',
            'val':  '/mnt/fs0/datasets/mscoco/val_tfrecords'}
KEY_LIST = ['bboxes', 'height', 'images', 'labels', 'num_objects', \
        'segmentation_masks', 'width']
BYTES_KEYs = ['images', 'labels', 'segmentation_masks', 'bboxes']
DATA_PATH = {}
for key_group in FOLDERs:
    for key_feature in KEY_LIST:
        DATA_PATH[ '%s/%s' % (key_group, key_feature) ] = os.path.join(FOLDERs[key_group], KEY_LIST[key_feature])


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

        source_dirs = [data_path['%s/%s' % (self.group, v)] for v in key_list]
        meta_dicts = [{v : {'dtype': tf.string, 'shape': []}} if v in BYTES_KEYs else {v : {'dtype': tf.int64, 'shape': []}} for v in key_list]

        super(COCO, self).__init__(
            source_dirs = source_dirs,
            meta_dicts = meta_dicts,
            batch_size=batch_size,
            n_threads=n_threads,
            shuffle = True,
            *args, **kwargs)


def pack_model(inputs, network = 'resnet50', weight_decay = 0.00005):



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

    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)
    BATCH_SIZE = args.batchsize

    train_data_param = {
                'func': COCO,
                'data_path': DATA_PATH,
                'group': 'train',
                'n_threads': n_threads,
                'batch_size': 1
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
