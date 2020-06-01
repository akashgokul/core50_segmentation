#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 23-07-2019                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Data Loader for the CORe50 Dataset """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# other imports
# import sys
# sys.path.append(".")
# import core50_segmentation
# from core50_segmentation.delete_background_multimages import process_img


import numpy as np
import pickle as pkl
import pandas as pd
import os
import logging
from hashlib import md5
from PIL import Image

from core50_helper_dataset import CORE50Helper
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CORE50(object):
    """ CORe50 Data Loader class

    Args:
        root (string): Root directory of the dataset where ``core50_128x128``,
            ``paths.pkl``, ``LUP.pkl``, ``labels.pkl``, ``core50_imgs.npz``
            live. For example ``~/data/core50``.
        preload (string, optional): If True data is pre-loaded with look-up
            tables. RAM usage may be high.
        scenario (string, optional): One of the three scenarios of the CORe50
            benchmark ``ni``, ``nc``, ``nic``, `nicv2_79`,``nicv2_196`` and
             ``nicv2_391``.
        train (bool, optional): If True, creates the dataset from the training
            set, otherwise creates from test set.
        cumul (bool, optional): If True the cumulative scenario is assumed, the
            incremental scenario otherwise. Practically speaking ``cumul=True``
            means that for batch=i also batch=0,...i-1 will be added to the
            available training data.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed as in the official benchmark.
        start_batch (int, optional): One of the training incremental batches
            from 0 to max-batch - 1. Remember that for the ``ni``, ``nc`` and
            ``nic`` we have respectively 8, 9 and 79 incremental batches. If
            ``train=False`` this parameter will be ignored.
    """

    nbatch = {
        'ni': 8,
        'nc': 9,
        'nic': 79,
        'nicv2_79': 79,
        'nicv2_196': 196,
        'nicv2_391': 391
    }

    def __init__(self, root='', preload=False, scenario='ni', task_type = 'classify', cumul=False,
                 run=0, start_batch=0):
        """" Initialize Object """

        self.root = os.path.expanduser(root)
        self.preload = preload
        self.scenario = scenario
        self.cumul = cumul
        self.run = run
        self.batch = start_batch
        self.task_type = task_type
        if(self.task_type == 'detect'):
            self.train_bbox_gt = pd.read_csv('/home/akash/core50/data/core50_train.csv')
            self.test_bbox_gt = pd.read_csv('/home/akash/core50/data/core50_test.csv')


        if preload:
            print("Loading data...")
            bin_path = os.path.join(root, 'core50_imgs.bin')
            if os.path.exists(bin_path):
                with open(bin_path, 'rb') as f:
                    self.x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(164866, 128, 128, 3)

            else:
                with open(os.path.join(root, 'core50_imgs.npz'), 'rb') as f:
                    npzfile = np.load(f)
                    self.x = npzfile['x']
                    print("Writing bin for fast reloading...")
                    self.x.tofile(bin_path)

        print("Loading paths...")
        with open(os.path.join(root, 'paths.pkl'), 'rb') as f:
            self.paths = pkl.load(f)
        paths = []
        for idx in range(len(self.paths)):
            curr_dir = os.path.join(self.root, self.paths[idx])
            if(os.path.isfile(curr_dir[:-4]+'_seg.png')):
                paths.append(self.paths[idx])
        print(len(paths))
        self.paths = paths

        print("Loading LUP...")
        with open(os.path.join(root, 'LUP.pkl'), 'rb') as f:
            self.LUP = pkl.load(f)

        print("Loading labels...")
        with open(os.path.join(root, 'labels.pkl'), 'rb') as f:
            self.labels = pkl.load(f)
        
    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """

        scen = self.scenario
        run = self.run
        batch = self.batch

        if self.batch == self.nbatch[scen]:
            raise StopIteration

        # Getting the right indexis
        if self.cumul:
            train_idx_list = []
            for i in range(self.batch + 1):
                train_idx_list += self.LUP[scen][run][i]
        else:
            train_idx_list = self.LUP[scen][run][batch]

        # loading data
        if self.preload:
            train_x = np.take(self.x, train_idx_list, axis=0)\
                      .astype(np.float32)
        else:
            print("Loading data...")
            # Getting the actual paths
            train_paths = []
            train_relative_paths = []
            for idx in train_idx_list:
                if(idx < len(self.paths)):
                    train_paths.append(os.path.join(self.root, self.paths[idx]))
                    train_relative_paths.append(self.paths[idx])
            # loading imgs
            train_x = self.get_batch_from_paths(train_paths).astype(np.float32)

        # In either case we have already loaded the y
        if self.cumul:
            train_y_label = []
            for i in range(self.batch + 1):
                train_y_label += self.labels[scen][run][i]
            
            train_y_label = np.asarray(train_y_label, dtype=np.float32)
            
            train_y_bbox = []
            if(self.task_type == 'detect'):
                train_y_bbox = [ [self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:]]['xmin'],
                    self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:]]['xmax'],   
                    self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:]]['ymin'],
                    self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:]]['xmin']] 
                    for img_path in train_relative_paths]
                train_y_bbox = torch.as_tensor(train_y_bbox)
            
            train_y_mask = np.array([])
            if(self.task_type == 'segment'):
                mask_paths = [path[:-4] +'_seg.png' for path in train_paths]
                train_y_mask = self.get_batch_from_paths(mask_paths, mask=True)

            train_y_label = np.asarray(train_y_label, dtype=np.float32)
            targets = {'bbox': train_y_bbox, 'label':train_y_label, 'mask':train_y_mask}
        else:
            train_y_label = self.labels[scen][run][batch]
            train_y_label = np.asarray(train_y_label, dtype=np.float32)
            train_y_bbox = np.array([])
            # if(self.task_type == 'detect'):
            #     train_y_bbox = [ [self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:].replace('png','jpg')]['xmin'].item(),
            #         self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:]]['xmax'],   
            #         self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:]]['ymin'],
            #         self.train_bbox_gt.loc[self.train_bbox_gt['Filename'] == img_path[-15:]]['xmin']] 
            #         for img_path in train_relative_paths]
            #     train_y_bbox = torch.as_tensor(train_y_bbox)

            train_y_mask = np.array([])
            if(self.task_type == 'segment'):
                mask_paths = [path[:-4] +'_seg.png' for path in train_paths]
                train_y_mask = self.get_batch_from_paths(mask_paths, mask=True)

            train_y_label = np.asarray(train_y_label, dtype=np.float32)
            targets = {'bbox': train_y_bbox, 'label':train_y_label, 'mask':train_y_mask}
        # train_y = process_img(train_relative_paths)
        #self.writer.add_image('mask', train_y,dataformats='HWC')

        # Update state for next iter
        self.batch += 1

        return (train_x, targets, self.batch)

    def get_test_set(self):
        """ Return the test set (the same for each inc. batch). """

        scen = self.scenario
        run = self.run

        test_idx_list = self.LUP[scen][run][-1]

        if self.preload:
            test_x = np.take(self.x, test_idx_list, axis=0).astype(np.float32)
        else:
            # test paths
            test_paths = []
            test_relative_paths = []
            for idx in test_idx_list:
                test_paths.append(os.path.join(self.root, self.paths[idx]))
                test_relative_paths.append(self.paths[idx])

            # test imgs
            test_x = self.get_batch_from_paths(test_paths).astype(np.float32)
    
        test_y_label = self.labels[scen][run][-1]
        test_y_bbox = []
        if(self.task_type == 'detect'):
            test_y_bbox = [ [self.test_bbox_gt.loc[self.test_bbox_gt['Filename'] == img_path[-15:]]['xmin'],
                self.test_bbox_gt.loc[self.test_bbox_gt['Filename'] == img_path[-15:]]['xmax'],   
                self.test_bbox_gt.loc[self.test_bbox_gt['Filename'] == img_path[-15:]]['ymin'],
                self.test_bbox_gt.loc[self.test_bbox_gt['Filename'] == img_path[-15:]]['xmin']] 
                for img_path in test_relative_paths]
            test_y_bbox = torch.as_tensor(test_y_bbox)


        test_y_mask = np.array([])
        if(self.task_type == 'segment'):
            mask_paths = [path[:-4] +'_seg.png' for path in test_paths]
            test_y_mask = self.get_batch_from_paths(mask_paths, mask=True)

        test_y_label = np.asarray(test_y_label, dtype=np.float32)
        targets = {'bbox': test_y_bbox, 'label':test_y_label, 'mask':test_y_mask}

        return test_x, targets

    next = __next__  # python2.x compatibility.

    @staticmethod
    def get_batch_from_paths(paths, compress=False, snap_dir='',
                             on_the_fly=True, verbose=False, mask=False):
        """ Given a number of abs. paths it returns the numpy array
        of all the images. """

        # Getting root logger
        log = logging.getLogger('mylogger')

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = len(paths)
        hexdigest = md5(''.join(paths).encode('utf-8')).hexdigest()
        log.debug("Paths Hex: " + str(hexdigest))
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, 'rb') as f:
                    npzfile = np.load(f)
                    x, y = npzfile['x']
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, 'rb') as f:
                    if(mask):
                        x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(num_imgs, 128, 128, 1)
                    else:
                        x = np.fromfile(f, dtype=np.uint8) \
                            .reshape(num_imgs, 128, 128, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            if(mask):
                x = np.zeros((num_imgs, 128, 128), dtype=np.uint8)
            else:
                x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end='')
                img = Image.open(path)
                img = np.array(img)
                x[i] = img

            if verbose:
                print()

            if not on_the_fly:
                # Then we save x
                if compress:
                    with open(file_path, 'wb') as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert (x is not None), 'Problems loading data. x is None!'

        return x




if __name__ == "__main__":

    # Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    # and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50(root='/home/akash/core50/data/core50_128x128', scenario="ni", task_type='segment')
    writer = SummaryWriter()

    # Get the fixed test set
    # test_x, test_y = dataset.get_test_set()

    # loop over the training incremental batches
    for i, train_batch in enumerate(dataset):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y, t = train_batch

        print("----------- batch {0} -------------".format(i))
        print("train_x shape: {}, train_y shape: {}, task: {}"
              .format(train_x.shape, train_y['mask'].shape, t))
        if(train_x.shape[0] > 0):
            print("TASK NOT EMPTY")
            img_1 = train_y['mask'][0,:,:]
            writer.add_image('task_img_'+str(t),train_x[0,:,:,:],dataformats='HWC')
            writer.add_image('task_img_'+str(t), img_1,dataformats='HW')

        # use the data
        pass

