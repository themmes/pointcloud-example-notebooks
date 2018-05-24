import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
from random import shuffle
import json

import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.delayed import delayed

from scipy.stats import binned_statistic_dd

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
#import provider
import tf_util
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=str, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--n_augmentations', type=int, default=1, help='Number of augmentations option: 1-6 [default: 1]')
parser.add_argument('--rgb', action='store_true', default=False, help='Use of RGB channels: True/False [default: True]')
parser.add_argument('--intensity', action='store_true', default=False, help='Use of intensity channels: True/False [default: False]')
parser.add_argument('--xyz', action='store_true', default=False, help='Add global XYZ channels: True/False [default: False]')
parser.add_argument('--xyzonly', action='store_true', default=False, help='Only use global XYZ channels: True/False [default: False]')
parser.add_argument('--trajectory', action='store_true', default=False, help='Add trajectory reference channels: True/False [default: False]')
parser.add_argument('--grid_size', type=int, default=1, help='Size of grid option: 1,2,5,10 [default: 1]')
parser.add_argument('--empty', action='store_true', default=False, help='Data quantity experiment')
parser.add_argument('--empty500', action='store_true', default=False, help='Data quantity experiment')
parser.add_argument('--empty100', action='store_true', default=False, help='Data quantity experiment')
parser.add_argument('--sample_method', default='random', help='random, grid or grideven [default: random]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
TEST_AREA = FLAGS.test_area
N_AUGMENTATIONS = FLAGS.n_augmentations
# channel selections
XYZ = FLAGS.xyz
XYZONLY = FLAGS.xyzonly
RGB = FLAGS.rgb
INTENSITY = FLAGS.intensity
TRAJECTORY = FLAGS.trajectory
GRID_SIZE = FLAGS.grid_size
hash_col = 'hash'+str(GRID_SIZE)+'m'
EMPTY = FLAGS.empty
EMPTY500 = FLAGS.empty500
EMPTY100 = FLAGS.empty500
SAMPLE_METHOD = FLAGS.sample_method

POINT_DIM = 3
if XYZ: POINT_DIM += 3
if RGB: POINT_DIM += 3
if INTENSITY: POINT_DIM += 1
if TRAJECTORY: POINT_DIM += 2

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train_custom.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 5
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def traintest_split(df, mode, path=None):
    if mode == 'random':
        hashes = df.index.unique().values
        # first split in train/test
        train_test_msk = np.random.rand(len(hashes))
        train_val_hashes = hashes[train_test_msk < 0.8]
        test_hashes = hashes[~(train_test_msk < 0.8)]
        # then split train again in train/val
        train_val_msk = np.random.rand(len(train_val_hashes))
        train_hashes = train_val_hashes[train_val_msk < 0.8]
        validation_hashes = train_val_hashes[~(train_val_msk < 0.8)]
    elif mode == 'spatial':
        hash_centers = df.groupby(hash_col).agg({'x_org':'mean', 'y_org':'mean', 'z_org':'mean'})
        # split in train and test along Y-axis
        train_val_hashes = hash_centers[hash_centers.y_org < hash_centers.y_org.quantile(q=.8)].index.values
        test_hashes = hash_centers[~(hash_centers.y_org < hash_centers.y_org.quantile(q=.8))].index.values
        # split train in train and validation randomly
        train_val_msk = np.random.rand(len(train_val_hashes))
        train_hashes = train_val_hashes[train_val_msk < 0.8]
        validation_hashes = train_val_hashes[~(train_val_msk < 0.8)]
    elif mode == 'load' and path != None:
        with open(path, 'r') as data_split:
            data_split_dict = json.load(data_split)
        train_hashes = np.array(data_split_dict['train'])
        if EMPTY: train_hashes = np.array(data_split_dict['train_empty'])
        if EMPTY500: train_hashes = np.array(data_split_dict['train_empty500'])
        if EMPTY100: train_hashes = np.array(data_split_dict['train_empty100'])
        validation_hashes = np.array(data_split_dict['validation'])
        test_hashes = np.array(data_split_dict['test'])
    else:
        print('No mode selected for split')
        raise
    
    return train_hashes, validation_hashes, test_hashes

def load_data_memory(data_dir, file, datasplit_path):
    """
    Function to load the data for training and testing.
    1. loads data from parquet
    2. gets all unique hash_codes with at least 100 points
    3. splits data in 80/20 train/test, splits train in 80/20 train/validation
    4. writes hash_codes from all three splits to json file in log directory
    IN: path for data directory, file name (combined full path)
    OUT: all data, hash subsets for train, test and validation
    """
    df = dd.read_parquet(os.path.join(data_dir, file))
    df = df.compute()
    df.reset_index(inplace=True)
    df.set_index(hash_col, inplace=True, drop=False)
    df = df.sort_index()
    df = df.groupby(hash_col).filter(lambda x: len(x) > 100)
    train_hashes, validation_hashes, test_hashes = traintest_split(df, mode='load', path=datasplit_path)
    with open(os.path.join(LOG_DIR, 'data_split.json'), 'w') as data_split:
        json.dump({'train': train_hashes.tolist(), 'validation': validation_hashes.tolist(), 'test': test_hashes.tolist()}, data_split)
    return df, train_hashes, test_hashes, validation_hashes

def generator(df, hashes, BATCH_SIZE, NUM_POINT, N_AUGMENTATIONS, shuffled=True):
    data_channels = [coord+str(GRID_SIZE)+'m' for coord in ['x_norm', 'y_norm', 'z_norm']]
    if XYZONLY: data_channels = ['x', 'y', 'z']
    if INTENSITY: data_channels = ['intensity'] + data_channels
    if RGB: data_channels = ['r', 'g', 'b'] + data_channels
    if XYZ: data_channels = ['x', 'y', 'z'] + data_channels
    if TRAJECTORY: data_channels.extend(['d_traj', 'h_traj'])
    
    seed_hash = []
    for seed in range(N_AUGMENTATIONS):
        for h in hashes:
            seed_hash.append((seed, h))
    shuffle(seed_hash)
    
    batches = [seed_hash[i:i+BATCH_SIZE] for i in range(0,len(seed_hash),BATCH_SIZE)]
    if len(batches[-1]) < BATCH_SIZE: batches = batches[:-1]
    if shuffled: [shuffle(batch) for batch in batches]
        
    def random_sample_block(group, seed):
        if len(group) > NUM_POINT:
            data_group = group.sample(n=NUM_POINT, replace=False, random_state=seed)
        else:
            data_group = group.sample(n=NUM_POINT, replace=True, random_state=seed)
        return data_group
    
    def grid_sample_block(group,seed):
        if len(group) < NUM_POINT: group = group.sample(n=NUM_POINT, replace=True,random_state=seed)
        group.reset_index(inplace=True, drop=True)
        xyz = group[['x_norm', 'y_norm', 'z_norm']].values
        binned = np.array([np.digitize(dim, bins=np.linspace(np.min(dim), np.max(dim), np.floor(np.cbrt(NUM_POINT)))) for dim in list(xyz.T)]).T
        uniq_bins, ids, uniq_counts = np.unique(binned, axis=0, return_inverse=True, return_counts=True)
        weights = 1/uniq_counts[ids]
        probabilities = weights / np.sum(weights)
        np.random.seed(seed)
        return group.iloc[np.random.choice(np.arange(xyz.shape[0]), NUM_POINT, p=probabilities, replace=True)]
    
    def grid_sample_even_block(group,seed):
        if len(group) < NUM_POINT: group = group.sample(n=NUM_POINT, replace=True,random_state=seed)
        group.reset_index(inplace=True, drop=True)
        xyz = group[['x_norm', 'y_norm', 'z_norm']].values
        binned = np.array([np.digitize(dim, bins=np.linspace(np.min(dim), np.max(dim), np.floor(np.cbrt(NUM_POINT)))) for dim in list(xyz.T)]).T
        uniq_bins, ids, uniq_counts = np.unique(binned, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(1+np.log(uniq_counts[ids]))
        probabilities = weights / np.sum(weights)
        np.random.seed(seed)
        return group.iloc[np.random.choice(np.arange(xyz.shape[0]), NUM_POINT, p=probabilities, replace=True)]

    for batch in batches:
        if SAMPLE_METHOD == 'random':
            df_batch = [random_sample_block(df.loc[h], s) for s,h in batch]
        elif SAMPLE_METHOD == 'grid':
            df_batch = [grid_sample_block(df.loc[h], s) for s,h in batch]
        elif SAMPLE_METHOD == 'grideven':
            df_batch = [grid_sample_even_block(df.loc[h], s) for s,h in batch]
        else:
            print('Sampling method not given')
        data = np.stack([b[data_channels].values for b in df_batch])
        label = np.stack([l.label.values for l in df_batch])
        yield data, label

data_dir = '/home/tom/vision/data'
file = 'oakland_norm'
datasplit_path = 'datasplit.json'
df, train_hashes, test_hashes, validation_hashes = load_data_memory(data_dir, file, datasplit_path)
        
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, POINT_DIM)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, POINT_DIM, bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)
            
            # Add confusion matrix to summary
            #tf.summary.tensor_summary('confusion_matrix', tf.confusion_matrix(labels=tf.reshape(tf.to_int64(labels_pl), [-1]), 
            #                                              predictions=tf.reshape(tf.argmax(pred, 2), [-1])))
                                                          # num_classes=NUM_CLASSES))

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    num_batches = 0
    for batch_data, batch_label in generator(df, train_hashes, BATCH_SIZE, NUM_POINT, N_AUGMENTATIONS):
        num_batches += 1 * BATCH_SIZE
        if num_batches % 10 == 0:
            print('Current batch num: {0}'.format(num_batches))
            
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    
    num_batches = 0
    for batch_data, batch_label in generator(df, validation_hashes, BATCH_SIZE, NUM_POINT, N_AUGMENTATIONS):
        num_batches += 1 * BATCH_SIZE
        if num_batches % 10 == 0:
            print('Current batch num: {0}'.format(num_batches))
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(BATCH_SIZE):
            for j in range(NUM_POINT):
                l = batch_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i, j] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    print('Correct classes', total_correct_class)
    print('Seen classes', total_seen_class)
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class),dtype=np.float)))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
    del df
    print('Done!')