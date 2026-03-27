# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# This software is delivered with Government Purpose Rights (GPR) under agreement number FA8750-16-2-0204.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) 
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
@author: davide.cozzolino
"""

import os
import numpy as np
import tensorflow as tf
import argparse
try:
    from . import train_utility as utility
    #from .Producer2 import setWorkers_mock    as setWorkers
    #from .Producer2 import setWorkers_process as setWorkers
    from .Producer2 import setWorkers_thread  as setWorkers
    from .FCnet import FullConvNet
except:
    import train_utility as utility
    #from Producer2 import setWorkers_mock    as setWorkers
    #from Producer2 import setWorkers_process as setWorkers
    from Producer2 import setWorkers_thread  as setWorkers
    from FCnet import FullConvNet

parser = argparse.ArgumentParser(description="Script to train the Noiseprint.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu'        , type=str , default='0'  , help='id of GPU to use')
parser.add_argument('--workers'    , type=int , default=4    , help='number of workers')
parser.add_argument('--output_dir' , type=str , default='../model/noiseprint_ori/', help='output directory')
parser.add_argument('--dataset_dir', type=str , default='../data/ori/'  , help='dataset directory')
parser.add_argument('--pretrain'   , type=str , default='../pretrain/denoiser_17_3_5', \
                    help='pre-trained weight (default: %(default)s, use ''../pretrain/denoiser_17_1_1'' to replicate original paper)')
parser.add_argument('--train_model', type=int , default=600  , help='number of models of the training set')
parser.add_argument('--valid_model', type=int , default=25   , help='number of models of the validation set')
parser.add_argument('--block_size' , type=int , default=48   , help='block size')
parser.add_argument('--inp_channel', type=int , default=3    , help='number of input channels (default: %(default)s, use 1 to replicate original paper)')
parser.add_argument('--reg_type'   , type=int , default=2    , help='type of regularization')
parser.add_argument('--reg_weight' , type=float,default=32.0 , help='weight of regularization')
parser.add_argument('--batch_model', type=int , default=25   , help='number of models in a batch')
parser.add_argument('--epoch_imgs' , type=int , default=128  , help='number of images for model in an epoch')
parser.add_argument('--batch_imgs' , type=int , default=4    , help='number of images for model in a batch')
parser.add_argument('--img_clips'  , type=int , default=2    , help='number of cuttings for image')
parser.add_argument('--alpha'      , type=float,default=1.0  , help='change it if you have the vanishing gradient problem')
parser.add_argument('--net_levels' , type=int , default=17   , help='number of levels of the network architecture')
parser.add_argument('--padding'    , type=str , default='VALID', help='padding (default: %(default)s, use ''SAME'' to replicate original paper)')

config = parser.parse_args()
print(config)

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

block_size  = config.block_size
nearImg     = config.batch_imgs
cliped_num  = config.img_clips
batch_model = config.batch_model
batch_size  = batch_model*nearImg
batch_num   = batch_size*cliped_num
numMaxImg   = config.epoch_imgs
num_levels  = config.net_levels
numWorkers  = config.workers
regularize_factor = config.reg_weight
regularize_type   = config.reg_type
assert(numMaxImg%nearImg==0)
list_lr = [1e-3,]*20+ [1e-4,]*20 + [1e-5,]*20 + [1e-6,]*20
problem_type = 0 # 0 variant, 1 invariant position

forderNET = os.path.join(config.output_dir, "net%d_%d%d_%d%d_%d" % (num_levels, problem_type, config.inp_channel, nearImg, cliped_num, regularize_type))
init_forderNET = config.pretrain

def onlyOpenImage(img, *other):
    return (img, *other)

def clipImage(img, wSize, crop0, crop1, indRot):
    left  = max(min(int(np.floor((img.shape[1]-wSize+1) * crop1)), img.shape[1]-wSize),0)
    upper = max(min(int(np.floor((img.shape[0]-wSize+1) * crop0)), img.shape[0]-wSize),0)
    right = left  + wSize
    lower = upper + wSize

    img   = img[upper:lower,left:right,:]
    indRot = indRot%8
    if indRot >=4:
        img = img[::-1,:,:]
    img = np.rot90(img, indRot%4, axes=(0,1))

    img = img.astype(np.float32)/256
    return img

if __name__ == '__main__':
    cm = utility.defLabelClass(batch_num, nearImg, problem_type*(cliped_num-1)+1)
    
    tf.reset_default_graph()
    
    #make list datasets
    imgs_train = [np.load(os.path.join(config.dataset_dir, 'train%04d.npy' % index)) for index in range(config.train_model)]
    imgs_valid = [np.load(os.path.join(config.dataset_dir, 'valid%04d.npy' % index)) for index in range(config.valid_model)]
    print(len(imgs_train), len(imgs_valid))
    assert(len(imgs_train) % batch_model == 0)
    assert(len(imgs_valid) % batch_model == 0)

    numEpoch     = len(list_lr)
    global_step  = tf.Variable(0, name="global_step", trainable=False)
    corr_alpha   = tf.Variable(config.alpha, dtype=tf.float32, name="corr_alpha", trainable=False)
    x_data       = tf.placeholder(tf.float32, [batch_num, block_size, block_size, config.inp_channel], name="x_data")
    learnin_rate = tf.placeholder(tf.float32, [], name="learnin_rate")
    flag_train   = tf.placeholder(tf.bool, [], name="flag_train")
    net  = FullConvNet(bnorm_decay=0.9, num_levels=num_levels)
    net_output = net(x_data, flag_train, padding=config.padding)
    print('output', net_output.get_shape())
    
    tf_whitening = utility.make_whitening(net_output, net_output.get_shape()[1], regularize_type)
    dist = utility.distmxt(net_output)
    corr = -corr_alpha*corr_alpha*dist
    loss = tf.reduce_sum(utility.my_loss_paper(corr,cm))

    print(dist.get_shape())
    print(cm.shape)
    dist_p = tf.reduce_sum(dist*(cm>0))/np.count_nonzero(cm>0)
    dist_n = tf.reduce_sum(dist*(cm<0))/np.count_nonzero(cm<0)
    
    optimizer = tf.train.AdamOptimizer(learnin_rate, name="opt")
    if regularize_factor>0:
        loss_and_reg = loss - regularize_factor*tf_whitening
    else:
        loss_and_reg = loss
    
    trainer_loss = optimizer.minimize(loss_and_reg, var_list=net.trainable_list)
    train_ops    = [trainer_loss] + net.extra_train
    train_op     = tf.group(*train_ops)
    
    configSess = tf.ConfigProto(); configSess.gpu_options.allow_growth = True
    #configSess = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
    sess = tf.Session(config=configSess)
    sess.run(tf.global_variables_initializer())
    if init_forderNET is not None:
        init_saver  = tf.train.Saver(net.variables_list)
        if os.path.isdir(init_forderNET):
            init_chkpt_fname = tf.train.latest_checkpoint(init_forderNET)
        else:
            init_chkpt_fname = init_forderNET
        init_saver.restore(sess, init_chkpt_fname)
        print('pretrain:', init_chkpt_fname)
    
    list_var_Saver = [global_step, corr_alpha] + net.variables_list
    saver = tf.train.Saver(list_var_Saver, max_to_keep=None)
    
    test_function  = lambda x_data_value: sess.run((loss, corr, tf_whitening),
                     feed_dict={flag_train: False, x_data: x_data_value})
    train_function = lambda x_data_value, learnin_rate_value: sess.run((loss, tf_whitening, train_op),
                     feed_dict={flag_train: True , x_data: x_data_value, learnin_rate: learnin_rate_value})
    
    os.makedirs(forderNET, exist_ok=True)
    
    chkpt_fname = tf.train.latest_checkpoint(forderNET)
    print(chkpt_fname)
    if chkpt_fname is not None:
        saver.restore(sess, chkpt_fname)
        #sess.run(tf.assign(global_step, 1))
        print('restored')
    else:
        saver.save(sess, os.path.join(forderNET, "model"), global_step = global_step)
    
    with open(os.path.join(forderNET, "log.txt"),'a') as filelog:
        for epoch in range(sess.run(global_step),numEpoch+1):
            if epoch>=0:
                list_valid = utility.genList(imgs_valid, numMaxImg, nearImg, 8)
                assert(len(list_valid)%batch_size==0)
                num_iter_valid  = len(list_valid)//batch_size
                print(len(list_valid), batch_size, num_iter_valid)
                
                list_valid = utility.genListWithCliped(list_valid, batch_model*nearImg, \
                                        cliped_num, block_size)
                
                cum_whitening_value = 0
                cum_loss = 0
                cum_scores = np.zeros((0,), dtype=np.float32)
                cum_labels = np.zeros((0,), dtype=np.float32)

                with setWorkers(list_valid, pre_fun=onlyOpenImage, app_fun=clipImage, \
                                buffer_size=numWorkers*100, workers=numWorkers) as iter_valid:
                    iteretion = 0
                    for x_data_value in utility.genBatchList(batch_num, iter_valid, flag_reset=False):
                        x_data_value = np.asarray(x_data_value)
                        if config.inp_channel == 1:
                            x_data_value = 0.299 * x_data_value[:, :, :, 0:1] + \
                                           0.587 * x_data_value[:, :, :, 1:2] + \
                                           0.114 * x_data_value[:, :, :, 2:3]

                        loss_value, corr_value, whitening_value = test_function(x_data_value)

                        print("Valid  epoch:%04d, %06d/%06d, loss:%6g, reg:%6g" % \
                              (epoch, iteretion, num_iter_valid, loss_value, whitening_value))
                        cum_scores = np.concatenate((cum_scores, corr_value.flatten()), 0)
                        cum_labels = np.concatenate((cum_labels, cm.flatten()), 0)
                        cum_loss += loss_value
                        cum_whitening_value += whitening_value
                        iteretion = iteretion+1
                cum_loss = cum_loss/num_iter_valid
                cum_whitening_value = cum_whitening_value/num_iter_valid
                acc_th,acc_val,mcc_th,mcc_val,auc = utility.getAUC(cum_scores,cum_labels)
                print("Valid  epoch:%04d, end, loss:%6g, auc:%6g, reg:%6g" % (epoch, cum_loss, auc, cum_whitening_value))
                print("                        acc:%6g, mcc:%6g, acc_th:%6g mcc_th:%6g" % (acc_val, mcc_val, acc_th, mcc_th))

                filelog.write("Valid  epoch:%04d, end, loss:%6g, auc:%6g, reg:%6g\n" % (epoch, cum_loss, auc, cum_whitening_value))
                filelog.write("                        acc:%6g, mcc:%6g, acc_th:%6g mcc_th:%6g\n" % (acc_val, mcc_val, acc_th, mcc_th))
                filelog.flush()
            
            if epoch < numEpoch:
                learnin_rate_value = list_lr[epoch]
                list_train = utility.genList(imgs_train, numMaxImg, nearImg, 1)
                assert(len(list_train)%batch_size==0)
                num_iter_train = len(list_train)//batch_size
                print(len(list_train), batch_size, num_iter_train)
                list_train = utility.genListWithCliped(list_train, batch_model*nearImg, \
                                               cliped_num, block_size)
                
                with setWorkers(list_train, pre_fun=onlyOpenImage, app_fun=clipImage, \
                                buffer_size=numWorkers*100, workers=numWorkers) as iter_train:
                    iteretion = 0
                    for x_data_value in utility.genBatchList(batch_num, iter_train, flag_reset=False):
                        x_data_value = np.asarray(x_data_value)
                        if config.inp_channel == 1:
                            x_data_value = 0.299 * x_data_value[:, :, :, 0:1] + \
                                           0.587 * x_data_value[:, :, :, 1:2] + \
                                           0.114 * x_data_value[:, :, :, 2:3]

                        loss_value, whitening_value, _ = train_function(x_data_value, learnin_rate_value)
                        
                        print("Train epoch:%04d, lr:%6g, %06d/%06d, loss:%6g, reg:%6g" % \
                          (epoch, learnin_rate_value, iteretion, num_iter_train, loss_value, whitening_value))
                        iteretion = iteretion+1
                sess.run(tf.assign(global_step, epoch + 1))
                saver.save(sess, os.path.join(forderNET, "model"), global_step=global_step)

