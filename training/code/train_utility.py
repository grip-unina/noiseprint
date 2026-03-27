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

import numpy as np
import tensorflow as tf

def distmxt(res):
    #numImg = int(res.get_shape()[0])
    numImg = tf.shape(res)[0]
    res1 = tf.reshape(res ,(numImg, 1, -1))
    res2 = tf.reshape(res1,(1, numImg, -1))

    T   = (res1 - res2)/np.sqrt(2)
    dist = tf.reduce_sum(tf.square(T), axis=2)
    return dist

def my_loss_paper(corr, cm):
    ninf = np.NINF * tf.ones_like(corr)
    corrp = tf.where(cm >0, corr, ninf)
    corra = tf.where(cm!=0, corr, ninf)
    loss = tf.reduce_logsumexp(corra, axis=0) - tf.reduce_logsumexp(corrp,axis=0)
    return tf.reduce_sum(loss)

def defLabelClass(batch_size, nearImg, repMat):
    atom_size = batch_size//nearImg//repMat
    cm = -np.ones((atom_size,atom_size), dtype=np.float32) + 2.0 * np.eye(atom_size,  dtype=np.float32)
    cm = np.kron(np.ones((repMat,repMat),  dtype=np.float32),   cm) 
    cm = np.kron(cm, np.ones((nearImg,nearImg),  dtype=np.float32)) 
    
    cm = cm - np.eye(batch_size,  dtype=np.float32)
    return cm

        
def make_whitening(x, block_size, regularize_type):
    from scipy import fft
    mtxfft_real = np.real(fft(np.eye(block_size),axis=0)).astype(np.float32)
    mtxfft_imag = np.imag(fft(np.eye(block_size),axis=0)).astype(np.float32)

    tf_fft1_real = tf.tensordot(mtxfft_real, x, [[1,],[1,]])
    tf_fft1_imag = tf.tensordot(mtxfft_imag, x, [[1,],[1,]])
    tf_fft2_real = tf.tensordot(mtxfft_real, tf_fft1_real, [[1,],[2,]]) - tf.tensordot(mtxfft_imag, tf_fft1_imag, [[1,],[2,]])
    tf_fft2_imag = tf.tensordot(mtxfft_real, tf_fft1_imag, [[1,],[2,]]) + tf.tensordot(mtxfft_imag, tf_fft1_real, [[1,],[2,]])
    if regularize_type==2:
        tf_fft2_abs2 = tf.reduce_mean(tf_fft2_real**2 + tf_fft2_imag**2, axis=2)
        print('reg med')
        tf_whitening = tf.reduce_mean(tf.log(tf_fft2_abs2 / tf.reduce_sum(tf_fft2_abs2, axis=(0,1,2), keep_dims=True)))
    elif regularize_type==1:
        tf_fft2_abs2 = (tf_fft2_real**2 + tf_fft2_imag**2)
        print('reg sig')
        tf_whitening = tf.reduce_mean(tf.log(tf_fft2_abs2 / tf.reduce_sum(tf_fft2_abs2, axis=(0,1,3), keep_dims=True)))
    else:
        tf_whitening = 0
    return tf_whitening

def getAUC(scores, labels):
    from sklearn import metrics
    inds = np.argsort(scores)
    labels = labels[inds]
    scores = scores[inds]
    tn = np.cumsum((labels==-1).astype(dtype=np.float32))
    fp = tn[-1]-tn
    fn = np.cumsum((labels==+1).astype(dtype=np.float32))
    tp = fn[-1]-fn
    acc = (tp/(tp+fn) + tn/(tn+fp))/2
    ind_acc = np.argmax(acc)
    mcc = (tp*tn-fn*fp)/np.maximum(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),0.0001)
    ind_mcc = np.argmax(mcc)
    #plt.plot(mcc)
    #plt.show()
    return scores[ind_acc], acc[ind_acc], scores[ind_mcc], mcc[ind_mcc], metrics.auc(fp/(tn+fp),tp/(fn+tp))

def getAUC_dict(scores, labels):
    from sklearn import metrics
    inds = np.argsort(scores)
    labels = labels[inds]
    scores = scores[inds]
    tn = np.cumsum((labels==-1).astype(dtype=np.float32))
    fp = tn[-1]-tn
    fn = np.cumsum((labels==+1).astype(dtype=np.float32))
    tp = fn[-1]-fn
    FPR = fp/(tn+fp)
    TPR = tp/(fn+tp)
    auc = metrics.auc(FPR,TPR)
    acc = (TPR+1.0-FPR)/2
    mcc = (tp*tn-fn*fp)/np.maximum(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),0.0001)
    inf_tnr10 = np.argmin(np.abs(TPR-0.9))
    inf_tpr10 = np.argmin(np.abs(FPR-0.1))
    ind_acc = np.argmax(acc)
    ind_mcc = np.argmax(mcc)
    out = dict()
    out['auc'] = auc
    out['TNR10_val'] = 1.0-FPR[inf_tnr10]
    out['TPR10_val'] = TPR[inf_tpr10]
    out['acc_val']   = acc[ind_acc]
    out['mcc_val']   = mcc[ind_mcc]
    out['acc_th']    = scores[ind_acc]
    out['mcc_th']    = scores[ind_mcc]
    out['TNR10_th']  = scores[inf_tnr10]
    out['TPR10_th']  = scores[inf_tpr10]
    
    return out

def genListWithCliped(list_input, element_num, cliped_num, wSize):
    list_output = list()

    for index0 in range(0, len(list_input), element_num):
        indRR  = np.random.random_sample([cliped_num, 2])
        indRot = np.random.randint(low=0, high=8)

        for indexR in range(cliped_num):
            for indexE in range(index0, index0 + element_num):
                list_output.append((list_input[indexE], wSize, indRR[indexR, 0], indRR[indexR, 1], indRot))

    return list_output

def genList(imgs, numMaxImg, nearImg, rip):
    listInd = list()
    numModel = len(imgs)
    ripModel = numMaxImg // nearImg
    assert numMaxImg % nearImg == 0

    numsImg = [len(imgs[indexC]) - (len(imgs[indexC]) % nearImg) for indexC in range(numModel)]
    listModel = list(range(numModel))
    listImg = [list(range(len(imgs[indexC]))) for indexC in range(numModel)]
    for i in range(rip):
        for indexR in range(ripModel):
            np.random.shuffle(listModel)
            for indexC in listModel:
                for indexI in range(nearImg * indexR, nearImg * indexR + nearImg):
                    indexF = indexI % numsImg[indexC]
                    if indexF == 0:
                        np.random.shuffle(listImg[indexC])
                    listInd.append(imgs[indexC][listImg[indexC][indexF]])
    return listInd


class genBatchList():
    def __init__(self, batch_size, iterator, flag_reset=True):
        self.batch_size = int(batch_size)
        self.num_samples = len(iterator)
        self.iterator = iter(iterator)
        self.num_batch = int(np.ceil(self.num_samples / self.batch_size))
        self.flag_reset = flag_reset
        print(self.num_samples, self.batch_size, self.num_batch)

    def getNumSamples(self):
        return self.num_samples

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        while True:
            offset = 0
            for b in range(self.num_batch):
                data_batch = list()
                for c in range(self.batch_size):
                    offset = offset + 1
                    data_batch.append(next(self.iterator))
                    if offset >= self.num_samples: break

                yield data_batch
                if offset >= self.num_samples: break
            if not self.flag_reset:
                break


class genBatch():
    def __init__(self, batch_size, iterator, flag_reset=True):
        self.batch_size = int(batch_size)
        self.num_samples = len(iterator)
        self.iterator = iter(iterator)
        self.num_batch = int(np.ceil(self.num_samples / self.batch_size))
        self.flag_reset = flag_reset
        print(self.num_samples, self.batch_size, self.num_batch)

    def getNumSamples(self):
        return self.num_samples

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        while True:
            offset = 0
            for b in range(self.num_batch):
                data_batch = list()
                for c in range(self.batch_size):
                    offset = offset + 1
                    data_batch.append(next(self.iterator))
                    if offset >= self.num_samples: break

                data_batch = np.asarray(data_batch)
                yield data_batch
                if offset >= self.num_samples: break
            if not self.flag_reset:
                break