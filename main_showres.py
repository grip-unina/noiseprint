# This code shows the result of noiseprint_blind output
#    python main_blind.py input.png output.mat
#    python main_showres.py input.png reference.png output.mat
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#
erodeKernSize  = 15
dilateKernSize = 11

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sys import argv
imgfilename = argv[1]
reffilename = argv[2]
outfilename = argv[3]

print(' %s' % imgfilename)
from noiseprint.utility.utilityRead import imread2f, computeMCC
from noiseprint.noiseprint_blind import genMappFloat
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter
from sklearn import metrics

img, mode = imread2f(imgfilename, channel = 3)
gt = imread2f(reffilename, channel = 1)[0]>0.5
print('size : ', img.shape)
assert(img.shape[0]==gt.shape[0])
assert(img.shape[1]==gt.shape[1])

gt1 = minimum_filter(gt, erodeKernSize)
gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
gtV = np.logical_or(gt0, gt1)
    
if outfilename[-4:] == '.mat':
    import scipy.io as sio
    dat = sio.loadmat(outfilename)
else:
    import numpy as np
    dat = np.load(outfilename)

time = dat['time'].flatten()
qf   = dat['QF'].flatten()

print('time : %g' % time)
print('qf   : %g' % qf)

valid   = dat['valid']
range0  = dat['range0'].flatten()
range1  = dat['range1'].flatten()
imgsize = dat['imgsize'].flatten()
mapp    = genMappFloat(dat['map'], valid, range0,range1, imgsize)

plt.figure(figsize=(3*5,2*5))
grid = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2, )
plt.subplot(grid[0,0])
plt.imshow(img, clim=[0,1])
plt.title('Input image')
plt.subplot(grid[0,1])
plt.imshow(gt, clim=[0,1], cmap='gray')
plt.title('Ground truth')
plt.subplot(grid[0,2])
plt.imshow(mapp, clim=[np.nanmin(mapp),np.nanmax(mapp)], cmap='jet')
plt.title('Heatmap')

mcc, ths = computeMCC(mapp, gt0, gt1)
plt.subplot(grid[1,:2])
plt.plot(ths,mcc)
plt.grid()
plt.xlabel('threshold'); plt.ylabel('|MCC|')
plt.legend(['max |MCC|=%5.3f'%np.max(mcc)])
plt.title('Matthews Correlation Coefficient')

ap1 = metrics.average_precision_score(gt1[gtV], +mapp[gtV])
ap2 = metrics.average_precision_score(gt1[gtV], -mapp[gtV])
smapp = mapp if ap1>=ap2 else -mapp
ap = max(ap1, ap2)
prec, racall, _ = metrics.precision_recall_curve(gt1[gtV], smapp[gtV])

plt.subplot(grid[1,2])
plt.plot(racall, prec)
plt.grid();
plt.xlabel('recall '); plt.ylabel('precision')
plt.legend(['AP=%5.3f'%ap])
plt.title('precision-recall curve')

plt.show()