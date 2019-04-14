# This code shows the noiseprint_blind output
#    python main_blind.py input.png output.mat
#    python main_showout.py input.png output.mat
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

import matplotlib.pyplot as plt
import numpy as np
from sys import argv
imgfilename = argv[1]
outfilename = argv[2]

print(' %s' % imgfilename)
from noiseprint.utility.utilityRead import imread2f
img, mode = imread2f(imgfilename, channel = 3)
print('size : ', img.shape)

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

if 'noiseprint' in dat.keys():
    res = dat['noiseprint']
    vmin = np.min(res[34:-34,34:-34])
    vmax = np.max(res[34:-34,34:-34])
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img, clim=[0,1])
    plt.title('input \n image (%s, %d)' % (mode, qf))
    plt.subplot(1,2,2)
    plt.imshow(res.clip(vmin,vmax), clim=[vmin,vmax], cmap='gray')
    plt.title('noiseprint')
    plt.show()


if 'map' in dat.keys():
    mapp    = dat['map']
    valid   = dat['valid']
    range0  = dat['range0'].flatten()
    range1  = dat['range1'].flatten()
    imgsize = dat['imgsize'].flatten()

    from noiseprint.noiseprint_blind import genMappUint8
    mapUint8 = genMappUint8(mapp, valid, range0,range1, imgsize)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img, clim=[0,1])
    plt.title('input \n image (%s, %d)' % (mode, qf))
    plt.subplot(1,2,2)
    plt.imshow(mapUint8, clim=[0,255], cmap='gray')
    plt.title('heatmap')
    plt.show()
