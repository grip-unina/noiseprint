# This code converts the noiseprint_blind output in a PNG image
#   python main_mat2uint8.py output.mat output.png
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

from sys import argv
mapfilename = argv[1]
outfilename = argv[2]

if mapfilename[-4:] == '.mat':
    import scipy.io as sio
    dat = sio.loadmat(mapfilename)
else:
    import numpy as np
    dat = np.load(mapfilename)

mapp    = dat['map']
valid   = dat['valid']
range0  = dat['range0'].flatten()
range1  = dat['range1'].flatten()
imgsize = dat['imgsize'].flatten()

from noiseprint.noiseprint_blind import genMappUint8
mapUint8 = genMappUint8(mapp, valid, range0, range1, imgsize)

from PIL import Image
Image.fromarray(mapUint8).save(outfilename)
