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
from time import time
from PIL import Image
import numpy as np
import argparse
try:
    from .db_utility import get_list_valid, get_list_train, jpeg_compression
except:
    from db_utility import get_list_valid, get_list_train, jpeg_compression

parser = argparse.ArgumentParser(description="Script to generate numpy files of dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_dir', type=str , default='../data/ori/', help='output directory')
parser.add_argument('--width'     , type=int , default=1792 , help='width of the extracted dataset')
parser.add_argument('--height'    , type=int , default=1280 , help='height of the extracted dataset')
parser.add_argument('--seed'      , type=int , default=3453 , help='manual seed')
parser.add_argument('--jpeg_comp' , type=bool, default=False, help='flag for JPEG compression')
parser.add_argument('--resize_img', type=int , default=None , help='size for image resizing, None for no resizing')
parser.add_argument('--low_comp'  , type=int , default=90   , help='minimum JPEG compression')
parser.add_argument('--high_comp' , type=int , default=90   , help='maximum JPEG compression')

opt = parser.parse_args()
print(opt)

seed       = opt.seed
outputdata = opt.output_dir
size_imgs  = (opt.width, opt.height) # (W,H)
jpegComp   = opt.jpeg_comp
lowComp    = opt.low_comp
highComp   = opt.low_comp
resizeImg  = opt.resize_img

os.makedirs(outputdata, exist_ok=True)

imgs = dict()
imgs["train"] = get_list_train()
imgs["valid"] = get_list_valid()

print("train  num.cam %3d; num img %d "% (len(imgs["train"]), sum([len(x) for x in imgs["train"]])))
print("valid  num.cam %3d; num img %d "% (len(imgs["valid"]), sum([len(x) for x in imgs["valid"]])))

seed = np.random.RandomState(seed)
for key in ["train", "valid"]:
    list_cams = imgs[key]

    timestamp = time()
    num_imgs = 0
    for index, list_imgs in enumerate(list_cams):
        lists_out = list()
        for filename in list_imgs:
            with Image.open(filename) as img_o:
                if (resizeImg is not None) and (img_o.size[0]>resizeImg):
                    outsize = (resizeImg, img_o.size[1]*resizeImg//img_o.size[0])
                    print(img_o.size, outsize)
                    img = img_o.resize(outsize, resample=Image.BILINEAR)
                else:
                    img = img_o.copy()

                if jpegComp:
                    quality = seed.randint(lowComp, highComp + 1)
                    img = jpeg_compression(img, quality)

                left  = (img.size[0] - size_imgs[0]) // 2
                upper = (img.size[1] - size_imgs[1]) // 2
                left  = left  - left  % 16
                upper = upper - upper % 16
                img = img.crop((left, upper, left + size_imgs[0], upper + size_imgs[1]))
                lists_out.append(np.asarray(img))
                num_imgs = num_imgs + 1
        print(key, index, num_imgs, (time() - timestamp) / num_imgs)

        np.save(os.path.join(outputdata,'%s%04d.npy'% (key, index)), lists_out)
