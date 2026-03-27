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

import pandas
import numpy as np
import os

dataset_root   = '../dataset/dpreview.com' #TODO set these variable
dataset_models = '../dataset/list_models.csv'
dataset_images = '../dataset/list_images.csv'

def get_list_valid():
    tab_models  = pandas.read_csv(dataset_models, sep=';', dtype={'id': str})
    tab_images  = pandas.read_csv(dataset_images, sep=';', dtype={'id': str, 'id_model':str})
    list_models = sorted(tab_models.loc[tab_models['valid']==1]['id'].tolist())
    imgs = list()
    for item in list_models:
        tab_one = tab_images[tab_images['id_model']==item]
        list_file = tab_one['id'].tolist()
        img1 = list()
        for name in list_file:
            filename = os.path.join(dataset_root, 'cam_'+item, name + '.jpg')
            img1.append(filename)

        imgs.append(img1)
    return imgs

def get_list_train():
    tab_models = pandas.read_csv(dataset_models,sep=';', dtype={'id': str})
    tab_images = pandas.read_csv(dataset_images,sep=';', dtype={'id': str, 'id_model':str})
    list_models = sorted(tab_models.loc[tab_models['train']==1]['id'].tolist())
    imgs = list()
    for item in list_models:
        tab_one = tab_images[tab_images['id_model']==item]
        list_file = tab_one['id'].tolist()
        img1 = list()
        for name in list_file:
            filename = os.path.join(dataset_root, 'cam_'+item, name+'.jpg')
            img1.append(filename)

        imgs.append(img1)
    return imgs


def jpeg_compression(img, quality):
    from PIL import Image
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, progression=False, optimize=False)
    return Image.open(buffer)

def jpeg_compression_numpy(x, quality):
    from PIL import Image
    from io import BytesIO
    with  BytesIO() as buffer:
        with Image.fromarray(x) as img_x:
            img_x.save(buffer, format="JPEG", quality=quality, progression=False, optimize=False)
        with Image.open(buffer) as img_y:
            y = np.asarray(img_y).copy()
    return y

if __name__ == '__main__':
    from tqdm import tqdm
    from PIL import Image
    
    #list_all = sum(get_list_valid(),list())
    list_all = sum(get_list_train(), list())
    
    for item in tqdm(list_all):
        try:
            img = Image.open(item)
            if img.mode=='RGB':
                ss = img.size
                img = np.copy(np.asarray(img))
                if not img.dtype==np.uint8:
                    print('\n\nB%s\n\n\n'%item, img.dtype)
                elif not len(img.shape)==3:
                    print('\n\nC%s\n\n\n'%item)
                elif not img.shape[2]==3:
                    print('\n\nD%s\n\n\n'%item)
                elif not img.shape[0]==ss[1]:
                    print('\n\nF%s\n\n\n'%item)
                elif not img.shape[1]==ss[0]:
                    print('\n\nG%s\n\n\n'%item)
                    #print('\n\n%s\n\n\n'%'ok')
            else:   
                print('\n\nA%s\n\n\n'%item)
        except:
            print('\n\nE%s\n\n\n'%item)



