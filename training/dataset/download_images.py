# Script in Python 3.x to download the images from dpreview.com
db_root = './dpreview.com/'        # directory where to save the downloaded images
file_images = './list_images.csv'  # list of images to download from dpreview.com

import requests
import pandas
import numpy as np
import time
import json
import os
import locale
from tqdm import tqdm
from urllib.request import urlretrieve
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def download(url, output_dir ):
    filename = os.path.basename(url).split('?')[0]
    output_file = os.path.join(output_dir, filename)
    trys = 0
    while trys<5:
        trys = trys+1
        try:
            urlretrieve(url, output_file)
            return trys
        except:
            time.sleep(10)
    return trys


list_images = pandas.read_csv(file_images, sep=';', dtype={'id': str,'id_model': str})
dat_models = dict()
print('Downloading images from dpreview.com ...')
for index, img in tqdm(list_images.iterrows(), total=len(list_images)):
    id_model = img['id_model']
    id_image = img['id']
    if id_model not in dat_models:
        output_dir = os.path.join(db_root, id_model)
        os.makedirs(output_dir, exist_ok=True)
        page0 = requests.get('https://www.dpreview.com/sample-galleries/data/get-gallery?galleryId=%s'%id_model)
        dat_models[id_model] = json.loads(page0.content)['images']
    
    info = [x for x in dat_models[id_model] if x['id']==id_image][0]
    download(info['url'], output_dir)
print('Downloading is DONE.')