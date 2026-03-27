# Noiseprint: a CNN-based camera model fingerprint
[Noiseprint](https://ieeexplore.ieee.org/document/8713484) is a CNN-based camera model fingerprint
extracted by a fully Convolutional Neural Network (CNN).

This folder contains the training code of Noiseprint defined in the following paper:

[A] D. Cozzolino and L. Verdoliva
''Noiseprint: A CNN-Based Camera Model Fingerprint'''
IEEE Transactions on Information Forensics and Security, 2020.

However, the provided dataset is related to the successive paper:

[B] D. Cozzolino, F. Marra, D. Gragnaniello, G. Poggi, and L. Verdoliva
''Combining PRNU and noiseprint for robust and efficient device sourceidentification''
EURASIP Journal on Information Security, 2020.


## License
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
This software is delivered with Government Purpose Rights (GPR) under agreement number FA8750-16-2-0204.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package)

## Usage

The provided dataset is a collection of images publicly available on "dpreviewer.com".
To download the dataset from "dpreviewer.com", run:

```
cd dataset
python download_images.py
```

To speed up the training, the dateset is converted in NPY files.
To create the NPY files, run:

```
cd ../code
python create_data.py
```

Noiseprint uses initial weights obtained training a denoiser.
The initial weights are included in the folder. 

To train the noiseprint with the parameters used in [A], run:

```
python train_noiseprint.py --gpu 0 --inp_channel 1 --pretrain ../pretrain/denoiser_17_1_1 --padding SAME
```

To train the noiseprint with the parameters used in [B], run:

```
python train_noiseprint.py --gpu 0
```

The whole code is tesetd with Python 3.x and Tensorflow 1.2.1 .

## Dependencies
Our code uses Python3.5, Cuda8, Cudnn5.
The following python packages are required:
- pillow==5.4.1
- numpy==1.16.1
- pandas==0.20.3
- scipy==0.19.1
- scikit-learn==0.19.1
- scikit-image==0.14.2
- tensorflow-gpu==1.2.1
- concurrent-iterator==0.2.6

## Reference

```js
@article{Cozzolino2020_Noiseprint,
  title={Noiseprint: A CNN-Based Camera Model Fingerprint},
  author={D. Cozzolino and L. Verdoliva},
  journal={IEEE Transactions on Information Forensics and Security},
  doi={10.1109/TIFS.2019.2916364},
  pages={144-159},
  year={2020},
  volume={15}
} 
```

```js
@article{cozzolino2020_Combining,
  title={Combining PRNU and noiseprint for robust and efficient device source identification},
  author={D. Cozzolino and F. Marra and D. Gragnaniello and G. Poggi and L. Verdoliva},
  journal={EURASIP Journal on Information Security},
  volume={2020},
  number={1},
  pages={1--12},
  year={2020},
  publisher={SpringerOpen}
}
```