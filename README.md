# Noiseprint: a CNN-based camera model fingerprint
[Noiseprint](https://ieeexplore.ieee.org/document/8713484) is a CNN-based camera model fingerprint
extracted by a fully Convolutional Neural Network (CNN).

## License
Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

## Installation
The code requires Python 3.x and Tensorflow 1.2.1 .

To install Python 3.x for Ubuntu, you can run:

```
apt-get update
apt-get install -y python3.5 python3.5-dev python3-pip python3-venv
```

We recommend to use a virtual environment: 

```
python3.5 -m venv ../venv
source ../venv/bin/activate
pip install --upgrade pip
```

### Installation with GPU
Install Cuda8 and Cudnn5, more informetion on sites:
- https://developer.nvidia.com/cuda-downloads
- https://developer.nvidia.com/cudnn

Then install the requested libraries using:
```
cat noiseprint/requirements-gpu.txt | xargs -n 1 -L 1 pip install
```

### Installation without GPU
Install the requested libraries using:
```
cat noiseprint/requirements-cpu.txt | xargs -n 1 -L 1 pip install
```


## Usage
To extract the noiseprint, run:

```
python main_extraction.py <input image> <output mat/npz file>
```

The noiseprint is saved in a file with extension mat or npz.
To show the saved noiseprint, run:

```
python main_showout.py <input image> <output mat/npz file>
```

While to execute the blind localization method, run:

```
python main_blind.py <input image> <output mat/npz file>
```

The heatmap is saved in a file with extension mat or npz.
To show the result, run:

```
python main_showres.py <input image> <gt image> <output mat/npz file>
```

To convert the heatmap in a png image, run:

```
python main_map2uint8.py <output mat/npz file> <output png file>
```


### Demo
To execute the demo, run the script

```
cd ./demo
./demo_extraction.sh
./demo_heatmap.sh
```

## Reference

```js
@article{Cozzolino2019_Noiseprint,
  title={Noiseprint: A CNN-Based Camera Model Fingerprint},
  author={D. Cozzolino and L. Verdoliva},
  journal={IEEE Transactions on Information Forensics and Security},
  doi={10.1109/TIFS.2019.2916364},
  pages={144-159},
  year={2020},
  volume={15}
} 
```
