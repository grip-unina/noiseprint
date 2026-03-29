# Noiseprint: a CNN-based camera model fingerprint

[Noiseprint](https://ieeexplore.ieee.org/document/8713484) is a CNN-based camera model fingerprint
extracted by a fully Convolutional Neural Network (CNN).

## License

Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt (included in this package).

## Installation

### Windows, Linux

The code requires Python >= 3.8 and Tensorflow >= 2.6.

To install Python 3.8 for Ubuntu, you can run:

```bash
apt-get update
apt-get install -y python3.8 python3.8-dev python3-pip python3-venv
```

We recommend to use a virtual environment. With `venv`, you can create it with:

```bash
python3 -m venv ../venv
source ../venv/bin/activate
pip install --upgrade pip
```

### Installation with GPU, NVIDIA-CUDA

Install Cuda8 and Cudnn5, more informetion on sites:

- https://developer.nvidia.com/cuda-downloads
- https://developer.nvidia.com/cudnn

Then install the requested libraries using:

```bash
cat noiseprint/requirements-gpu.txt | xargs -n 1 -L 1 pip install
```

### Installation without GPU

Install the requested libraries using:

```bash
cat noiseprint/requirements-cpu.txt | xargs -n 1 -L 1 pip install
```

## MacOS (Apple Silicon)

Noiseprint also works on ARM-based Apple Silicon devices. We recommend using `conda` and the provided requirements file. First, download the official Metal PluggableDevice from Apple:

```bash
https://developer.apple.com/metal/tensorflow-plugin/
```

Then install the required libraries:

```bash
conda create --name noiseprint --file requirements_conda_m1.txt
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

These two commands will create a conda environment named `noiseprint`, then install the required libraries and finally download from PyPI `tensorflow-macos` and `tensorflow-metal`.

An additional `requirements-m1.txt` file is provided for those who wish not to use `conda`.

Finally, macOS behaviour on Intel devices is currently untested. Any feedback is welcome.

## Usage

To extract the noiseprint, run:

```bash
python main_extraction.py <input image> <output mat/npz file>
```

The noiseprint is saved in a file with extension mat or npz.
To show the saved noiseprint, run:

```bash
python main_showout.py <input image> <output mat/npz file>
```

While to execute the blind localization method, run:

```bash
python main_blind.py <input image> <output mat/npz file>
```

The heatmap is saved in a file with extension mat or npz.
To show the result, run:

```bash
python main_showres.py <input image> <gt image> <output mat/npz file>
```

To convert the heatmap in a png image, run:

```bash
python main_map2uint8.py <output mat/npz file> <output png file>
```

### Demo

To execute the demo, run the script

```bash
cd ./demo
./demo_extraction.sh
./demo_heatmap.sh
```

## Reference

```bibtex
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
