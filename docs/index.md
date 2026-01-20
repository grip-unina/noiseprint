---
layout: paper
paper: "Noiseprint: a CNN-based camera model fingerprint"
github_url: https://github.com/grip-unina/noiseprint
authors:  
  - name: Davide Cozzolino
    link: https://www.grip.unina.it/members/cozzolino
    index: 1
  - name: Luisa Verdoliva
    link: https://www.grip.unina.it/members/verdoliva
    index: 1
affiliations: 
  - name: University Federico II of Naples, Italy
    index: 1
links:
    paper: https://ieeexplore.ieee.org/document/8713484
    arxiv: https://arxiv.org/abs/1808.08396
    code: https://github.com/grip-unina/noiseprint
---

<center><img src="./header.gif" alt="header" width="50%" /></center>

Forensic analyses of digital images rely heavily on the traces of in-camera and out-camera processes left on the acquired images.
Such traces represent a sort of camera fingerprint.
If one is able to recover them, by suppressing the high-level scene content and other disturbances, a number of forensic tasks
can be easily accomplished. A notable example is the PRNU pattern, which can be regarded as a device fingerprint, and has received great attention in multimedia forensics.
In this paper we propose a method to extract a camera model fingerprint, called noiseprint, where the scene content is largely suppressed and model-related artifacts are enhanced.
This is obtained by means of a Siamese network, which is trained with pairs of image patches coming from the same (label +1) or different (label −1) cameras.
Although noiseprints can be used for a large variety of forensic tasks, here we focus on image forgery localization.
Experiments on several datasets widespread in the forensic community show noiseprint-based methods to provide state-of-the-art performance.

## News

*   2019-05-13: Paper was published in IEEE Transactions on Information Forensics and Security.


## Bibtex

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
