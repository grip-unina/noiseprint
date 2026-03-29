import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "noiseprint",
    version = "1.0.0",
    author = "Davide Cozzolino, Luisa Verdoliva",
    description = ("Forensic analyses of digital images rely heavily on the traces of in-camera and out-camera processes left on the acquired images. Such traces represent a sort of camera finger- print. If one is able to recover them, by suppressing the high-level scene content and other disturbances, a number of forensic tasks can be easily accomplished. A notable example is the PRNU pattern, which can be regarded as a device fingerprint, and has received great attention in multimedia forensics. In this paper, we propose a method to extract a camera model fingerprint, called noiseprint, where the scene content is largely suppressed and model-related artifacts are enhanced."),
    license = "Custom",
    keywords = "camera forensics fingerprint",
    url = "https://github.com/grip-unina/noiseprint",
    packages=['noiseprint', 'noiseprint/feat_spam', 'noiseprint/nets', 'noiseprint/utility'],
    long_description=read('README.md')
)