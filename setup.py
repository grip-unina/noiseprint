import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "binghamton-camerafp",
    version = "1.0.0",
    author = "M. Goljan",
    author_email = "mgoljan@binghamton.edu",
    description = ("Python3 implementation of digital camera fingerprint extraction (maximum likelihood formula), removal of non-unique artifacts (NUA) from fingerprints, extraction of noise residual from images, and calculation of Peak-correlation-to-correlation-ratio (PCE) detection statistic is shared. "),
    license = "Creative Commons Attribution-NonCommercial 4.0 International License",
    keywords = "camera forensics fingerprint",
    url = "http://dde.binghamton.edu/download/camera_fingerprint/",
    packages=['camerafp'],
    long_description=read('README.md')
)