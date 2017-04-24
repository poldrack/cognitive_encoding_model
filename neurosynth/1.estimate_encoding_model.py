# setup database for encoding model

import os
import tarfile
import pandas,numpy
import pickle
import time
import nibabel
import glob

from Bio import Entrez
import nilearn.image
import nilearn.input_data
from sklearn.linear_model import ElasticNet

data=numpy.load('data/imgdata.npy')
desmtx=pandas.read_csv('data/desmtx.csv')
en=ElasticNet()
en.fit(data,desmtx)
