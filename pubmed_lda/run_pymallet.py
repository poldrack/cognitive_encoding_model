#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymallet.py - main class definition

Created on Sat May 13 13:50:13 2017

@author: poldrack
"""

import os,sys,random,shutil,glob
import subprocess
from sklearn.model_selection import KFold
import numpy
from pymallet.pymallet import PyMallet

ndims=int(sys.argv[1])
if 1:
    # commands to run smoke test
    pm=PyMallet('/work/01329/poldrack/wrangler/software_wrangler/mallet-2.0.8/',verbose=False)
    pm.import_data(infile='/work/01329/poldrack/wrangler/code/cognitive_encoding_model/data/pubmed/psych_pubmed_raw.txt')
    pm.train_topics(ndims,pm.datafile,'../models/lda')
    
    
