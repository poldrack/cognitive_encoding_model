#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make single file with cleaned abstracts
for mallet

Created on Sat May 13 16:43:35 2017

@author: poldrack
"""

import pickle

infile='../data/neurosynth/ns_abstracts_cleaned.pkl'
a=pickle.load(open(infile,'rb'))
with open('../data/neurosynth/neurosynth_abstracts_raw.txt','w') as f:
    for abs in a.keys():
        f.write('%d EN %s\n'%(abs,' '.join(a[abs])))
