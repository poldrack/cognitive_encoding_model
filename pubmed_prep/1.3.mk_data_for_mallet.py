#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make single file with cleaned abstracts
for mallet

Created on Sat May 13 16:43:35 2017

@author: poldrack
"""

import pickle

infile='../data/pubmed/doc_td.pkl'
a=pickle.load(open(infile,'rb'))
with open('../data/pubmed/psych_pubmed_raw.txt','w') as f:
    for abs in a:
        f.write('%s EN %s\n'%(abs.tags[0],' '.join(abs.words)))
