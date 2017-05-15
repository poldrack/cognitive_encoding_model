#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make single file with cleaned abstracts
for mallet

Created on Sat May 13 16:43:35 2017

@author: poldrack
"""

import pickle

infile='/Users/poldrack/code/cognitive_encoding_model/pubmed_word2vec/doc_td.pkl'
a=pickle.load(open(infile,'rb'))
with open('/Users/poldrack/code/cognitive_encoding_model/pubmed_word2vec/psych_pubmed_raw.txt','w') as f:
    for abs in a:
        f.write('%s EN %s\n'%(abs.tags[0],' '.join(abs.words)))
