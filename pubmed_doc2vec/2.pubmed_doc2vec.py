"""
create two-dimensional embedding for pubmed abstracts
identified from psychology journals using 1.get_articles.py
"""

import pickle,os,sys
import string,re
import gensim.models
import collections

import random
import numpy,pandas
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.doc2vec import TaggedDocument
from nltk.stem import WordNetLemmatizer
import nltk

from joblib import Parallel, delayed
sys.path.insert(0,'../utils')
from utils import text_cleanup, get_journals
datadir='../data/pubmed'
modeldir='../models/doc2vec'
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
use_dbow=False
if use_dbow:
    modelname='_pvdbow'
    dm=0
else:
    modelname='_pvdm'
    dm=1
doc_td=pickle.load(open('%s/doc_td.pkl'%datadir,'rb'))

# fit model

ndims=300
force_new=False


if os.path.exists('%s/model.txt'%modeldir):
    os.remove('%s/model.txt'%modeldir)

if os.path.exists('%s/doc2vec%s_trigram_%ddims.model'%(modeldir,modelname,ndims)) and not force_new:
    print('using saved model')
    model_docs=Doc2Vec.load('%s/doc2vec%s_trigram_%ddims.model'%(modeldir,modelname,ndims))
else:
    if os.path.exists('%s/doc2vec%s_trigram_%ddims_vocab.model'%(modeldir,modelname,ndims)) and not force_new:
        print("using saved vocabulary")
        model_docs=Doc2Vec.load('%s/doc2vec%s_trigram_%ddims_vocab.model'%(modeldir,modelname,ndims))
    else:
        print('learning vocabulary')
        model_docs=Doc2Vec(dm=dm, size=ndims, window=15, negative=5,
                hs=0, min_count=5, workers=14,iter=100,sample=1e-5,
                alpha=0.025, min_alpha=0.025,dbow_words=1)
        model_docs.build_vocab(doc_td)
        model_docs.save('%s/doc2vec%s_trigram_%ddims_vocab.model'%(modeldir,modelname,ndims))
    print('learning model')
    for epoch in range(10):
        random.shuffle(doc_td)
        print('training on',model_docs.alpha)
        model_docs.train(doc_td,total_examples=model_docs.corpus_count,
                            epochs=model_docs.iter)
        model_docs.alpha-=.002
        model_docs.min_alpha=model_docs.alpha
        model_docs.save('%s/doc2vec%s_trigram_%ddims.model'%(modeldir,modelname,ndims))
        with open('%s/model.txt'%modeldir,'a') as f:
            f.write('%f\n'%model_docs.alpha)
