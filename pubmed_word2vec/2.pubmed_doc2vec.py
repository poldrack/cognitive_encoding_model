"""
create two-dimensional embedding for pubmed abstracts
identified from psychology journals using 1.get_articles.py
"""

import pickle,os
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

from utils import text_cleanup, get_journals

if not os.path.exists('models'):
    os.mkdir('models')

print('using saved text')
doc_td=pickle.load(open('doc_td.pkl','rb'))

# fit model

ndims=300
force_new=True


if os.path.exists('model.txt'):
    os.remove('model.txt')

if os.path.exists('models/doc2vec_trigram_%ddims.model'%ndims) and not force_new:
    print('using saved model')
    model_docs=Doc2Vec.load('models/doc2vec_trigram_%ddims.model'%ndims)
else:
    if os.path.exists('models/doc2vec_trigram_%ddims_vocab.model'%ndims) and not force_new:
        print("using saved vocabulary")
        model_docs=Doc2Vec.load('models/doc2vec_trigram_%ddims_vocab.model'%ndims)
    else:
        print('learning vocabulary')
        model_docs=Doc2Vec(dm=0, size=ndims, window=15, negative=5,
                hs=0, min_count=5, workers=46,iter=100,sample=1e-5,
                alpha=0.025, min_alpha=0.025,dbow_words=1)
        model_docs.build_vocab(doc_td)
        model_docs.save('models/doc2vec_trigram_%ddims_vocab.model'%ndims)
    print('learning model')
    for epoch in range(10):
        random.shuffle(doc_td)
        print('training on',model_docs.alpha)
        model_docs.train(doc_td,total_examples=model_docs.corpus_count,
                            epochs=model_docs.iter)
        model_docs.alpha-=.002
        model_docs.min_alpha=model_docs.alpha
        model_docs.save('models/doc2vec_trigram_%ddims.model'%ndims)
        with open('model.txt','a') as f:
            f.write('%f\n'%model_docs.alpha)
