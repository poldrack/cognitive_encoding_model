"""
use crossvalidation to estimate dimensionality
of word2vec model
"""

import pickle,os,sys
import string,re
import gensim.models

import random
import pandas,numpy
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

print('using saved text')
doc_td=pickle.load(open('doc_td.pkl','rb'))
docs=list(range(len(doc_td)))

kf = KFold(n_splits=2)

try:
    ndims=int(sys.argv[1])
except:
    ndims=100
    print('using default ndims=',ndims)

print('learning vocabulary')
model_docs=Doc2Vec(dm=1, size=ndims, window=5, negative=0,
        hs=1, min_count=50, workers=22,iter=20,
        alpha=0.025, min_alpha=0.025,dbow_words=1)
model_docs.build_vocab(doc_td)

scores=numpy.zeros(len(doc_td))

for train, test in kf.split(docs):
    train_docs=[doc_td[i] for i in train]
    test_docs=[doc_td[i] for i in test]
    print('learning model')
    for epoch in range(10):
        random.shuffle(train_docs)
        print(ndims,'training on',model_docs.alpha)
        model_docs.train(train_docs,total_examples=len(train_docs),
                            epochs=model_docs.iter)
        model_docs.alpha-=.002
        model_docs.min_alpha=model_docs.alpha
    td=[test_docs[i].words for i in range(len(test_docs))]
    scores[test]=model_docs.score(td)
numpy.save('scores_%d_dims.npy'%ndims,scores)
