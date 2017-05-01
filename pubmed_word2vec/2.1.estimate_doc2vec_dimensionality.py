"""
use crossvalidation to estimate dimensionality
of word2vec model
"""

import pickle,os
import string,re
import gensim.models

import random
import pandas
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

print('using saved text')
doc_td=pickle.load(open('doc_td.pkl','rb'))
docs=list(range(len(doc_td)))

kf = KFold(n_splits=2)
ndims=200

if os.path.exists('doc2vec_trigram_vocab.model'):
    print("using saved vocabulary")
    model_docs=Doc2Vec.load('doc2vec_trigram_vocab.model')
else:
    print('learning vocabulary')
    model_docs=Doc2Vec(dm=1, size=ndims, window=5, negative=0,
            hs=1, min_count=50, workers=22,iter=20,
            alpha=0.025, min_alpha=0.025,dbow_words=1)
    model_docs.build_vocab(doc_td)
    model_docs.save('doc2vec_trigram_vocab.model')

ndims=100

model_docs.vector_size=ndims

for train, test in kf.split(docs):
    train_docs=[doc_td[i] for i in train]
    test_docs=[doc_td[i] for i in test]
    print('learning model')
    for epoch in range(10):
        random.shuffle(train_docs)
        print('training on',model_docs.alpha)
        model_docs.train(train_docs,total_examples=len(train_docs),
                            epochs=model_docs.iter)
        model_docs.alpha-=.002
        model_docs.min_alpha=model_docs.alpha
    asdf
    
