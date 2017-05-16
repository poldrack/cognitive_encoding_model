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

print('using saved model')
model_docs=Doc2Vec.load('models/doc2vec_trigram_%ddims.model'%ndims)

# check the model
# from https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
n_to_check=1000
print('checking model performance for %d random selected abstracts'%n_to_check)
ranks = []

def get_ranks(doc,model_docs):
    inferred_vector = model_docs.infer_vector(doc.words)
    sims = model_docs.docvecs.most_similar([inferred_vector]) #, topn=len(model_docs.docvecs))
    return [doc.tags[0],int(sims[0][0]==doc.tags[0]),sims[0][1]]

results=[]
docs_to_check=numpy.random.randint(0,len(doc_td),n_to_check)
for i in docs_to_check:
    results.append(get_ranks(doc_td[i],model_docs))
    print(i,results[-1])
pickle.dump(results,open('model_check_results.pkl','wb'))
