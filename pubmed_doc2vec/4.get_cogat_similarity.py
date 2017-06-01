"""
compute similarity between all cogat terms using doc2vec model
"""

import pandas,numpy
from gensim.models.doc2vec import Doc2Vec,TaggedDocument


m=Doc2Vec.load('../models/doc2vec/doc2vec_trigram_300dims.model')

desmtx=pandas.read_csv('../data/neurosynth/desmtx.csv',index_col=0)

concept_similarity=numpy.zeros((desmtx.shape[1],desmtx.shape[1]))

for i in range(desmtx.shape[1]):
    for j in range(i+1,desmtx.shape[1]):
        concepts=[]
        for c in [i,j]:
            concepts.append(desmtx.columns[c].replace(' ','_'))
        concept_similarity[i,j]=m.wv.similarity(concepts[0],concepts[1])
if not os.path.exists('../data/doc2vec'):
    os.mkdir('../data/doc2vec')
numpy.save(concept_similarity,'../data/doc2vec/cogat_doc2vec_similarity.npy')
