"""
compute simlarity of inferred vectors between all neurosynth datasets
"""

import os,pickle
import numpy,pandas
from joblib import Parallel, delayed,load,dump

def vector_corr(i,data):
    """
    find the real image that most closely matches the index image
    ala Kay et al., 2008
    """
    corr_all=numpy.zeros(data.shape[0])
    for j in range(i,data.shape[0]):
        corr_all[j]=numpy.corrcoef(data[i,:],data[j,:])[0,1]
    return corr_all

data=pandas.read_csv('../data/neurosynth/ns_doc2vec_300dims_projection.csv',index_col=0)
data=data.values

testmode=False

if testmode:
    data=data[:5,:]

dump(data,'ns_vectors.mm')

data = load('ns_vectors.mm', mmap_mode='r')

results=Parallel(n_jobs=24)(delayed(vector_corr)(i,data) for i in range(data.shape[0]))

pickle.dump(results,open('../data/neurosynth/ns_vector_similarity_results.pkl','wb'))

os.remove('ns_vectors.mm')
