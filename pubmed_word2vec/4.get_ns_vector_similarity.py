"""
compute simlarity of inferred vectors between all neurosynth datasets
"""

import os
import numpy,pandas
from joblib import Parallel, delayed

def vector_corr(i,data):
    """
    find the real image that most closely matches the index image
    ala Kay et al., 2008
    """
    corr_all=numpy.zeros(data.shape[0])
    for j in range(data.shape[0]):
        corr_all[j]=numpy.corrcoef(data[i,:],data[j,:])[0,1]
    return corr_all

data=numpy.load('ns_inferred_vectors.npy')
results=Parallel(n_jobs=20)(delayed(vector_corr)(i,data) for i in range(data.shape[0]))

pickle.dump(results,open('ns_vector_similarity_results.pkl','wb'))
