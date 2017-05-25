"""
compute simlarity of images between all neurosynth datasets
"""

import os,pickle
import numpy,pandas
from joblib import Parallel, delayed
from sklearn.metrics import f1_score,jaccard_similarity_score

njobs=24

data=pickle.load(open('../data/neurosynth/neurosynth_reduced_cleaned.pkl','rb'))
coords=[]
for i in range(data.shape[0]):
    for j in range(i,data.shape[0]):
        if i==j:
            continue
        coords.append((i,j))

data=(data>0).astype('int')
results=Parallel(n_jobs=njobs)(delayed(jaccard_similarity_score)(data[i,:],data[j,:]) for i,j in coords)

pickle.dump(results,open('../data/neurosynth/ns_image_similarity_results.pkl','wb'))
