"""
compute simlarity of images between all neurosynth datasets
"""

import os,pickle
import numpy,pandas
from joblib import Parallel, delayed,dump,load
from sklearn.metrics import f1_score,jaccard_similarity_score

njobs=24

data=pickle.load(open('../data/neurosynth/neurosynth_reduced_cleaned.pkl','rb'))
data=(data>0).astype('int')
dump(data,'data.mm')
data = load('data.mm', mmap_mode='r')

coords=[]
for i in range(data.shape[0]):
    for j in range(i,data.shape[0]):
        if i==j:
            continue
        coords.append((i,j))

def get_similarity(data,c):
    i,j=c
    return jaccard_similarity_score(data[i,:],data[j,:])

results=Parallel(n_jobs=njobs)(delayed(get_similarity)(data,c) for  in coords)

pickle.dump(results,open('../data/neurosynth/ns_image_similarity_results.pkl','wb'))
os.remove('data.mm')
