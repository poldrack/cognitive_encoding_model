"""
compute accuracy for pairs of images in each test set

"""

import pickle
import pandas,numpy
from sklearn.metrics import f1_score,jaccard_similarity_score
import glob
from joblib import Parallel, delayed

expanded=True

if expanded:
    results=pickle.load(open('results/fitted_lrcv_reduced_expanded.pkl','rb'))
    expflag='_expanded'

else:
    results=pickle.load(open('results/fitted_lrcv_reduced.pkl','rb'))
    expflag=''

# load data and compute dice for each study
desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
data=pickle.load(open('neurosynth_reduced.pkl','rb'))
s=numpy.sum(data,1)
data=data[s>0,:]
desmtx=desmtx.ix[s>0]
data=(data>0).astype('int')

# dupes=desmtx.duplicated()
# desmtx=desmtx.ix[dupes==False]
# data=data[dupes.values==False,:]
#pred=results[0][numpy.where(dupes==False)]
pred=results[0]


def test_match(data1,data2,pred1,pred2,scorer=jaccard_similarity_score):
    f1_d1_p1=scorer(data1,pred1)
    f1_d2_p1=scorer(data2,pred1)
    f1_d1_p2=scorer(data1,pred2)
    f1_d2_p2=scorer(data2,pred2)
    if (f1_d1_p1 > f1_d2_p1) and (f1_d2_p2>f1_d1_p2):
        return 1
    else:
        return 0

# compare all possible combinations of images
print('getting coordinates')
coords=[]
assert len(results[3])==4

for fold in range(len(results[3])):
    test=results[3][fold]
    for i in range(test.shape[0]):
        for j in range(i+1, test.shape[0]):
            coords.append((test[i],test[j]))


#coords=coords[:8]
n_jobs=128
print("computing accuracies")
accuracy_list=Parallel(n_jobs=n_jobs)(delayed(test_match)(data[i,:],data[j,:],pred[i,:],pred[j,:]) for i,j in coords)
pickle.dump((coords,accuracy_list),open('results/pred_accuracy_list%s.pkl'%expflag,'wb'))
