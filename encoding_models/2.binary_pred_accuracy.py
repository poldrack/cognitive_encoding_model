"""
compute accuracy for pairs of images in each test set

"""

import pickle,sys,os
import pandas,numpy
from sklearn.metrics import f1_score,jaccard_similarity_score
import glob
from joblib import Parallel, delayed

infile=sys.argv[1]
n_jobs=48

assert os.path.exists(infile)
outfile=infile.replace('results','predacc')
assert outfile is not infile

print('will save results to:',outfile)
results=pickle.load(open(infile,'rb'))
# load data and compute dice for each study
desmtx=pandas.read_csv('../data/neurosynth/desmtx_cleaned.csv',index_col=0)
data=pickle.load(open('data/neurosynth/neurosynth_reduced_cleaned.pkl','rb'))

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

print("computing accuracies")
accuracy_list=Parallel(n_jobs=n_jobs)(delayed(test_match)(data[i,:],data[j,:],pred[i,:],pred[j,:]) for i,j in coords)
pickle.dump((coords,accuracy_list),open(outfile,'wb'))
