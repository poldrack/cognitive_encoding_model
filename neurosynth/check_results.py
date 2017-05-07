import pickle
import pandas,numpy
from sklearn.metrics import f1_score

results=pickle.load(open('results/fitted_lrcv_reduced.pkl','rb'))

import glob
shuf_files=glob.glob('results/fitted_lrcv_reduced_shuffle_*.pkl')


for i,f in enumerate(shuf_files):
    _,r,_=pickle.load(open(f,'rb'))
    r_ar=numpy.array(r)
    if i==0:
        simdata=r_ar[:,2]
    else:
        simdata=numpy.vstack((simdata,r_ar[:,2]))

simmax=simdata.max(0)
r_ar=numpy.array(results[1])
print(numpy.sum(r_ar[:,2]>simmax))

# load data and compute dice for each study     
desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
data=pickle.load(open('neurosynth_reduced.pkl','rb'))
s=numpy.sum(data,1)
data=data[s>0,:]
desmtx=desmtx.ix[s>0]
data=(data>0).astype('int')
print('computing f scores')
pred=results[0]
pred_scores=numpy.zeros(data.shape[0])
for i in data.shape[0]:
   pred_scores[i]=f1_score(data[i,:],pred[i,:])
numpy.save('pred_scores.npy',pred_scores)

