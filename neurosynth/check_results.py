import pickle
import pandas,numpy
from sklearn.metrics import f1_score
import glob
from joblib import Parallel, delayed

results=pickle.load(open('results/fitted_lrcv_reduced.pkl','rb'))

shuf_files=glob.glob('results/fitted_lrcv_reduced_shuffle_*.pkl')

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
for i in range(data.shape[0]):
   pred_scores[i]=f1_score(data[i,:],pred[i,:])
numpy.save('pred_scores.npy',pred_scores)

pred_scores_shuf=numpy.zeros((data.shape[0],len(shuf_files)))
print('computing f scores for shuffled data (slow!)')

for i,f in enumerate(shuf_files):
    p,r,_=pickle.load(open(f,'rb'))
    pred_scores_shuf[:,i]=Parallel(n_jobs=20)(delayed(f1_score)(data[i,:],p[i,:]) for i in range(data.shape[0]))

numpy.save('pred_scores_shuf.npy',pred_scores_shuf)
