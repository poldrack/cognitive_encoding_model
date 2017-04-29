# estimate encoding model on full dataset
# and test identification accuracy
# baby step towards crossvalidation- if it doesn't work here
# then it certainly won't work on out-of-sample data

import os
import pandas,numpy
import pickle

from sklearn.linear_model import ElasticNet,ElasticNetCV,MultiTaskElasticNetCV

expanded=False
force_new=True

# load data and estimate model on full dataset
data=pickle.load(open('neurosynth_reduced.pkl','rb'))
if expanded:
    desmtx=pandas.read_csv('data/desmtx_expanded.csv',index_col=0)
    outfile='data/fitted_mtencv_reduced_expanded.pkl'
else:
    desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
    outfile='data/fitted_mtencv_reduced.pkl'

if os.path.exists(outfile) and not force_new:
    en=pickle.load(open(outfile,'rb'))
    print('using cached elastic net model')
else:
    print('estimating elastic net model')
    n_jobs=1
    l1_ratio=[.1, .3, .5, .7]
    n_alphas=25
    mtencv=MultiTaskElasticNetCV(n_jobs=n_jobs,l1_ratio=l1_ratio,
                        normalize=True,verbose=1,
                        n_alphas=n_alphas)
    mtencv.fit(desmtx,data)
    pickle.dump(mtencv,open(outfile,'wb'))



# estimate map for each study using forward model
print('estimating maps using forward model')
estimated_maps=numpy.zeros(data.shape)
#for i in range(data.shape[0]):
p=mtencv.predict(desmtx)
numpy.save('pred_mtencv.npy',p)

def find_match(idx,pred,actual):
    """
    find the real image that most closely matches the index image
    ala Kay et al., 2008
    """
    for i in range(actual.shape[0]):
        pass
