# estimate encoding model on full dataset
# and test identification accuracy
# baby step towards crossvalidation- if it doesn't work here
# then it certainly won't work on out-of-sample data

import os
import pandas,numpy
import pickle

from sklearn.linear_model import ElasticNet,ElasticNetCV,MultiTaskElasticNetCV
from joblib import Parallel, delayed

def find_match(i,p,data):
    """
    find the real image that most closely matches the index image
    ala Kay et al., 2008
    """
    corr_all=numpy.zeros(p.shape[0])
    for j in range(p.shape[0]):
        corr_all[j]=numpy.corrcoef(p[i,:],data[j,:])[0,1]
    corr_true=corr_all[i]
    corr_max=numpy.nanmax(corr_all)
    if corr_true==corr_max:
        print('success!')
        success=1
    else:
        success=0
    corr_rank=numpy.nanmean(corr_true>corr_all)
    return ([corr_true,corr_max,corr_rank,success])


if __name__=="__main__":
    expanded=False
    force_new=False

    # load data and estimate model on full dataset
    data=pickle.load(open('neurosynth_reduced.pkl','rb'))
    if expanded:
        desmtx=pandas.read_csv('data/desmtx_expanded.csv',index_col=0)
        outfile='data/fitted_mtencv_reduced_expanded.pkl'
    else:
        desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
        outfile='data/fitted_mtencv_reduced.pkl'

    # remove datasets with no signals in ROIs

    s=numpy.sum(data,1)
    data=data[s>0,:]
    desmtx=desmtx.ix[s>0]

    dupes=desmtx.duplicated()
    desmtx=desmtx.ix[dupes==False]
    data=data[dupes.values==False,:]

    if os.path.exists(outfile) and not force_new:
        mtencv=pickle.load(open(outfile,'rb'))
        print('using cached elastic net model')
    else:
        print('estimating elastic net model')
        n_jobs=20
        l1_ratio=[.1, .3, .5, .7]
        n_alphas=25
        mtencv=MultiTaskElasticNetCV(n_jobs=n_jobs,l1_ratio=l1_ratio,
                            normalize=True,verbose=1,
                            n_alphas=n_alphas)
        mtencv.fit(desmtx,data)
        pickle.dump(mtencv,open(outfile,'wb'))



    # estimate map for each study using forward model

    if os.path.exists('pred_mtencv.npy'):
        print('loading predictions')
        p=numpy.load('pred_mtencv.npy')
    else:
        print('estimating predictions using forward model')
        p=mtencv.predict(desmtx)
        numpy.save('pred_mtencv.npy',p)

    # assess similarity of predicted to actual
    results=Parallel(n_jobs=20)(delayed(find_match)(i,p,data) for i in range(p.shape[0]))

    pickle.dump(results,open('similarity_results.pkl','wb'))
