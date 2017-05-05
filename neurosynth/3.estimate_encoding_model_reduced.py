# estimate encoding model on full dataset
# and test identification accuracy
# baby step towards crossvalidation- if it doesn't work here
# then it certainly won't work on out-of-sample data

import os
import pandas,numpy
import pickle

from sklearn.linear_model import ElasticNet,ElasticNetCV,RidgeCV,LassoCV
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

def fit_en(idx,desmtx,data,model='ridgecv',
            n_jobs=-1,
            l1_ratio=[.1, .3, .5, .7],
            ridge_alphas=[0.5,1.0,2.5,5.0,7.5,10.0,12.5,15.,17.5,20.0,22.5,25.],
            n_alphas=25):
        assert model in ['ridgecv','encv']
        if model=='encv':
            cv=ElasticNetCV(n_jobs=n_jobs,l1_ratio=l1_ratio,
                            normalize=True,verbose=1,
                            n_alphas=n_alphas)
        elif model=='ridgecv':
            cv=RidgeCV(normalize=True,alphas=ridge_alphas)
        elif model=='lasso':
            cv.LassoCV(n_alphas=25)
        cv.fit(desmtx,data[:,idx])
        return cv

if __name__=="__main__":
    expanded=True
    force_new=True
    model='ridgecv'

    # load data and estimate model on full dataset
    data=pickle.load(open('neurosynth_reduced.pkl','rb'))
    if expanded:
        desmtx=pandas.read_csv('data/desmtx_expanded.csv',index_col=0)
        outfile='data/fitted_mtencv_reduced_expanded.pkl'
        expflag='_expanded'
    else:
        desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
        outfile='data/fitted_mtencv_reduced.pkl'
        expflag=''

    # remove datasets with no signals in ROIs

    s=numpy.sum(data,1)
    data=data[s>0,:]
    desmtx=desmtx.ix[s>0]

    dupes=desmtx.duplicated()
    desmtx=desmtx.ix[dupes==False]
    data=data[dupes.values==False,:]

    if os.path.exists(outfile) and not force_new:
        print('using cached elastic net model')
        models=pickle.load(open(outfile,'rb'))
        print('loading predictions')
        p=numpy.load('data/pred_encv%s.npy'%expflag)
    else:
        print('estimating elastic net model')
        models={}
        output=[]
        p=numpy.zeros(data.shape)
        for i in range(data.shape[1]):

            models[i]=fit_en(i,desmtx,data,model=model)
            p[:,i]=models[i].predict(desmtx)
            output.append([i,models[i].alpha_,numpy.corrcoef(data[:,i],p[:,i])[0,1]])
            print(output[-1])
        pickle.dump(models,open(outfile,'wb'))

        numpy.save('data/pred_encv%s_%s.npy'%(expflag,model),p)

    # estimate map for each study using forward model

    asdf

    # assess similarity of predicted to actual
    results=Parallel(n_jobs=20)(delayed(find_match)(i,p,data) for i in range(p.shape[0]))

    pickle.dump(results,open('similarity_results%s.pkl'%expflag,'wb'))
