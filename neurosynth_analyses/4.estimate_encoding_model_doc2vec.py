# estimate encoding model on full dataset
# using projection into doc2vec space

import os,time,sys
import argparse
import random

import pandas,numpy
import pickle

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score
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

def fit_lr(desmtx,data,solver='lbfgs',
            penalty='l2',
            n_jobs=-1,
            n_Cs=25):
        cv=LogisticRegressionCV(Cs=n_Cs,penalty=penalty,
                                class_weight='balanced',
                                solver=solver)
        cv.fit(desmtx,data)
        return cv

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--verbose", help="increase output verbosity",
                        default=0, action='count')
    parser.add_argument("--solver", help="solver for LR",
                        default='lbfgs')
    parser.add_argument('-j',"--n_jobs", help="number of processors",type=int,
                            default=-1)
    parser.add_argument("--n_folds", help="number of CV folds",type=int,
                            default=4)
    parser.add_argument("--n_dims", help="number of model dims",type=int,
                            default=50)
    parser.add_argument("--n_Cs", help="number of C values",type=int,
                            default=25)
    parser.add_argument("--penalty", help="penalty type for LR",
                        default='l2')
    parser.add_argument('-p',"--prototype", help="use limited number of cycles for prototyping",
                        action='store_true')
    parser.add_argument('-f',"--force_true", help="force recomputation",
                        action='store_true')
    parser.add_argument('-s',"--shuffle", help="shuffle target variable",
                        action='store_true')
    args=parser.parse_args()

    if args.verbose:
        print('ARGS:',args)

    if args.shuffle:
        shuf_flag='_shuffle_%s'%'%08x'%random.getrandbits(32)
        if args.verbose:
            print('SHUFFLING DATA')
    else:
        shuf_flag=''
    ndims=args.n_dims
    # load data and estimate model on full dataset
    data=pickle.load(open('data/neurosynth_reduced_cleaned.pkl','rb'))
    desmtx=pandas.read_csv('data/ns_doc2vec_%ddims_projection.csv'%ndims,index_col=0)
    outfile='results/fitted_lrcv_doc2vec%s.pkl'%shuf_flag

    # make sure output directory exists
    if os.path.dirname(outfile):
        if not os.path.exists(os.path.dirname(outfile)):
            os.mkdir(os.path.dirname(outfile))

    if args.verbose:
        print('binarizing data')
    data=(data>0).astype('int')

    if args.verbose:
        print('estimating logistic regression model')
    models={}
    output=[]
    p=numpy.zeros(data.shape)
    t=time.time()
    skf = KFold(n_splits=args.n_folds,shuffle=True)


    if args.prototype:
        print('using prototype, only 1 variable')
        nvars=1
    else:
        nvars=data.shape[1]

    testsplits=[]
    for train, test in skf.split(desmtx):
        testsplits.append(test)
        for i in range(nvars):
            traindata=data[train,i].copy()
            testdata=data[test,i].copy()
            Xtrain=desmtx.iloc[train]
            Xtest=desmtx.iloc[test]
            cv=LogisticRegressionCV(Cs=args.n_Cs,penalty=args.penalty,
                                            class_weight='balanced',
                                            solver=args.solver,
                                            n_jobs=args.n_jobs)
            if args.shuffle==True:
                numpy.random.shuffle(traindata)
            cv.fit(Xtrain,traindata)
            p[test,i]=cv.predict(Xtest)
        output.append([i,cv.C_[0],f1_score(data[:,i],p[:,i])])
        if args.verbose:
            print(output[-1])
            print('time:',time.time() - t)
        t=time.time()
    pickle.dump((p,output,args,testsplits),open(outfile,'wb'))
