# estimate encoding model on full dataset
# here we binarize (given the zero-inflated nature of the data)
# and use logistic regression
# save test splits for use in later quantification of accuracy

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
    parser.add_argument("--n_Cs", help="number of C values",type=int,
                            default=25)
    parser.add_argument("--penalty", help="penalty type for LR",
                        default='l2')
    parser.add_argument("--getsim", help="compute similarity (slow)",
                        action='store_true')
    parser.add_argument('-p',"--prototype", help="use limited number of cycles for prototyping",
                        action='store_true')
    parser.add_argument('-e',"--expanded", help="use expanded design matrix",
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

    # load data and estimate model on full dataset
    data=pickle.load(open('neurosynth_reduced.pkl','rb'))
    if args.expanded:
        desmtx=pandas.read_csv('data/desmtx_expanded.csv',index_col=0)
        expflag='_expanded'
    else:
        desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
        expflag=''
    outfile='results/fitted_lrcv_reduced%s%s.pkl'%(expflag,shuf_flag)

    # make sure output directory exists
    if os.path.dirname(outfile):
        if not os.path.exists(os.path.dirname(outfile)):
            os.mkdir(os.path.dirname(outfile))

    # remove datasets with no signals in ROIs

    s=numpy.sum(data,1)
    data=data[s>0,:]
    desmtx=desmtx.ix[s>0]

    #dupes=desmtx.duplicated()
    #desmtx=desmtx.ix[dupes==False]
    #data=data[dupes.values==False,:]
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

    for train, test in skf.split(desmtx):
        testsplits=[]
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

    if args.getsim:
        # assess similarity of predicted to actual
        results=Parallel(n_jobs=args.n_jobs)(delayed(find_match)(i,p,data) for i in range(p.shape[0]))

        pickle.dump(results,open('results/similarity_results%s%s.pkl'%(expflag,shuf_flag),'wb'))
