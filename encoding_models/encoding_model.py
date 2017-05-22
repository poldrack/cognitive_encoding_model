"""
class definition for encoding models
"""

import os,datetime,sys
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score
from joblib import Parallel, delayed

import random
import socket

import pandas,numpy
import pickle



def get_timestamp():
    return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())


class EncodingModel:
    def __init__(self,desmtx_file,flag,verbose=True,n_jobs=-1,minstudies=0,
                 shuffle=False,prototype=False,datadir=None,outdir=None,
                 binarize=True,n_folds=4,n_Cs=16,solver='lbfgs',
                 penalty='l2',c_minexp=-4,c_maxexp=2):
        self.desmtx_file=desmtx_file
        assert os.path.exists(self.desmtx_file)
        self.flag=flag
        assert len(self.flag)>0
        self.verbose=verbose
        self.binarize=binarize
        self.n_jobs=n_jobs
        self.minstudies=minstudies
        self.shuffle=shuffle
        self.n_folds=n_folds
        self.prototype=prototype
        self.timestamp=get_timestamp()
        self.hash='%08x'%random.getrandbits(32)
        self.n_Cs=n_Cs
        self.c_minexp=c_minexp
        self.c_maxexp=c_maxexp
        self.solver=solver
        self.penalty=penalty
        if datadir is None:
            self.datadir='../data'
        else:
            self.datadir=datadir
        if outdir is None:
            self.outdir='../models/encoding/%s'%self.flag
        else:
            self.outdir=outdir
        if not os.path.exists(self.outdir):
            if self.verbose:
                print('making output directory:',self.outdir)
            os.makedirs(self.outdir)
        self.logfile=os.path.join(self.outdir,'%s_log_%s'%(self.flag,self.hash))
        self.log_info('%s: start'%get_timestamp())
        for v in vars(self):
            self.log_info('%s:%s'%(v,str(vars(self)[v])))
        self.log_info('Hostname:'+socket.gethostbyaddr(socket.gethostname())[0])
        self.data=None
        self.desmtx=None



    def log_info(self,info):
        # write info to log file for run
        with open(self.logfile,'a') as f:
            f.write(info+'\n')

    def load_data(self,infile='neurosynth/neurosynth_reduced_cleaned.pkl'):
        self.data=pickle.load(open(os.path.join(self.datadir,infile),'rb'))
        self.log_info('loaded datafile: %s'%os.path.join(self.datadir,infile))

    def load_desmtx(self):
        if os.path.basename(self.desmtx_file).split('.')[-1]=='csv':
            self.desmtx=pandas.read_csv(self.desmtx_file,index_col=0)
        self.log_info('loaded desmtx:%s'%self.desmtx_file)

    def clean_data_and_design(self):
        if self.data is None:
            self.load_data()

        if self.desmtx is None:
            self.load_desmtx()
        # remove studies with no signals in ROIs from dataset and design

        s=numpy.sum(self.data,1)
        self.data=self.data[s>0,:]
        self.desmtx=self.desmtx.ix[s>0]

        # remove desmtx columns with too few observations
        if self.minstudies>0:
            self.desmtx=self.desmtx.ix[:,self.desmtx.sum(0)>self.minstudies]

        #dupes=desmtx.duplicated()
        #desmtx=desmtx.ix[dupes==False]
        #data=data[dupes.values==False,:]
        if self.binarize:
            if self.verbose:
                print('binarizing data')
            self.data=(self.data>0).astype('int')

        if self.verbose:
            print('found %d good datasets'%self.data.shape[0])


    def estimate_model(self,save_results=True):
        if self.verbose:
            print('estimating logistic regression model')
        models={}

        output=[]

        p=numpy.zeros(self.data.shape)
        skf = KFold(n_splits=self.n_folds,shuffle=True)


        if self.prototype:
            print('using prototype, only 1 variable')
            nvars=1
        else:
            nvars=self.data.shape[1]

        if self.shuffle:
            if self.verbose:
                print('shuffling data')
            shufflag='_shuffle'
        else:
            shufflag=''

        Cs=numpy.logspace(self.c_minexp,self.c_maxexp,self.n_Cs)
        testsplits=[]

        # shuffles along first axis
        if self.shuffle==True:
            numpy.random.shuffle(self.data)

        for train, test in skf.split(self.desmtx):
            testsplits.append(test)
            for i in range(nvars):
                traindata=self.data[train,i].copy()
                testdata=self.data[test,i].copy()
                Xtrain=self.desmtx.iloc[train]
                Xtest=self.desmtx.iloc[test]
                cv=LogisticRegressionCV(Cs=Cs,penalty=self.penalty,
                                        class_weight='balanced',
                                        solver=self.solver,
                                        n_jobs=self.n_jobs)


                cv.fit(Xtrain,traindata)
                p[test,i]=cv.predict(Xtest)
        output.append([i,f1_score(self.data[:,i],p[:,i])])
        if self.verbose:
            print(output[-1])
        if save_results:
            pickle.dump((p,output,testsplits),open(os.path.join(self.outdir,'results_%s_%s.pkl'%(shufflag,self.hash)),'wb'))
        return (p,output,testsplits)

    def estimate_model_nocv(self,save_results=True):
        if self.verbose:
            print('estimating logistic regression model on full dataset (no CV)')

        if self.prototype:
            print('using prototype, only 1 variable')
            nvars=1
        else:
            nvars=self.data.shape[1]

        coefs=numpy.zeros((self.data.shape[1],self.desmtx.shape[1]))
        Cs=numpy.logspace(self.c_minexp,self.c_maxexp,self.n_Cs)

        p=numpy.zeros(self.data.shape)
        penalties=numpy.zeros(nvars)
        for i in range(nvars):
            cv=LogisticRegressionCV(Cs=Cs,penalty=self.penalty,
                                    class_weight='balanced',
                                    solver=self.solver,
                                    n_jobs=self.n_jobs)
            cv.fit(self.desmtx.values,self.data[:,i])
            coefs[i,:]=cv.coef_
            p[:,i]=cv.predict(self.desmtx.values)
            penalties[i]=cv.C_[0]
        if save_results:
            pickle.dump((p,coefs,penalties),open(os.path.join(self.outdir,'fulldata_results_%s.pkl'%self.hash),'wb'))
        return (p,coefs,penalties)
