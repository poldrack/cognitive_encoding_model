# estimate encoding model on full dataset
# here we binarize (given the zero-inflated nature of the data)
# and use logistic regression
# save test splits for use in later quantification of accuracy
# this version is generalized across all the different encoding models

import os,datetime,sys
import argparse
import random
from inspect import signature

import pandas,numpy
import pickle

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold,StratifiedKFold

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

def get_timestamp():
    return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())


                                                
class EncodingModel:
    def __init__(self,desmtx_file,flag,verbose=True,n_jobs=-1,minstudies=0,
                 shuffle=False,prototype=False,datadir=None,outdir=None,
                 binarize=True,n_folds=4,n_Cs=25,solver='lbfgs',
                 penalty='l2'):
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
        self.data=None
        self.desmtx=None
        
    
        
    def log_info(self,info):
        # write info to log file for run
        with open(self.logfile,'a') as f:
            f.write(info+'\n')
        
    def load_data(self,infile='neurosynth/neurosynth_reduced.pkl'):
        self.data=pickle.load(open(os.path.join(self.datadir,infile),'rb'))
        self.log_info('datafile: %s'%os.path.join(self.datadir,infile))
               
    def load_desmtx(self):
        if os.path.basename(self.desmtx_file).split('.')[-1]=='csv':
            self.desmtx=pandas.read_csv(self.desmtx_file,index_col=0)
        self.log_info('desmtx:%s'%self.desmtx_file)

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
            
        testsplits=[]
        for train, test in skf.split(self.desmtx):
            testsplits.append(test)
            for i in range(nvars):
                traindata=self.data[train,i].copy()
                testdata=self.data[test,i].copy()
                Xtrain=self.desmtx.iloc[train]
                Xtest=self.desmtx.iloc[test]
                cv=LogisticRegressionCV(Cs=self.n_Cs,penalty=self.penalty,
                                        class_weight='balanced',
                                        solver=self.solver,
                                        n_jobs=self.n_jobs)
                if self.shuffle==True:
                    # shuffle training data 
                    numpy.random.shuffle(traindata)
                cv.fit(Xtrain,traindata)
                p[test,i]=cv.predict(Xtest)
        output.append([i,f1_score(self.data[:,i],p[:,i])])
        if self.verbose:
            print(output[-1])
        if save_results:
            pickle.dump((p,output,testsplits),open(os.path.join(self.outdir,'results_%s_%s_%s.pkl'%(shufflag,self.hash,self.timestamp)),'wb'))
        return (p,output,testsplits)
        
if __name__=="__main__":

    en=EncodingModel('../data/neurosynth/desmtx.csv','test',
                     prototype=True,n_jobs=1,shuffle=True)
    en.load_data()
    en.load_desmtx()
    en.clean_data_and_design()
    results=en.estimate_model()

if __name__=="__foo__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--verbose", help="increase output verbosity",
                        default=0, action='count')
    parser.add_argument("--solver", help="solver for LR",
                        default='lbfgs')
    parser.add_argument('-j',"--n_jobs", help="number of processors",type=int,
                            default=-1)
    parser.add_argument("--n_folds", help="number of CV folds",type=int,
                            default=4)
    parser.add_argument("--minstudies", help="minimum number of nonzero entries in desmtx",type=int,
                            default=0)
    parser.add_argument("--n_Cs", help="number of C values",type=int,
                            default=25)
    parser.add_argument("--penalty", help="penalty type for LR",
                        default='l2')
    parser.add_argument('-b',"--binarize", help="binarize data",
                        default=True,action='store_false')
    parser.add_argument('-p',"--prototype", help="use limited number of vars for prototyping",
                        action='store_true')
    parser.add_argument('-t',"--force_true", help="force recomputation",
                        action='store_true')
    parser.add_argument('-s',"--shuffle", help="shuffle target variable",
                        action='store_true')
    parser.add_argument('-d',"--desmtx", help="design matrix file",
                        required=True)
    parser.add_argument('-f',"--flag", help="flag for output naming",
                        required=True)
    args=parser.parse_args()

    args.prototype=True
    args.minstudies=200
    if args.verbose:
        print('ARGS:',args)

    if args.shuffle:
        shuf_flag='_shuffle_%s'%'%08x'%random.getrandbits(32)
        if args.verbose:
            print('SHUFFLING DATA')
    else:
        shuf_flag=''

    em=EncodingModel(args.desmtx,args.flag,verbose=args.verbose,
                n_jobs=args.n_jobs,minstudies=args.minstudies,
                shuffle=args.shuffle,prototype=args.prototype,
                n_folds=args.n_folds)
    em.load_data()
    em.load_desmtx()
    em.clean_data_and_design()

 
