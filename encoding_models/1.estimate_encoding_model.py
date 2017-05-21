# estimate encoding model on full dataset
# here we binarize (given the zero-inflated nature of the data)
# and use logistic regression
# save test splits for use in later quantification of accuracy
# this version is generalized across all the different encoding models

import os,datetime,sys
import argparse

from encoding_model import EncodingModel


if __name__=="__test__":

    en=EncodingModel('../data/neurosynth/desmtx.csv','test',
                     prototype=True,n_jobs=1,shuffle=True)
    en.load_data()
    en.load_desmtx()
    en.clean_data_and_design()
    results=en.estimate_model()

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

    if args.verbose:
        print('ARGS:',args)

    em=EncodingModel(args.desmtx,args.flag,verbose=args.verbose,
                n_jobs=args.n_jobs,minstudies=args.minstudies,
                shuffle=args.shuffle,prototype=args.prototype,
                n_folds=args.n_folds)
    em.load_data()
    em.load_desmtx()
    em.clean_data_and_design()
    results=en.estimate_model()
