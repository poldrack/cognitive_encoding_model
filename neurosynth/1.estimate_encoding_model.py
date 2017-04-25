# estimate encoding model on full dataset
# and test identification accuracy
# baby step towards crossvalidation- if it doesn't work here
# then it certainly won't work on out-of-sample data

import os
import pandas,numpy
import pickle

from sklearn.linear_model import ElasticNet

# load data and estimate model on full dataset
data=numpy.load('data/imgdata.npy')
desmtx=pandas.read_csv('data/desmtx.csv')
if os.path.exists('data/fitted_en.pkl'):
    en=pickle.load(open('data/fitted_en.pkl','rb'))
    print('using cached elastic net model')
else:
    print('estimating elastic net model')
    en=ElasticNet(verbose=True)
    en.fit(data,desmtx)
    pickle.dump(en,open('data/fitted_en.pkl','wb'))

# estimate map for each study using forward model
print('estimating maps using forward model')
estimated_maps=numpy.zeros(data.shape)
#for i in range(data.shape[0]):
