# estimate encoding model on full dataset
# and test identification accuracy
# baby step towards crossvalidation- if it doesn't work here
# then it certainly won't work on out-of-sample data

import os
import pandas,numpy
import pickle

from sklearn.linear_model import ElasticNet

try:
   expanded=sys.argv[1]
except:
   expanded=False

# load data and estimate model on full dataset
data=numpy.load('data/imgdata.npy')
if expanded:
    desmtx=pandas.read_csv('data/desmtx_expanded.csv',index_col=0)
    outfile='data/fitted_en_expanded.pkl'
else:
    desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
    outfile='data/fitted_en.pkl'

if os.path.exists(outfile):
    en=pickle.load(open(outfile,'rb'))
    print('using cached elastic net model')
else:
    print('estimating elastic net model')
    en=ElasticNet()
    en.fit(desmtx,data)
    pickle.dump(en,open(outfile,'wb'))


# estimate map for each study using forward model
print('estimating maps using forward model')
estimated_maps=numpy.zeros(data.shape)
#for i in range(data.shape[0]):
p=en.predict(desmtx)
if expanded:
    numpy.save('pred_en_expanded.npy',p)
else:
    numpy.save('pred_en.npy',p)

