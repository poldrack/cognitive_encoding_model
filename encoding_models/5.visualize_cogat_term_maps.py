"""
for each term, generate a map showing relative loadings of ROIs
"""



import os,pickle
import numpy,pandas
import nibabel

model='cogat'
outdir='../images/%s'%model
if not os.path.exists(outdir):
    os.makedirs(outdir)

parcellation=nibabel.load('../data/neurosynth/parcellation.nii.gz')
parc_data=parcellation.get_data()

if model=='cogat':
    desmtx_file='../data/neurosynth/desmtx_cleaned.csv'
    desmtx=pandas.read_csv(desmtx_file,index_col=0)
    embedding=pickle.load(open('../models/encoding/cogat/fulldata_results_2a4f8c5d.pkl','rb'))
    roimap=embedding[1]  # 2000 ROIs X 290 concepts

for c in range(roimap.shape[1]):
    cname=desmtx.columns[c]
    outfile=os.path.join(outdir,'%s.nii.gz'%'_'.join(cname.split(' ')))
    print(cname,outfile)
    data=numpy.zeros(parc_data.shape)
    for i in range(roimap.shape[0]):
        idx= parc_data==i
        data[idx]=roimap[i,c]
    outimg=nibabel.Nifti1Image(data,parcellation.affine,parcellation.header)
    outimg.to_filename(outfile)
