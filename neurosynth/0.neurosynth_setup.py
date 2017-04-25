# setup database for encoding model

import os
import tarfile
import pandas,numpy
import pickle
import time
import nibabel
import glob

from Bio import Entrez
import nilearn.image
import nilearn.input_data
from sklearn.linear_model import ElasticNet

Entrez.email='poldrack@stanford.edu'

import neurosynth as ns
from neurosynth.base.dataset import Dataset
from neurosynth import meta

from cognitiveatlas.api import get_concept

def intersect(a, b):
    return list(set(a) & set(b))

class Neurosynth:
    def __init__(self,datadir='data',verbose=True,
                    ma_count_thresh=16,
                    meta_image='consistency_z',
                    resolution=3):
        self.dataset=None
        self.concepts=None
        self.concepts_df=None
        self.concept_pmids={}
        self.datadir=datadir
        self.datafile=os.path.join(datadir,'database.txt')
        self.verbose=verbose
        self.ma_count_thresh=ma_count_thresh
        self.meta_image=meta_image
        self.resolution=resolution
        self.imagedir_resampled=None
        self.image_concepts=None
        self.desmtx=None

        if not os.path.exists(self.datadir):
            print('downloading neurosynth data')
            ns.dataset.download(path='/tmp', unpack=True)
            print('extracting data')
            tfile = tarfile.open("/tmp/current_data.tar.gz", 'r:gz')
            if not os.path.exists(self.datadir):
                os.mkdir(self.datadir)
            tfile.extractall(self.datadir)
            os.remove("/tmp/current_data.tar.gz")
            print('done creating dataset in',self.datadir)

        self.imagedir=os.path.join(self.datadir,'ma_images')
        if not os.path.exists(self.imagedir):
            os.mkdir(self.imagedir)

    def get_dataset(self,force_load=False):
        if os.path.exists(os.path.join(self.datadir,'dataset.pkl')) and not force_load:
            print('loading database from',os.path.join(self.datadir,'dataset.pkl'))
            self.dataset=Dataset.load(os.path.join(self.datadir,'dataset.pkl'))
        else:
            print('loading database - this takes a few minutes')
            self.dataset = Dataset('data/database.txt')
            self.dataset.add_features('data/features.txt')
            self.dataset.save(os.path.join(self.datadir,'dataset.pkl'))

    def get_concepts(self,force_load=False):
        if os.path.exists(os.path.join(self.datadir,'concepts_df.csv')) and not force_load:
            print('using cached cognitive atlas concepts')
            self.concepts_df=pandas.read_csv(os.path.join(self.datadir,'concepts_df.csv'))
        else:
            self.concepts_df=get_concept().pandas
            self.concepts_df.to_csv(os.path.join(self.datadir,'concepts_df.csv'))
        self.concepts=self.concepts_df.name.tolist()

    def get_concept_pmids(self,retmax=2000000,force_load=False):
        # get the pmids for each concept that are in neurosynth
        # for single-word concepts we use the neurosynth search tool
        # for phrases we use pubmed
        if os.path.exists(os.path.join(self.datadir,'concept_pmids.pkl')) and not force_load:
            print('using cached concept_pmids')
            self.concept_pmids=pickle.load(open(os.path.join(self.datadir,'concept_pmids.pkl'),'rb'))
            return

        print('loading all neurosynth pmids')
        all_neurosynth_ids=self.dataset.image_table.ids.tolist()
        for id in self.concepts:
            time.sleep(0.5)
            handle = Entrez.esearch(db="pubmed", retmax=retmax,term='"%s"'%id)
            record = Entrez.read(handle)
            handle.close()
            # make sure we got all the records - rerun if we didn't
            if int(record['Count'])>retmax:
                handle = Entrez.esearch(db="pubmed", retmax=int(record['Count']),term='"%s"'%id)
                record = Entrez.read(handle)
                handle.close()
            records_int=[int(i) for i in record['IdList']]
            ns_pmids=intersect(all_neurosynth_ids,records_int)
            print('pubmed found',len(ns_pmids),'matching pmids for',id)
            self.concept_pmids[id]=ns_pmids
        pickle.dump(self.concept_pmids,open(os.path.join(self.datadir,'concept_pmids.pkl'),'wb'))


    def get_concept_images(self, force_load=False):

        for c in self.concept_pmids.keys():
            if not force_load and os.path.exists(os.path.join(self.imagedir,
                                            '%s_specificity_z.nii.gz'%c.replace(' ','-'))):
                continue
            if len(self.concept_pmids[c])<self.ma_count_thresh:
                #print('skipping',c,len(self.concept_pmids[c]),'pmids')
                continue
            print('running meta-analysis for',c)
            ma = meta.MetaAnalysis(self.dataset, self.concept_pmids[c])
            ma.save_results(self.imagedir, c.replace(' ','-'))

        if force_load or not os.path.exists(os.path.join(self.imagedir,'mask_image.nii.gz')):
            # make mask of voxels with zero standard deviation
            concept_images=glob.glob(os.path.join(self.imagedir,
                                            '*_%s.nii.gz'%self.meta_image))

            imgdata=numpy.zeros((91,109,91,len(concept_images)))
            print('loading concept images to compute std')
            for i,c in enumerate(concept_images):
                tmp=nibabel.load(c).get_data()
                imgdata[:,:,:,i]=tmp

            imgstd=numpy.std(imgdata,axis=3)
            maskdata=(imgstd>0).astype('int')
            maskimg=nibabel.Nifti1Image(maskdata,affine=nibabel.load(c).affine)
            maskimg.to_filename(os.path.join(self.imagedir,'mask_image.nii.gz'))

    def get_resampled_images(self,shape=None,affine=None,force_load=False):
        # use 3 mm as default
        if not shape:
            shape=[60,72,60]
            affine=numpy.array([[-3,0,0,90],[0,3,0,-126],[0,0,3,-72],[0,0,0,1]])
            self.resolution=affine[1,1].astype('int')
        print('resampling data to %d mm'%self.resolution)
        self.imagedir_resampled=os.path.join(self.datadir,'ma_images_%dmm'%self.resolution)
        if not os.path.exists(self.imagedir_resampled):
            os.mkdir(self.imagedir_resampled)
        concept_images=glob.glob(os.path.join(self.imagedir,
                                            '*_%s.nii.gz'%self.meta_image))
        for c in concept_images:
            if force_load or not os.path.exists(os.path.join(self.imagedir_resampled,os.path.basename(c))):
                img=nilearn.image.resample_img(c, target_affine=affine, target_shape=shape)
                img.to_filename(os.path.join(self.imagedir_resampled,os.path.basename(c)))

        if not os.path.exists(os.path.join(self.datadir,'mask_%dmm.nii.gz'%self.resolution)):
            # make MNI mask at chosen resolution
            mask=os.path.join(os.environ['FSLDIR'],'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
            maskimg=nilearn.image.resample_img(mask, target_affine=affine, target_shape=shape)
            maskimg.to_filename(os.path.join(self.datadir,'mask_%dmm.nii.gz'%self.resolution))

    def load_concept_images(self,force_load=True):
        concept_images=glob.glob(os.path.join(self.imagedir_resampled,
                                            '*_%s.nii.gz'%self.meta_image))
        concept_images.sort()
        self.image_concepts=[os.path.basename(i).split('_')[0] for i in concept_images]
        if os.path.exists(os.path.join(self.datadir,'imgdata_%dmm.npy'%self.resolution)):
            self.imgdata=numpy.load(os.path.join(self.datadir,'imgdata_%dmm.npy'%self.resolution))
            # make sure it's the right size
            if self.imgdata.shape[1]==len(concept_images):
                print('using cached concept image data')
                return

        masker=nilearn.input_data.NiftiMasker(
            mask_img=os.path.join(self.datadir,'mask_%dmm.nii.gz'%self.resolution),
            target_shape=[60,72,60],
            target_affine=numpy.array([[-3,0,0,90],[0,3,0,-126],[0,0,3,-72],[0,0,0,1]]))
        print('loading concept image data')
        self.imgdata=masker.fit_transform(concept_images)
        numpy.save(os.path.join(self.datadir,'imgdata_%dmm.npy'%self.resolution),self.imgdata)

    def save(self):
        with open('data/neurovault_%dmm.pkl'%self.resolution,'wb') as f:
            pickle.dump(self,f)

    def build_design_matrix(self,force_load=False):
        if not force_load and os.path.exists('data/desmtx.csv'):
            self.desmtx=pandas.DataFrame.from_csv('data/desmtx.csv')
            print('using cached design matrix')
            return
        print('building design matrix')
        all_concept_pmids=[]
        for k in self.concept_pmids.keys():
            all_concept_pmids=all_concept_pmids + self.concept_pmids[k]
        all_concept_pmids=list(set(all_concept_pmids))
        all_concept_pmids.sort()

        self.desmtx=pandas.DataFrame(data=0,index=all_concept_pmids,columns=all_concepts)

        for k in self.concept_pmids.keys():
            pmids=self.concept_pmids[k]
            self.desmtx[k][pmids]=1
        # drop columns with no matches
        self.desmtx=self.desmtx.ix[:,self.desmtx.sum()>0]
        self.desmtx.to_csv('data/desmtx.csv')

if __name__=='__main__':
    # setup
    resolution=3
    if os.path.exists('data/neurovault_%dmm.pkl'%resolution):
        print('loading cached structure')
        n=pickle.load(open('data/neurovault_%dmm.pkl'%resolution,'rb'))
    else:
        n=Neurosynth(resolution=resolution)
        n.get_dataset()
        n.get_concepts()
        n.get_concept_pmids()
        # turns out we probably don't need this
        #n.get_concept_images()
        #n.get_resampled_images()
        #n.load_concept_images()
        n.save()

    # fit encoding model
    n.build_design_matrix()
    # first, build design matrix
    print('loading dataset')
    # put into nsamples X nfeatures
    data=n.dataset.get_image_data(list(n.desmtx.index)).T
    numpy.save('data/imgdata.npy',data)
