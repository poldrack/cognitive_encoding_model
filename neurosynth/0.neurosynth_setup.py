# setup database for encoding model

import os
import tarfile
import pandas
import pickle
import time
from Bio import Entrez

Entrez.email='poldrack@stanford.edu'

import neurosynth as ns
from neurosynth.base.dataset import Dataset
from cognitiveatlas.api import get_concept

def intersect(a, b):
    return list(set(a) & set(b))

class Neurosynth:
    def __init__(self,datadir='data',verbose=True):
        self.dataset=None
        self.concepts=None
        self.concepts_df=None
        self.concept_pmids={}
        self.datadir=datadir
        self.datafile=os.path.join(datadir,'database.txt')
        self.verbose=verbose

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

    def get_concept_pmids(self,retmax=500000):
        # get the pmids for each concept that are in neurosynth
        # for single-word concepts we use the neurosynth search tool
        # for phrases we use pubmed
        print('loading all neurosynth pmids')
        all_neurosynth_ids=self.dataset.image_table.ids.tolist()
        for id in self.concepts:
            if len(id.split(' '))>1:
                time.sleep(0.5)
                handle = Entrez.esearch(db="pubmed", retmax=500000,term='"%s"'%id)
                record = Entrez.read(handle)
                handle.close()

                if len(record['IdList'])>0:
                    records_int=[int(i) for i in record['IdList']]
                    self.concept_pmids[id]=intersect(all_neurosynth_ids,records_int)
                    if self.verbose:
                        print('found',len(self.concept_pmids[id]),'matching pmids for',id)
                else:
                    print('no PMIDs for %s'%id)
            else:
                try:
                    self.concept_pmids[id]=self.dataset.get_studies(features=id)
                    if self.verbose:
                        print('found',len(self.concept_pmids[id]),'matching pmids for',id)
                except TypeError:
                    print('problem searching neurosynth for',id)
if __name__=='__main__':
    n=Neurosynth()
    n.get_dataset()
    n.get_concepts()
    n.get_concept_pmids()
