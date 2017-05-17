"""
project neurosynth articles into word2vec space
"""
import os
import pickle
import pandas,numpy
from joblib import Parallel, delayed
from text_cleanup import text_cleanup
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.stem import WordNetLemmatizer
import nltk

from Bio import Entrez

Entrez.email='poldrack@stanford.edu'
force_new=False
datadir='../data/neurosynth'

if os.path.exists('%s/abstract_text.pkl'%datadir) and not force_new:
    print('loading existing abstracts')
    abstracts,bad_pmids=pickle.load(open('%s/abstract_text.pkl'%datadir,'rb'))
else:
    print('getting abstracts')
    desmtx=pandas.read_csv('%s/desmtx.csv'%datadir,index_col=0)
    pmids=list(desmtx.index)
    abstracts={}
    bad_pmids=[]
    for retstart in [0,6000]:
        good_record=False
        tryctr=0
        while not good_record:
            tryctr+=1
            try:
                handle = Entrez.efetch(db="pubmed", retstart=retstart,retmax=6001,id=",".join(['%d'%i for i in pmids]), retmode="xml")
                records=Entrez.read(handle)
            except:
                print('retrying...')
                if tryctr>5:
                    break

        for i in records['PubmedArticle']:
            pmid=int(i['MedlineCitation']['PMID'])
            if pmid in abstracts or pmid in bad_pmids:
                continue
            if 'Abstract' in i['MedlineCitation']['Article']:
                abstracts[pmid]=i['MedlineCitation']['Article']['Abstract']['AbstractText']
            else:
                bad_pmids.append(pmid)

    print(': found %d abstracts from %d keys'%(len(abstracts),len(pmids)))
    print('saving abstracts')
    pickle.dump((abstracts,bad_pmids),open('%s/abstract_text.pkl'%datadir,'wb'))


try:
    assert not force_new
    desmtx=pandas.read_csv('%s/desmtx_cleaned.csv'%datdir,index_col=0)
    data=pickle.load(open('%s/neurosynth_reduced_cleaned.pkl'%datadir,'rb'))
    print('loading cleaned data')
except:
    desmtx=pandas.read_csv('%s/desmtx.csv'%datadir,index_col=0)
    data=pickle.load(open('%s/neurosynth_reduced.pkl'%datadir,'rb'))
    # remove pmids without abstracts
    for p in bad_pmids:
        print('removing bad pmid:',p)
        data=data[desmtx.index!=p,:]
        desmtx=desmtx.drop(p)
    assert data.shape[0]==desmtx.shape[0]
    # remove pmids with no activation across all ROIs
    s=numpy.sum(data,1)
    data=data[s>0,:]
    desmtx=desmtx.ix[s>0]
    desmtx.to_csv('%s/desmtx_cleaned.csv'%datadir)
    pickle.dump(data,open('%s/neurosynth_reduced_cleaned.pkl'%datadir,'wb'))
    print('created and saved cleaned data')

assert data.shape[0]==desmtx.shape[0]

# clean abstracts

def clean_abstract(a):
    """
    abstracts are in separate sentences - combine and clean
    """
    abstract=''
    wordnet_lemmatizer=WordNetLemmatizer()
    for sent in a:
        abstract+=str(sent)
    abstract=text_cleanup(abstract)
    docsplit=[wordnet_lemmatizer.lemmatize(i) for i in nltk.tokenize.word_tokenize(abstract) if len(i)>1]
    return docsplit,abstract

if os.path.exists('%s/ns_abstracts_cleaned.pkl'%datadir):
    abstracts_cleaned=pickle.load(open('%s/ns_abstracts_cleaned.pkl'%datadir,'rb'))
    print('loaded ')
else:
    print('cleaning abstracts')
    abstracts_cleaned={}
    for k in abstracts.keys():
        abstracts_cleaned[k],_=clean_abstract(abstracts[k])
    pickle.dump(abstracts_cleaned,open('%s/ns_abstracts_cleaned.pkl'%datadir,'wb'))
# get vector projection for each abstract
ndims=50
print('loading Doc2Vec model')
model_docs=Doc2Vec.load('../data/models/doc2vec/doc2vec_trigram_%ddims.model'%ndims)

print('getting vector projections')
pmids=list(desmtx.index)
inferred_vectors=pandas.DataFrame()
for i,pmid in enumerate(pmids):
    if numpy.mod(i,1000)==0:
        print(i)
    inferred_vectors[pmid]=model_docs.infer_vector(abstracts_cleaned[pmid])
inferred_vectors=inferred_vectors.T
inferred_vectors.to_csv('%s/ns_doc2vec_%ddims_projection.csv'%(datadir,ndims))
