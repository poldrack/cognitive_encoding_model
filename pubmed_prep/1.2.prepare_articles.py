"""
create two-dimensional embedding for pubmed abstracts
identified from psychology journals using 1.get_articles.py
"""

import pickle,os
import string,re
import gensim.models
import collections
from nltk.stem import WordNetLemmatizer

import random
import numpy,pandas
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.doc2vec import TaggedDocument
import nltk

from joblib import Parallel, delayed

from utils import text_cleanup, get_journals

use_cogat_phrases=True # also transform 3+ word cogat Phrases

journals=get_journals()
datadir='../data/pubmed'
nsdatadir='../data/neurosynth'

# get list of all abstracts for training of bigram detector
if os.path.exists(os.path.join(datadir,'cleaned_abstracts.pkl')):
    print('using saved clean abstracts')
    cleaned_abstracts,all_cleaned_abstracts=pickle.load(open(os.path.join(datadir,'cleaned_abstracts.pkl'),'rb'))
else:
    print('cleaning up text')
    abstracts_raw=pickle.load(open(os.path.join(datadir,'abstracts.pkl'),'rb'))
    cleaned_abstracts={}
    all_cleaned_abstracts=[]

    for j in journals:
        cleaned_abstracts[j]={}
        for pmid in abstracts_raw[j].keys():
            abstract=text_cleanup(abstracts_raw[j][pmid][0])
            all_cleaned_abstracts.append(abstract.split(' '))
            cleaned_abstracts[j][pmid]=abstract

    pickle.dump((cleaned_abstracts,all_cleaned_abstracts),open(os.path.join(datadir,'cleaned_abstracts.pkl'),'wb'))


# note: these get added to all_cleaned_abstracts for generating of the
# bigram/trigram transformers, but not to the documents for modeling
if use_cogat_phrases:
    wordnet_lemmatizer=WordNetLemmatizer()
    desmtx_df=pandas.read_csv(os.path.join(nsdatadir,'data/desmtx.csv'),index_col=0)
    cogat_concepts=[i.lower() for i in list(desmtx_df.columns)]
    # kludge - create enough documents with each concept for it to end up
    # in the n-gram list
    cleaned_cogat_concepts={}
    for c in cogat_concepts:
        if ')' in c:
            continue
        else:
            cleaned_cogat_concepts[c]=text_cleanup(c)
            for i in range(100):
                all_cleaned_abstracts.append(cleaned_cogat_concepts[c])
    pickle.dump(cleaned_cogat_concepts,open(os.path.join(datadir,'cleaned_cogat_concepts.pkl'),'wb'))


if os.path.exists(os.path.join(datadir,'trigram_transformer.pkl')):
    print('using trained trigram transformer')
    bigram_transformer=gensim.models.phrases.Phraser.load(os.path.join(datadir,'bigram_transformer.pkl'))
    trigram_transformer=gensim.models.phrases.Phraser.load(os.path.join(datadir,'trigram_transformer.pkl'))
else:
    print('training bigram detector')
    bigrams = gensim.models.Phrases(all_cleaned_abstracts,min_count=50)
    bigram_transformer=gensim.models.phrases.Phraser(bigrams)
    bigram_transformer.save(os.path.join(datadir,'bigram_transformer.pkl'))
    print('training trigram detector')
    trigrams=gensim.models.Phrases(bigram_transformer[all_cleaned_abstracts],
                        min_count=50,threshold=2)
    trigram_transformer=gensim.models.phrases.Phraser(trigrams)
    trigram_transformer.save(os.path.join(datadir,'trigram_transformer.pkl'))


doc_td=[]
for j in cleaned_abstracts.keys():
    print(j)
    for pmid in cleaned_abstracts[j].keys():
        docsplit=[i for i in cleaned_abstracts[j][pmid].split(' ') if len(i)>0]
        doc_td.append(TaggedDocument(trigram_transformer[bigram_transformer[docsplit]],[pmid]))

pickle.dump(doc_td,open(os.path.join(datadir,'doc_td.pkl'),'wb'))
