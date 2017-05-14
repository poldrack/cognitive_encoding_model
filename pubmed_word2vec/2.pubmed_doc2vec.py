"""
create two-dimensional embedding for pubmed abstracts
identified from psychology journals using 1.get_articles.py
"""

import pickle,os
import string,re
import gensim.models
import collections

import random
import pandas
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.doc2vec import TaggedDocument
from nltk.stem import WordNetLemmatizer
import nltk

from joblib import Parallel, delayed

from utils import text_cleanup, get_journals

if not os.path.exists('models'):
    os.mkdir('models')

use_cogat_phrases=True # also transform 3+ word cogat Phrases

journals=get_journals()

# preprocess and clean up text
if os.path.exists('doc_td.pkl'):
    print('using saved text')
    doc_td=pickle.load(open('doc_td.pkl','rb'))
else:
    # get list of all abstracts for training of bigram detector
    if os.path.exists('cleaned_abstracts.pkl'):
        print('using saved clean abstracts')
        cleaned_abstracts,all_cleaned_abstracts=pickle.load(open('cleaned_abstracts.pkl','rb'))
    else:
        print('cleaning up text')
        abstracts_raw=pickle.load(open('abstracts.pkl','rb'))
        cleaned_abstracts={}
        all_cleaned_abstracts=[]
        wordnet_lemmatizer=WordNetLemmatizer()
        for j in journals:
            cleaned_abstracts[j]={}
            for pmid in abstracts_raw[j].keys():
                abstract=text_cleanup(abstracts_raw[j][pmid][0])
                docsplit=[wordnet_lemmatizer.lemmatize(i) for i in nltk.tokenize.word_tokenize(abstract)]
                cleaned_abstracts[j][pmid]=' '.join(docsplit)
                all_cleaned_abstracts.append(docsplit)
        pickle.dump((cleaned_abstracts,all_cleaned_abstracts),open('cleaned_abstracts.pkl','wb'))

    # note: these get added to all_cleaned_abstracts for generating of the
    # bigram/trigram transformers, but not to the documents for modeling
    if use_cogat_phrases:
        wordnet_lemmatizer=WordNetLemmatizer()
        desmtx_df=pandas.read_csv('../neurosynth/data/desmtx.csv',index_col=0)
        cogat_concepts=[i.lower() for i in list(desmtx_df.columns)]
        # kludge - create enough documents with each concept for it to end up
        # in the n-gram list
        cleaned_cogat_concepts={}
        for c in cogat_concepts:
            if ')' in c:
                continue
            else:
                c_cleaned=text_cleanup(c)
                c_lemm=[wordnet_lemmatizer.lemmatize(i) for i in nltk.tokenize.word_tokenize(c_cleaned)]
                cleaned_cogat_concepts[c]=c_lemm
                for i in range(100):
                    all_cleaned_abstracts.append(c_lemm)
        pickle.dump(cleaned_cogat_concepts,open('cleaned_cogat_concepts.pkl','wb'))


    if os.path.exists('trigram_transformer.pkl'):
        print('using trained trigram transformer')
        bigram_transformer=gensim.models.phrases.Phraser.load('bigram_transformer.pkl')
        trigram_transformer=gensim.models.phrases.Phraser.load('trigram_transformer.pkl')
    else:
        print('training bigram detector')
        bigrams = gensim.models.Phrases(all_cleaned_abstracts,min_count=50)
        bigram_transformer=gensim.models.phrases.Phraser(bigrams)
        bigram_transformer.save('bigram_transformer.pkl')
        print('training trigram detector')
        trigrams=gensim.models.Phrases(bigram_transformer[all_cleaned_abstracts],
                            min_count=50,threshold=2)
        trigram_transformer=gensim.models.phrases.Phraser(trigrams)
        trigram_transformer.save('trigram_transformer.pkl')


    doc_td=[]
    for j in cleaned_abstracts.keys():
        print(j)
        for pmid in cleaned_abstracts[j].keys():
            docsplit=cleaned_abstracts[j][pmid].split(' ')
            doc_td.append(TaggedDocument(trigram_transformer[bigram_transformer[docsplit]],[pmid]))

    pickle.dump(doc_td,open('doc_td.pkl','wb'))

# fit model

ndims=300

if os.path.exists('model.txt'):
    os.remove('model.txt')

if os.path.exists('models/doc2vec_trigram_%ddims.model'%ndims):
    print('using saved model')
    model_docs=Doc2Vec.load('models/doc2vec_trigram_%ddims.model'%ndims)
else:
    if os.path.exists('models/doc2vec_trigram_%ddims_vocab.model'%ndims):
        print("using saved vocabulary")
        model_docs=Doc2Vec.load('models/doc2vec_trigram_%ddims_vocab.model'%ndims)
    else:
        print('learning vocabulary')
        model_docs=Doc2Vec(dm=0, size=ndims, window=15, negative=5,
                hs=0, min_count=5, workers=46,iter=100,sample=1e-5,
                alpha=0.025, min_alpha=0.025,dbow_words=1)
        model_docs.build_vocab(doc_td)
        model_docs.save('models/doc2vec_trigram_%ddims_vocab.model'%ndims)
    print('learning model')
    for epoch in range(10):
        random.shuffle(doc_td)
        print('training on',model_docs.alpha)
        model_docs.train(doc_td,total_examples=model_docs.corpus_count,
                            epochs=model_docs.iter)
        model_docs.alpha-=.002
        model_docs.min_alpha=model_docs.alpha
        model_docs.save('models/doc2vec_trigram_%ddims.model'%ndims)
        with open('model.txt','a') as f:
            f.write('%f\n'%model_docs.alpha)

# check the model
# from https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
n_to_check=1000
print('checking model performance for %d random selected abstracts'%n_to_check)
ranks = []

def get_ranks(doc,model_docs):
    inferred_vector = model_docs.infer_vector(doc.words)
    sims = model_docs.docvecs.most_similar([inferred_vector]) #, topn=len(model_docs.docvecs))
    return [doc.tags[0],int(sims[0][0]==doc.tags[0]),sims[0][1]]

results=[]
docs_to_check=numpy.random.randint(0,len(doc_td),n_to_check)
for i in docs_to_check:
    results.append(get_ranks(doc_td[i],model_docs))
pickle.dump(results,open('model_check_results.pkl','wb'))
