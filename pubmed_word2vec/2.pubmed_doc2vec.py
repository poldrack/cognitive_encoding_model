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

from text_cleanup import text_cleanup

use_cogat_phrases=True # also transform 3+ word cogat Phrases


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
        for j in abstracts_raw.keys():
            cleaned_abstracts[j]={}
            for pmid in abstracts_raw[j].keys():
                abstract=text_cleanup(abstracts_raw[j][pmid][0])
                docsplit=[wordnet_lemmatizer.lemmatize(i) for i in nltk.tokenize.word_tokenize(abstract)]
                cleaned_abstracts[j][pmid]=' '.join(docsplit)
                all_cleaned_abstracts.append(docsplit)
        pickle.dump((cleaned_abstracts,all_cleaned_abstracts),open('cleaned_abstracts.pkl','wb'))

    if use_cogat_phrases:
        desmtx_df=pandas.read_csv('../neurosynth/data/desmtx.csv',index_col=0)
        cogat_concepts=[i for i in list(desmtx_df.columns) if len(i.split(' '))>1]
        # kludge - create enough documents with each concept for it to end up
        # in the n-gram list
        for i in range(100):
            for c in cogat_concepts:
                if ')' in c:
                    continue
                else:
                    all_cleaned_abstracts.append(c)


    if os.path.exists('trigram_transformer.pkl'):
        trigram_transformer=gensim.models.Phraser().load('trigram_transformer.pkl')
    else:
        print('training bigram detector')
        bigrams = gensim.models.Phrases(all_cleaned_abstracts,min_count=50)
        bigram_transformer=gensim.models.phrases.Phraser(bigrams)
        print('training trigram detector')
        trigrams=gensim.models.Phrases(bigram_transformer[all_cleaned_abstracts],min_count=50)
        trigram_transformer=gensim.models.phrases.Phraser(trigrams)
        trigram_transformer.save('trigram_transformer.pkl')
    asdf

    doc_td=[]
    for j in cleaned_abstracts.keys():
        print(j)
        for pmid in cleaned_abstracts[j].keys():
            docsplit=cleaned_abstracts[j][pmid].split(' ')
            doc_td.append(TaggedDocument(trigram_transformer[docsplit],[pmid]))

    pickle.dump(doc_td,open('doc_td.pkl','wb'))

# fit model

ndims=50


if os.path.exists('doc2vec_unigram.model'):
    print('using saved model')
    model_docs=Doc2Vec.load('doc2vec_unigram.model')
else:
    if os.path.exists('doc2vec_unigram_vocab.model'):
        print("using saved vocabulary")
        model_docs=Doc2Vec.load('doc2vec_unigram_vocab.model')
    else:
        print('learning vocabulary')
        model_docs=Doc2Vec(dm=1, size=ndims, window=5, negative=5,
                hs=0, min_count=2, workers=22,iter=20,
                alpha=0.025, min_alpha=0.025,dbow_words=1)
        model_docs.build_vocab(doc_td)
        model_docs.save('doc2vec_unigram_vocab.model')
    print('learning model')
    for epoch in range(10):
        random.shuffle(doc_td)
        print('training on',model_docs.alpha)
        model_docs.train(doc_td,total_examples=model_docs.corpus_count,
                            epochs=model_docs.iter)
        model_docs.alpha-=.002
        model_docs.min_alpha=model_docs.alpha
        model_docs.save('doc2vec_unigram.model')
        with open('model.txt','a') as f:
            f.write('%f\n'%model_docs.alpha)

# check the model
# from https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
print('checking model performance')
ranks = []
def get_ranks(doc_id,model_docs,doc_td):
    inferred_vector = model_docs.infer_vector(doc_td[doc_id].words)
    sims = model_docs.docvecs.most_similar([inferred_vector]) #, topn=len(model_docs.docvecs))
    return [doc_td[doc_id].tags[0],int(sims[0][0]==doc_td[doc_id].tags[0]),sims[0][1]]

results=Parallel(n_jobs=20)(delayed(get_ranks)(i,model_docs,doc_td) for i in range(len(doc_td)))
pickle.dump(results,open('model_check_results.pkl','wb'))

# compute similarity between all documents in doc2vec space
