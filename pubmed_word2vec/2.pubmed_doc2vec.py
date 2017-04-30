"""
create two-dimensional embedding for pubmed abstracts
identified from psychology journals using 1.get_articles.py
"""

import pickle,os
import string,re

import random
import pandas
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.doc2vec import TaggedDocument
from nltk.stem import WordNetLemmatizer
import nltk

from text_cleanup import text_cleanup



# preprocess and clean up text
if os.path.exists('doc_td.pkl'):
    print('using saved text')
    doc_td=pickle.load(open('doc_td.pkl','rb'))
else:
    print('cleaning up text')
    abstracts_raw=pickle.load(open('abstracts.pkl','rb'))
    doc_td=[]
    wordnet_lemmatizer=WordNetLemmatizer()
    stopwords=nltk.corpus.stopwords.words('english')
    for j in abstracts_raw.keys():
        print(j)
        for pmid in abstracts_raw[j].keys():
            abstract=text_cleanup(abstracts_raw[j][pmid][0])
            # strip stopwords
            #abstract=' '.join([i for i in abstract.split(' ') if not i in stopwords])
            #abstract=abstracts[a]
            docsplit=[wordnet_lemmatizer.lemmatize(i) for i in nltk.tokenize.word_tokenize(abstract)]
            doc_td.append(TaggedDocument(docsplit,[pmid]))

    pickle.dump(doc_td,open('doc_td.pkl','wb'))


# fit model

ndims=50

if os.path.exists('doc2vec_unigram_vocab.model'):
    print("using saved vocabulary")
    model_docs=Doc2Vec.load('doc2vec_unigram_vocab.model')
else:
    print('learning vocabulary')
    model_docs=Doc2Vec(dm=1, size=ndims, window=5, negative=5,
            hs=0, min_count=2, workers=32,
            alpha=0.025, min_alpha=0.025)
    model_docs.build_vocab(doc_td)
    model_docs.save('doc2vec_unigram_vocab.model')

if os.path.exists('doc2vec_unigram.model'):
    print('using saved model')
    model_docs=Doc2Vec.load('doc2vec_unigram.model')
else:
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

# compute similarity between all documents in doc2vec space
