from gensim.models.doc2vec import Doc2Vec,TaggedDocument

m=Doc2Vec.load('models/doc2vec_trigram_300dims.model')

from utils import *
# http://stackoverflow.com/questions/3788870/how-to-check-if-a-word-is-an-english-word-with-python

from nltk.corpus import words as nltk_words
def is_english_word(word):
    # creation of this dictionary would be done outside of 
    #     the function because you only need to do it once.
    dictionary = dict.fromkeys(nltk_words.words(), None)
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False

authors=load_authors()
short_terms=[]

for i in m.wv.vocab:
   if len(i)<5:
        short_terms.append(i)
with open('short_terms.txt','w') as f:
   for s in short_terms:
        if not s in authors and not is_english_word(s):
           f.write(s+'\n')
