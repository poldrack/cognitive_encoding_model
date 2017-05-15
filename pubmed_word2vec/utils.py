"""
util functions
"""

import string,re
import pickle
import nltk
from nltk.stem import WordNetLemmatizer


def load_acroynms(acronym_file='acronyms.txt'):
    with open(acronym_file) as f:
        lines=[i.strip() for i in f.readlines()]
    return lines

def load_authors(authors_file='authors_cleaned.pkl'):
    a=pickle.load(open(authors_file,'rb'))
    return a

def text_cleanup(text,strip_stopwords=True,strip_acronyms=True,
                strip_names=True,lemmatize=True):

    text=text.lower()
    translate_table = dict((ord(char), ' ') for char in string.punctuation)
    #text=' '.join([i.decode('utf-8') for i in text])
    text=text.translate(translate_table)
    text=re.sub(r'\<([^\]]+)\>'," ",text) # replace bracketed items
    text=text.translate({ord(ch): '' for ch in '0123456789'})
    stripped = lambda s: "".join(i for i in s if 31 < ord(i) < 127)
    text=stripped(text)

    if strip_stopwords:
        stopwords=nltk.corpus.stopwords.words('english')
        text = ' '.join([i for i in text.split(' ') if not i in stopwords])
    if strip_acronyms:
        acronyms=load_acroynms()
        text = ' '.join([i for i in text.split(' ') if not i in acronyms])
    if strip_names:
        authors=load_authors()
        text = ' '.join([i for i in text.split(' ') if not i in acronyms])
    if lemmatize:
        wordnet_lemmatizer=WordNetLemmatizer()
        docsplit=[wordnet_lemmatizer.lemmatize(i) for i in nltk.tokenize.word_tokenize(text)]
        text=' '.join(docsplit)
    return text


def get_journals():
    with open('journals.txt') as f:
        journals=[i.strip() for i in f.readlines()]
    return journals
