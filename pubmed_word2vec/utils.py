"""
util functions
"""

import string,re
import nltk

def load_acroynms(acronym_file='acronyms.txt'):
    with open(acronym_file) as f:
        lines=[i.strip() for i in f.readlines()]
    return lines

def text_cleanup(text,strip_stopwords=True,strip_acronyms=True):

    translate_table = dict((ord(char), ' ') for char in string.punctuation)
    #text=' '.join([i.decode('utf-8') for i in text])
    text=text.translate(translate_table)
    text=re.sub(r'\<([^\]]+)\>'," ",text) # replace bracketed items
    text=text.translate({ord(ch): '' for ch in '0123456789'})
    stripped = lambda s: "".join(i for i in s if 31 < ord(i) < 127)
    if strip_stopwords:
        stopwords=nltk.corpus.stopwords.words('english')
        text = ' '.join([i for i in text.split(' ') if not i in stopwords])
    if strip_acronyms:
        acronyms=load_acroynms()
        text = ' '.join([i for i in text.split(' ') if not i in acronyms])
    return stripped(text).lower()

def text_cleanup_punct(text,strip_stopwords=True):
    """ leave in punctuation but surround with spaces"""
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        text = text.replace(char, ' ' + char + ' ')
    text=text.replace('-',' ').replace('/',' ')
    text=text.translate({ord(ch): '' for ch in '0123456789'})
    stripped = lambda s: "".join(i for i in s if 31 < ord(i) < 127)
    if strip_stopwords:
        stopwords=nltk.corpus.stopwords.words('english')
        text = ' '.join([i for i in text.split(' ') if not i in stopwords])
    return stripped(text).lower()


def get_journals():
    with open('journals.txt') as f:
        journals=[i.strip() for i in f.readlines()]
    return journals
