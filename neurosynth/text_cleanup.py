
import string,re
import nltk


def text_cleanup(text,strip_stopwords=True):

    translate_table = dict((ord(char), ' ') for char in string.punctuation)
    #text=' '.join([i.decode('utf-8') for i in text])
    text=text.translate(translate_table)
    text=re.sub(r'\<([^\]]+)\>'," ",text) # replace bracketed items
    text=text.translate({ord(ch): '' for ch in '0123456789'})
    stripped = lambda s: "".join(i for i in s if 31 < ord(i) < 127)
    if strip_stopwords:
        stopwords=nltk.corpus.stopwords.words('english')
        text = ' '.join([i for i in text.split(' ') if not i in stopwords])
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
