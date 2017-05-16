import gensim.models

import pickle,os

cleaned_abstracts,all_cleaned_abstracts=pickle.load(open('cleaned_abstracts.pkl','rb'))
bigram_transformer=gensim.models.phrases.Phraser.load('bigram_transformer.pkl')

trigram_transformer=gensim.models.phrases.Phraser.load('trigram_transformer.pkl')

print(list(trigram_transformer[bigram_transformer[[['spatial','working','memory']]]]))


