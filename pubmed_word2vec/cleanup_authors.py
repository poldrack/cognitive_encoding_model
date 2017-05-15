import pickle

a=pickle.load(open('authors.pkl','rb'))
authors=[i.lower() for i in list(set(a))]
pickle.dump(authors,open('authors_cleaned.pkl','wb'))
