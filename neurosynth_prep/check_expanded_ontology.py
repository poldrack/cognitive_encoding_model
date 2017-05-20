"""
check results of ontology expansion
"""


import pandas
desmtx=pandas.read_csv('../data/neurosynth/desmtx_cleaned.csv',index_col=0)
desmtx_exp=pandas.read_csv('../data/neurosynth/desmtx_cleaned_expanded.csv',index_col=0)

pmids=list(desmtx.index)

for pmid in pmids[:20]:
    orig=list(desmtx.columns[desmtx.ix[pmid].nonzero()])
    exp=list(desmtx_exp.columns[desmtx_exp.ix[pmid].nonzero()])
    if set(orig) != set(exp):
        print(pmid,'intersection:')
        print(set(orig).intersection(set(exp)))
        print('difference')
        print(list(set(exp)-set(orig)))
        print('')
