"""
get abstracts for selected journals from pubmed
"""

import os
import tarfile
import pandas,numpy
import pickle
import time
import nibabel
import glob

from Bio import Entrez

Entrez.email='poldrack@stanford.edu'

with open('journals.txt') as f:
    journals=[i.strip() for i in f.readlines()]


retmax=2000000
delay=0.5 # delay for pubmed api
terms_to_exclude=["neural","neuroimaging","brain","positron emission tomography","fMRI", "functional MRI", "functional magnetic resonance imaging"]
extra_terms=''
for t in terms_to_exclude:
    extra_terms=extra_terms+'NOT "%s" '%t
force_new=True

# get matching pubmed ids
if os.path.exists('pmids.pkl') and not force_new:
    print('using cached pmids')
    pmids=pickle.load(open('pmids.pkl','rb'))
else:
    pmids={}

for j in journals:
    if j in pmids:
        continue
    else:
        print('adding',j)
    time.sleep(0.5)
    pmids[j]=[]
    searchterm='"%s"[TA]%s'%(j,extra_terms)
    #print(searchterm)
    handle = Entrez.esearch(db="pubmed", retmax=retmax,term=searchterm)
    record = Entrez.read(handle)
    handle.close()
    pmids[j]=[int(i) for i in record['IdList']]
    print('found %d records for'%len(pmids[j]),j)
pickle.dump(pmids,open('pmids.pkl','wb'))

# get neurosynth data so that we can exclude any papers in the database
nsdata=pandas.read_csv('../neurosynth/data/database.txt',index_col=0,sep='\t')
ns_pmids=list(set(nsdata.index))

# get abstract text
# also save authors for later filtering

if os.path.exists('abstracts.pkl') and not force_new:
    print('using cached abstracts')
    abstracts=pickle.load(open('abstracts.pkl','rb'))
else:
    abstracts={}
    authors=[]

for j in journals:
    if j in abstracts:
        continue
    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(['%d'%i for i in pmids[j]]), retmode="xml")
        time.sleep(delay)
        records=Entrez.read(handle)
        abstracts[j]={}
        for i in records['PubmedArticle']:
            pmid=int(i['MedlineCitation']['PMID'])
            if pmid in ns_pmids:
                print('skippping neurosynth pmid',pmid)
                continue
            if 'AuthorList' in i['MedlineCitation']['Article']:
                for au in i['MedlineCitation']['Article']['AuthorList']:
                    if 'LastName' in au:
                        authors.append(au['LastName'])
                    else:
                        print('hmm, no last name',au)
            if 'Abstract' in i['MedlineCitation']['Article']:
                abstracts[j][pmid]=i['MedlineCitation']['Article']['Abstract']['AbstractText']
            else:
                pass #print('no abstract for',j,str(i['MedlineCitation']['PMID']))

        print(j,': found %d abstracts from %d keys'%(len(abstracts[j]),len(pmids[j])))
    except:
        e = sys.exc_info()[0]
        print('problem with',j,e)
pickle.dump(abstracts,open('abstracts.pkl','wb'))
pickle.dump(authors,open('authors.pkl','wb'))
