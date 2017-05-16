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
datadir='../data/pubmed'
nsdatadir='../data/neurosynth'
if not os.path.exists(datadir):
   os.makedirs(datadir)
if not os.path.exists(nsdatadir):
   os.makedirs(nsdatadir)

retmax=2000000
delay=0.5 # delay for pubmed api
terms_to_exclude=["neural","neuroimaging","brain","positron emission tomography","fMRI", "functional MRI", "functional magnetic resonance imaging"]
force_new=True

with open('%s/journals.txt'%datadir) as f:
    journals=[i.strip() for i in f.readlines()]


extra_terms=''
for t in terms_to_exclude:
    extra_terms=extra_terms+'NOT "%s" '%t

# get matching pubmed ids
if os.path.exists('%s/pmids.pkl'%datadir) and not force_new:
    print('using cached pmids')
    pmids=pickle.load(open('%s/pmids.pkl'%datadir,'rb'))
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
pickle.dump(pmids,open('%s/pmids.pkl'%datadir,'wb'))

# get neurosynth data so that we can exclude any papers in the database
nsdata=pandas.read_csv('%s/database.txt'%nsdatadir,index_col=0,sep='\t')
ns_pmids=list(set(nsdata.index))

# get abstract text
# also save authors for later filtering

if os.path.exists('%s/abstracts.pkl'%datadir) and not force_new:
    print('using cached abstracts')
    abstracts=pickle.load(open('%s/abstracts.pkl'%datadir,'rb'))
else:
    abstracts={}
    authors=[]

for j in journals:
    if j in abstracts:
        continue
    good_record=None
    maxtries=5
    tryctr=0
    while not good_record:
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(['%d'%i for i in pmids[j]]), retmode="xml")
            time.sleep(delay)
            records=Entrez.read(handle)
            good_record=True
        except:
            e = sys.exc_info()[0]
            print('problem with',tryctr,j,e)
    if not good_record:
        raise Exception('unsolvable problem with',j)
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
pickle.dump(abstracts,open('%s/abstracts.pkl'%datadir,'wb'))
#pickle.dump(authors,open('%s/authors.pkl'%datadir,'wb'))
authors_cleaned=[i.lower() for i in list(set(authors))]
pickle.dump(authors_cleaned,open('%s/authors_cleaned.pkl'%datadir,'wb'))
