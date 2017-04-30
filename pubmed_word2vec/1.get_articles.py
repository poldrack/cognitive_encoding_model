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

journals=['J Exp Psychol Learn Mem Cogn','Cognition','Mem Cognit',
            'J Exp Psychol Gen','J Exp Psychol Appl',
            'J Exp Psychol Hum Percept Perform','Cogn Psychol',
            'Cogn Sci','Atten Percept Psychophys',
            'Psychon Bull Rev',
            'Cogn Emot','J Vis','Vision Res',
            'Hear Res','Psychol Rev','Psychol Bull',
            'Psychol Sci','Pers Soc Psychol Bull',
            'J Pers Soc Psychol','J Res Pers',
            'J Exp Soc Psychol','Emotion','Motiv Emot']
            # rejected:
            # 'J Acoust Soc Am' (too many)
            # 'Judgm Decis Mak' (too few)
            # 'Q J Exp Psychol' - no abstracts present

retmax=2000000
delay=0.5 # delay for pubmed api

# get matching pubmed ids
if os.path.exists('pmids.pkl'):
    print('using cached pmids')
    pmids=pickle.load(open('pmids.pkl','rb'))
else:
    pmids={}
    for j in journals:
        time.sleep(0.5)
        pmids[j]=[]
        handle = Entrez.esearch(db="pubmed", retmax=retmax,term='"%s"[TA]'%j)
        record = Entrez.read(handle)
        handle.close()
        pmids[j]=[int(i) for i in record['IdList']]
        print('found %d records for'%len(pmids[j]),j)
    pickle.dump(pmids,open('pmids.pkl','wb'))

# get abstract text
if os.path.exists('abstracts.pkl'):
    print('using cached abstracts')
    abstracts=pickle.load(open('abstracts.pkl','rb'))
else:
    abstracts={}
    for j in journals:
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(['%d'%i for i in pmids[j]]), retmode="xml")
            time.sleep(delay)
            records=Entrez.read(handle)
            abstracts[j]={}
            for i in records['PubmedArticle']:
                pmid=int(i['MedlineCitation']['PMID'])
                if 'Abstract' in i['MedlineCitation']['Article']:
                    abstracts[j][pmid]=i['MedlineCitation']['Article']['Abstract']['AbstractText']
                else:
                    pass #print('no abstract for',j,str(i['MedlineCitation']['PMID']))

            print(j,': found %d abstracts from %d keys'%(len(abstracts[j]),len(pmids[j])))
        except:
            print('problem with',j)
    pickle.dump(abstracts,open('abstracts.pkl','wb'))
