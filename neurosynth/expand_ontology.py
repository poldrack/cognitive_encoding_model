# use neo4j cog atlas database to expand an ontology mapping

from py2neo import Graph, Node, Relationship
import pandas

graph = Graph("http://0.0.0.0:7474/db/data/")


def flatten(l):
    # parent_concepts returns some funky nested lists
    # this function cleans them up
    output=[]
    for i in l:
        if isinstance(i, str):
            output.append(i)
            #print('appending',i)
        else:
            #print('passing off',i)
            output = output + flatten(i)
    return list(set(output))

def parent_concepts(id=None,name=None,flatten_output=True,verbose=False):
    assert id or name
    # in case a list is passed, run over each element
    if name:
        parents=graph.cypher.execute("MATCH (child:concept)-[:KINDOF]->(parent:concept) WHERE child.name='{}' RETURN parent".format(name))
    elif id:
        parents=graph.cypher.execute("MATCH (child:concept)-[:KINDOF]->(parent:concept) WHERE child.id='{}' RETURN parent".format(id))
    if len(parents)==0:
        return []
    parent_names=[]
    for p in parents:
        parent_names.append(p['parent']['name'])
    if verbose:
        print(name,': found',parent_names)
    if flatten_output:
        return flatten(parent_names + [parent_concepts(name=i,flatten_output=flatten_output) for i in parent_names])
    else:
        return parent_names + [parent_concepts(name=i) for i in parent_names]


def expand_desmtx(d):
    for i in d.index:
        tmp=d.ix[i]
        tmp=tmp[tmp>0]
        orig_concepts=list(tmp.index)
        expanded_concepts=[]
        for c in orig_concepts:
            expanded_concepts=expanded_concepts+parent_concepts(name=c)
        expanded_concepts=list(set(expanded_concepts))
        for e in expanded_concepts:
            d.ix[i][e]=1
    return d

if __name__=='__main__':
    desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
    dm=expand_desmtx(desmtx)
    i=26095530
    tmp=desmtx.ix[i]
    tmp[tmp>0]
    tmpe=dm.ix[i]
    tmpe[tmpe>0]
