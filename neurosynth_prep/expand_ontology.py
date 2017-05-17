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

def parent_concepts(name=None,id=None,flatten_output=True,verbose=False):
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

def list_tasks():
    tasks = graph.cypher.execute("MATCH (t:task) RETURN t")
    return tasks

def list_contrasts():
    contrasts = graph.cypher.execute("MATCH (c:contrast) RETURN c")
    return contrasts

def get_task_id(name):
    match=graph.cypher.execute("MATCH (c:task) WHERE c.name='{}' RETURN c".format(name))
    assert len(match)==1
    return match[0]['c']['id']

def get_contrasts_for_task(name):
    id=get_task_id(name)
    conditions=graph.cypher.execute("MATCH (t:task)-[:HASCONDITION]->(c:condition) WHERE t.name='{}' RETURN c".format(name))
    for c in conditions:
        print(c['t'])

    match=graph.cypher.execute("MATCH (c:task)WHERE c.name='{}' RETURN c".format(name))
    print(match)

# tests
def test_expansion():
    desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
    i=26110429
    tmp=desmtx.ix[i]
    t=tmp[tmp>0]
    print(t)
    for c in t.index:
        print(parent_concepts(c))
    dm=expand_desmtx(desmtx.copy())
    tmpe=dm.ix[i]
    print(tmpe[tmpe>0])

def test_tasks():
    tasks = list_tasks()
    for task in tasks:
        print(task['t'].properties)

if __name__=='__main__':
    desmtx=pandas.read_csv('data/desmtx.csv',index_col=0)
    dm=expand_desmtx(desmtx.copy())
    dm.to_csv('data/desmtx_expanded.csv')
