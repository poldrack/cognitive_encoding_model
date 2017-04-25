# use neo4j cog atlas database to expand an ontology mapping

from py2neo import Graph, Node, Relationship

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

def parent_concepts(id=None,name=None):
    if name is None and id is None:
        raise Exception('Must specify either name or id input variable')
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
    print(name,': found',parent_names)
    return flatten(parent_names + [parent_concepts(name=i) for i in parent_names])
