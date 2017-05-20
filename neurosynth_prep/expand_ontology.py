# use neo4j cog atlas database to expand an ontology mapping
# first need to run neo4j using docker, ala these instructions from Ross
"""
Code is on my tasks_api branch:

https://github.com/rwblair/cogat/tree/tasks_api

Once cloned run a `docker-compose build` or the like. This will take awhile, the mongo, and neo images are quite large. I am also trying to import some of the data on container build so after awhile you might see some output that looks like tasks and such being created. Once this is done do a `docker-compose up` and go to either the neo4j end point at 192.168.99.100:7474 or the website itself on port 80 to verify that some nodes were imported.

The next step is to get the remaining information imported. This requires mysql. Hopefully it will be as simple as :

brew install mysql
brew services start mysql
mysql -uroot -dcogat < cogat-2017-03-23.sql

I will send you a dropbox invite to the sql file for that database backup.

Next setup or use an existing python3 environment, and install py2neo and pymysql from pip. The cogat containers should be up and running for this. From the root code directory run `python mysql2neo.py` You should see a variety of types of nodes being created.

pip install "py2neo<3"

For the IP address I assume you are still using boot to docker, if the IP of your containers is different you will need to update line 7 in mysql2neo.py.

Forgot to add, if there are no nodes in the database after the containers are built you can run the following to import data:

docker-compose run --rm uwsgi python scripts/migrate_database.py

If it complains about cognitiveatlas library not being installed then run:

docker-compose run --rm uwsgi bash
> $pip install cognitiveatlas
> $python scripts/migrate_database.py

also see this gist: https://gist.github.com/anonymous/37fd3d97793140caf69d9d438aeb55b3


"""

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
    desmtx=pandas.read_csv('../data/neurosynth/desmtx_cleaned.csv',index_col=0)
    dm=expand_desmtx(desmtx.copy())
    dm.to_csv('../data/neurosynth/desmtx_cleaned_expanded.csv')
