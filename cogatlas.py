"""
class to represent cognitive concepts
and perform expansion using cog atlas hierarchy
"""

import os
import pandas
import glob

def load_dump_file(filename):
        # first read loads the column names, which ahve a different delimiter
        a=pandas.read_csv(filename,sep=';')
        return pandas.read_csv(filename,sep=';',names=a.columns[0].split(','),
            skiprows=1)

class Cogatlas:
    def __init__(self,dump_dir='cogat_dumps'):
        self.dump_dir=dump_dir
        self.concepts_df=None
        self.concepts={}
        self.contrasts_df=None
        self.assertions_df=None
        self.relations_df=None
        self.tasks_df=None
        self.tasks={}
        self.concept_graph={}
        self.isa_dict=[]
        self.partof_dict=[]
        self.measuredby_dict=[]

    def load_cogat(self):
        # load dumps
        assert os.path.exists(self.dump_dir)
        assertion_file=glob.glob(os.path.join(self.dump_dir,'Dump_assertion*csv'))
        assert len(assertion_file)==1
        self.assertions_df=load_dump_file(assertion_file[0])

        concept_file=glob.glob(os.path.join(self.dump_dir,'Dump_concept*csv'))
        assert len(concept_file)==1
        self.concepts_df=load_dump_file(concept_file[0])
        self.concepts_df['id']=[i.split('/')[-1] for i in c.concepts_df.url]
        for i in range(self.concepts_df.shape[0]):
            tmp=self.concepts_df.iloc[i,:]
            id=tmp.id
            del tmp['id']
            assert id not in self.concepts
            self.concepts[id]=tmp.to_dict()

        relation_file=glob.glob(os.path.join(self.dump_dir,'Dump_relation*csv'))
        assert len(relation_file)==1
        self.relations_df=load_dump_file(relation_file[0])

        task_file=glob.glob(os.path.join(self.dump_dir,'Dump_task*csv'))
        assert len(task_file)==1
        self.tasks_df=load_dump_file(task_file[0])
        self.tasks_df['id']=[i.split('/')[-1] for i in c.tasks_df.url]
        for i in range(self.tasks_df.shape[0]):
            tmp=self.tasks_df.iloc[i,:]
            id=tmp.id
            del tmp['id']
            assert id not in self.tasks
            self.tasks[id]=tmp.to_dict()

        contrast_file=glob.glob(os.path.join(self.dump_dir,'Dump_contrast*csv'))
        assert len(contrast_file)==1
        self.contrasts_df=load_dump_file(contrast_file[0])

    def create_relations(self):
        for i in range(self.assertions_df.shape[0]):
            tmp=self.assertions_df.iloc[i,:]
            if tmp.id_relation=='T1':
                # is-a-kind-of
                if not tmp.id_subject in self.concepts:
                    print('oops - ',tmp.id_subject,'not in concepts dict')
                    continue
            #if not tmp.id_relation in ['T1','T2']
    def create_graph(self):
        pass

if __name__=='__main__':
    c=Cogatlas()
    c.load_cogat()
