#!/usr/bin/env python


import pickle
import os
import rdflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle
import os
import biomapy as bp
from collections import Counter

#HANDLING ONTOLOGIES
from nxontology.imports import from_file
import pronto
from nxontology import NXOntology
from nxontology.imports import pronto_to_multidigraph, multidigraph_to_digraph
#SKLEARN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import argparse

parser=argparse.ArgumentParser(description='Perform the experiment of investigating the effect of preprocessing and  adding different ontologies to the KG')
parser.add_argument('-e','--embs',help = """Choose which algorithm of embeddings creation to use, Dlemb and M2V are implemented""",type = str,default = "Dlemb")
parser.add_argument('-a','--algorithm',help = """Choose which algorithm of classification to use possibilities are LR for logistic regression and SVM for support vector machine""",type = str,default = "SVM")
args = parser.parse_args()


def ProcessOntology(mode = None):
    if mode == 'raw':
        Kg = rdflib.Graph()
        ontology_path = './datasets/hp.owl'
        annotations_path = '../datasets/HP_Annotations.tsv'
        Kg.parse(ontology_path, format='xml')

        file_annot_hpo = open(annotations_path, 'r')
        for annot in file_annot_hpo:
            ent, hpo_term_list = annot[:-1].split('\t')
    #             print(ent, hpo_term_list)

            url_ent = "http://purl.obolibrary.org/obo/" + ent


            for url_hpo_term in hpo_term_list.split(';'):
                Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                        rdflib.term.URIRef(url_hpo_term)))



        print('RAW KG created')
        file_annot_hpo.close()
        print('Building triples')
        dic_nodes = {}
        id_node = 0
        id_relation = 0
        dic_relations = {}
        list_triples = []

        for (subj, predicate, obj) in Kg:
            if str(subj) not in dic_nodes:
                dic_nodes[str(subj)] = id_node
                id_node = id_node + 1
            if str(obj) not in dic_nodes:
                dic_nodes[str(obj)] = id_node
                id_node = id_node + 1
            if str(predicate) not in dic_relations:
                dic_relations[str(predicate)] = id_relation
                id_relation = id_relation + 1
            list_triples.append([str(subj) , str(predicate), str(obj)])
        list_triples_df = pd.DataFrame(list_triples,columns=['source','rel_type','target'])
        
        return list_triples_df
    
    elif mode == 'processed_hp':
        hp_annotations = pd.read_csv('../datasets/HP_Annotations.tsv',sep='\t',header = None)
        hp_processed = from_file('../datasets/hp.owl')
        hp_graph = hp_processed.graph

        # Keep only HP subset
        hp_nodes = [node for node in list(hp_graph.nodes()) if 'HP' in node]
        hp_subgraph = hp_graph.subgraph(hp_nodes)
        
        hp_annotations[1] = hp_annotations[1].apply(lambda x : x.split(';'))
        hp_annotations = hp_annotations.explode(1)
        hp_annotations[1] = hp_annotations[1].apply(lambda x: x.split('/')[-1])
        hp_annotations[1] = hp_annotations[1].astype(str)
        edges_to_add = pd.DataFrame(list(zip(hp_annotations[1].tolist(),
                                     ['has_annotate' for _ in range(hp_annotations.shape[0])],
                                     hp_annotations[0].tolist())), columns= ['source','rel_type','target'])
        edgelist = nx.to_pandas_edgelist(hp_subgraph)
        edgelist.insert(1,'rel_type',['is_a' for _ in range(edgelist.shape[0])])
        edgelist['source'] = edgelist['source'].apply(lambda x: x.replace(':','_'))
        edgelist['target'] = edgelist['target'].apply(lambda x: x.replace(':','_'))

        final_edgelist = pd.concat([edgelist,edges_to_add])
        
        return final_edgelist
    
    elif mode == 'processed_hp_go':
        # HP
        hp_annotations = pd.read_csv('../datasets/HP_Annotations.tsv',sep='\t',header = None)
        hp_processed = from_file('../datasets/hp.owl')
        hp_graph = hp_processed.graph

        # Keep only HP subset
        hp_nodes = [node for node in list(hp_graph.nodes()) if 'HP' in node]
        hp_subgraph = hp_graph.subgraph(hp_nodes)
        
        hp_annotations[1] = hp_annotations[1].apply(lambda x : x.split(';'))
        hp_annotations = hp_annotations.explode(1)
        hp_annotations[1] = hp_annotations[1].apply(lambda x: x.split('/')[-1])
        hp_annotations[1] = hp_annotations[1].astype(str)
        edges_to_add = pd.DataFrame(list(zip(hp_annotations[1].tolist(),
                                     ['has_annotate' for _ in range(hp_annotations.shape[0])],
                                     hp_annotations[0].tolist())), columns= ['source','rel_type','target'])
        edgelist = nx.to_pandas_edgelist(hp_subgraph)
        edgelist.insert(1,'rel_type',['is_a' for _ in range(edgelist.shape[0])])
        edgelist['source'] = edgelist['source'].apply(lambda x: x.replace(':','_'))
        edgelist['target'] = edgelist['target'].apply(lambda x: x.replace(':','_'))
        
        #GO
        go_pronto = pronto.Ontology(handle='../datasets/go-basic.owl')
        go_multidigraph = pronto_to_multidigraph(go_pronto)
        go_edgelist=pd.DataFrame([(source,rel,target) for (source,rel,target) in go_multidigraph.edges(keys=True)],columns=['source','target','rel_type'])

        go_edgelist = go_edgelist[['source','rel_type','target']]
        go_edgelist.relation=go_edgelist.rel_type.apply(lambda x: x.replace(' ','_'))
        with open('../datasets/goa_human.gaf', 'r') as f:
            goa=f.readlines()

        goa=goa[41:]
        goa=[line.rstrip('\t\n').split('\t') for line in goa]
        goa=pd.DataFrame(goa)
        goa=goa[[2,3,4]]
        goa.columns=['gene','rel_type','function']

        goa=goa[goa.gene!=''].reset_index(drop=True)

        mappedgenes=bp.gene_mapping_many(goa.gene.tolist(),'symbol','entrez')
        goa['geneId']=mappedgenes

        goa=goa.dropna(subset=['geneId','function'])
        goa.geneId=goa.geneId.astype(int)
        goa.geneId=goa.geneId.astype(str)
        
        g=pd.DataFrame(columns=['geneId','rel_type','function'])
        for rel in set(goa.rel_type.tolist()):
            tp=goa[goa.rel_type==rel]

            tp=tp.groupby('geneId').agg({'rel_type':'first',
                                     'function':set})
            tp.reset_index(inplace=True)

            tp=tp.explode('function')    

            g=pd.concat([g,tp])
        
        g.columns = ['source','rel_type','target']
        
        final_edgelist = pd.concat([edgelist,edges_to_add,g])
        
        return final_edgelist


def CreateEmbeddings(edgelist, mode = None):
    if mode == 'raw':
        edgelist.to_csv('kg_edgelist_rvp.csv')
        print('creating_embeddings')
        if args.embs == "Dlemb":
            os.system('../tools/DLemb-main/./DLemb.py --edgelist ./kg_edgelist_rvp.csv --output ./kg_embeddings_rvp.pickle -e 15')
        elif args.embs == "M2V":
            os.system('../tools/runMetapath2Vec/./Metapath2vec.py --edgelist ./kg_edgelist_rvp.csv --output ./kg_embeddings_rvp.pickle')
        
        with open('kg_embeddings_rvp.pickle','rb') as f:
            embdict = pickle.load(f)
        
        entities = list(embdict.keys())
        gene_entities=[key for key in entities if ( key.split('/')[-1].isnumeric())&('obolibrary' in key)]
        disease_entities=[key for key in entities if ('C' in key)&(key.split('/')[-1].split('C')[-1].isnumeric())&('obolibrary' in key)]
        Entities=gene_entities+disease_entities
        new_embdict = {e:k for e,k in embdict.items() if e in Entities}
        nbd = {k.split('/')[-1]:v for k,v in new_embdict.items()}
        return nbd

    else:
        edgelist.to_csv('kg_edgelist_rvp.csv')
        print('creating_embeddings')
        if args.embs == "Dlemb":
            os.system('../tools/DLemb-main/./DLemb.py --edgelist ./kg_edgelist_rvp.csv --output ./kg_embeddings_rvp.pickle -e 15')
        elif args.embs == "M2V":
            os.system('../tools/runMetapath2Vec/./Metapath2vec.py --edgelist ./kg_edgelist_rvp.csv --output ./kg_embeddings_rvp.pickle')
        with open('kg_embeddings_rvp.pickle','rb') as f:
            embdict = pickle.load(f)
        return embdict
        


def CreateTrainTest(Train, Test, embdict ):
    Id2Vec=embdict
    Train['dis_emb']=list(map(Id2Vec.get,Train.diseaseId.tolist()))
    Train['gene_emb']=list(map(Id2Vec.get,Train.geneId.astype(str).tolist()))
    Test['dis_emb']=list(map(Id2Vec.get,Test.diseaseId.tolist()))
    Test['gene_emb']=list(map(Id2Vec.get,Test.geneId.astype(str).tolist()))
    Train.dropna(inplace=True)
    Test.dropna(inplace = True)
    Train['Concatenation']=Train.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
    Test['Concatenation']=Test.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
    
    X_train = np.stack(Train["Concatenation"].values)
    y_train = Train["Label"].astype(int).values
    X_test = np.stack(Test["Concatenation"].values)
    y_test = Test["Label"].astype(int).values
    
    
    return X_train, y_train, X_test, y_test


def EvaluatePerformance(model,X_train,y_train,X_test,y_test,ax1,ax2,exp):
    
    model.fit(X_train,y_train)
    
    
    # PLOT TEST AUC ###################################################
    auc_test=RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name="ROCAUC_"+exp,
            alpha=1,
            lw=2,
            ax=ax1)
    
    # PLOT TEST PRAUC ###################################################
    prauc_curve_test = PrecisionRecallDisplay.from_estimator(model,
                                                      X_test,
                                                      y_test,
                                                      name="PRAUC_"+exp,
                                                      alpha=1,
                                                      lw=2,
                                                      ax=ax2)

def Main():
    

    if args.algorithm == 'SVM':
        clf = SVC(**{'C': 10, 'kernel': 'rbf'})
    elif args.algorithm == "LR":
        clf = LogisticRegression()
    model_name = type(clf).__name__
    
    
    exps = ['raw','processed_hp','processed_hp_go']
    
    fig, (ax1,ax2) = plt.subplots(2, 1,sharex=True,figsize=(20,12))
    
    for exp in exps:
        dis_train=pd.read_csv('../TrainingSets/Dis_Train.csv')
        dis_test=pd.read_csv('../TrainingSets/Dis_Test.csv')
        dis_train=dis_train.astype(str)
        dis_test=dis_test.astype(str)
        print(f'startin exp {exp}') 
        edgelist = ProcessOntology(mode = exp)
        print('edgelist created') 
        embs = CreateEmbeddings(edgelist, mode = exp)
        
        print(f'embeddings created') 
        X_tr,y_tr,X_te,y_te = CreateTrainTest(dis_train,dis_test, embs)
        
        EvaluatePerformance(clf,X_tr,y_tr,X_te,y_te,ax1,ax2,exp)
        
        print('Performance Evaluated writing results')
    path= '../outputs/RawVsProcessed/'
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(path+f'{args.embs}_{model_name}.png',dpi=300,bbox_inches = "tight")

        

    
    
if __name__ == '__main__':
    Main()
