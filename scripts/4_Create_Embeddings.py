#!/usr/bin/env python3

import os
from itertools import product
import networkx as nx
from collections import Counter
import numpy as np
import pandas as pd
import subprocess
from stellargraph import StellarGraph
import warnings
import graphlot as gp
import random
from stellargraph.data import UniformRandomMetaPathWalk
import pickle
from gensim.models import Word2Vec
import argparse
from multiprocessing import cpu_count
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline


parser=argparse.ArgumentParser(description='Creates embeddings out of a Knowledge graph')

parser.add_argument('--workers',help = "number of cores to use with word2vec default is the number of the cores of the machine -1 " ,type = int)
parser.add_argument('--path',help = "path of the KG for creating the embeddings" ,type = str)

args = parser.parse_args()


if args.workers:
    nofcores=args.workers
else:
    nofcores = cpu_count() - 1 



### Define some global variables

train = pd.read_csv('../TrainingSets/Dis_Train.csv')
test = pd.read_csv('../TrainingSets/Dis_Test.csv')
train = train.astype(str)
test = test.astype(str)


def LoadData():
    with open(args.path,'rb') as f:
        kg=pickle.load(f)

    kg.remove_nodes_from(list(nx.isolates(kg)))
    return kg

def CreateToynet():
    toynet = gp.CreateNetworkFromRandomClasses([200,230,450],3000)
    edgelist = nx.to_pandas_edgelist(toynet)
    rel_types = [random.sample(['a','b','c'],1)[0] for n in range(edgelist.shape[0])]
    rel_types_dict = dict(zip(toynet.edges(),rel_types))
    nx.set_edge_attributes(toynet,rel_types_dict,'label')
    edgelist.insert(1,'rel_type',rel_types)
    node_attributes = nx.get_node_attributes(toynet,'Type')
    nx.set_node_attributes(toynet,node_attributes,'label')
    return toynet,edgelist

def ParseKG(kg,subnet = False):
    nodes = list(kg.nodes())
    node_attributes = nx.get_node_attributes(kg,'tipo')
    edge_attributes = nx.get_edge_attributes(kg,'rel_type')
    nx.set_node_attributes(kg,node_attributes,'label')
    nx.set_edge_attributes(kg,edge_attributes,'label')
    if subnet:
        randomnodes = random.sample(nodes,subnet)
        subnet_kg = kg.subgraph(randomnodes)
        subnet_kg = nx.Graph(subnet_kg)
        subnet_kg.remove_nodes_from(list(nx.isolates(subnet_kg)))
        return subnet_kg
    else:
        return kg


def typology(i):
    if i.isnumeric():
        return 'protein'
    elif 'GO' in i:
        return 'function'
    elif 'C' in i:
        return 'phenotype'
    elif 'DB' in i:
        return 'drug'
    else:
        print(i)

def Convertkgtostellar(kg):
    stellar_kg = StellarGraph.from_networkx(kg)
    return stellar_kg
  
def CreateRandomEmbs(kg): 
    entities = list(kg.nodes())

    rand_features = np.random.rand(len(entities),100)

    Id2Vec = dict(zip(entities,rand_features)) 
    return Id2Vec

def run_metapath2vec(graph):
    # specify the metapath schemas as a list of lists of node types.
    types = list(set([typology(h) for h in graph.nodes()]))
    metapaths= list(product(types,repeat = 3))
    metapaths = [list(m) for m in metapaths]
    metapaths = [a for a in metapaths if a[0]==a[2]]

    # Create the random walker
    rw = UniformRandomMetaPathWalk(graph)

    walks = rw.run(
        nodes=list(graph.nodes()),  # root nodes
        length=50,  # maximum length of a random walk
        n=3,  # number of random walks per root node
        metapaths=metapaths,  # the metapaths
    )
    model = Word2Vec(window = 5, sg = 1, hs = 0,
                 negative = 5, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,workers=7,min_count=1)

    
    model.build_vocab(walks, progress_per=2)

    model.train(walks, total_examples = model.corpus_count, epochs=15, report_delay=1)

    print("Number of random walks:{}".format(len(walks)))
    Id2Vec=dict(zip(model.wv.index_to_key,model.wv.vectors))
    return Id2Vec

def RunRGCN(kg):
    edgelist = nx.to_pandas_edgelist(kg)
    edgelist = edgelist[['source','rel_type','target']]
    Train=edgelist.iloc[:int(edgelist.shape[0]*0.70)]
    Test=edgelist.iloc[int(edgelist.shape[0]*0.70):int(edgelist.shape[0]*0.85)]
    Vali=edgelist.iloc[int(edgelist.shape[0]*0.85):]
    path = "./data/bioKG/"
    if not os.path.isdir(path):
        os.makedirs(path)
    Train.to_csv(path + '/train.txt',sep='\t',index = False,header=False)
    Test.to_csv(path +'/test.txt',sep='\t',index = False,header=False)
    Vali.to_csv(path +'/valid.txt',sep='\t',index = False,header=False)
    kg_nodes = list(kg.nodes())
    kg_relations = list(set(edgelist.rel_type.tolist()))


    with open(path + '/entities.dict','w') as file:
        for i,n in enumerate(kg_nodes):
            file.write(str(i)+'\t'+str(n)+'\n')


    with open(path + '/relations.dict','w') as file:
        for i,n in enumerate(kg_relations):
            file.write(str(i)+'\t'+str(n)+'\n')
    
    subprocess.call('../tools/RGCN-master/main.py --n-epochs 15 --evaluate-every 2 --graph-batch-size 100'.split(' '))
    model =  torch.load("./best_mrr_model.pth")
    embs = dict(model['state_dict'])['entity_embedding.weight']
    embs = np.array(embs)
    Id2Vec = dict(zip(kg_nodes,embs))
    return Id2Vec


def runRotatE(kg):
    #CREATE MAPPING DICITONARIES
    Id2Node=dict(enumerate(list(kg.nodes)))
    Node2Id={v:k for k,v in Id2Node.items()}
    Id2Int=dict(enumerate(list(set([edge[2]['rel_type'] for edge in kg.edges(data=True)]))))
    Int2Id={v:k for k,v in Id2Int.items()}


    #  CREATE TRIPLES FACTORY 
    edgelist=list(nx.to_edgelist(kg))
    set_of_triples=[[Node2Id[node1],Int2Id[interaction['rel_type']],Node2Id[node2]] for (node1,node2,interaction) in edgelist]
    set_of_triples_Long=torch.LongTensor(set_of_triples)
    tf=TriplesFactory(set_of_triples_Long,entity_to_id=Node2Id,relation_to_id=Int2Id)
    tr, te, va = tf.split([.8, .1, .1])
    results = pipeline(
        training=tr,
        testing=te,
        validation=va,
        model='RotatE',
        model_kwargs=dict(embedding_dim=100),
        epochs=15
        )
    
    model=results.model
    entity_tensor= model.entity_representations[0]().detach().cpu()
    embeddings_real = np.array([[float(complex_number.real) for complex_number in complex_vector] for complex_vector in entity_tensor])
    Id2Vec=dict(zip(list(kg.nodes()),embeddings_real))
    return Id2Vec
    
def RunBioKG():

    
    subprocess.call("""../tools/knowalk-main/KW2VEC.py -e ./data/edgelist.csv -w {('drug','protein'):0,('protein','function'):10,('function','phenotype'):100} -s True -d True --epochs 15 -c 1 -o ../outputs/kg_embeddings_kw.pickle""".split(' '))
    
    with open('../outputs/kg_embeddings_kw.pickle','rb') as f:
        id2vec = pickle.load(f)
    return id2vec

def RunDLemb():
    
    subprocess.call("../tools/DLemb-main/DLemb.py -i ./data/edgelist.csv -e 15 -o ../outputs/kg_embeddings_dl.pickle".split(' '))
   
    with open('../outputs/kg_embeddings_dl.pickle','rb') as f:
        id2vec = pickle.load(f)
    return id2vec

def RunTransE(kg):
    #CREATE MAPPING DICITONARIES
    Id2Node=dict(enumerate(list(kg.nodes)))
    Node2Id={v:k for k,v in Id2Node.items()}
    Id2Int=dict(enumerate(list(set([edge[2]['rel_type'] for edge in kg.edges(data=True)]))))
    Int2Id={v:k for k,v in Id2Int.items()}


    #  CREATE TRIPLES FACTORY 
    edgelist=list(nx.to_edgelist(kg))
    set_of_triples=[[Node2Id[node1],Int2Id[interaction['rel_type']],Node2Id[node2]] for (node1,node2,interaction) in edgelist]
    set_of_triples_Long=torch.LongTensor(set_of_triples)
    tf=TriplesFactory(set_of_triples_Long,entity_to_id=Node2Id,relation_to_id=Int2Id)
    tr, te, va = tf.split([.8, .1, .1])
    results = pipeline(
        training=tr,
        testing=te,
        validation=va,
        model='TransE',
        model_kwargs=dict(embedding_dim=100),
        epochs=15
        )
    
    model=results.model
    entity_tensor= model.entity_representations[0]().detach().cpu()
    embeddings_real = np.array([[float(complex_number.real) for complex_number in complex_vector] for complex_vector in entity_tensor])
    Id2Vec=dict(zip(list(kg.nodes()),embeddings_real))
    return Id2Vec

def RunN2V(kg):
    # Produce the Node2Vec inputs
    Num2Node=dict(enumerate(list(kg.nodes)))
    Node2Num={v:k for k,v in Num2Node.items()}

    EdgeList=pd.DataFrame(([(Node2Num[s],Node2Num[t]) for (s,t) in list(kg.edges)]))
    EdgeList.to_csv('./data/Node2Vec_kg_input.txt',sep=' ',header=False,index=False)


    #Run Node2Vec
    os.system('../tools/node2vec/./node2vec -i:./data/Node2Vec_kg_input.txt -o:./Node2Vec_kg_output.txt -e:15 -l:50 -d:100 -r:5 -dr -v')

    with open('./Node2Vec_kg_output.txt','r') as f:
        NodEmbs=f.readlines()

    NodEmbs=[s.split('\n') for s in NodEmbs ]
    NodEmbs=dict(zip([s[0].split(' ')[0] for s in NodEmbs[1:]],[s[0].split(' ')[1:] for s in NodEmbs[1:]]))
    Id2Vec={Num2Node[int(NodeNumber)]:np.array([float(number) for number in v]) for NodeNumber,v in NodEmbs.items()}
    return Id2Vec
    
    

def RunDistMult(kg):
    #CREATE MAPPING DICITONARIES
    Id2Node=dict(enumerate(list(kg.nodes)))
    Node2Id={v:k for k,v in Id2Node.items()}
    Id2Int=dict(enumerate(list(set([edge[2]['rel_type'] for edge in kg.edges(data=True)]))))
    Int2Id={v:k for k,v in Id2Int.items()}


    #  CREATE TRIPLES FACTORY 
    edgelist=list(nx.to_edgelist(kg))
    set_of_triples=[[Node2Id[node1],Int2Id[interaction['rel_type']],Node2Id[node2]] for (node1,node2,interaction) in edgelist]
    set_of_triples_Long=torch.LongTensor(set_of_triples)
    tf=TriplesFactory(set_of_triples_Long,entity_to_id=Node2Id,relation_to_id=Int2Id)
    tr, te, va = tf.split([.8, .1, .1])
    results = pipeline(
        training=tr,
        testing=te,
        validation=va,
        model='DistMult',
        model_kwargs=dict(embedding_dim=100),
        epochs=15
        )
    
    model=results.model
    entity_tensor= model.entity_representations[0]().detach().cpu()
    embeddings_real = np.array([[float(complex_number.real) for complex_number in complex_vector] for complex_vector in entity_tensor])
    Id2Vec=dict(zip(list(kg.nodes()),embeddings_real))
    return Id2Vec
    


    


def WriteEmbeddings(Id2Vec,path,dis_train=train,dis_test=test):
    warnings.filterwarnings(action='ignore')

    dis_train = dis_train[(dis_train.diseaseId.isin(list(Id2Vec.keys())))&(dis_train.geneId.isin(list(Id2Vec.keys())))]
    dis_test = dis_test[(dis_test.diseaseId.isin(list(Id2Vec.keys())))&(dis_test.geneId.isin(list(Id2Vec.keys())))]

    print('Shape train and test before embedding mapping',(dis_train.shape,dis_test.shape))
    
    #ADD EMBEDDINGS (FEATURES) TO THE TRAINING SET
    disembs = list(map(Id2Vec.get,dis_train.diseaseId.tolist()))
    genembs = list(map(Id2Vec.get,dis_train.geneId.tolist()))
    dis_train['dis_emb']= disembs
    dis_train['gene_emb']=genembs
    dis_train.dropna(inplace=True)

    somma = dis_train.apply(lambda x: np.sum((x['gene_emb'],x['dis_emb']),axis=0),axis=1)
    concat = dis_train.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
    average = dis_train.apply(lambda x: np.sum((x['gene_emb'],x['dis_emb']),axis=0)/2,axis=1)
    hada = dis_train.apply(lambda x: np.multiply(x['gene_emb'],x['dis_emb']),axis=1)
    dis_train['Sum']= somma
    dis_train['Concatenation']= concat
    dis_train['Average']= average
    dis_train['Hadmard']= hada
    
    
    #ADD EMBEDDINGS (FEATURES) TO THE TEST SET
    disembs = list(map(Id2Vec.get,dis_test.diseaseId.tolist()))
    genembs = list(map(Id2Vec.get,dis_test.geneId.tolist()))
    dis_test['dis_emb']= disembs
    dis_test['gene_emb']= genembs
    dis_test.dropna(inplace=True)

    somma = dis_test.apply(lambda x: np.sum((x['gene_emb'],x['dis_emb']),axis=0),axis=1)
    concat = dis_test.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
    average = dis_test.apply(lambda x: np.sum((x['gene_emb'],x['dis_emb']),axis=0)/2,axis=1)
    hada = dis_test.apply(lambda x: np.multiply(x['gene_emb'],x['dis_emb']),axis=1)
    dis_test['Sum']= somma
    dis_test['Concatenation']= concat
    dis_test['Average']= average
    dis_test['Hadmard']= hada
    
    print(dis_test.shape,dis_train.shape)

    
    if not os.path.isdir(path):
        os.makedirs(path)


    with open(path + 'Id2Vec.pickle','wb') as f:
        pickle.dump(Id2Vec,f)
    with open(path + 'Train_embs.pickle','wb') as f:
        pickle.dump(dis_train,f)
    with open(path + 'Test_embs.pickle','wb') as f:
        pickle.dump(dis_test,f)

      

    
    
    

def Main():
    # LOAD DATA
    kg = LoadData()
    kg_parsed = ParseKG(kg)
    print(f'KG number of nodes: {kg_parsed.number_of_nodes()} KG number of edges: {kg_parsed.number_of_edges()}')
    print(dict(Counter([typology(e) for e in set(kg.nodes())])))
    stellar_kg = Convertkgtostellar(kg_parsed)
    edgelist = nx.to_pandas_edgelist(kg_parsed)
    edgelist = edgelist[['source','rel_type','target']]
    edgelist['source_type'] =  edgelist.source.apply(lambda x: typology(x))
    edgelist['target_type'] =  edgelist.target.apply(lambda x: typology(x))
    edgelist.to_csv('./data/edgelist.csv',index = False)

    # RUN METAPATH2VEC

    print(""" ===========================================================================
                                    Create Random Embs
    ===========================================================================""".center(20))
    id2vec_rand = CreateRandomEmbs(kg_parsed)
    print(dict(Counter([typology(e) for e in list(id2vec_rand.keys())])))
    WriteEmbeddings(id2vec_rand,'../Embeddings/Random/')
    # RUN METAPATH2VEC

    print(""" ===========================================================================
                                    Running Metapath2vec
    ===========================================================================""".center(20))
    id2vec_meta = run_metapath2vec(stellar_kg)
    print(dict(Counter([typology(e) for e in list(id2vec_meta.keys())])))
    WriteEmbeddings(id2vec_meta,'../Embeddings/Metapath2Vec_100D/')
    # RUN RGCN
    print(""" ===========================================================================
                                Running RGCN
    ===========================================================================""".center(20))
    id2vec_rgcn = RunRGCN(kg_parsed)
    print(dict(Counter([typology(e) for e in list(id2vec_rgcn.keys())])))
    WriteEmbeddings(id2vec_rgcn,'../Embeddings/RGCN/')

    # RUN RotatE
    print(""" ===========================================================================
                                    Running RotatE
    ===========================================================================""".center(20))
    id2vec_rotate = runRotatE(kg_parsed)
    print(dict(Counter([typology(e) for e in list(id2vec_rotate.keys())])))

    WriteEmbeddings(id2vec_rotate,'../Embeddings/RotatE/')
    # RUN KNOWALK
    print(""" ===========================================================================
                                    Running K2VEC
    ===========================================================================""".center(20))
    id2vec_knwa = RunBioKG()
    print(dict(Counter([typology(e) for e in list(id2vec_knwa.keys())])))

    WriteEmbeddings(id2vec_knwa,'../Embeddings/BioKG2Vec/')
    # RUN DLEMB
    print(""" ===========================================================================
                                    Running Dlemb
    ===========================================================================""".center(20))
    id2vec_dlemb = RunDLemb()
    print(dict(Counter([typology(e) for e in list(id2vec_dlemb.keys())])))

    WriteEmbeddings(id2vec_dlemb,'../Embeddings/Dlemb/')
    # RUN TRANSE
    print(""" ===========================================================================
                                    Running TransE
    ===========================================================================""".center(20))
    id2vec_trans = RunTransE(kg_parsed)
    print(dict(Counter([typology(e) for e in list(id2vec_trans.keys())])))

    WriteEmbeddings(id2vec_trans,'../Embeddings/TransE/')
    # RUN DISTMULT
    print("""===========================================================================
                                    Running DistMult
    ===========================================================================""".center(20))
    id2vec_dist = RunDistMult(kg_parsed)
    print(dict(Counter([typology(e) for e in list(id2vec_dist.keys())])))

    WriteEmbeddings(id2vec_dist,'../Embeddings/DistMult/')
    #RUN Node2Vec
    print(""" ===========================================================================
                                    Running N2V
    ===========================================================================""".center(20))
    id2vec_n2v = RunN2V(kg_parsed)
    print(dict(Counter([typology(e) for e in list(id2vec_n2v.keys())])))

    WriteEmbeddings(id2vec_n2v,'../Embeddings/N2V/')    
    
    
if __name__ == '__main__':
    Main()
    
