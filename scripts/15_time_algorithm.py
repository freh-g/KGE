#!/usr/bin/env python

import time
import subprocess
import random
import os
import numpy as np
import pickle
from collections import Counter
import networkx as nx
from stellargraph import StellarGraph
from itertools import product
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import matplotlib.pyplot as plt



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


def Convertkgtostellar(kg):
    stellar_kg = StellarGraph.from_networkx(kg)
    return stellar_kg
  

def LoadData():
    with open('../KGs/KG_100.pickle','rb') as f:
        kg=pickle.load(f)

    kg.remove_nodes_from(list(nx.isolates(kg)))
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


def RunDLemb():

    subprocess.call("../tools/DLemb-main/DLemb.py -i ./data/edgelist.csv -e 15 -o ../outputs/kg_embeddings_dl.pickle".split(' '))
   
    with open('../outputs/kg_embeddings_dl.pickle','rb') as f:
        id2vec = pickle.load(f)
    return id2vec

def RunBioKG(kg):
    edgelist = nx.to_pandas_edgelist(kg)
    edgelist = edgelist[['source','rel_type','target']]
    edgelist['source_type'] =  edgelist.source.apply(lambda x: typology(x))
    edgelist['target_type'] =  edgelist.target.apply(lambda x: typology(x))
    edgelist.to_csv('./data/edgelist.csv',index = False)
    
    subprocess.call("""../tools/knowalk-main/KW2VEC.py -e ./data/edgelist.csv -w {('drug','protein'):0,('protein','function'):10,('function','phenotype'):100} -s True -d True --epochs 15 -c 1 -o ../outputs/kg_embeddings_kw.pickle""".split(' '))
    
    with open('../outputs/kg_embeddings_kw.pickle','rb') as f:
        id2vec = pickle.load(f)
    return id2vec

def Main():
    number_of_experiments = 10
    fig,ax = plt.subplots()
    times_metapath2vec = []
    times_BioKG2Vec = []
    times_dlemb = []
    for exp in range(number_of_experiments):
        kg = LoadData()
        kg_parsed = ParseKG(kg)
        nodes = list(kg_parsed.nodes())
        sampled_nodes = random.sample(nodes,10000)
        subgraph = nx.subgraph(kg_parsed,sampled_nodes)
        kg_parsed = subgraph
        edgelist = nx.to_pandas_edgelist(subgraph)
        edgelist[['source','target','rel_type']].to_csv('./data/edgelist.csv',index = False)
        print(f'KG number of nodes: {kg_parsed.number_of_nodes()} KG number of edges: {kg_parsed.number_of_edges()}')
        print(dict(Counter([typology(e) for e in set(kg.nodes())])))
        stellar_kg = Convertkgtostellar(kg_parsed)
        start = time.time()
        run_metapath2vec(stellar_kg)
        end = time.time()
        elapsed_time = end - start
        times_metapath2vec.append(elapsed_time)
        print("Metapath2Vec Time",elapsed_time)
        start = time.time()
        RunBioKG(kg_parsed)
        end = time.time()
        elapsed_time = end - start
        times_BioKG2Vec.append(elapsed_time)
        print("BioKG2Vec Time", elapsed_time)
        start = time.time()
        RunDLemb()
        end = time.time()
        elapsed_time = end - start
        times_dlemb.append(elapsed_time)
        print("DLemb Time", elapsed_time)
    ind = np.arange(number_of_experiments)
    width = 0.25
    bar1 = plt.bar(ind, times_metapath2vec, width) 
    
    bar2 = plt.bar(ind+width, times_BioKG2Vec, width) 
    
    bar3 = plt.bar(ind+width*2, times_dlemb, width) 
    plt.legend( (bar1, bar2, bar3), ('Metapath2Vec', 'BioKG2Vec', 'DLemb') )
    plt.ylabel('Seconds')
    path = '../outputs/time_comparison/'
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(path+'Time.png',dpi=300)
    mean_time_metapath = np.mean(times_metapath2vec)
    mean_time_biokg = np.mean(times_BioKG2Vec)
    mean_time_dlemb = np.mean(times_dlemb)
    print(f"Mean time metapath2vec: {mean_time_metapath} seconds \n Mean time BioKG2Vec: {mean_time_biokg} seconds \n Mean time DLemb: {mean_time_dlemb} seconds")
    increased_BioKG2Vec = ((mean_time_biokg - mean_time_metapath)/mean_time_biokg) * 100
    increased_DLemb = ((mean_time_dlemb - mean_time_metapath)/mean_time_dlemb) * 100

    print(f"Percentage of increased performance BioKG2Vec: {np.absolute(increased_BioKG2Vec)} % \n Percentage of increased performance DLemb: {np.absolute(increased_DLemb)} %")

if __name__=="__main__":
    Main()


