#!/usr/bin/env python
import networkx as nx
import pandas as pd
from stellargraph import StellarGraph
from gensim.models import Word2Vec
import pickle
from stellargraph.data import UniformRandomMetaPathWalk
import argparse
from multiprocessing import cpu_count
from itertools import product

parser=argparse.ArgumentParser(description='Creates embeddings out of a Knowledge graph')
parser.add_argument('--workers',help = "number of cores to use with word2vec default is the number of the cores of the machine -1 " ,type = int)
parser.add_argument('--edgelist',help = "edgelist in csv format")
parser.add_argument('--epochs',help = "epochs to train the model for",default=15,type=int)
parser.add_argument('--output',help = "outpath")

args = parser.parse_args()

if args.workers:
    nofcores=args.workers
else:
    nofcores = cpu_count() - 1 


def from_edgelist_to_nx(edg):
    edgelist = pd.read_csv(edg)
    graph = nx.from_pandas_edgelist(edgelist)
    return graph
    
    

def Convertkgtostellar(kg):
    stellar_kg = StellarGraph.from_networkx(kg)
    return stellar_kg

def typology(i):
    if str(i).isnumeric():
        return 'protein'
    elif 'GO' in str(i):
        return 'function'
    elif (('C' in str(i)) & (len(str(i)) == 8)) | ("UMLS" in str(i)):
        return 'phenotype'
    elif 'HP' in str(i):
        return 'phenotype'
    elif (('DB' in str(i)) & (len(str(i)) == 7)):
        return 'drug'
    else:
        return 'other'

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

    model.train(walks, total_examples = model.corpus_count, epochs=args.epochs, report_delay=1)

    print("Number of random walks:{}".format(len(walks)))
    Id2Vec=dict(zip(model.wv.index_to_key,model.wv.vectors))
    return Id2Vec

def Main():
    print("Building Network")
    network = from_edgelist_to_nx(args.edgelist)
    attrs = dict(zip(network.nodes(),[typology(n) for n in network.nodes()]))
    nx.set_node_attributes(network,attrs,name = 'label')
    network_stellar = Convertkgtostellar(network)
    print(network_stellar.info())
    id2vec = run_metapath2vec(network_stellar)
    with open(args.output,'wb') as f:
        pickle.dump(id2vec,f)


if __name__ == '__main__':
    Main()
    
