#!/usr/bin/env python
from K2V_Walkers import KRW
import pandas as pd
import pickle
import networkx as nx
from tqdm import tqdm
import argparse
from gensim.models import Word2Vec
import ast
from multiprocessing import cpu_count, Process, Queue
from math import ceil



parser=argparse.ArgumentParser(description='Creates embeddings out of a Knowledge graph')
parser.add_argument('-e','--edgelist',help = """path of the edgelist, it have to be in csv format, columns names have to be [source, target, relation_type, source_type, target_type] and having no index""",type = str)
parser.add_argument('-w','--weights',help = """user specific weights to be assigned to the walker, they have to be assigned in dictionary form ex: "{('drug','protein'): 0,('protein','function'): 10,('function','phenotype'):100}" """,type = str, default = "{}")
parser.add_argument('-i','--iterations',help = "how many walks every node",type = int, default = 3)
parser.add_argument('-l','--length',help = 'length of the walks',type = int, default = 50)
parser.add_argument('-r','--restart', help = 'restart from the beginning to complete the walk if no outedges are present',type = str, default = 'True')
parser.add_argument('-t','--edgetype', help = 'if true feed edge types to Word2Vec in order to build the embeddings',type = str, default = 'True')
parser.add_argument('-v','--verbose',help = "the algorithm sends feedback during the walk",type = str, default = 'False' )
parser.add_argument('-s','--save',help = "if true save the walks in KNWalks.txt",type = str, default = 'False')
parser.add_argument('-o','--output',help = "output file in which to store the embeddings, it will be a pickled python dictionary",type = str)
parser.add_argument('-d','--directed',help = "if True considers only out edges of a node as possible path else all neighbors are considered ",type = str, default = 'True')
parser.add_argument('-c','--min_word_count',help = "The minimum occurrence of words used from W2V to create the embeddings",type = int, default = 5)

parser.add_argument('--workers',help = "number of cores to use with word2vec default is the number of the cores of the machine -1 " ,type = int)
parser.add_argument('--window',help = "window for Word2Vec training, it will predict words in a window of this size",type = int, default = 5)
parser.add_argument('--skipgram',help = "if using skipgram model",type = int, default = 1)
parser.add_argument('--hs',help = "if 1 will use hierarchical softmax if 0 negative sampling will be used ",type = int, default = 0)
parser.add_argument('--negatives',help = "how many negatives used as negative examples",type = int, default = 5)
parser.add_argument('--alpha',help = "the learning rate",type = int, default = 0.03)
parser.add_argument('--vector_size',help = "length of the embeddings",type = int, default = 100)
parser.add_argument('--epochs',help = "how many epochs to train Word2Vec",type = int, default = 30)


args = parser.parse_args()



def LoadEdges():
    edgelist = pd.read_csv(args.edgelist)
    return edgelist.astype(str)
    

def CreateNetworkFromEdgelist(edgelist):
    kg = nx.MultiDiGraph()
    nodes = list(set(list(zip(edgelist.source.tolist(),edgelist.source_type.tolist()))+list(zip(edgelist.target.tolist(), edgelist.target_type.tolist()))))
    nodes = [(node,dict(type = tipo)) for (node,tipo) in nodes]
    edges = list(zip(edgelist.source.tolist(),edgelist.target.tolist(),[dict(rel_type=tipo) for tipo in edgelist.rel_type.tolist()]))
    
    kg.add_nodes_from(nodes)
    kg.add_edges_from(edges)
    
    return kg

def chunk_into_n(lst, n):
    size = ceil(len(lst) / n)
    res = list(map(lambda x: lst[x * size:x * size + size],list(range(n))))
    return res

def MakeWalks(nodes,kg,q,p,probabilities= ast.literal_eval(args.weights)):
    random_walks = []
    for n in tqdm(nodes,position=p):
        biorw=KRW(n,kg,Iterations=args.iterations,
                 Depth=args.length,
                 NodeAttributeName='type',
                 EdgeAttributeName='rel_type',
                 DictOfProb=probabilities,
                 restart=args.restart,
                 EdgeType=args.edgetype,
                 verbose = args.verbose,
                 directed = args.directed)
        random_walks.extend(biorw)
    
    q.put(random_walks)


def ProduceEmbeddings(model,random_walks,epochs = args.epochs):
    
    model.build_vocab(random_walks, progress_per=2)

    model.train(random_walks, total_examples = model.corpus_count, epochs=epochs)
    
    Id2Vec=dict(zip(model.wv.index_to_key,model.wv.vectors))
    
    return Id2Vec


def Main():
    
    if args.workers:
        nofcores=args.workers
    else:
        nofcores = cpu_count() - 1 
    
    print(f'\n\n\t\t\t|=====> USING {nofcores} CPUs <=====|')
    
    
    print('\n\n\t\t\t|=====> CREATING KG <=====|')
    
    edgelist = LoadEdges()
    
    kg = CreateNetworkFromEdgelist(edgelist)
    
    print(f'\n\n\t\t\t|=====> KG CREATED, NUMBER OF NODES:{kg.number_of_nodes()} NUMBER OF EDGES:{kg.number_of_edges()} <=====|')
    
    print('\n\n\t\t\t|=====> START WALKING <=====|')
    all_nodes = list(kg.nodes)
    all_nodes = chunk_into_n(all_nodes,nofcores)
    processes = []
    walks = []
    q = Queue()
    
    for i,chunk in enumerate(all_nodes):
        p = Process(target = MakeWalks,args=(chunk,kg,q,i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        walk = q.get() # will block
        walks.append(walk)
    
    for p in processes:
        p.join()  
     
    
    walks = list(sum(walks,[]))
   
    print(f"\n\n\t\t\t|=====> {len(walks)} WALKS HAVE BEEN PRODUCED  <=====|")
    
    if args.save:
        with open('KNWalks.txt','w') as f:
            for i,walk in enumerate(walks):
                f.write(f'WALK {i} ===> ')
                for word in walk:
                    f.write(word + ' ')
                f.write('\n')
 
    print('\n\n\t\t\t|=====> PRODUCING THE EMBEDDINGS <=====|')
    model = Word2Vec(window = args.window, sg = args.skipgram, hs = args.hs,
                 negative = args.negatives, # for negative sampling
                alpha=args.alpha,vector_size = args.vector_size,min_count = args.min_word_count,workers = nofcores)

    Id2Vec = ProduceEmbeddings(model,walks)            
    
    print('\n\n\t\t\t|=====> SAVING THE EMBEDDINGS <=====|')
    
    with open(args.output,'wb') as f:
        pickle.dump(Id2Vec,f)

if __name__ == '__main__':
    Main()
