#!/usr/bin/env python
import pandas as pd
import pickle
import argparse 
import torch
import random
import numpy as np
from keras.layers import Input, Embedding, Dot, Reshape
from keras.models import Model

parser=argparse.ArgumentParser(description='Creates embeddings out of a Knowledge graph')
parser.add_argument('-i','--edgelist',help = """path of the edgelist, it have to be in csv format, columns names have to be [source, target, relation_type, source_type, target_type] and having no index""",type = str)
parser.add_argument('-p','--positives',help = """ number of positive training instances (entity pairs) to train the algorithm""",type = int,default = 1500)
parser.add_argument('-n','--negative_ratio',help = "ratio of negatives, default 1 i.e. same number of positives",type = int, default = 1)
parser.add_argument('-s','--embedding_size',help = 'length of the embeddings',type = int, default = 100)
parser.add_argument('-e','--n_epochs', help = 'number of epochs to train the model for',type = int, default = 100)
parser.add_argument('-t','--epoch_steps', help = 'Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch, default is n_triplets/n_positive_triplets)',type = int)
parser.add_argument('-o','--output',help = "where to save the embedding dictionary in pickle format",type = str)
args = parser.parse_args()



def generate_batch(triplets, n_positive, negative_ratio ):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
            
    entities=set([t[0] for t in triplets]+[t[2] for t in triplets])
    pairs = [(t[0],t[2]) for t in triplets]
    set_of_pairs = set(pairs)
    
    batch = np.zeros((batch_size, 3))

 
    neg_label = -1

    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (head, tail) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (head, tail, 1)

        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:

            # random selection
            random_head = random.randrange(len(entities))
            random_tail = random.randrange(len(entities))


            # Check to make sure this is not a positive example
            if (random_head, random_tail) not in set_of_pairs:

                # Add to batch and increment index
                batch[idx, :] = (random_head, random_tail, neg_label)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
    
        yield {'head': batch[:, 0], 'tail': batch[:, 1]}, batch[:, 2]

        
def DLemb(node_emb_size,input_dimension):   
    # Inputs for h,t
    h = Input(name = 'head', shape = [1])
    t = Input(name = 'tail',shape = [1])
        
    nodes_embedding_layer = Embedding(name = 'node_embeddings',
                                input_dim = input_dimension,
                                output_dim = node_emb_size)

    head_embedding = nodes_embedding_layer(h)
    tail_embedding = nodes_embedding_layer(t)
    
    dotted_embedding = Dot(name = 'dot_product',normalize = True, axes = 2)([head_embedding,
                                                                                         tail_embedding])
    merged = Reshape(target_shape = [1])(dotted_embedding)

            
    model = Model(inputs = [h, t], outputs = merged)

    model.compile(optimizer = 'Adam', loss = 'mse')

    return model



def Main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Detected device for training: {device}')
    edgelist = pd.read_csv(args.edgelist)
    edgelist = edgelist.astype(str)
    entities= list(set(edgelist.source.tolist()+edgelist.target.tolist()))
    interactions=list(set(edgelist.rel_type.tolist()))


    id_to_entity=dict(enumerate(entities))
    id_to_relation=dict(enumerate(interactions))

    entity_to_id={v:k for k,v in id_to_entity.items()}
    relation_to_id={v:k for k,v in id_to_relation.items()}

    heads_ids=list(map(entity_to_id.get,edgelist.source.tolist()))

    relations_ids=list(map(relation_to_id.get,edgelist.rel_type.tolist()))

    tails_ids=list(map(entity_to_id.get,edgelist.target.tolist()))


    triplets=list(zip(heads_ids,relations_ids,tails_ids))
    
    model = DLemb(args.embedding_size,len(entity_to_id))
    model.summary()

    gen = generate_batch(triplets, args.positives, args.negative_ratio)

    if args.epoch_steps:
        steps = args.epoch_steps
    else:
        steps = len(triplets) // args.positives
    
    # Train
    model.fit(gen, epochs = args.n_epochs, 
                            steps_per_epoch = steps,
                            verbose = 2)
    
    # Extract embeddings
    node_embeddings = model.get_layer('node_embeddings')
    node_embeddings = node_embeddings.get_weights()[0]
    node_embeddings.shape

    Id2Vec=dict(zip(id_to_entity.values(),node_embeddings))
    
    with open(args.output,'wb') as f:
        pickle.dump(Id2Vec,f)
if __name__ == '__main__':
    Main()
        