#!/usr/bin/env python
import pandas as pd
import pickle
import os
import numpy as np
import biomapy as bp
import argparse

parser=argparse.ArgumentParser(description='Train the model and make predictions')
parser.add_argument('-m','--model',help='path of the trained model for GDA predictions')
parser.add_argument('-e','--embeddings',help='path in which the Id2Vec dictionary is')
parser.add_argument('-f','--features', default = 'Concatenation', help = 'GDA representation type used for predictions, it can be Sum, Concatenation, Hadmard, Average, the model have to be trained with the same representation')
parser.add_argument('-c','--CUI', help = 'cui code for whichrun the predictions', type = str)
parser.add_argument('-o','--output',help = 'path in which to save the predictions')

args=parser.parse_args()




def Loaddata(path):
    files = os.listdir(path)
    for f in files:
        if 'id2vec' in f.lower():
            id2vec_name = f
    
    with open(path + id2vec_name,'rb') as f:
        Id2Vec = pickle.load(f)
 
    
    return Id2Vec


def Main():

    # Load the embeddings
    Id2Vec = Loaddata(args.embeddings)
    
    # Load the model
    
    with open(args.model, 'rb') as handle:
        clf = pickle.load(handle)
    
    
    print('BUILDING THE DATASET')

    # Get the random embedding of IDD
    disease_embedding=Id2Vec[args.CUI]

    # all the genes for which I have embeddings
    
    genes_with_embeddings=[gen for gen in list(Id2Vec.keys()) if gen.isnumeric()]
    
    # Make de Dataset
    data =pd.DataFrame(zip(genes_with_embeddings,
                        [Id2Vec[gen] for gen in genes_with_embeddings], 
                        [disease_embedding for _ in range(len(genes_with_embeddings))]), columns=['geneId','geneEmb','disEmb'])
    
    
    if args.features == 'Concatenation':
        data[args.features]=data.apply(lambda x: np.concatenate((x['geneEmb'],x['disEmb'])),axis=1)
    elif args.features == 'Hadamard':
        data[args.features]=data.apply(lambda x: np.multiply(x['geneEmb'],x['disEmb']),axis=1)
    elif args.features == 'Sum':
        data[rgs.features]=data.apply(lambda x: np.sum((x['geneEmb'],x['disEmb']),axis=0),axis=1)
    elif args.features == 'Average':
        data[args.features]=data.apply(lambda x: np.sum((x['geneEmb'],x['disEmb']),axis=0)/2,axis=1)
    
    data = data.dropna()
    
    print('MAKING PREDICTIONS')
    
    # PREDICT #
    pred=clf.predict(data[args.features].tolist())
    predprob=clf.predict_proba(data[args.features].tolist())
    
    data['CUI'] = [args.CUI for _ in range(data.shape[0])]
    data['Predicted']=pred
    data['Proba'] = predprob[:,1]
    data['GeneSymbol']=bp.gene_mapping_many([int(g) for g in data.geneId.tolist()],'entrez','symbol')
    
    # SAVING THE PREDICTIONS
    
    data = data.sort_values(by='Proba', ascending = False)
    data.reset_index(drop = True).to_csv(args.output)
    
    data.to_csv('../outputs/IDD_predictions.csv')

    
    
    
if __name__ == '__main__':
    Main()
    