#!/usr/bin/env python
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import argparse

parser=argparse.ArgumentParser(description='Perform the experiment of investigating the effect of preprocessing and  adding different ontologies to the KG')
parser.add_argument('-a','--algorithm',help = """Choose which algorithm of embeddings creation to use, Dlemb and M2V are implemented""",type = str,default = "Dlemb")
args = parser.parse_args()

def Main():
    
    outpath = "../outputs/randomVsEmbs/"
    if not os.path.isdir(outpath):
        os.makedir(outpath)
    
    DisCur = pd.read_csv('../datasets/disgenet_parsed.csv')
    DisNCur = pd.read_csv('../datasets/all_gene_disease_associations.tsv',sep='\t')

    DiseaseClasses = DisCur.groupby('diseaseId').agg({'geneSymbol':'count'})
    
    if args.algorithm == 'Dlemb':
        paths = [
                '../Embeddings/Dlemb_final/Id2Vec.pickle', 
                '../Embeddings/Random/Id2Vec.pickle'
        ]
    elif args.algorithm == 'M2V':
        paths = [
                '../Embeddings/Metapath2Vec_final/Id2Vec.pickle', 
                '../Embeddings/Random/Id2Vec.pickle'
        ]
        



    
    DiseaseClasses = DisCur.groupby('diseaseId').agg(GDA_number = ('geneSymbol','count'))
    all_associations = list(zip(DisNCur.geneId.tolist(),DisNCur.diseaseId.tolist()))


    # We want classes every n GDAs
    size_of_class = 20
    max_number_of_associations = DiseaseClasses.GDA_number.max()
    n_of_classes = max_number_of_associations / size_of_class
    number_of_associations = np.linspace(1,max_number_of_associations,round(n_of_classes),dtype=int)


    print('Creating Disease Classes')

    DiseaseClasses['AssociationClass'] = ''
    for nass in tqdm(number_of_associations,position=0, leave=True):
        for i,r in DiseaseClasses.iterrows():
            if nass <= r['GDA_number'] <= nass + size_of_class:
                DiseaseClasses.at[i,'AssociationClass'] = nass
                

    print(DiseaseClasses.AssociationClass.value_counts())
    association_class_dict = dict(zip(DiseaseClasses.index,DiseaseClasses.AssociationClass.tolist()))

    all_associations = list(zip(DisNCur.geneId.tolist(),DisNCur.diseaseId.tolist()))

    # Map the disease classes to Disgenet
    DisCur['diseaseClass'] = list(map(association_class_dict.get,DisCur.diseaseId.tolist()))

    list_of_scores = {}
    
    # Create a whole dataset and shuffle it
    dataset = DisCur[['geneId','diseaseId','diseaseClass']]
    dataset = dataset.sample(frac=1)

    clfs = [
        SVC(**{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}, probability=True),
        # LogisticRegression()
        ]

    for clf in clfs:
        for path in paths:
            algorithm_name = path.split('/')[2].split("_")[0]
            with open(path,'rb') as f:
                Id2Vec = pickle.load(f)
            genes_in_dataset = [g for g in list(Id2Vec.keys()) if g.isnumeric()]
            clas = list(set(DiseaseClasses.AssociationClass.tolist()))
            clas = [e for e in clas if e !='']

            for c in sorted(clas)[1:]:
                
                try:
                    # Select Just One class
                    dataset_single = dataset[dataset.diseaseClass == c]
                    
                    # Random a sample disease (the one most associated) from the dataset
                    max_number_of_association = dataset_single.diseaseId.value_counts().values[0]
                    most_associated_cui = dataset_single.diseaseId.value_counts().index[0]
                    
                    # Filter the dataframe for that disease 
                    dataset_single = dataset_single[dataset_single.diseaseId == most_associated_cui]
                    
                    #Create the negatives
                    negative_associations = []
                    print(f'Creating Negatives for class {c}, Total number of associations: {dataset_single.shape[0]}')
                    i = 0

                    total_iterations = dataset_single.shape[0]

                    with tqdm(total = total_iterations, position=0, leave=True) as pbar:
                        while i <= dataset_single.shape[0]:
                            random_association = (random.choice(genes_in_dataset),most_associated_cui)
                            if random_association not in all_associations:
                                negative_associations.append(random_association)
                                i += 1
                                pbar.update(1)
                            else:
                                pass

                    # Merge negatives and positives
                    negative_association_df = pd.DataFrame(negative_associations,columns=['geneId','diseaseId'])        
                    dataset_new = dataset_single[['geneId','diseaseId']]

                    dataset_new['Label'] = 1
                    negative_association_df['Label'] = 0


                    TrainingSet = pd.concat([dataset_new,negative_association_df])

                    TrainingSet = TrainingSet.sample(frac = 1).astype(str)
                    
                    TrainingSet['Label'] = TrainingSet['Label'].astype(int)
                    
                    TrainingSet['dis_emb']=list(map(Id2Vec.get,TrainingSet.diseaseId.tolist()))
                    TrainingSet['gene_emb']=list(map(Id2Vec.get,TrainingSet.geneId.tolist()))
                    print(TrainingSet.head())
                    print(f'size of data before mapping: {TrainingSet.shape[0]}')
                    TrainingSet.dropna(inplace = True)
                    print(f'size of data after mapping: {TrainingSet.shape[0]}')
                    
                    print(f'Ratio of positives/negatives of data: {TrainingSet.Label.sum()/len(TrainingSet.Label.tolist())}')

                    #CREATE EMBEDDING FRATURES

                    TrainingSet['Sum']=TrainingSet.apply(lambda x: np.sum((x['gene_emb'],x['dis_emb']),axis=0),axis=1)
                    TrainingSet['Concatenation']=TrainingSet.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
                    TrainingSet['Average']=TrainingSet.apply(lambda x: np.sum((x['gene_emb'],x['dis_emb']),axis=0)/2,axis=1)
                    TrainingSet['Hadmard']=TrainingSet.apply(lambda x: np.multiply(x['gene_emb'],x['dis_emb']),axis=1)
                    
                
                    X = np.stack(TrainingSet['Concatenation'].values)
                    y = TrainingSet["Label"].astype(int).values
                    
                    
                    print('evaluating')
                    scores = cross_validate(clf,X,y,scoring=['accuracy','precision','recall','f1','roc_auc'])
                    
                    list_of_scores[(most_associated_cui,max_number_of_association)] = scores
                except Exception as e:
                    print(e)
                    pass
                    
            with open(f"../outputs/randomVsEmbs/{algorithm_name}_{str(clf)}.pickle",'wb') as f:
                pickle.dump(list_of_scores,f)


    

    
    
if __name__ == '__main__':
    Main()   
