#!/usr/bin/env python3


import pandas as pd
import random
import os
def Main():
    print("\n\t.....LOADING DATA.....")
    #Load Disgenet
    disnc= pd.read_csv('../datasets/all_gene_disease_associations.tsv',sep='\t')
    discur=pd.read_csv('../datasets/disgenet_parsed.csv')
    discur.geneId=discur.geneId.astype(str)
    disnc.geneId=disnc.geneId.astype(str)
    discur=discur[['geneId','diseaseId']]
    disnc=disnc[['geneId','diseaseId']]
    print(f'disgenet curated shape {discur.shape}')
    print(f'disgenet non-curated shape {disnc.shape}')

    # ADD RANDOMLY GENERATED NEGATIVES IF THEY AREN'T IN DISGENET NON CURATED

    print("\n\t.....MAKE RANDOM NEGATIVE ASSOCIATIONS.....")

    PositiveAssociations=list(zip(disnc.geneId.tolist(),disnc.diseaseId.tolist()))
    NegativeAssociations=[]
    Genes=list(set(discur.geneId.tolist()))
    Diseases=list(set(discur.diseaseId.tolist()))
    for i in range(discur.shape[0]):
        added=False
        while not added:
            RandGen=random.choice(Genes)
            RandDis=random.choice(Diseases)
            RandomAss=(RandGen,RandDis)
            if RandomAss not in PositiveAssociations:
                NegativeAssociations.append(RandomAss)
                added=True
            else:
                print(f'Association {RandomAss} is positive, length of Negative associations list: {len(NegativeAssociations)}')
            

    # MAKE A DATAFRAME OUT OF THEM
    discur['Label']=1
    NegAssDf=pd.DataFrame(NegativeAssociations,columns=['geneId','diseaseId'])
    NegAssDf['Label']=0
    AssDf=pd.concat([discur,NegAssDf])
    AssDf=AssDf.sample(frac=1)
    AssDf.reset_index(drop=True,inplace=True)


    # SPLIT IN TRAIN AND TEST MAKING SHURE THAT I HAVE ALL THE SAME GENES IN THE TRAINING AND IN THE TEST
    
    print("\n\t.....TRAIN TEST SPLIT.....")

    Train=AssDf.iloc[:int(AssDf.shape[0]*0.80)]
    Test=AssDf.iloc[int(AssDf.shape[0]*0.80):]

    #FILTER THE TEST IN ORDER THAT ALL THE GENES AND DISEASE ARE REPRESENTED IN THE 80% OF THE DATASET (SO THAT WE CAN PRODUCE THE EMBEDDINGS)
    Test=Test[(Test.geneId.isin(Train.geneId.tolist()))&(Test.diseaseId.isin(Train.diseaseId.tolist()))]
    #CHECK THE BALANCING OF THE DATASET
    print(f'Ration of the positive to negative labels after filtering = {Test.Label.value_counts().values[0]/Test.Label.value_counts().values[1]}')


    # SAVE THE DATASETS
    if not os.path.isdir('../TrainingSets/'):
        os.makedirs('../TrainingSets/')
                          
    Train.to_csv('../TrainingSets/Dis_Train.csv',index=False)
    Test.to_csv('../TrainingSets/Dis_Test.csv',index=False)
if __name__ == '__main__':
    Main()

