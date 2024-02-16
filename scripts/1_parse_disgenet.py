#!/usr/bin/env python3

import pandas as pd

def Main():
    disgenet = pd.read_csv('../datasets/curated_gene_disease_associations.tsv',sep='\t')
    disgenet_grouped = disgenet.groupby('diseaseId').agg({'geneId':list,'diseaseName':'first','geneSymbol':'first'})
    diseases = disgenet_grouped.index.tolist()
    nofassociations = 20
    for disease in diseases:
        try:
            #if we cannot find the diseases means it has been dropped
            associations = disgenet_grouped[disgenet_grouped.index == disease].geneId.tolist()[0]
            if len(associations) > nofassociations:
                for ix,ass,dis in zip(range(disgenet_grouped.shape[0]),disgenet_grouped.geneId.tolist(),disgenet_grouped.index.tolist()): #checking all the other associations
                    if ((dis != disease)&(len(set(associations).intersection(set(ass))) > 0.90*nofassociations)):
                        disgenet_grouped = disgenet_grouped.drop(dis)
                        # print('dropped',dis)
        except Exception as e:
            pass
    di2 = disgenet_grouped.explode('geneId')
    di2.reset_index(inplace=True)
    
    di2.to_csv('../datasets/disgenet_parsed.csv',columns=['diseaseId','geneId','geneSymbol'])

if __name__=='__main__':
    Main()

    
