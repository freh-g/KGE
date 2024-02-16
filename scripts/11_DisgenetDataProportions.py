#!/usr/bin/env python

import pickle
import pandas as pd
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
import argparse

parser=argparse.ArgumentParser(description='Carry out the experiment of data proportion')
parser.add_argument('-g','--generate_kg',help = """if true generates KGS with different DisGeNET proportions""",type = str,default = "False")
parser.add_argument('-e','--embs',help = """Which embeddings to test, available options are Dlemb and M2V""",type = str,default = "Dlemb")
parser.add_argument('-a','--algorithm',help = """classification algorithm to use, the possibilities are either SVM for support vector machines or LR for logistic regression""",type = str,default = "SVM")
args = parser.parse_args()

def CreateEmbeddings(kg):
    
    edgelist = nx.to_pandas_edgelist(kg)
    
    edgelist.to_csv('kg_edgelist_de.csv')
    print('creating_embeddings')
    if args.embs == 'Dlemb':
        os.system('../tools/DLemb-main/./DLemb.py -i ./kg_edgelist_de.csv -o ./kg_embeddings_de.pickle -e 1')
    elif args.embs == 'M2V':
        os.system('../tools/runMetapath2Vec/./Metapath2vec.py --edgelist ./kg_edgelist_de.csv --output ./kg_embeddings_de.pickle')
    with open('kg_embeddings_de.pickle','rb') as f:
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



def EvaluatePerformance(model,X_train,y_train,X_test,y_test,ax1,ax2,pctg,savepath):
    
    model.fit(X_train,y_train)
    
    
    # PLOT TEST AUC ###################################################
    auc_test=RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name="ROCAUC_"+pctg+"%_DisGeNET",
            alpha=1,
            lw=2,
            ax=ax1)
    
    # PLOT TEST PRAUC ###################################################
    prauc_curve_test = PrecisionRecallDisplay.from_estimator(model,
                                                      X_test,
                                                      y_test,
                                                      name="PRAUC_"+pctg+"%_DisGeNET",
                                                      alpha=1,
                                                      lw=2,
                                                      ax=ax2)
    plt.savefig(savepath, dpi=300)


    

    
def Main():
    
    pctgs_kg = [20,50,80,100]
    if args.generate_kg == "True":
        print("Creating KGs")
        for pc in pctgs_kg:
            os.system(f"./3_Create_KG.py --Dpcgt {pc}")
    if args.algorithm == "SVM":
        clf = SVC(**{'C': 10, 'kernel': 'rbf'})
    elif args.algorithm == "LR":
        clf = LogisticRegression()
    model_name = type(clf).__name__
    kg_path = '../KGs/'

    kgs = os.listdir(kg_path)
    
    pctgs = [a.split('_')[1].rstrip('.pickle') for a in kgs]
    print(pctgs)
    fig, (ax1,ax2) = plt.subplots(2, 1,sharex=True,figsize=(20,12))
    
    for kg_name,pctg in zip(kgs,pctgs):
        dis_train=pd.read_csv('../TrainingSets/Dis_Train.csv')
        dis_test=pd.read_csv('../TrainingSets/Dis_Test.csv')
        print(pctg)
        with open(kg_path+kg_name,'rb') as f:
            kg = pickle.load(f)
        
        embs = CreateEmbeddings(kg)
        
        X_tr,y_tr,X_te,y_te = CreateTrainTest(dis_train,dis_test, embs)
        
        EvaluatePerformance(clf,X_tr,y_tr,X_te,y_te,ax1,ax2,
                           pctg,f'../outputs/plot_{pctg}.jpg')
    
    plt.savefig(f'../outputs/DataExp/{args.embs}_{model_name}.png',dpi=300,bbox_inches = "tight")

        

    
    
if __name__ == '__main__':
    Main()
