#!/usr/bin/env python

from collections import Counter
from datetime import datetime
from tqdm import tqdm
import pickle
import pandas as pd
import random
import seaborn as sns
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

parser=argparse.ArgumentParser(description='Run Generalization experiment')
parser.add_argument('-p','--path',help = """path of the id2vec embedding dictionary""",type = str,default = "../Embeddings/Metapath2Vec_final/Id2Vec.pickle")
parser.add_argument('-a','--algorithm',help = """type of classification algorithm can be either SVM for support vector machine or LR for logistic regression""",type = str,default = "SVM")

args = parser.parse_args()

def CreateTrainTest(Id2VecPath, TrainPath = '../TrainingSets/Dis_Train.csv', TestPath = '../TrainingSets/Dis_Test.csv' ):
    with open(Id2VecPath,'rb') as f:
        Id2Vec=pickle.load(f)
        
    Train=pd.read_csv(TrainPath)
    Test=pd.read_csv(TestPath)
    
    Train['dis_emb']=list(map(Id2Vec.get,Train.diseaseId.tolist()))
    Train['gene_emb']=list(map(Id2Vec.get,Train.geneId.astype(str).tolist()))
    Test['dis_emb']=list(map(Id2Vec.get,Test.diseaseId.tolist()))
    Test['gene_emb']=list(map(Id2Vec.get,Test.geneId.astype(str).tolist()))
    Train.dropna(inplace=True)
    Test.dropna(inplace = True)
    Train['Concatenation']=Train.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
    Test['Concatenation']=Test.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
 
    
    TrainingSet = pd.concat([Train,Test])

    
    with open('../TrainingSets/GDA.pickle','wb') as f:
        pickle.dump(TrainingSet,f)

   
def SimulatedExperiments(model,PathData = '../TrainingSets/GDA.pickle' , n_of_experiments = 100, path_out = '../outputs/GeneralizationExp/',exp_name = None):
    
    if not os.path.isdir(path_out):
        os.makedir(path_out)
    model_name = type(model).__name__

    with open(PathData,'rb') as f:
        TrainingSet = pickle.load(f)
    TotalCuis=list(set(TrainingSet.diseaseId.tolist()))
    precisions=[]
    recalls=[]
    aucs=[]
    for i in range(n_of_experiments):
        print(f"\tstarting experiment {i}")
        start = datetime.now()
        DiseaseSubsample=random.sample(TotalCuis,1000)
        TrainingSetNew=TrainingSet[TrainingSet.diseaseId.isin(DiseaseSubsample)]
        TrainingSetNoDis=TrainingSet[~TrainingSet.diseaseId.isin(DiseaseSubsample)]


        X= np.array(TrainingSetNoDis.Concatenation.tolist())
        y= np.array(TrainingSetNoDis.Label.tolist())
        X_new=np.array(TrainingSetNew.Concatenation.tolist())
        y_new=np.array(TrainingSetNew.Label.tolist())


        print("\t\tfitting the model")
        # Fit the models
        model.fit(X,y)


        print("\t\tmaking predictions")
        Predicted=model.predict(X_new)
        


        precisions.append(precision_score(Predicted,y_new))
        recalls.append(recall_score(Predicted,y_new))
        aucs.append(roc_auc_score(Predicted,y_new))
        end = datetime.now()
        print(f'\tExperiment {i} completed, started at {start.time()} ended at {end.time()}')


    print('Average Recall of the model is: ',np.mean(recalls),'Average Precision of the model is: ',np.mean(precisions),'Average AUC of the model is: ',np.mean(aucs))
    
    fig,ax=plt.subplots(figsize=(15,8))
    ax.plot(recalls,alpha=0.5)
    ax.plot(precisions,alpha=0.5)
    ax.axhline(np.mean(aucs),linestyle='--',c='r')
    ax.legend(labels=['recall','precision','mean_AUC'])
    ax.set_xlabel('Experiment',fontdict=dict(size=20))
    plt.tight_layout()

    plt.savefig(path_out + f'SimulatedExperiments_{model_name}.jpg',dpi=300)
    

def DiseaseClassMapper(ListOfIcds):
    DiseaseClasses={139: "infectious and parasitic diseases",
            239: "neoplasms",
            279: "endocrine, nutritional and metabolic diseases, and immunity disorders",
            289: "diseases of the blood and blood-forming organs",
            319: "mental disorders",
            389: "diseases of the nervous system and sense organs",
            459: "diseases of the circulatory system",
            519: "diseases of the respiratory system",
            579: "diseases of the digestive system",
            629: "diseases of the genitourinary system",
            679: "complications of pregnancy, childbirth, and the puerperium",
            709: "diseases of the skin and subcutaneous tissue",
            739: "diseases of the musculoskeletal system and connective tissue",
            759: "congenital anomalies",
            779: "certain conditions originating in the perinatal period",
            799: "symptoms, signs, and ill-defined conditions",
            999: "injury and poisoning"}
    DiseaseClassesList=[]
    for ix,icd in enumerate(ListOfIcds):
        Class=int(str(icd).split('.')[0])
        for i,(k,v) in enumerate(DiseaseClasses.items()):
            if i==0:
                range_lower=0
                range_upper=k
                if range_lower<=int(Class)<=range_upper:
                    DiseaseClassesList.append(i)
            else:
                range_lower=list(DiseaseClasses.items())[i-1][0]+1
                range_upper=k
                if range_lower<=int(Class)<=range_upper:
                    DiseaseClassesList.append(i)
    return DiseaseClassesList    
    
    

    
def CreateDiseaseClassExperimentDataset(Id2VecPath, PathData = '../TrainingSets/GDA.pickle' ,path_out='../TrainingSets/Data_for_heatmap.pickle'):
    with open('../datasets/cuicdmapper.pickle','rb') as handle:
        CuIcdMapper=pickle.load(handle)
    with open(PathData,'rb') as handle:
        TrainingSet=pickle.load(handle)
    with open(Id2VecPath,'rb') as handle:
        Id2Vec=pickle.load(handle)
        
    
    
    #Map disease to disease class    
    TrainingSet['ICD']=list(map(CuIcdMapper.get,TrainingSet.diseaseId.tolist()))
    TrainingSet=TrainingSet.dropna()
    TrainingSet = TrainingSet.reset_index()
    # Drop the rows that could not be mapped
    for ix,r in TrainingSet.iterrows():
        try:
            float(r['ICD'])
        except:
            TrainingSet=TrainingSet.drop(ix)
    TrainingSet['ICDClass']=DiseaseClassMapper(TrainingSet.ICD.tolist())
    
    #Load Disgenet
    DisgenetNonCurated=pd.read_csv('../datasets/all_gene_disease_associations.tsv',sep='\t')
    #Cuis in the trainingset (for which I have the IcdClass)
    CuisInTrainingset=TrainingSet.diseaseId.astype(str).tolist()
    GenesInTrainingset = TrainingSet.geneId.astype(str).tolist()
    #These are my associations coming from disgenet non curated (more data)
    PositiveGDA=list(zip(DisgenetNonCurated.geneId.astype(str).tolist(),DisgenetNonCurated.diseaseId.astype(str).tolist()))


    all_positives = TrainingSet[TrainingSet.Label == 1 ].shape[0]
    all_negatives = TrainingSet[TrainingSet.Label == 0 ].shape[0]
    negatives_to_add = all_positives - all_negatives
    
    print(f"The TrainingSet has {all_positives} and {all_negatives} negatives, adding {negatives_to_add} associations")

    
    #Create the negatives
    NegativeGDA=[]
    i=0
    pbar = tqdm(total = negatives_to_add)
    while i < negatives_to_add:
        disgene=(random.choice(GenesInTrainingset),random.choice(CuisInTrainingset))
        if disgene not in PositiveGDA:
            i+=1
            NegativeGDA.append(disgene)
            pbar.update(1)
    pbar.close()
    # Add Negatives and their embeddings to the trainingset
    NegDf=pd.DataFrame(NegativeGDA,columns=['geneId','diseaseId'])
    NegDf['geneId'] = NegDf['geneId'].astype(str)
    NegDf['gene_emb']=list(map(Id2Vec.get,NegDf.geneId.tolist()))
    NegDf['dis_emb']=list(map(Id2Vec.get,NegDf.diseaseId.tolist()))
    NegDf['Label']=0
    NegDf.dropna(subset = ['gene_emb','dis_emb'],inplace=True)
    NegDf['Concatenation']=NegDf.apply(lambda x: np.concatenate((x['gene_emb'],x['dis_emb'])),axis=1)
    TrainingSet=pd.concat([TrainingSet,NegDf])
    
    
    # Add the ICD name and Class to the new Added
    TrainingSet['ICD']=list(map(CuIcdMapper.get,TrainingSet.diseaseId.tolist()))
    TrainingSet['ICDClass']=DiseaseClassMapper(TrainingSet.ICD.tolist())
    
    TrainingSet=TrainingSet.sample(frac=1)
    
    #Drop DiseaseClasses With few examples
    TrainingSet=TrainingSet[(TrainingSet.ICDClass!=14) & (TrainingSet.ICDClass!=10)]
    
    final_positives = TrainingSet[TrainingSet.Label == 1 ].shape[0]
    final_negatives = TrainingSet[TrainingSet.Label == 0 ].shape[0]
    print(f"The final TrainingSet has {final_positives} and {final_negatives} negatives")
    
    GTS = TrainingSet.groupby("ICDClass").agg({"Label":list})
    for i,r in GTS.iterrows():
        LabelNumber = Counter(r["Label"])
        print(f"for DiseaseClass {i} the positives and negatives are {LabelNumber}")
    
    with open(path_out,'wb') as f:
        pickle.dump(TrainingSet,f)
        
        
def ProduceHeatmap(model, data_path = '../TrainingSets/Data_for_heatmap.pickle', out_path='../outputs/GeneralizationExp/',exp_name = None):
    
    DiseaseClasses={139: "infectious and parasitic diseases",
            239: "neoplasms",
            279: "endocrine, nutritional and metabolic diseases, and immunity disorders",
            289: "diseases of the blood and blood-forming organs",
            319: "mental disorders",
            389: "diseases of the nervous system and sense organs",
            459: "diseases of the circulatory system",
            519: "diseases of the respiratory system",
            579: "diseases of the digestive system",
            629: "diseases of the genitourinary system",
            679: "complications of pregnancy, childbirth, and the puerperium",
            709: "diseases of the skin and subcutaneous tissue",
            739: "diseases of the musculoskeletal system and connective tissue",
            759: "congenital anomalies",
            779: "certain conditions originating in the perinatal period",
            799: "symptoms, signs, and ill-defined conditions",
            999: "injury and poisoning"}
    
    
    model_name = type(clf).__name__
    with open(data_path,'rb') as handle:
        TrainingSet=pickle.load(handle)    # Validate the Model By training It on a single class and Validate it on the rest 
    
    icd_classes = list(set(TrainingSet.ICDClass.tolist()))
    Heatmap=pd.DataFrame(columns=icd_classes,index=icd_classes)
    
    
    for j in Heatmap.columns:
        start = datetime.now()
        #Define the temporary training and test set
        TmpTset=TrainingSet[TrainingSet.ICDClass==j]
        X= np.array(TmpTset.Concatenation.tolist())
        y= np.array(TmpTset.Label.tolist())
        # Fit the Model
        model.fit(X,y)

        # Validate the Model on Other Diseases
        for i in Heatmap.index:
            TTest=TrainingSet[TrainingSet.ICDClass==i]
            X_test=np.array(TTest.Concatenation.tolist())
            y_test=np.array(TTest.Label.tolist())

            # Predict the associations
            ypred= model.predict(X_test)
            Heatmap.at[i,j] = roc_auc_score(y_test,ypred)
        end = datetime.now()

        print(f'Experiment for disease class {j} completed, started at {start.time()}, ended at {end.time()}')




    Heatmap=Heatmap.astype(float)
    fig,ax =plt.subplots(figsize=(15,10))
    sns.heatmap(Heatmap,ax=ax,annot=True,cmap='crest',cbar_kws={'label': 'AUC'})

    HeatmapLab=[list(DiseaseClasses.values())[i] for i in Heatmap.index]
    ax.set_xticklabels(HeatmapLab,rotation=45,ha='right')
    ax.set_yticklabels(HeatmapLab,rotation=360)

    plt.tight_layout()
    plt.savefig(out_path+'_'+model_name+'_'+exp_name +'.jpg',dpi=300,bbox_inches = "tight")


            
            
def Main(verbose = True):
    if not os.path.isdir('../outputs/GeneralizationExp/'):
        os.makedir('../outputs/GeneralizationExp/')
    id2vec_path = args.path
    experiment_name = id2vec_path.split('/')[2]
    
    if args.algorithm == "SVM":
        clf = SVC(**{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'})
    elif args.algorithm == "LR":
        clf = LogisticRegression(max_iter=1000)
    if verbose:
        print(f'Creating Data for simulated Experiments with model {clf}... \n')
    CreateTrainTest(id2vec_path)
    
    # n_of_experiments = 100
    # if verbose:
    #     print(f'Running {n_of_experiments} Simulations... \n')
    # SimulatedExperiments(clf,n_of_experiments=n_of_experiments,exp_name=experiment_name)
    
    if verbose:
        print('Creating Data for Disease Classes Experiment... \n')
    CreateDiseaseClassExperimentDataset(id2vec_path)
    
    if verbose:
        print('Producing Heatmap \n')
    ProduceHeatmap(clf,exp_name = experiment_name)


if __name__ == '__main__':
    Main()
