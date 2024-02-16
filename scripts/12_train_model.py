#!/usr/bin/env python
import pandas as pd
import pickle
import ast
import os
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF


parser=argparse.ArgumentParser(description='Train and save the model on specific embeddings')

parser.add_argument('-m','--model',help='model type, it can be SVM RF and LR for using respectively support vector machine, random forest or logistic regression')
parser.add_argument('-p','--parameters',help = """optional: parameters to be passed to the model, they can be passed in a from of a python dictionary in string from ex : "{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}"  """,type = str)
parser.add_argument('-f','--features', default = 'Concatenation', help = 'feature type to use, it can be Sum, Concatenation, Hadmard, Average')
parser.add_argument('-e','--embeddings',help='path in which the Id2Vec dictionary is')
parser.add_argument('-o','--output',help = 'path in which to save the model')

args=parser.parse_args()


def Loaddata(path):
    files = os.listdir(path)
    for f in files:
        if 'train' in f.lower():
            train_name = f
        elif 'test' in f.lower():
            test_name = f
    with open(path + train_name,'rb') as f:
        Train = pickle.load(f)
        
    with open(path + test_name,'rb') as f:
        Test = pickle.load(f)
    
        
    df = pd.concat([Train,Test])
    
    return df


def Main():
    
    TrainSet = Loaddata(args.embeddings)
    TrainSet.Label = TrainSet.Label.astype(int)
    
    # # DEFINE TRAINING AND TEST SET
    X= np.array(TrainSet[args.features].tolist())
    y= np.array(TrainSet.Label.tolist())
    
    
    
    if args.model == 'SVC':
        if args.parameters:
            parameters = ast.literal_eval(args.parameters)
            clf = SVC(**parameters,probability = True)
        else:
            clf = SVC(probability = True)
    
    elif args.model == 'RF':
        if args.parameters:
            parameters = ast.literal_eval(args.parameters)
            clf = RF(**parameters)
        else:
            clf = RF()
    if args.model == 'LR':
        if args.parameters:
            parameters = ast.literal_eval(args.parameters)
            clf = LR(**parameters)
        else:
            clf = LR()
            
    print (f'SELECTED MODEL = {clf}')
    
    # # FIT THE MODEL ON THE TRAINING SET 
    clfit = clf.fit(X,y)
    
    # SAVE THE FITTED MODEL
    with open(args.output,'wb') as f:
        pickle.dump(clfit,f)

        
        
if __name__ == '__main__':
    Main()
        