import pandas as pd
import pickle
import math
import multiprocessing

import numpy as np
import os


#KERAS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout


# CLASSIFIERS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#MODEL SELECTION
from sklearn.model_selection import GridSearchCV


#EVALUATION METRICES
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore")




def DefineClassifiers(input_dim_for_keras_model):
    # LOGISTIC REGRESSION
    lr=LogisticRegression()

    #RANDOM FOREST
    rf=RandomForestClassifier()

    #GRADIENT BOOSTING FOREST
    xgb=XGBClassifier()

    #SUPPORT VECTOR MACHINE
    svm=SVC()

    
    def create_Keras_model(n_layers, first_layer_nodes, last_layer_nodes, activation_func, loss_func,input_dim=input_dim_for_keras_model,dropout_rate=0.2):
    
        def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
            layers = []

            nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
            nodes = first_layer_nodes
            for i in range(1, n_layers+1):
                layers.append(math.ceil(nodes))
                nodes = nodes + nodes_increment

            return layers

        model=Sequential()

        n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)

        model.add(Input(shape=(input_dim,)))
        for i in range(1, n_layers):
            if i==1:
                model.add(Dense(first_layer_nodes,activation=activation_func))
            else:
                model.add(Dense(n_nodes[i-1], activation=activation_func))

        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',loss=loss_func, metrics = ["AUC"])

        return model
    ##Wrap model into scikit-learn
    BinaryFFN =  KerasClassifier(build_fn=create_Keras_model)
    BinaryFFN._estimator_type = "classifier" 
    
    
    
    
    
    models=[BinaryFFN,lr,rf,xgb,svm]
    
    return models


def DefineSearchSpaces():
    lr_parameters={
    'penalty': ['l1', 'l2'],
    'C':[0.001,0.01,1,5,10,25]
    }

    rf_parameters={
        'max_depth': [2, 4, 6, None],
        'n_estimators': [20, 50, 100]
    }
    xgb_parameters={
        "colsample_bytree": [0.7, 0.3],
        "gamma": [0, 0.5],
        "learning_rate": [0.03, 0.3],
        "max_depth": [2, 6], 
        "n_estimators": [100, 150], 
        "subsample": [0.6, 0.4]

    }

    svm_parameters={
        'C': [0.1,1, 10], 
        'gamma': [0.1,0.01,0.001],
        'kernel': ['rbf', 'poly']
        
    }
    
    
    BinaryFFN_parameters={
        'n_layers': [2, 3], 
        'first_layer_nodes': [150, 100, 50], 
        'last_layer_nodes': [50,20], 
        'activation_func': ['sigmoid','relu', 'tanh'], 
        'loss_func': ['binary_crossentropy', 'hinge'], 
        'batch_size': [30,100], 'epochs': [20, 60]
        }
    
    parameters=[BinaryFFN_parameters,lr_parameters,rf_parameters,xgb_parameters,svm_parameters]
    
    return parameters


    
    

def load_data(root_dir,FeaturesName="Concatenation",LabelName="Label"):
    
    files=os.listdir(root_dir)
    
    for file in files:
        if 'train' in file.lower():
            with open(root_dir+'/' + file,'rb') as f:
                Train=pickle.load(f)
        elif 'test' in file.lower():
            with open(root_dir+'/' + file,'rb') as f:
                Test=pickle.load(f)
    
    X_train=np.array(Train[FeaturesName].tolist())
    y_train=np.array(Train[LabelName].astype(int).tolist())
    X_test=np.array(Test[FeaturesName].tolist())
    y_test=np.array(Test[LabelName].astype(int).tolist())
    
    return X_train, y_train, X_test, y_test
 
def GridSearch(Classifier,parameters,X_train,y_train,n_of_cores):
    grid = GridSearchCV(Classifier, parameters, scoring='roc_auc', n_jobs=n_of_cores, verbose=5,refit=True)
    grid.fit(X_train, y_train)
    best_params=grid.best_params_
    return grid.best_estimator_ , best_params

def EvaluateOnTest(best_model,X_test,y_test):
    y_pred=best_model.predict(X_test)
    metric_list=[]
    waf = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    prauc=average_precision_score(y_test,y_pred)
    auc=roc_auc_score(y_test, y_pred)
    metric_list.append(waf)
    metric_list.append(precision)
    metric_list.append(recall)
    metric_list.append(accuracy)
    metric_list.append(auc)
    metric_list.append(prauc)

    return metric_list

def WriteBestParams(file_path,string_to_write):
    with open(file_path,'a') as f:
        f.write(string_to_write)
        
def Main():
    print("Number of available cores ",multiprocessing.cpu_count())
    working_dir= '../../Embeddings/'
    classifier_names=['ffn','lr','rf','xgb','svm']
    metrics_headers=['F1','PRECISION','RECALL','ACCURACY','AUC','PRAUC']
    metrics_headers_str = '\t'.join(metrics_headers)
#     evaluation_dataframe=pd.DataFrame(columns=['F1','PRECISION','RECALL','ACCURACY','AUC','PRAUC'])
    with open('./outputs/metrics.txt','w') as f:
        f.write('Grid_Instance' + '\t' + metrics_headers_str)
    datasets=os.listdir(working_dir)
    operations=['Concatenation','Sum','Average','Hadmard']
    parameters=DefineSearchSpaces()
    for data in datasets:
        for combination_type in operations:
            if combination_type == 'Concatenation':
                input_size = 200
            else:
                input_size = 100

            classifiers = DefineClassifiers(input_size)
            
            X_train, y_train, X_test, y_test=load_data(working_dir+data,FeaturesName=combination_type)
            for (classifier,grid,classifier_name) in zip(classifiers,parameters,classifier_names):
                algorithm_name = data.split('_')[0]
                print(f"OOOOOOOOOOOOOOOOOOOOOOO testing {algorithm_name} - {combination_type} - {classifier_name} OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                best_classifier,best_params=GridSearch(classifier,grid,X_train,y_train,-1)
                metric_list=EvaluateOnTest(best_classifier,X_test,y_test)
                index_string=algorithm_name + '_' + combination_type + '_' + classifier_name
#                 evaluation_dataframe.loc[index_string] = metric_list
                metric_list_str = '\t'.join(str(m) for m in metric_list)
                print('======================================>,',str(index_string),str(best_params))
                WriteBestParams('../outputs/GSCV_log.txt','\n' + index_string+'\t'+str(best_params))
                WriteBestParams('../outputs/metrics_GSCV.txt', '\n' + index_string+ '\t' + metric_list_str )
#    evaluation_dataframe.to_csv('./outputs/CV_Metrics.csv')
            
        
        
if __name__=='__main__':
    Main()
    
