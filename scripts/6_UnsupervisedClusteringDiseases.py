#!/usr/bin/env python3


import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import random
import ast
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics.cluster import homogeneity_score

parser=argparse.ArgumentParser(description='Reduce dimension of the embeddings')

parser.add_argument('--method',help = "method for dimensionality reduction" ,type = str,default='PCA')
parser.add_argument('--dc',help = "disease classes to plot, numbers based on the dictionary that is printed",type = str,default=str(list(range(0,17))))

args = parser.parse_args()



def reduce_dim(embeddings, components = 3, method = 'TSNE'):
    """Reduce dimension of embeddings"""
    if method == 'TSNE':
        x_red = TSNE(components, metric = 'cosine').fit_transform(embeddings)
        return x_red
    elif method == 'UMAP':
        # Might want to try different parameters for UMAP
        x_red = umap.UMAP(n_components=components, metric = 'cosine', 
                    init = 'random', n_neighbors = 5,min_dist = 0).fit_transform(embeddings)
        return x_red
    elif method == 'PCA':
        pca = PCA(n_components=components)
        x_red = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        # print(f'explained variance ratio: {exp_var_pca}' +'\n' + f'cumulative variance ratio: {cum_sum_eigenvalues}')
        return x_red

def PrepareData(cuis,red,cui2name,diseasesclasses=None,columns = ['pc1','pc2','pc3'],DiseaseClasses = dict):

    names=[]
    for cui in cuis:
        try:
            names.append(cui2name[cui])
        except:
            names.append(None)

    Redf=pd.DataFrame(red,columns=columns)
    Redf['Name']=names
    Redf['CUI']=cuis
    Redf=Redf[~Redf.Name.isna()]
    with open('../datasets/cuicdmapper.pickle','rb') as handle:
        CuIcdMapper=pickle.load(handle)


    def dictmapper(dictio,element):
        try:
            return(dictio[element])
        except:
            return None
    Redf['Icd']=Redf.CUI.apply(lambda x: dictmapper(CuIcdMapper,x))
    Redf=Redf.dropna()
    DisRdata=Redf

    DisRdata=DisRdata.reset_index(drop=True)


    rowtodrop=[]
    for i,element in enumerate(DisRdata.Icd.tolist()):
        try:
            f=float(element)
        except:
            rowtodrop.append(i)
    DisRdata.drop(rowtodrop,axis=0,inplace=True)
    DisRdata=DisRdata.reset_index(drop=True)

    DiseaseClassesList=[]
    for ix,icd in enumerate(DisRdata.Icd.tolist()):
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
    DisRdata['DiseaseClass']=DiseaseClassesList
    DisRdata['DiseaseClassName'] = [list(DiseaseClasses.values())[i] for i in DiseaseClassesList]
    if diseasesclasses:
        return DisRdata[DisRdata.DiseaseClass.isin(diseasesclasses)]
    else:
        return DisRdata
    

def plot_dimensions(DisRdata,diseaseclasses=[],dim1=0,dim2=1,annotate=[],neighbors=False,save=False,figsize=(20,15),legend=True,title = False,axes=None,figure = None,DiseaseClasses = dict):
    
    
    NUM_COLORS = len(DisRdata.DiseaseClass.value_counts().index)
    DISEASE_CLASSES=DisRdata.DiseaseClass.value_counts().index
    LISTOFCOLORS=['#4363d8', '#f58231', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    colorindex=dict(zip(DISEASE_CLASSES,[n for n in range(len(DISEASE_CLASSES))]))


    def plot_neighbors(disease,n=10):
        disease_vector=DisRdata[DisRdata.Name==disease][['pc1','pc2','pc3']].values
        vectors_reduced=DisRdata[['pc1','pc2','pc3']].values
        disease_index=DisRdata[DisRdata.Name==disease].index[0]
        norma = np.linalg.norm(vectors_reduced, axis=1)
        normalized = (vectors_reduced.T / norma).T
        dists = np.dot(normalized, normalized[disease_index])
        closest = np.argsort(dists)[-n:]
        neighbors=[]
        for c in reversed(closest):
            neighbors.append(DisRdata.iloc[c].Name)
        return neighbors

            
    
    
    
    for i in range(NUM_COLORS):
        setofDiseases=DisRdata[DisRdata.DiseaseClass==DISEASE_CLASSES[i]]
        disease_to_annotate=set(annotate).intersection(setofDiseases.Name.tolist())
        if DISEASE_CLASSES[i] in diseaseclasses:
            axes.scatter(setofDiseases.values[:,dim1],
                       setofDiseases.values[:,dim2],
                       c=LISTOFCOLORS[i],
                       marker='P',
                       label=DiseaseClasses[list(DiseaseClasses)[DISEASE_CLASSES[i]]],
                       s=300,
                      linewidth=1,
                      edgecolors='black')
        else:
            axes.scatter(setofDiseases.values[:,dim1],
                       setofDiseases.values[:,dim2],
                       c=LISTOFCOLORS[i],
                       label=DiseaseClasses[list(DiseaseClasses)[DISEASE_CLASSES[i]]],
                      s=30)
        if len(disease_to_annotate)>0:
            if neighbors:
                for dis in disease_to_annotate:
                    neigh=plot_neighbors(dis,neighbors)
                    xses=[]
                    yses=[]
                    labels=[]
                    box_colors=[]
                    for i,n in enumerate(neigh):
                        if i%2==0:
                            factor=10*random.random()
                        else:
                            factor=-10*random.random()
                        x=DisRdata[DisRdata.Name==n].values[:,dim1][0]
                        y=DisRdata[DisRdata.Name==n].values[:,dim2][0]
                        ax.annotate(n,(x,y),xytext=(x+factor,y+factor),
                                    weight='bold',fontsize=12,ha='center',va='center', bbox= dict(boxstyle="round4", 
                                    fc=LISTOFCOLORS[colorindex[DisRdata[DisRdata.Name==n].DiseaseClass.values[0]]],alpha=0.5))
        
                        
#                     xses.append(DisRdata[DisRdata.Name==n].values[:,dim1][0])
#                     yses.append(DisRdata[DisRdata.Name==n].values[:,dim2][0])
#                     labels.append(n)
#                     color=LISTOFCOLORS[colorindex[DisRdata[DisRdata.Name==n].DiseaseClass.values[0]]]
#                     gr.labelling_without_overlapping(xses,yses,labels,
#                                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=3),
#                                      weight='bold',fontsize=12,ha='center',va='center', 
#                                     bbox= dict(boxstyle="round4", fc=box_colors[i]  ,alpha=0.5,linewidth=0.1),ax=ax)

            else:
                for dis in disease_to_annotate:
                    x=DisRdata[DisRdata.Name==dis].values[:,dim1]
                    y=DisRdata[DisRdata.Name==dis].values[:,dim2]
                    ax.annotate(dis,(x,y),fontsize=13)
    if legend:
        figure.legend(
            loc='center left', 
                   bbox_to_anchor=(1, 0.5),markerscale=1.5,fontsize=13
                 )
    if title: 
        plt.title(title,fontdict=dict(fontsize=20))
    axes.set_xticks([])
    axes.set_yticks([])
    if save:
        plt.savefig(save,bbox_inches = 'tight',dpi = 300)



def Main():
    dc={139: "infectious and parasitic diseases",
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
    
    print(dict(enumerate(dc.values())))
    Homogeneity_df = pd.DataFrame(columns = ["CUIS_homogeneity_score"])
    # Produce multiple heatmaps wit facet plots
    columns = 2
    rows = 3
    row = 0
    col = 0
    disease_to_plot = ast.literal_eval(args.dc)
    facet_disease, facet_axes_disease = plt.subplots(rows,columns, sharex=True,sharey = True,figsize = (10,10))
    heatmap_fig_dis, heatmap_axes_dis = plt.subplots(rows,columns, sharex=True,sharey = True,figsize = (10,10))
    shilouette_fig_dis, shilouette_axes_dis = plt.subplots(rows,columns, sharex=True,figsize = (10,10))

    with open('../datasets/CUI_NAME.RRF','rb') as f:
        cuiname=f.readlines()
    cuiname=[stri.decode('latin-1') for stri in cuiname]
    CuiToName=dict(zip([st.split('|')[0] for st in cuiname[1:]],[st.split('|')[1] for st in cuiname[1:]]))

    for element in os.listdir('../Embeddings/'):
        

        # DISEASE PART
        path = f'../Embeddings/{element}/'
        id2vec = [ele for ele in os.listdir(path) if 'id2vec' in ele.lower()][0]
        Id2VecPath = path + id2vec
        # Get all the data
        with open(Id2VecPath,'rb') as f:
            Id2Vec=pickle.load(f)
        CuiDict={k:v for k,v in Id2Vec.items() if 'C' in k}
        cui_vectors=np.array(list(CuiDict.values()),dtype=float)
        cuis=[k for k,v in Id2Vec.items() if 'C' in k]
        emb_name = element.split('_')[0]
        
        if emb_name !='Random':

            red=reduce_dim(cui_vectors,method=args.method)
            DisRdata=PrepareData(cuis,red, CuiToName,diseasesclasses=disease_to_plot,DiseaseClasses=dc)
            plot_dimensions(DisRdata,figsize=(10,8),axes=facet_axes_disease[row][col],legend=False,figure=facet_disease,DiseaseClasses = dc)
            
            facet_axes_disease[row][col].set_title(emb_name)
            # facet_axes_disease[row][col].set_xticklabels(data.columns.tolist(),rotation = 45,ha= 'right')
            facet_axes_disease[row][col].tick_params(top=False,
                    bottom=False,
                    left=False,
                    right=False)

            KmeansDisData=PrepareData(cuis,cui_vectors,CuiToName,diseasesclasses=list(range(16)),columns=None,DiseaseClasses=dc)
            KmeansDisData.dropna(inplace=True)
            KmeansFeatures = KmeansDisData.iloc[:,:100].values
            kmeans = KMeans(
                init="random",
                n_clusters=16,
                n_init=10,
                max_iter=300,
                random_state=42)
            y_km = kmeans.fit_predict(KmeansFeatures)

            KmeansDisData['Kmeans_y'] = y_km

            # Visualize the heatmap
            Disease_Classes = list(set(KmeansDisData.DiseaseClassName.tolist()))
            data = pd.DataFrame(index=list(range(16)),columns=Disease_Classes)

            for d in Disease_Classes:
                tmp = KmeansDisData[KmeansDisData.DiseaseClassName == d].Kmeans_y.value_counts()
                data[d] = tmp



            sns.heatmap(data, ax = heatmap_axes_dis[row][col],cmap='crest')
            heatmap_axes_dis[row][col].set_title(emb_name)
            heatmap_axes_dis[row][col].set_ylabel('Kmeans_clusters')
            heatmap_axes_dis[row][col].set_xticklabels(data.columns.tolist(),rotation = 45,ha= 'right')
            heatmap_axes_dis[row][col].tick_params(top=False,
                    bottom=False,
                    left=False,
                    right=False)    
            
        
            # Calculate the Homogeneity score

            hs = homogeneity_score(KmeansDisData.DiseaseClass.tolist(),KmeansDisData.Kmeans_y.tolist())

            Homogeneity_df.at[emb_name,"CUIS_homogeneity_score"] = hs

            # Calculate shilouhette score
            distortions = []
            shilouettes = []
            for i in range(10, 20):
                km = KMeans(n_clusters=i, 
                            init='random', 
                            n_init=10, 
                            max_iter=300, 
                            random_state=0)
                predictions = km.fit_predict(KmeansFeatures)
                distortions.append(km.inertia_)
                try:
                    sil_scor = np.mean(silhouette_samples(KmeansFeatures,predictions))
                except Exception as e:
                    print(e)
                    sil_scor = 0
                shilouettes.append(sil_scor)

            shilouette_axes_dis[row][col].plot(range(10, 20), shilouettes, marker='x')

            shilouette_axes_dis[row][col].axvline(16,c='r',linestyle='--',label='true number of clusters')
            shilouette_axes_dis[row][col].set_xlabel('Number of clusters')

            shilouette_axes_dis[row][col].set_ylabel('Silhouette score')

            shilouette_axes_dis[row][col].legend()
            shilouette_axes_dis[row][col].set_title(emb_name)
        
            col += 1
            if col == 2:
                row += 1
                col = 0
                
        
        else:
            pass




    
    
    plt.tight_layout()
    facet_axes_disease[0][0].legend(bbox_to_anchor=(1.03, 1.12), loc='lower center')
    facet_disease.savefig(f'../outputs/Clustering/diseases_facet_{args.method}.png',dpi=300,bbox_inches = 'tight')
    heatmap_fig_dis.savefig('../outputs/Clustering/facet_diseases_heatmap.png',dpi = 300,bbox_inches='tight')
    shilouette_fig_dis.savefig('../outputs/Clustering/facet_diseases_shilouette.png',dpi = 300,bbox_inches='tight')
    Homogeneity_df.to_csv(f'../outputs/Clustering/homogeneity_diseaes.csv')

if __name__ == '__main__':
    Main()
        
