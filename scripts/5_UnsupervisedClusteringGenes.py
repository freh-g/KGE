#!/usr/bin/env python3
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import biomapy as bp
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import ast
import itertools
import argparse
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics.cluster import homogeneity_score

parser=argparse.ArgumentParser(description='Reduce dimension of the embeddings')

parser.add_argument('--method',help = "method for dimensionality reduction" ,type = str,default='PCA')
parser.add_argument('--gc',help = "gene classes to plot, numbers based on the dictionary called gene_class_mapper.pickle in the data folder",type = str)
args = parser.parse_args()




def reduce_dim(embeddings, components = 3, method = 'TSNE'):
    """Reduce dimension of embeddings"""
    if method == 'TSNE':
        x_red = TSNE(components, metric = 'cosine',perplexity=50).fit_transform(embeddings)
        return x_red
    elif method == 'UMAP':
        # Might want to try different parameters for UMAP
        x_red = umap.UMAP(n_components=components, metric = 'cosine', 
                    init = 'random', n_neighbors = 10,min_dist = 0).fit_transform(embeddings)
        return x_red
    elif method == 'PCA':
        pca = PCA(n_components=components)
        x_red = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        # print(f'explained variance ratio: {exp_var_pca}' +'\n' + f'cumulative variance ratio: {cum_sum_eigenvalues}')
        return x_red


def plot_genes(red_genes,figsize=(20,15),GeneClasses=dict ,geneclasses=[],dim1=0,dim2=1,annotate=[],neighbors=False,save=False,legend=True,title =False,axes=None):

    NUM_COLORS = len(red_genes.GeneClass.value_counts().index)
    GENE_CLASSES=red_genes.GeneClass.value_counts().index
    LISTOFCOLORS=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    colorindex=dict(zip(GENE_CLASSES,[n for n in range(len(GENE_CLASSES))]))

    def plot_neighbors(symbol,n=10):
        gene_vector=red_genes[red_genes.Symbol==symbol][['pc1','pc2','pc3']].values
        vectors_reduced=red_genes[['pc1','pc2','pc3']].values
        gene_index=red_genes[red_genes.Symbol==symbol].index[0]
        norma = np.linalg.norm(vectors_reduced, axis=1)
        normalized = (vectors_reduced.T / norma).T
        dists = np.dot(normalized, normalized[gene_index])
        closest = np.argsort(dists)[-n:]
        neighbors=[]
        for c in reversed(closest):
            neighbors.append(red_genes.iloc[c].Name)
        return neighbors

            
    
    
    
    for i in range(NUM_COLORS):
        SetOfGenes=red_genes[red_genes.GeneClass==GENE_CLASSES[i]]
        genes_to_annotate=set(annotate).intersection(SetOfGenes.Symbol.tolist())

        if GENE_CLASSES[i] in geneclasses:
            axes.scatter(SetOfGenes.values[:,dim1],
                       SetOfGenes.values[:,dim2],
                       c=LISTOFCOLORS[i],
                       marker='P',
                       label=GeneClasses[list(GeneClasses)[GENE_CLASSES[i]]],
                       s=300,
                      linewidth=1,
                      edgecolors='black')
        else:
            axes.scatter(SetOfGenes.values[:,dim1],
                       SetOfGenes.values[:,dim2],
                       c=LISTOFCOLORS[i],
                       label=GeneClasses[list(GeneClasses)[GENE_CLASSES[i]]],
                      s=30)
        if len(genes_to_annotate)>0:
            if neighbors:
                for gen in genes_to_annotate:
                    neigh=plot_neighbors(gen,neighbors)
                    xses=[]
                    yses=[]
                    labels=[]
                    box_colors=[]
                    for i,n in enumerate(neigh):
                        if i%2==0:
                            factor=-2
                        else:
                            factor=1.5
                        x=red_genes[red_genes.Symbol==n].values[:,dim1][0]
                        y=red_genes[red_genes.Symbol==n].values[:,dim2][0]
                        axes.annotate(n,(x,y),xytext=(x+i/2*factor,y-i/2*factor),
                                    weight='bold',fontsize=12,ha='center',va='center', bbox= dict(boxstyle="round4", 
                                    fc=LISTOFCOLORS[colorindex[red_genes[red_genes.Symbol==n].GeneClass.values[0]]],alpha=0.5))
        

            else:
                for gen in genes_to_annotate:
                    x=red_genes[red_genes.Symbol==gen].values[:,dim1]
                    y=red_genes[red_genes.Symbol==gen].values[:,dim2]
                    axes.annotate(gen,(x,y),fontsize=13)
    
    if legend:
        plt.legend(
            loc='center left', 
                   bbox_to_anchor=(1, 0.5),markerscale=1.5,fontsize=13
                  )
    if title:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if save:
        plt.savefig(save,bbox_inches = 'tight',dpi = 300)
        
        
        
def Main():
    # Produce multiple heatmaps wit facet plots
    columns = 2
    rows = 3
    row = 0
    col = 0
    
    Homogeneity_df = pd.DataFrame(columns=['genes_homogeneity_score'])
    heatmap_fig, heatmap_axes = plt.subplots(rows,columns, sharex=True,sharey = True,figsize = (10,10))
    shilouette_fig, shilouette_axes = plt.subplots(rows,columns, sharex=True,figsize = (10,10))
    facet_genes, facet_axes_genes = plt.subplots(rows,columns, sharex=True,sharey = True,figsize = (10,10))

    # Gather Data for Genes
    ncbidb=pd.read_csv('https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz',sep='\t')
    ncbidb=ncbidb[ncbidb['#tax_id']==9606]
    ncbidb['GeneID']=ncbidb['GeneID'].astype(str)
    different_classes = []
    gene_classes = pd.read_csv('../datasets/HPA_protein_classes/gene_classes.csv',index_col=0)
    gene_classes
    with open('../datasets/HPA_protein_classes/gene_class_mapper.pickle','rb') as f:
            GC = pickle.load(f)
    GC = dict(zip(['Transcription_factor' if e == 'Transcription' else e for e in GC.keys()],GC.values()))
            
    print(GC)
    for c in GC.values():
        tmp_set = set(gene_classes[gene_classes.gene_class == c].Gene.tolist())
        different_classes.append(tmp_set)
        
    # Keep unique elements of every class
    c = Counter(itertools.chain.from_iterable(different_classes)) 
    unique_elements = [k for k,v in c.items() if v==1]
    
    #Filter protein atlas in order to have a dict mapper of gene:gene-class
    gene_classes = gene_classes[gene_classes.Gene.isin(unique_elements)]
    gene_classes_mapper = dict(zip(gene_classes.Gene.tolist(),['Transcription_factor' if e == 'Transcription' else e for e in gene_classes.gene_class.tolist()]))
    # REVERSE THE GENE CLASS DICTIONARY
    GC_reversed = {v:k for k,v in GC.items()}  

    for element in os.listdir('../Embeddings/'):  
        
        # PLOT GENES
        path = f'../Embeddings/{element}/'
        id2vec = [ele for ele in os.listdir(path) if 'id2vec' in ele.lower()][0]
        Id2VecPath = path + id2vec
        # Get all the data
        with open(Id2VecPath,'rb') as f:
            Id2Vec=pickle.load(f)
        GenDict={k:v for k,v in Id2Vec.items() if k.isnumeric()}
        gene_vectors=np.array(list(GenDict.values()),dtype=float)
        emb_name = element.split('_')[0]
        if emb_name !='Random':  

            
            # Filter the dataset 
            genes=list(GenDict.keys())
            red_genes=reduce_dim(gene_vectors,method=args.method)
            red_genes=pd.DataFrame(red_genes,columns=['pc0','pc1','pc2'])
            red_genes['Entrez']=genes
            red_genes['Symbol']=bp.gene_mapping_many([int(gen) for gen in genes],'entrez','symbol')

            red_genes = red_genes[red_genes.Symbol.isin(unique_elements)]
            red_genes=red_genes.dropna()
            red_genes['GeneClass'] = list(map(gene_classes_mapper.get,red_genes.Symbol.tolist()))
            red_genes=red_genes.dropna()
            red_genes.GeneClass=red_genes.GeneClass.astype(int)
            red_genes['embs'] = [GenDict[g] for g in red_genes.Entrez.tolist()]
            if args.gc:
                gc = ast.literal_eval(args.gc)
                red_genes = red_genes[red_genes.GeneClass.isin(gc)]
            else:
                gc = set(red_genes.GeneClass.tolist())
                red_genes = red_genes[red_genes.GeneClass.isin(gc)]
            plot_genes(red_genes,figsize=(10,8),legend=False,axes=facet_axes_genes[row][col],GeneClasses=GC_reversed)
            # POPULATE THE DIMENSIONALITY REDUCTION PLOT
            facet_axes_genes[row][col].set_title(emb_name)
            # facet_axes_disease[row][col].set_xticklabels(data.columns.tolist(),rotation = 45,ha= 'right')
            facet_axes_genes[row][col].tick_params(top=False,
                    bottom=False,
                    left=False,
                    right=False)
            
            # HEATMAP PLOT
            vectors_filtered = red_genes.embs.tolist()
            vectors_filtered = np.stack(vectors_filtered)
    
            kmeans = KMeans(
            init="random",
            n_clusters=len(gc),
            n_init=10,
            max_iter=300 )
            y_km = kmeans.fit_predict(vectors_filtered)
            red_genes['Kmeans_results'] = y_km
            

            data = pd.DataFrame(index=list(range(len(gc))))

            for i in gc:
                tmp = red_genes[red_genes.GeneClass == i].Kmeans_results.value_counts()
                data[i] = tmp
            
            data.columns=[GC_reversed[a] for a in gc]
            
            # populate the facet of heatmaps
            
            sns.heatmap(data, ax = heatmap_axes[row][col],cmap='crest')
            heatmap_axes[row][col].set_title(emb_name)
            heatmap_axes[row][col].set_ylabel('Kmeans_clusters')

            heatmap_axes[row][col].set_xticklabels(data.columns.tolist(),rotation = 45,ha= 'right')
            heatmap_axes[row][col].tick_params(top=False,
                    bottom=False,
                    left=False,
                        right=False)
            # HOMOGENEITY SCORE
            hs = homogeneity_score(red_genes.GeneClass.tolist(),red_genes['Kmeans_results'].tolist())
            
            Homogeneity_df.at[emb_name,'genes_homogeneity_score'] = hs
            
            # SHILOUETTE SCORE
            
            distortions = []
            shilouettes = []
            for i in range(1, len(gc) + 5):
                km = KMeans(n_clusters=i, 
                            init='random', 
                            n_init=10, 
                            max_iter=300, 
                            random_state=0)
                predictions = km.fit_predict(vectors_filtered)
                distortions.append(km.inertia_)
                try:
                    sil_scor = np.mean(silhouette_samples(vectors_filtered,predictions))
                except Exception as e:
                    print(e)
                    sil_scor = 0
                shilouettes.append(sil_scor)

            
            shilouette_axes[row][col].plot(range(1, len(gc) + 5), shilouettes, marker='x')

            shilouette_axes[row][col].axvline(len(gc),c='r',linestyle='--',label='true number of clusters')
            shilouette_axes[row][col].set_xlabel('Number of clusters')

            shilouette_axes[row][col].set_ylabel('Silhouette score')

            shilouette_axes[row][col].legend()
            shilouette_axes[row][col].set_title(emb_name)
                    
            col += 1
            if col == 2:
                row += 1
                col = 0
        else:
            pass
    plt.tight_layout()
    facet_axes_genes[0][0].legend(bbox_to_anchor=(1.03, 1.12), loc='lower center')
    Homogeneity_df.to_csv('../outputs/Clustering/homogeneity_df_genes.csv')
    facet_genes.savefig(f'../outputs/Clustering/genes_facet_{args.method}.png',dpi=300,bbox_inches = 'tight')
    heatmap_fig.savefig('../outputs/Clustering/facet_gene_heatmap.png',dpi = 300,bbox_inches='tight')
    shilouette_fig.savefig('../outputs/Clustering/silhouette_facet.png',dpi = 300,bbox_inches='tight')

if __name__ == '__main__':
    Main()