#!/usr/bin/env python
import networkx as nx
import os
import igraph as ig
import numpy as np
from gprofiler import GProfiler
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.legend import Legend
import argparse


parser=argparse.ArgumentParser(description='Train the model and make predictions')
parser.add_argument('-p','--prob',help='probability of association from the algorithm',default=0.95,type=float)
args=parser.parse_args()


def labelling_without_overlapping(x,y,list_of_annotations,ax,verbose=False,**kwargs):
    
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    
    def doOverlap(ret1,ret2):
        l1 = Point(ret1[0,0],ret1[1,1])
        r1 = Point(ret1[1,0],ret1[0,1])
        l2 = Point(ret2[0,0],ret2[1,1])
        r2 = Point(ret2[1,0],ret2[0,1])

        # If one rectangle is on left side of other
        if l1.x >= r2.x or l2.x >= r1.x:
            return False

        # If one rectangle is above other
        if(r1.y >= l2.y or r2.y >= l1.y):
            return False

        return True

    annotations_coord=[]
    for i, dot in enumerate(y):
        x_coords=x[i]
        y_coords=y[i]
        annotation=ax.annotate(str(list_of_annotations[i]),
                                xy=(x[i],y[i]),
                                 xytext=(x_coords,y_coords),
                                    **kwargs)

        ax.figure.canvas.draw()
        bbox=Text.get_window_extent(annotation)
        bbox_data = ax.transData.inverted().transform(bbox)
        factor=0.2*(bbox_data[0,0]-bbox_data[1,0])
        annotations_coord.append(bbox_data)
        ##BUILD THE SPIRAL##
        theta=np.radians(np.linspace(1,360*200,500))
        r=np.linspace(0,max(max(zip(x,y))),len(theta))
        x_2 = r*np.cos(theta)+x_coords#move the spiral onto the data point
        y_2 = r*np.sin(theta)+y_coords
        n=0
        keep_cycling=True
        while keep_cycling:
            keep_cycling=False
            if verbose==True:
                print('start checking box %s'% i)
            for ind, box in enumerate (annotations_coord[0:-1]):
                if verbose:
                    print('checking %s and %s' % (i,ind))
                if doOverlap(box,bbox_data):
                    if verbose:
                        print('%s and %s overlap' % (i,ind))
                    annotation.set_x(x_2[n])
                    annotation.set_y(y_2[n])
                    n+=1
                    ax.figure.canvas.draw()
                    bbox=Text.get_window_extent(annotation)
                    bbox_data = ax.transData.inverted().transform(bbox)
                    annotations_coord.pop()
                    annotations_coord.append(bbox_data)
                    if verbose:
                        print('new coords (x=%i,y=%i)'%(x_coords,y_coords))
                        print('new bbox data',bbox_data)
                        print('annotation coordinates',box)
                        print('restart iteration')
                    keep_cycling=True
                    break



                    
def plot_enrichment_analisys_network(df,pvalue,colormap='cividis',edgecolor='red',mkcolor='grey',mkfsize=10000,layout='spring',
                                     mklinewidths=2,alpha=1,figsize=(40,20),savefig=False,factor=1,k=10,title_fontsize = 35,
                                     cbarfontsize=10,labelling=True,legend=False, legend_fontsize = 20, legend_titlefontsize = 25,
                                     legend_col = 6, legend_labelspacing = 1.5, legend_title = '',
                                     legend_columnspacing=1.5, legend_handlelength = 3, size_legend_nofelements=3, cbar_orientation= 'horizontal',
                                     cbar_loc=(1, 0.5),**kwargs):

    maxpv=max([-np.log10(p) for p in df.p_value.tolist()])
    for i, (s,v) in enumerate(zip(df.source.value_counts().index,df.source.value_counts())):
        data=df[(df.source==s)&(-np.log10(df.p_value)>pvalue)].reset_index()
        if data.shape[0]==0:
            continue
        else:
            
            nxen=nx.Graph()
            #add nodes
            for i,r in data.iterrows():
                 nxen.add_node(r['name'],size=r['intersection_size'],pvalue=-np.log10(r['p_value']))

            #add edges
            for i,r in data.iterrows():
                for index,row in data.iloc[i+1:].reset_index().iterrows():
                    if len(set(r['intersections']).intersection(set(row['intersections'])))>0:
                        nxen.add_edge(r['name'],
                                    row['name'], 
                                    weight= len(set(r['intersections']).intersection(set(row['intersections']))))
            # Get positions for the nodes in G
            if layout=='spring':
                pos_ = nx.spring_layout(nxen,k)
            
            elif layout=='auto':
                ig_subgraph=ig.Graph.from_networkx(nxen)
                pos_= dict(zip([v['_nx_name'] for v in ig_subgraph.vs],[coord for coord in ig_subgraph.layout_auto()]))
                    
                    


            

            #Normalize connections
            connections=[]
            for edge in nxen.edges(data=True):
                connections.append(edge[2]['weight'])
            if len(connections)!=0:
                
                if ((max(connections)-min(connections)==0) | (len(connections)==0)):
                    norm_connections=[x/100 for x in connections]
                else:
                    norm_connections=[(x-min(connections))/(max(connections)-min(connections)) for x in connections]
            else:
                connections=norm_connections


            #Normalize sizes
            markers=[]
            for node in nxen.nodes(data=True):
                markers.append(node[1]['size'])
            if len(markers)!=0:
                
                if ((max(markers)-min(markers)==0) | (len(markers)==0)):
                    norm_markers=[x/100 for x in markers]
                else:
                    norm_markers=[(x-min(markers))/(max(markers)-min(markers)) for x in markers]
            else:
                markers=norm_markers
            
               
            norm_markers=np.clip(norm_markers,0.3, 1)
            
            
            fig,ax=plt.subplots(figsize=figsize)
            
            ##Plot the nodes
            xses,yses=[],[]
            lab=[]
            colors=[]
            for node in nxen.nodes(data=True):
                xses.append(pos_[node[0]][0])
                yses.append(pos_[node[0]][1])
                lab.append(node[0])
                colors.append(node[1]['pvalue'])
            
            nodez_for_legend = ax.scatter(xses,yses,s=markers)
            nodez=ax.scatter(xses,yses,s=[mkfsize*size for size in norm_markers],
                           c=colors,cmap=colormap,vmax=maxpv,alpha=alpha,edgecolors=mkcolor,
                             linewidths=mklinewidths,clip_on=False,zorder=1)

            number_lab = [str(e[0]) for e in list(enumerate(lab))]
            
            ##Mark the labels
            if labelling:
                labelling_without_overlapping(xses,yses,number_lab,ax,**kwargs)
            
            

            ##Plot the edges
            for indx, edge in enumerate(nxen.edges(data=True)):
                if edge[2]['weight'] > 0:
                    path_1 = edge[0]#prepare the data to insert in make edge
                    path_2 = edge[1]
                    x0, y0 = pos_[path_1]
                    x1, y1 = pos_[path_2]
                    edgez=ax.plot(np.linspace(x0,x1),np.linspace(y0,y1),
                            color=edgecolor,
                            linewidth = 3*norm_connections[indx]**4,
                                 zorder=0)

            cbar=plt.colorbar(nodez,ax=ax,orientation=cbar_orientation,location = 'left')
            cbar.set_label(r'$-log_{10}(p-value)$',fontsize=cbarfontsize+4)
            cbar.ax.tick_params(labelsize=cbarfontsize)
            
            
            if legend:
                class TextHandlerB(HandlerBase):
                    def create_artists(self, legend, text ,xdescent, ydescent,
                        width, height, fontsize, trans):
                        tx = Text(width/2.,height/2, text, fontsize=fontsize,
                                  ha="center", va="center", fontweight="bold")
                        return [tx]
                    
                Legend.update_default_handler_map({str : TextHandlerB()})

                # Create a legend for the first line.
                first_legend = fig.legend(number_lab,lab, bbox_to_anchor=(1,0.5),loc = "lower left")

                # Add the legend manually to the current Axes.
                plt.gca().add_artist(first_legend)

                
                
                handles, _ = nodez.legend_elements(prop="sizes", alpha=0.6, num = size_legend_nofelements)
                _, label_markers = nodez_for_legend.legend_elements(prop="sizes", alpha=0.6)


                legend=fig.legend(handles, label_markers,fontsize = legend_fontsize,loc = "upper left",
                                    bbox_to_anchor=(1,0.5),ncol=legend_col,labelspacing = legend_labelspacing,
                                   columnspacing=legend_columnspacing,handlelength=legend_handlelength,frameon = False)

                legend.set_title(legend_title,prop={'size':legend_titlefontsize})
                
                

            
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_title(s,fontsize=title_fontsize)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            axis = plt.gca()
            # maybe smaller factors work as well, but 1.1 works fine for this minimal example
            axis.set_xlim([factor*x for x in axis.get_xlim()])
            axis.set_ylim([factor*y for y in axis.get_ylim()])
            fig.tight_layout()
            
            if savefig:
                fig.savefig(savefig+str(s)+'enrichment_analysis.jpeg', dpi=300,
                            bbox_inches="tight"
                            )
            
            
            


def Main():
    
    pred = pd.read_csv('../outputs/IDD_predictions.csv',index_col = 0)
    pathout = '../outputs/Enrichment/'
    if not os.path.isdir(pathout):
        os.makedirs(pathout)




    top_genes = pred[pred.Proba>=args.prob].GeneSymbol.tolist()
    print(f"A total of {len(top_genes)} genes has been selected for undergoing functional enrichment")
    gp = GProfiler(return_dataframe=True)
    functions=gp.profile(organism='hsapiens',
                query=top_genes,
                                    significance_threshold_method='bonferroni',
                                    no_iea=True,
                                    no_evidences=False)
    
    plot_enrichment_analisys_network(functions,20,fontsize=12,layout = 'auto',colormap='Blues',edgecolor='grey',
                                    mkcolor='grey',figsize=(15,8),k=100,factor=1.1,cbarfontsize=10,mkfsize=1200,
                                    ha='center',va='center',legend=True,legend_col=1, legend_columnspacing=1.2,legend_labelspacing=2,title_fontsize = 15,
                                    legend_title='Number of Genes',legend_titlefontsize=12,legend_fontsize=12,
                                    cbar_loc=(2,-2),cbar_orientation= 'vertical',savefig=pathout)


if __name__ == '__main__':
    Main()