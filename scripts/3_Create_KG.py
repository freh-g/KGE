#!/usr/bin/env python3

import networkx as nx
import pandas as pd
import pickle
import os
import graphlot as gr
import biomapy as bp
from collections import Counter
import pronto
import warnings
from nxontology.imports import pronto_to_multidigraph
import argparse

parser=argparse.ArgumentParser(description='Create a KG from multiple data sources')

parser.add_argument('--Dpcgt',help = "proportion of disgenet to add to the KG, default value is Train i.e. the same gdas that ", type=str,default="Train")
args = parser.parse_args()

def typology(i):
    if i.isnumeric():
        return 'protein'
    elif 'GO' in i:
        return 'function'
    elif 'C' in i:
        return 'phenotype'
    elif 'DB' in i:
        return 'drug'
    else:
        return i

def Main():
    warnings.filterwarnings(action='ignore')

    ### ADD HP

    hp_obo=pronto.Ontology('../datasets/hp.obo')
    hp_multidigraph = pronto_to_multidigraph(hp_obo)
    print(f"""
            Type and number of relationship in HPO: {Counter(key for _, _, key in hp_multidigraph.edges(keys=True))} 
            
            
            Number of phenotypes in HPO: {len(hp_multidigraph.nodes())}""")

    #CREATE THE EDGELIST

    hp_edgelist=pd.DataFrame([(source,rel,target) for (source,rel,target) in hp_multidigraph.edges(keys=True)],columns=['source','target','relation'])

    # BUILD A MAPPER BETWEEN NAME AND HP NAME

    hpid_to_name = {id_: data.get('name') for id_, data in hp_multidigraph.nodes(data=True)}
    name_to_hpid= {v:k for k,v in hpid_to_name.items()}

    # INSERT A SLASH BETWEEN IS A RELATIONSHIP
    hp_edgelist.relation=hp_edgelist.relation.apply(lambda x: x.replace(' ','_'))

    hpcuimapper=pd.read_csv('../datasets/MedGen_HPO_Mapping.txt',sep='|')
    hpcuimapgroup=hpcuimapper.groupby('SDUI').agg({'#CUI':'first'})
    hpcuimap_dict=dict(zip(hpcuimapgroup.index.tolist(),hpcuimapgroup['#CUI'].tolist()))
    hp_edgelist['source_cui']=list(map(hpcuimap_dict.get,hp_edgelist.source.tolist()))
    hp_edgelist['target_cui']=list(map(hpcuimap_dict.get,hp_edgelist.target.tolist()))
    hp_edgelist=hp_edgelist.dropna()
    hp_edgelist.head()

    ### GO

    go_pronto = pronto.Ontology(handle='../datasets/go-basic.owl')
    go_multidigraph = pronto_to_multidigraph(go_pronto)
    Counter(key for _, _, key in go_multidigraph.edges(keys=True))

    go_edgelist=pd.DataFrame([(source,rel,target) for (source,rel,target) in go_multidigraph.edges(keys=True)],columns=['source','target','relation'])

    go_edgelist.relation=go_edgelist.relation.apply(lambda x: x.replace(' ','_'))

    print(f"""
            Type and number of relationship in the GO: {Counter(key for _, _, key in go_multidigraph.edges(keys=True))} 
            
            
            Number of functions in GO: {len(go_multidigraph.nodes())}""")

    goid_to_name = {id_: data.get('name') for id_, data in go_multidigraph.nodes(data=True)}
    name_to_goid= {v:k for k,v in goid_to_name.items()}


    ### DO

    do_pronto = pronto.Ontology(handle='../datasets/HumanDO.owl')
    do_multidigraph = pronto_to_multidigraph(do_pronto)
    Counter(key for _, _, key in do_multidigraph.edges(keys=True))

    do_edgelist=pd.DataFrame([(source,rel,target) for (source,rel,target) in do_multidigraph.edges(keys=True)],columns=['source','target','relation'])

    # INSERT A SLASH BETWEEN IS A RELATIONSHIP
    do_edgelist.relation=do_edgelist.relation.apply(lambda x: x.replace(' ','_'))


    # MAP DO TO CUI 

    with open('../datasets/DoidCuimapper.pickle','rb') as f:
        doidcui=dict(pickle.load(f))
    doidcui={k.replace('_',':'):v for k,v in doidcui.items()}

    source_cuis=list(map(doidcui.get,do_edgelist.source.tolist()))
    target_cuis=list(map(doidcui.get,do_edgelist.target.tolist()))
    do_edgelist['source_cui']=source_cuis
    do_edgelist['target_cui']=target_cuis

    do_edgelist=do_edgelist.dropna()
    do_edgelist=do_edgelist.explode(column=['source_cui'])
    do_edgelist=do_edgelist.explode(column=['target_cui'])
    print(do_edgelist.shape)
    do_edgelist.head()

    #### GO ANNOTATIONS

    with open('../datasets/goa_human.gaf', 'r') as f:
        goa=f.readlines()

    goa=goa[41:]
    goa=[line.rstrip('\t\n').split('\t') for line in goa]
    goa=pd.DataFrame(goa)
    goa=goa[[2,3,4]]
    goa.columns=['gene','relationship','function']

    goa=goa[goa.gene!=''].reset_index(drop=True)

    mappedgenes=bp.gene_mapping_many(goa.gene.tolist(),'symbol','entrez')
    goa['geneId']=mappedgenes

    goa=goa.dropna(subset=['geneId','function'])
    goa.geneId=goa.geneId.astype(int).astype(str)


    g=pd.DataFrame(columns=['geneId','relationship','function'])
    for rel in set(goa.relationship.tolist()):
        tp=goa[goa.relationship==rel]
        
        tp=tp.groupby('geneId').agg({'relationship':'first',
                                'function':set})
        tp.reset_index(inplace=True)
        
        tp=tp.explode('function')    
        
        g=g._append(tp)
        
    goa = g

    ### DISGENET
    if args.Dpcgt == "Train":
        dis_train = pd.read_csv('../TrainingSets/Dis_Train.csv')
        dis_train = dis_train[dis_train.Label == 1]
        dis_train_pos = dis_train[['geneId','diseaseId']]
        dis_train_pos.geneId = dis_train_pos.geneId.astype(str)
    else:
        
        dis_train = pd.read_csv('../datasets/disgenet_parsed.csv')
        dis_train_pos = dis_train[['geneId','diseaseId']]
        dis_train_pos.geneId = dis_train_pos.geneId.astype(str)
        pctg = (int(args.Dpcgt)/100)*int(dis_train_pos.shape[0])
        dis_train_pos = dis_train_pos.iloc[:int(pctg)]
        
    print(f"A total of {dis_train_pos.shape[0]} GDAs will be added to the KG")
    ## HP annotations

    with open('../datasets/genes_to_phenotype.txt','r') as f:
        annothp=f.readlines()
    annothp=pd.DataFrame([line.rstrip('\n').split('\t')for line in annothp[1:]])
    annothp=annothp[[1,2]]

    annothp=annothp.groupby(2).agg({1:set})
    annothp.reset_index(inplace=True)
    annothp=annothp.explode(1)

    annothp.columns=['Ph','gene']

    annothp['Cui_PH']=list(map(hpcuimap_dict.get,annothp.Ph.tolist()))
    annothp['Entrez']=bp.gene_mapping_many(annothp.gene.tolist(),'symbol','entrez')
    annothp.dropna(inplace=True)
    annothp.Entrez=annothp.Entrez.astype(int).astype(str)

    ### PPI multiscale interactome 

    ppi=pd.read_csv('../datasets/multiscale-interactome/3_protein_to_protein.tsv',sep='\t')

    ppi.node_1=ppi.node_1.astype(str)
    ppi.node_2=ppi.node_2.astype(str)
    
    # ADD the new version of ppi
    huri = pd.read_csv('../datasets/huri_last.csv')
    huri = huri[['source_entrez','target_entrez']]
    huri.columns = ['node_1','node_2']
    
    huri = huri.astype(int).astype(str)
    ppi.node_1=ppi.node_1.astype(int).astype(str)
    ppi.node_2=ppi.node_2.astype(int).astype(str)
    ppi = ppi[['node_1','node_2']]
    ppi = ppi._append(huri)
    ppi = ppi.drop_duplicates()

    

    ### Drug2disease multiscale interactome
    dtd=pd.read_csv('../datasets/multiscale-interactome/6_drug_indication_df.tsv',sep='\t')
    dtd = dtd[dtd['drug'].str.contains('DB')]


    # ### Drug2protein multiscale interactome
    # dtp=pd.read_csv('../datasets/multiscale-interactome/1_drug_to_protein.tsv',sep='\t')

    ### DEFINE THE GRAPH
    kg=nx.MultiDiGraph()

    ### PPI NODES MI
    kg.add_nodes_from(list(set(ppi.node_1.tolist()+ppi.node_2.tolist())),tipo='protein')


    #PPI EDGES
    kg.add_edges_from(list(zip(ppi.node_1.tolist(),ppi.node_2.tolist())), rel_type='interacts_with')
    kg.add_edges_from(list(zip(ppi.node_2.tolist(),ppi.node_1.tolist())), rel_type='interacts_with')

    ### GO NODES
    kg.add_nodes_from(list(set(go_edgelist.source.tolist()+go_edgelist.target.tolist())),tipo='function')

    ### GO EDGES

    for rel in go_edgelist.relation.value_counts().index:
        slicedf=go_edgelist[go_edgelist.relation==rel]
        kg.add_edges_from(list(zip(slicedf.source.tolist(),slicedf.target.tolist())), rel_type=rel)
        

    ### PHENOTYPE NODES
    kg.add_nodes_from(list(set(hp_edgelist.source_cui.tolist()+hp_edgelist.target_cui.tolist())),tipo='phenotype')

    ### PHENOTYPE EDGES
    cui_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='phenotype']
    cui_in_do=list(set(do_edgelist.source_cui.tolist()+do_edgelist.target_cui.tolist()))
    cui_not_in_the_graph=set(cui_in_do)-set(cui_already_in_the_graph)

    # ADD THE PHENOTYPES FROM DO
    kg.add_nodes_from(cui_not_in_the_graph,tipo='phenotype')

    # ADD DO EDGES
    kg.add_edges_from(list(zip(do_edgelist.source_cui.tolist(),do_edgelist.target_cui.tolist())), rel_type='is_a')


    ## ADD GO ANNOTATIONS TO THE KG
    proteins_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='protein']
    proteins_in_the_annotation_file=goa.geneId.tolist()
    proteins_not_in_the_graph=set(proteins_in_the_annotation_file)-set(proteins_already_in_the_graph)


    kg.add_nodes_from(proteins_not_in_the_graph,tipo='protein')

    functions_already_in_the_graph=[fun[0] for fun in kg.nodes(data=True) if fun[1]['tipo']=='function']
    functions_in_the_annotation_file=goa.function.tolist()
    functions_not_in_the_graph=set(functions_in_the_annotation_file)-set(functions_already_in_the_graph)

    kg.add_nodes_from(functions_not_in_the_graph,tipo='function')

    for rel in goa.relationship.value_counts().index:
        slicedf=goa[goa.relationship==rel]
        kg.add_edges_from(list(zip(slicedf.geneId.tolist(),slicedf.function.tolist())), rel_type=rel)
        

    #ADD HP ANNOTATIONS 

    proteins_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='protein']
    proteins_in_the_annotation_file=annothp.Entrez.tolist()
    proteins_not_in_the_graph=set(proteins_in_the_annotation_file)-set(proteins_already_in_the_graph)


    # ADD HP_MISSING PROTEINS AS NODES
    kg.add_nodes_from(proteins_not_in_the_graph,tipo='protein')

    phenotypes_already_in_the_graph=[p[0] for p in kg.nodes(data=True) if p[1]['tipo']=='phenotype']
    phenotypes_in_the_annotation_file=annothp.Cui_PH.tolist()
    phenotypes_not_in_the_graph=set(phenotypes_in_the_annotation_file)-set(phenotypes_already_in_the_graph)
    phenotypes_not_in_the_graph

    # ADD HP_MISSING phenotypes AS NODES
    kg.add_nodes_from(phenotypes_not_in_the_graph, tipo='phenotype')

    # HP annotations
    kg.add_edges_from(list(zip(annothp['Cui_PH'].tolist(),annothp['Entrez'].tolist())), rel_type='has_annotated')
    kg.add_edges_from(list(zip(annothp['Entrez'].tolist(),annothp['Cui_PH'].tolist())), rel_type='is_annotated_to')

    p2d=pd.read_csv('../datasets/hpomapped.tsv',sep='\t')
    p2d=p2d[['CUI_phen','CUI_dis']]

    cui_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='phenotype']
    cui_in_p2d=list(set(p2d.CUI_dis.tolist()+p2d.CUI_phen.tolist()))
    cui_not_in_the_graph=set(cui_in_p2d)-set(cui_already_in_the_graph)
    print(f"there's {len(cui_not_in_the_graph)} cuis that are in h2d but not in the graph")


    # ADD THE PHENOTYPES FROM H2D
    kg.add_nodes_from(cui_not_in_the_graph,tipo='phenotype')

    kg.add_edges_from(list(zip(p2d.CUI_dis.tolist(),p2d.CUI_phen.tolist())), rel_type='has_phenotype')

    #GDAS
    ### 1) Check if the proteins present in Disgenet are already represented in the graph 
    proteins_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='protein']
    proteins_in_disgenet=dis_train_pos.geneId.tolist()
    proteins_not_in_the_graph=set(proteins_in_disgenet)-set(proteins_already_in_the_graph)
    print(f"there's {len(proteins_not_in_the_graph)} proteins that are in disgenet but not in the graph")
    proteins_not_in_the_graph


    # ADD THE PROTEINS FROM DISGENET
    kg.add_nodes_from(proteins_not_in_the_graph,tipo='protein')

    ### 2)Check if the diseases are already present in the graph
    cui_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='phenotype']
    cui_in_disgenet=dis_train_pos.diseaseId.tolist()
    cui_not_in_the_graph=set(cui_in_disgenet)-set(cui_already_in_the_graph)
    print(f"there's {len(cui_not_in_the_graph)} cuis that are in disgenet but not in the graph")
    cui_not_in_the_graph

    # ADD THE PHENOTYPES FROM DISGENET
    kg.add_nodes_from(cui_not_in_the_graph,tipo='phenotype')

    # ADD GENE DISEASE ASSOCIATIONS FROM DISGENET
    kg.add_edges_from(list(zip(dis_train_pos.diseaseId.tolist(),dis_train_pos.geneId.tolist())), rel_type='has_annotated')
    kg.add_edges_from(list(zip(dis_train_pos.geneId.tolist(),dis_train_pos.diseaseId.tolist())), rel_type='associated_with')
    print(len(kg.nodes),len(kg.edges),set([nod[1]['tipo']for nod in kg.nodes(data=True)]),set([ed[2]['rel_type'] for ed in kg.edges(data=True)]))


    #DRUG2DISEASE
    print(dtd.drug.isna().any(),dtd.indication.isna().any())


    cui_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='phenotype']
    cui_in_dtd=dtd.indication.tolist()
    cui_not_in_the_graph=set(cui_in_dtd)-set(cui_already_in_the_graph)
    cui_not_in_the_graph

    ## ADD MISSING CUIS FROM DRUG TO DISEASE

    kg.add_nodes_from(cui_not_in_the_graph,tipo='phenotype')

    # ADD DRUG NODES

    kg.add_nodes_from(set(dtd.drug.tolist()), tipo='drug')

    # ADD DRUG 2 DISEASE EDGES

    kg.add_edges_from(list(zip(dtd.drug.tolist(),dtd.indication.tolist())),rel_type='treats')

    
    ##DRUGBANK

    drugbank=pd.read_csv('../datasets/parsed_drugbank.tsv',sep='\t',index_col=0)
    drugbank['entrez']=bp.gene_mapping_many(drugbank.Target.tolist(), 'symbol','entrez')
    drugbank.dropna(inplace=True)
    drugbank.entrez=drugbank.entrez.astype(int)
    drugbank.entrez=drugbank.entrez.astype(str)

    proteins_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='protein']
    proteins_in_drugbank=drugbank.entrez.tolist()
    proteins_not_in_the_graph=set(proteins_in_drugbank)-set(proteins_already_in_the_graph)

    # ADD THE PROTEINS FROM DISGENET
    kg.add_nodes_from(proteins_not_in_the_graph,tipo='protein')

    drugs_already_in_the_graph=[nod[0] for nod in list(kg.nodes(data=True)) if nod[1]['tipo']=='drug']
    drugs_in_drugbank=drugbank['Drugbank id'].tolist()
    drugs_not_in_the_graph=set(drugs_in_drugbank)-set(drugs_already_in_the_graph)

    kg.add_nodes_from(drugs_not_in_the_graph, tipo='drug')

    for rel in drugbank.Action.value_counts().index:
        slicedf=drugbank[drugbank.Action==rel]
        kg.add_edges_from(list(zip(slicedf['Drugbank id'].tolist(),slicedf.entrez.tolist())), rel_type=rel)
    
    
    kg.remove_nodes_from(list(nx.isolates(kg)))

    types = ['protein','function','phenotype','drug']
    akwardnodes = [typology(x) for x in kg.nodes()]
    akwardnodes = [a for a in akwardnodes if a not in types]
    for node in akwardnodes:
        print(node)
        kg.remove_node(node)
    
    if not os.path.isdir('../KGs/'):
        os.mkdir('../KGs/')

    with open(f'../KGs/KG_{args.Dpcgt}.pickle','wb') as f:
        pickle.dump(kg,f)

    if not os.path.isdir('../img/'):
        os.mkdir('../img/')
    gr.plot_degree_distribution(kg,save_fig = f'../img/kg_degree_distribution_{args.Dpcgt}.jpeg' )
    
    print(f'KG number of nodes {kg.number_of_nodes()} number of edges {kg.number_of_edges()}')


if __name__ == '__main__':
    Main()