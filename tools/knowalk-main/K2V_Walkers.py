import random
from tqdm import tqdm
from collections import Counter

def KRW(Node,Graph,NodeAttributeName,EdgeAttributeName,DictOfProb,Iterations=5,Depth=70,restart='False',EdgeType='True',verbose = 'False', directed = 'False'):
    """This RW simulates the biological dogma in which drug - protein - function - disease path is prioritized
    - the dictionary of probabilities is compose by series of tuple:probabilities in which the tuple is the transaction between node types and
      the number is the probability increase that I want to get to go to that specific node type to the other
    - NodeAttributeName name is a string with the name in which is stored the node type
    - EdgeAttributeName is a string with the name in which is stored the edge type"""
    RandomWalks=[]
    for iteration in range(Iterations):
        Pointer=Node
        out_edges = Graph.out_edges(Pointer,data=True)
        in_edges = Graph.in_edges(Pointer,data=True)
        nodes = Graph.nodes
        type_of_prioritization = DictOfProb.keys()   
        if verbose == 'True':
            print(f'\n Starting at {iteration} starting node is: {Pointer}')
            
        RandomWalk=[]
        RandomWalk.append(Pointer)
        n = 0 
        while n < Depth:
            if directed == 'True':
                PossiblePaths=[ed for ed in out_edges]
            else:
                PossiblePaths=[ed for ed in out_edges] + [(b,a,c) for (a,b,c) in in_edges]
                
            PossiblePathsNodes=[ed[1] for ed in PossiblePaths]
            
            if (len(PossiblePaths) == 0) & (n == 0):
                if verbose == 'True':
                    print(f"first node doesn't have adjacent edges can't start the walk here {Pointer}")
                n = 10
            
            elif len(PossiblePaths)>0:
                TypeOfNode=nodes[Pointer][NodeAttributeName]
                if verbose == 'True':
                    print('currently on node:',Pointer,TypeOfNode)
                if TypeOfNode in [e[0] for e in type_of_prioritization]:
                    TypeOfNodeToPrioritize=list(DictOfProb)[[e[0] for e in type_of_prioritization].index(TypeOfNode)][1]
                    if verbose == 'True':
                        print(f'there are node to prioritize: {TypeOfNodeToPrioritize}')
                    TypesOfNeighbors=[nodes[n][NodeAttributeName] for n in PossiblePathsNodes]
                    NumberOfNeighbors=len(set(TypesOfNeighbors))
                    Neighbors=Counter(TypesOfNeighbors)
                    if verbose == 'True':
                        print('I have these type of Neighbors so see if weights are calculated in an appropriated way', Neighbors)
                    Weight=list(DictOfProb.values())[[e[0] for e in type_of_prioritization].index(TypeOfNode)]
                    NumberOfEdgesOfInterest=Neighbors[TypeOfNodeToPrioritize]
                    Weights=[1+Weight/NumberOfEdgesOfInterest if Type == TypeOfNodeToPrioritize else 1 for Type in TypesOfNeighbors ]
                    if verbose == 'True':
                        print(f'These are the Number of edges that leads to {TypeOfNodeToPrioritize}: {NumberOfEdgesOfInterest} so the calculate Weights are {Weights}')
                else:
                    if verbose == 'True':
                        print(f'No prioritization since the current node is {TypeOfNode}')
                    if directed=='True':    
                        NOfNeighbors=len(out_edges)
                        Weights=[1 for _ in range(NOfNeighbors)]
                    else:
                        NOfNeighbors=len(out_edges) + len(in_edges)
                        Weights=[1 for _ in range(NOfNeighbors)]

                path=random.choices(PossiblePaths,Weights)
                Edge=path[0][2][EdgeAttributeName]
                Pointer=path[0][1]
                if EdgeType == 'True':
                    RandomWalk.append(Edge)
                RandomWalk.append(Pointer)
                if verbose == 'True':
                    print('Node and edge stored, new node:',Pointer)
                n += 1

            else:
                if verbose == 'True':
                    print('NO OUT EDGES!!!')
                if restart == 'True':
                    if verbose == 'True':
                        print('restarting')
                    Pointer=Node
                    n+=1
                    if verbose == 'True':
                        print(f'I restarted so now the starting point is {Pointer}')
                else:
                    if verbose == 'True':
                        print(f'I didnt restart and I am just continuing so the node in which I am is {Pointer}')
                    n+=1
            if verbose =='True':
                print(f'After {n} number of steps I am finishing an iteration and I built a rw that is {len(RandomWalk)} ')
        RandomWalks.append(RandomWalk)
    return RandomWalks  

