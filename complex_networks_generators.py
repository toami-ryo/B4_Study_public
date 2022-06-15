import networkx as nx
import numpy as np
import random 
from scipy.stats import poisson
import itertools

# 頂点非活性化モデル 参考：複雑ネットワーク 基礎から応用まで 増田直紀 今野紀雄 著
# N:全体頂点数 m:辺の追加数 a:選択への次数の影響しにくさ
# 例:(N,m,a)=(100,4,2)
def node_deactivation_model(N,m,a) :
    G = nx.complete_graph(m)
    random.seed(1)
    act = list(G.nodes) 
    pr = np.zeros(m)
    for i in range(m,N) : 
        # print(act)
        prsum = 1/(m+a)
        for j in range(m) : # 辺を張る
            G.add_edge(i,act[j])
            pr[j] = 1/(G.degree[act[j]]+a)
            prsum += pr[j]
        
        dea = random.uniform(0,prsum) # 非活性化の全体量
        """ print(pr)
        print(prsum)
        print(dea)
        print() """
        for j in range(m) : # 非活性化
            prsum -= pr[j]
            if prsum <= dea :
                act[j] = i
                break
    
    return G

def random_weighting(G,lower,upper): # テスト用に [lower,upper] のランダムな重みをつける
    for path in list(G.edges):
        G.edges[path]['weight']=random.uniform(lower,upper)
    
    return G

# 食材ネットワークを表現するネットワークモデルの試作品 ####################################
def trial_graph_1(nlist,mulist,trial):
    N=len(nlist)
    nodelist = []
    for i in range(N):
        nodelist.append(['g{}i{}'.format(i,j) for j in range(nlist[i]) ])
    G = nx.Graph()

    random.seed(1)
    klist = []
    for t in range(trial):
        complete_nodes = []
        for i in range(N):
            r = random.uniform(0,1)
            rv = poisson(mulist[i])
            bar = 0
            M = -1
            for j in range(30):
                bar += rv.pmf(j)
                if r <= bar :
                    M = j
                    break
            if M == -1 :
                print('Error1')
                return G
            complete_nodes += random.sample(nodelist[i],M)

        e = list(itertools.combinations(complete_nodes,2))
        G.add_edges_from(e)
        for (u,v) in e :
            if nx.is_weighted(G,edge=(u,v)) :
                G[u][v]['weight']+=1
            else :
                G[u][v]['weight']=1
    return G

def trial_graph_2(N,m):
    G = nx.complete_graph(m)
    for (u,v) in G.edges():
        G[u][v]['weight'] = 1
    strengthes = [m-1 for i in range(m)]

    for i in range(m,N):
        alist = random_attachment(m-1,list(G.nodes),strengthes)
        G.add_node(i)
        e = list(itertools.combinations([i]+alist,2))
        
        G.add_edges_from(e)
        for (u,v) in e :
            if nx.is_weighted(G,edge=(u,v)) :
                G[u][v]['weight']+=1
            else :
                G[u][v]['weight']=1
        strengthes += [m-1]
        for j in alist:
            strengthes[j] += m-1            
    
    for i in G.nodes():
        G.nodes[i]['strength']=strengthes[i]
    
    return G


######################################################################################

# ノード集合 nlist から plist に比例する確率で重複なしに m 個選ぶ
def random_attachment(m,nlist,plist=[0]):
    plist = plist.copy()
    Length = len(nlist)

    if len(nlist) != len(plist) :
        if plist == [0]:
            plist = np.ones(len(nlist))
        else:
            print('Error : Lengthes of lists are different.')
            return
    if m > len(nlist):
        print('Error : Too many items are reqired.')
        return
    
    alist = []
    i=0
    counter = 0
    Limit = Length*1.1
    while i < m :
        L = sum(plist)
        counter += 1
        if counter > Limit :
            print('Error : Too much trial.')
            return
        
        l = 0
        r = random.uniform(0,L)
        for j in range(len(nlist)):
            l += plist[j]
            if r < l :
                alist.append(nlist[j])
                plist[j]=0
                break
        
        i += 1
        
    return alist
