import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math


###--------------------------- attributes --------------------------------------------###

# G の枝に betweenness の attributes を加える
def betweenness(G):
    G = G.copy()
    print('Adding betweenness attribute to the graph')
    N=len((G.nodes()))
    for i,j in G.edges():
        G[i][j]['betweenness']=0
    print('all nodes : '+str(len(G.nodes()))+'\n')
    count = 0
    for n in G.nodes():
        # if count > 4 :
        #     break
        T = bet_SPT(G,n)
        for i,j in G.edges():
            G[i][j]['betweenness'] += T[i][j]['SPT']/N
        print(str(count)+' : '+str(n))
        count += 1

    return G

def bet_SPT(G,r):
    N=len((G.nodes()))
    T = nx.Graph(G.edges)
    for i,j in G.edges():
        w=G[i][j]['weight']
        T[i][j]['distance']=1/w
        T[i][j]['SPT']=0
        
    paths=nx.shortest_path(G,source=r,weight='distance')
    
    for k,path in paths.items():
        for i in range(len(path)-1):
            T[path[i]][path[i+1]]['SPT'] += 1/N
    return T

# G の枝に salience の attributes を加える
def salience(G): 
    G = G.copy()
    print('Adding salience attribute to the graph')
    N = len(G.nodes())
    for i,j in G.edges():
        G[i][j]['salience']=0
    print('all nodes : '+str(len(G.nodes()))+'\n')
    count = 0
    for n in G.nodes():
        # if count > 4 :
        #     break
        T = SPT(G,n)
        for i,j in G.edges():
            if T[i][j]['SPT'] == 1:
                G[i][j]['salience'] += 1/N
        print(str(count)+' : '+str(n))
        count += 1

    return G

# G の枝をもとに、SPT の枝に含まれるなら1、含まれないなら0を、SPTの値として持つグラフ
def SPT(G,r): 
    T = nx.Graph(G.edges)
    for i,j in G.edges():
        w=G[i][j]['weight']
        T[i][j]['distance']=1/w
        T[i][j]['SPT']=0

    paths=nx.shortest_path(T,source=r,weight='distance')

    for m,path in paths.items():
        for i in range(len(path)-1):
            T[path[i]][path[i+1]]['SPT']=1
    """ print(paths.items())
    print(T)
    print() """
    return T 

# G の頂点に強度の attributes を加える
def strength(G,w='weight'):
    G = G.copy()
    for u in G.nodes():
        s = 0
        for v in G.adj[u]:
            s += G[u][v][w]
        G.nodes[u]['strength']=s

    return G

# 閾値以下の salience を 0 とみなす
def salience_threshold(G,s):
    for i,j,d in G.edges(data=True):
        if d['salience'] <= s :
            G[i][j]['salience']=0
    
    return G

# HSS の betweenness 版
def HBS(G,thr,B='betweenness'):
    G = G.copy()

    elist = list(G.edges())
    for e in elist:
        i,j=e
        if G[i][j][B] < thr :
            G.remove_edge(i,j)
    return G

# High Salience Skeleton 
def HSS(G,thr=0.95,S='salience'):
    # networkx のグラフは関数に代入されるとき、デフォルトでC言語の参照渡しと同じ処理をする
    # 関数内の操作をグローバルにまで持ち出したくない場合は copy メソッドで複製すること
    G = G.copy()

    elist = list(G.edges())
    for e in elist:
        i,j=e
        if G[i][j][S] < thr :
            G.remove_edge(i,j)
    return G

# HBS +α のグラフを返す
def b_hierarchy(G,thr):

    H = HBS(G,thr)

    flag = True
    for u in H.nodes():
        if H.degree(u) > 0 :
            H.nodes[u]['b_hierarchy']=0
            flag = False
        else :
            H.nodes[u]['b_hierarchy']=-1

    if flag:
        for u in H.nodes():
            cand = [None,0]
            for d, v in G.edges(u):
                if G[u][v]['betweenness']>cand[1]:
                    cand = [v,G[u][v]['betweenness']]
            if cand[0] != None :
                H.add_edges_from([(u,cand[0],G[u][cand[0]])])
                H.nodes[u]['b_hierarchy'] = 1

    flag = True
    while flag :
        flag = False
        add_list = []
        for u in H.nodes():
            if H.nodes[u]['b_hierarchy'] != -1 :
                continue
            cand = [None,0]
            for d, v in G.edges(u):
                if H.degree(v) > 0 and G[u][v]['betweenness']>cand[1]:
                    cand = [v,G[u][v]['betweenness']]
            if cand[0] != None :
                add_list.append((u,cand[0]))
                flag = True
        
        for u,v in add_list :
            H.add_edges_from([(u,v,G[u][v])])
            H.nodes[u]['b_hierarchy']=H.nodes[v]['b_hierarchy']+1
    
    return H

# HSS +α のグラフを返す
# HSS に含まれる頂点の階級を 0 とし、それらの頂点から距離 k で連結する頂点の階級を
# k とする
def s_hierarchy(G,thr=0.99):

    H = HSS(G,thr=thr)

    flag = True
    for u in H.nodes():
        if H.degree(u) > 0 :
            H.nodes[u]['s_hierarchy']=0
            flag = False
        else :
            H.nodes[u]['s_hierarchy']=-1
    
    if flag:
        for u in H.nodes():
            cand = [None,0]
            for d, v in G.edges(u):
                if G[u][v]['salience']>cand[1]:
                    cand = [v,G[u][v]['salience']]
            if cand[0] != None :
                H.add_edges_from([(u,cand[0],G[u][cand[0]])])
                H.nodes[u]['s_hierarchy'] = 1

    flag = True
    while flag :
        flag = False
        add_list = []
        for u in H.nodes():
            if H.nodes[u]['s_hierarchy'] != -1 :
                continue
            cand = [None,0]
            for d, v in G.edges(u):
                if H.degree(v) > 0 and G[u][v]['salience']>cand[1]:
                    cand = [v,G[u][v]['salience']]
            if cand[0] != None :
                add_list.append((u,cand[0]))
                flag = True
        
        for u,v in add_list :
            H.add_edges_from([(u,v,G[u][v])])
            H.nodes[u]['s_hierarchy']=H.nodes[v]['s_hierarchy']+1
    
    return H

#任意の頂点集合を核とした階層構造を返す
def selected_core_hierarchy(G,nodes,attr):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    flag = True
    for u in H.nodes():
        H.nodes[u]['hierarchy']=-1
    for u in nodes:
        H.nodes[u]['hierarchy']=0
        flag = False
    
    if flag:
        for u in H.nodes():
            cand = [None,0]
            for v in G.neighbors(u):
                if G[u][v][attr]>cand[1]:
                    cand = [v,G[u][v][attr]]
            if cand[0] != None :
                H.add_edges_from([(u,cand[0],G[u][cand[0]])])
                H.nodes[u]['hierarchy'] = 1

    flag = True
    while flag :
        flag = False
        add_list = []
        for u in H.nodes():
            if H.nodes[u]['hierarchy'] != -1 :
                continue
            cand = [None,0]
            for v in G.neighbors(u):
                if H.nodes[v]['hierarchy'] > -1 and G[u][v][attr]>cand[1]:
                    cand = [v,G[u][v][attr]]
            if cand[0] != None :
                add_list.append((u,cand[0]))
                flag = True
        
        for u,v in add_list :
            H.add_edges_from([(u,v,G[u][v])])
            H.nodes[u]['hierarchy']=H.nodes[v]['hierarchy']+1
    
    return H

def hierarchy_list(G,attr):
    l = []
    N = len(G.nodes())
    i = 0
    while True :
        nlist = [u for u in G.nodes() if G.nodes[u][attr]==i]
        l.append(nlist)
        if sum([len(u) for u in l]) >= len(G.nodes()):
            break
        i += 1
    
    return l

# 比較的離散な salience 値のとき、ひとつづつ降順に値を参照し、hierarchy をつくる
# SbS : step by step
# G : グラフ,  N : step by step で生成する操作のステップ数
def SbS_s_hierarchy(G,N):
    n_list = number_list(salience_list(G),rev=True)
    H = HSS(G,thr=n_list[0])

    for u in H.nodes():
        if H.degree(u) > 0 :
            H.nodes[u]['s_hierarchy']=0
        else :
            H.nodes[u]['s_hierarchy']=-1

    if N > 1 :
        for i in range(N-1):
            n = n_list[i+1]
            e_list = []
            for u, v in G.edges(): 
                if G[u][v]['salience']==n:
                    H.add_edges_from([(u,v,G[u][v])])
                    if H.nodes[u]['s_hierarchy'] == -1 :
                        H.nodes[u]['s_hierarchy'] = i+1
                    if H.nodes[u]['s_hierarchy'] == -1 :
                        H.nodes[u]['s_hierarchy'] = i+1

    left_nodes = []
    for u in H.nodes() :
        if (H.nodes[u]['s_hierarchy'] == -1 and G.degree(u) > 0) :
            left_nodes.append(u)


    flag = True
    while flag :
        flag = False
        add_list = []
        for u in left_nodes :
            cand = [None,0]
            for v in G.neighbors(u):
                if H.degree(v) > 0 and G[u][v]['salience']>cand[1]:
                    cand = [v,G[u][v]['salience']]
            if cand[0] != None :
                add_list.append((u,cand[0]))
                flag = True
        
        for u,v in add_list :
            H.add_edges_from([(u,v,G[u][v])])
            H.nodes[u]['s_hierarchy']=N
            left_nodes.remove(u)

    return H

# 4つの list 生成関数。描画に用いる
def weight_list(G):
    weights = [d['weight'] for u, v, d in G.edges(data=True)]
    return weights

def betweenness_list(G):
    betweenness=[d['betweenness'] for u, v, d in G.edges(data=True)]
    return betweenness

def degree_list(G):
    deg=[]
    for m,n in G.degree:
        deg.append(n)
    return deg   
    
def salience_list(G):
    saliences=[d['salience'] for u, v, d in G.edges(data=True)]
    return saliences

###------------------------------------------------------------------------###


###---------------------- drawing -----------------------------------------###

# network を描画
# 参考 : networkx.org の draw_networkx
def draw(G,attr = None):
    attr = 'weight'
    pos_arg = nx.spring_layout(G)
    node_size_arg = 20
    if attr == None :
        width_arg = 0.5
    else :
        width_arg = np.array([d[attr] for u, v, d in G.edges(data=True)])*0.2

    font_size_arg = 4

    nx.draw_networkx(G,pos=pos_arg,node_size=node_size_arg,width=width_arg,font_size=font_size_arg)
    plt.show()

###-----------------------------------------------------------------------###


###--------------------- distribution ----------------------------------------###

# 分布を描画するため、[数リスト,相対頻度リスト] を生成
# relative = False とすれば、絶対度数を出力する
def distribution(list1,relative=True):
    sorted_list=sorted(list1)
    L=len(sorted_list)
    if relative :
        N=L
    else :
        N=1
    l=1
    list_s=[]
    list_p=[]

    for i in range(L-1):
        if sorted_list[i]==sorted_list[i+1]:
            l=l+1
        else :
            p=l/N
            s=sorted_list[i]
            list_s.append(s)
            list_p.append(p)
            l=1
    list_s.append(sorted_list[L-1])
    list_p.append(l/N)
    a=[list_s]+[list_p]      

    return a

# distribution() を (s,p)　のペアでまとめたもの
def dist_pairs(x_list,relative=True):
    list_s, list_p = distribution(x_list,relative=relative)
    dp = []

    for i in range(len(list_s)):
        dp.append((list_s[i],list_p[i]))
    
    dp.sort(key= lambda pair : pair[0])

    return dp

# 数リストのみ
def number_list(x_list,rev = False):
    sorted_list=sorted(x_list)
    L=len(sorted_list)
    list_s=[]

    for i in range(L-1):
        if sorted_list[i]!=sorted_list[i+1]:
            list_s.append(sorted_list[i])
    list_s.append(sorted_list[L-1])

    list_s.sort(reverse=rev)

    return list_s

# ヒストグラムの要領で, ある区間にあてはまる個数を数え上げた [区間の中点リスト,相対頻度リスト]
# interval はスケール全体の幅、bins は区間の分割数
# 値を丸めている箇所もあるので注意
def hist_dist(list1,interval,bins=20,log=False) :
    if log :
        sorted_list=sorted(list1)
        L=len(sorted_list)
        l=0
        list_s=[]
        list_p=[]
        width = math.pow(interval[1]/interval[0],1/bins)
        upper = interval[0]*width
        i = 0
        # print('width = {}'.format(width))

        for n in range(bins-1):
            # print('{} : upper = {}'.format(n,upper))
            while (i < L) and (sorted_list[i] <= upper) :
                l+=1
                i+=1
            list_s.append(upper/math.pow(width,1/2))
            list_p.append(l/L)
            l=0
            upper *= width
        list_s.append(upper/math.pow(width,1/2))
        list_p.append(1-sum(list_p))

        a=[list_s]+[list_p]   

    else :
        sorted_list=sorted(list1)
        L=len(sorted_list)
        l=0
        list_s=[]
        list_p=[]
        width = (interval[1]-interval[0])/bins
        upper = interval[0] + width
        i = 0
        # print('width = {}'.format(width))

        for n in range(bins-1):
            # print('{} : upper = {}'.format(n,upper))
            while (i < L) and (sorted_list[i] <= upper) :
                l+=1
                i+=1
            list_s.append(upper-width/2)
            list_p.append(l/L)
            l=0
            upper += width

        list_s.append(upper-width/2)
        list_p.append(1-sum(list_p))

        a=[list_s]+[list_p]      

    return a

# 離散的な累積分布を描画するため、[数リスト,累積分布リスト]を生成
def cumulative_dist(list1,relative=True):
    sorted_list=sorted(list1)
    L=len(sorted_list)
    list_s = [sorted_list[0]]
    if relative :
        list_p = [1.0]
    else :
        list_p = [L]

    for i in range(1,L):
        if sorted_list[i-1] != sorted_list[i] :
            list_s.append(sorted_list[i])
            if relative :
                list_p.append((L-i)/L)
            else:
                list_p.append(L-i)

    a=[list_s]+[list_p]      

    return a

# 次数分布をプロット
def degree_dist(G):
    degrees = [G.degree(u) for u in G.nodes()]
    s, p = distribution(degrees)

    plt.figure()
    plt.scatter(s,p)
    plt.title('degree distribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('p')

    plt.savefig('degree_distribution.png') 

    plt.show()

# 重み分布をプロット
def weight_distribution(G):
    weights=weight_list(G)
    weights30=[]

    for w in weights: # w <= 300 で打ち切る
        if w<300:
            weights30.append(w)
    Weights=sorted(weights)

    weight_distribution=distribution(Weights)
    x=np.log(weight_distribution[0])
    y=np.log(weight_distribution[1])

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    X=np.array([Weights[0], Weights[-1]]) # 負のインデックスは最後尾からの順番を意味する
    Y=(math.e**c)*np.power(X,m)
    print(m,c)

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('w')
    plt.ylabel('p')
    
    plt.scatter(distribution(weights)[0],distribution(weights)[1])
    plt.title('Weight Distribution')
    # plt.plot(X,Y,color='r')
    # plt.tight_layout()
    plt.savefig('weight_simu.png') 
    plt.show()

# betweenness 分布をプロット
def betweenness_dist(G):
    betweennesses=[d['betweenness'] for u, v, d in G.edges(data=True)]

    scale, freq = distribution(betweennesses)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(scale, freq)
    ax.set_title('Betweenness Distribution')
    ax.set_xlabel('b')
    ax.set_ylabel('p')
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig('betweenness_distribution.png') 

    plt.show()

# betweenness 分布をヒストグラム的にプロット
# b > 0 のみ考慮する
def betweenness_hist_dist(G):
    betweennesses=[d['betweenness'] for u, v, d in G.edges(data=True)]
    betweennesses=[x for x in betweennesses if x>0]

    lower=min(betweennesses)
    upper=max(betweennesses)
    scale, freq = hist_dist(betweennesses,[lower,upper],log=True)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(scale, freq)
    ax.set_title('Betweenness Distribution')
    ax.set_xlabel('b')
    ax.set_ylabel('p')
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig('betweenness_distribution.png') 

    plt.show()

# salience 分布をプロット
def salience_hist_dist(G):
    saliences=[d['salience'] for u, v, d in G.edges(data=True)]

    n_bins = 20
    scale, freq = hist_dist(saliences,[0,1],bins=n_bins)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scale, freq)
    ax.set_title('Salience Distribution')
    ax.set_xlabel('s')
    ax.set_ylabel('p')
    ax.set_xlim(-0.05,1.05)

    plt.savefig('salience_histogram_distribution.png') 

    plt.show()

# salience 分布をプロット（非ゼロ）
def salience_non0_hist_dist(G):
    saliences=[d['salience'] for u, v, d in G.edges(data=True)]

    n_bins = 20

    ## hist_dist を転用したスクリプト ###
    sorted_list=sorted(saliences)
    L=len(sorted_list)
    N=L
    l=0
    list_s=[]
    list_p=[]
    width = 1/n_bins
    upper = width
    i = 0

    while sorted_list[0] <= 0 :
        sorted_list.pop(0)
        N -= 1
    #print(sorted_list)

    for n in range(n_bins):
        while (i < N) and (sorted_list[i] <= upper) :
            l+=1
            i+=1
        list_s.append(upper-width/2)
        list_p.append(l/L)
        l=0
        upper += width

    scale = list_s
    freq = list_p
    # print('\n')
    for i in range(len(list_s)):
        print(list_s[i],list_p[i])
    # print('pのsum = {}'.format(sum(list_p)))
    # print('N/L = {}'.format(N/L))
    ###################################

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scale, freq)
    ax.set_title('Salience Distribution')
    ax.set_xlabel('s')
    ax.set_ylabel('p')
    ax.set_xlim(-0.05,1.05)

    plt.savefig('salience_non0_distribution.png')

    plt.show()

# 0.99 >= s の salience 分布をプロット
def salience_dist_over99(G,relative=False):
    saliences=[d['salience'] for u, v, d in G.edges(data=True) if d['salience'] >= 0.99]

    scale, freq = distribution(saliences,relative=relative)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scale, freq)
    ax.set_title('Salience Distribution')
    ax.set_xlabel('s')
    ax.set_xlim(0.9895,1.0005)

    plt.savefig('salience_distribution_over99.png') 

    plt.show()

# 強度分布をプロット
def strength_dist(G):
    strengthes = [d['strength'] for u, d in G.nodes(data=True)]
    s, p = distribution(strengthes)

    plt.figure()
    plt.scatter(s,p)
    plt.title('strength distribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('strength')
    plt.ylabel('p')

    plt.savefig('strength_distribution.png')

    plt.show()

# 強度の累積分布をプロット
def cumulative_degree_dist(G,relative=True):
    degrees = [G.degree(u) for u in G.nodes()]

    x, y =cumulative_dist(degrees,relative=relative)

    plt.figure()
    plt.title('degree cumulative distribution')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('k')
    plt.ylabel('F')
    plt.scatter(x,y)

    plt.savefig('degree_cumulative_distribution.png')

    plt.show()

# 重みの累積分布をプロット
def cumulative_weight_dist(G,relative=True):
    weights=weight_list(G)

    weight_distribution=cumulative_dist(weights,relative=relative)
    x, y = weight_distribution

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('w')
    plt.ylabel('F')
    
    plt.scatter(x,y)
    plt.title('weight cumulative distribution')
    # plt.plot(X,Y,color='r')
    # plt.tight_layout()
    plt.savefig('weight_cumulative_distribution.png') 
    plt.show()

# 強度の累積分布をプロット
def cumulative_strength_dist(G,relative=True):
    strengthes = [d['strength'] for u, d in G.nodes(data=True)]

    x, y =cumulative_dist(strengthes,relative=relative)

    plt.figure()
    plt.title('strength cumulative distribution')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('strength')
    plt.ylabel('F')
    plt.scatter(x,y)

    plt.savefig('strength_cumulative_distribution.png') 

    plt.show()
###------------------------------------------------------------------------###


###----------------------- analysis ---------------------------------------###

# 上位N番目までの degree 値をもつ node を列挙
def topN_degree_ranking(G,N):
    if N > len(G.nodes()):
        print('Error : Too many items are required')
        return
    
    degrees = []
    for i in G.nodes():
        degrees += [G.degree(i)]
    
    ### 参考 : distribution() ###
    degrees.sort()
    L=len(degrees)
    l=1
    list_s=[]
    list_l=[]

    for i in range(L-1):
        if degrees[i]==degrees[i+1]:
            l=l+1
        else :
            s=degrees[i]
            list_s.append(s)
            list_l.append(l)
            l=1
    list_s.append(degrees[L-1])
    list_l.append(l)
    ############################

    list_s.reverse()
    list_l.reverse()
    L = 0
    rank = 1

    print('rank : degree : node\n')
    for j in range(len(list_s)):
        s = list_s[j]
        for i in G.nodes():
            if G.degree(i) == s :
                print('{} : {} : {}'.format(rank,s,i))
                rank += 1
        L += list_l[j]
        if L >= N :
            break

def topN_degree_list(G,N):
    if N > len(G.nodes()):
        print('Error : Too many items are required')
        return
    
    degrees = []
    for i in G.nodes():
        degrees += [G.degree(i)]
    
    ### 参考 : distribution() ###
    degrees.sort()
    L=len(degrees)
    l=1
    list_s=[]
    list_l=[]

    for i in range(L-1):
        if degrees[i]==degrees[i+1]:
            l=l+1
        else :
            s=degrees[i]
            list_s.append(s)
            list_l.append(l)
            l=1
    list_s.append(degrees[L-1])
    list_l.append(l)
    ############################

    list_s.reverse()
    list_l.reverse()
    L = 0

    l = []
    for j in range(len(list_s)):
        s = list_s[j]
        for i in G.nodes():
            if G.degree(i) == s :
                l.append((i,s))
        L += list_l[j]
        if L >= N :
            break

    return l

###以下２つは連結な部分グラフを得るための関数###
#連結成分それぞれの頂点数と頂点集合の組を得る
def number_and_example_of_connected_components(G):
    pairs = []
    for l in nx.connected_components(G):
        pairs.append((len(l),l))
    
    pairs.sort(key=lambda p:p[0],reverse=True)
    
    return pairs

# 基準とする頂点 ref を含む、できる限り大きな連結部分グラフを返す
def connected_subgraph(G,ref):
    H = nx.subgraph(G,nx.node_connected_component(G,ref))
    return H
############################################

# degree = 0 の頂点数と頂点集合を返す
def degree_0_nodes(G):
    counter = 0
    l0 =[]
    for i in G.nodes():
        if G.degree(i) == 0:
            counter += 1
            l0.append(i)
    
    return (counter,l0)

# 隣接頂点対の次数の組を図にプロット
def degree_pairs_plot(G):
    d = nx.node_degree_xy(G)
    x = []
    y = []
    for u,v in d:
        x.append(u)
        y.append(v)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.set_title('degrees of linked nodes pairs')

    plt.show()

# 隣接平均次数をプロット
def assortativity_plot(G,log=False):
    knn = nx.average_degree_connectivity(G)
    x = []
    y = []
    for k in knn:
        x.append(k)
        y.append(knn[k])
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.set_title('average degree connectivity')
    ax.set_xlabel('k')
    ax.set_ylabel('k_nn')
    if log :
        plt.xscale('log')
        plt.yscale('log')

    plt.savefig('average_degree_connectivity.png') 

    plt.show()

# 隣接平均強度をプロット
def assortativity_strength_plot(G,log=False):
    knn = average_neighbor_strength(G)
    x = []
    y = []
    for k in knn:
        x.append(k)
        y.append(knn[k])
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.set_title('average strength connectivity')
    ax.set_xlabel('strength')
    ax.set_ylabel('k_nn')
    if log :
        plt.xscale('log')
        plt.yscale('log')

    plt.savefig('average_strength_connectivity.png') 

    plt.show()

# 隣接する頂点の平均強度を求める
def average_neighbor_strength(G):
    H = strength(G)
    ave_dict={}
    n_dict={}
    for u in H.nodes():
        st_sum = 0
        N = 0
        for v in H.neighbors(u):
            st_sum += H.nodes[v]['strength']
            N += 1
        
        u_st = H.nodes[u]['strength']
        if u_st in ave_dict :
            ave_dict[u_st] += st_sum/N
            n_dict[u_st] += 1
        else :
            ave_dict[u_st] = st_sum/N
            n_dict[u_st] = 1
    
    for i in ave_dict:
        ave_dict[i] /= n_dict[i]

    return ave_dict

# 上位N番目までの attribute をもつ node を列挙
def topN_node_ranking(G,N,attr):
    if N > len(G.nodes()):
        print('Error : Too many items are required')
        return
    
    attributes = []
    for i in G.nodes():
        attributes += [G.nodes[i][attr]]
    
    ### 参考 : distribution() ###
    attributes.sort()
    L=len(attributes)
    l=1
    list_s=[]
    list_l=[]

    for i in range(L-1):
        if attributes[i]==attributes[i+1]:
            l=l+1
        else :
            s=attributes[i]
            list_s.append(s)
            list_l.append(l)
            l=1
    list_s.append(attributes[L-1])
    list_l.append(l)
    ############################

    list_s.reverse()
    list_l.reverse()
    L = 0
    rank = 1

    print('rank : value : node\n')
    for j in range(len(list_s)):
        s = list_s[j]
        for i in G.nodes():
            if G.nodes[i][attr] == s :
                print('{} : {} : {}'.format(rank,s,i))
                rank += 1
        L += list_l[j]
        if L >= N :
            break

def topN_node_list(G,N,attr):
    if N > len(G.nodes()):
        print('Error : Too many items are required')
        return
    
    attributes = []
    for i in G.nodes():
        attributes += [G.nodes[i][attr]]
    
    ### 参考 : distribution() ###
    attributes.sort()
    L=len(attributes)
    l=1
    list_s=[]
    list_l=[]

    for i in range(L-1):
        if attributes[i]==attributes[i+1]:
            l=l+1
        else :
            s=attributes[i]
            list_s.append(s)
            list_l.append(l)
            l=1
    list_s.append(attributes[L-1])
    list_l.append(l)
    ############################

    list_s.reverse()
    list_l.reverse()
    L = 0

    l = []
    for j in range(len(list_s)):
        s = list_s[j]
        for i in G.nodes():
            if G.nodes[i][attr] == s :
                l.append((i,s))
        L += list_l[j]
        if L >= N :
            break

    return l

# 連結でない部分グラフそれぞれの代表頂点をピックアップ
# リストの並びは部分グラフの大きさの降順
# G : グラフ,  N : 各部分グラフからピックアップする最大個数,  attr : 参照する頂点の特徴量
def representatives_of_components(G,N,attr):

    if attr == 'degree' :
        G = G.copy()
        for u in G.nodes():
            G.nodes[u]['degree'] = G.degree[u]
    
    # nx.connected_components() は、set 型 を生成する generator オブジェクトを返す
    compo_list = sorted(list(nx.connected_components(G)), key = len, reverse = True)

    n_list = []
    if N >= 2 :
        for d in compo_list:
            picup_list = []
            d = list(d) # set 型から list 型に変換
            d.sort(key = lambda u : G.nodes[u][attr], reverse = True)
            for i in range(len(d)):
                if i >= N :
                    break
                picup_list.append(d[i])
            n_list.append(picup_list)
    elif N == 1:
        for d in compo_list:
            d = list(d)
            d.sort(key = lambda u : G.nodes[u][attr], reverse = True)
            n_list.append(d[0])
    
    return n_list

###------------------------------------------------------------------###   
    

