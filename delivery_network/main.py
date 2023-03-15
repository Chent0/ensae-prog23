from graph import Graph, graph_from_file
import time
import random
import os
'''
data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g)
'''

#Séance 2  

'''
On implémente une fonction permettant d'évaluer un temps moyens un graphe
'''
def duration(filename, src, dest):
    g= graph_from_file(filename)
    start=time.perf_counter()
    g.min_power(src,dest)
    stop=time.perf_counter()
    return(stop-start)      

def duration_allroute(filename1, filename2,N):
    with open(filename1, "r") as file:
        nb_trajet=list(map(int, file.readline().split()))[0]
        avg_time=0
        for _ in range (N):
            src, dest, cost=list(map(int, file.readline().split()))
            avg_time+=duration(filename2, src, dest)/N
            print(avg_time)
        tot_time=nb_trajet*avg_time
    return tot_time

#print(duration_allroute(filename1_1,filename1_2, 10))
#print(duration_allroute(filename2_1,filename2_2, 10))
#print(duration_allroute(filename3_1,filename3_2, 5))
#print(duration_allroute(filename4_1,filename4_2, 5))
#print(duration_allroute(filename5_1,filename5_2, 1))
#print(duration_allroute(filename6_1,filename6_2, 1))
#print(duration_allroute(filename7_1,filename7_2, 1))
#print(duration_allroute(filename8_1,filename8_2, 1))
#print(duration_allroute(filename9_1,filename9_2, 1))
'''
NB: les distances du fichier routes.10 n'étant pas sous le bon format, il faut modifier graph_from_file.
(nous n'avons pas eu le temps de le faire encore)
Temps d'exécution des fichiers routes
routes.1 ~20 s
routes.2 ~96268.21900019424 s
routes.3 ~64055446.440028034 s
routes.4 ~28624969.50000059 s
routes.5 ~4612768.610000057 s
routes.6 ~289569851.34999925 s
routes.7 ~776184143.4499947 s
routes.8 ~152330837.85000053 s
routes.9 ~340250437.5000062 s
routes.10 ~ 
'''

'''
Question 12

On définit les fonctions permettant d'appliquer la méthode UnionFind: makeset permettant de créer des 
singletons contenant chaque noeud du graphe, find permettant de trouver quel est le noeud parent (la
racine) d'un noeud et union permettant d'unir deux sous arbres, sans qu'ils ne forment de cycle.
    
'''

def makeset(parent, rank, n): #fonction qui créé initialement n singletons, où n est le nombre de noeuds
    for i in range(1,n+1):
        parent[i]=i
        rank[i]=0

#Fonction permettant de trouver la racine d'un noeud. Si plusieurs noeuds ont la même racine, ils 
#appartiennent au même sous arbre. On choisit pour racine commune de ces noeuds, le noeud le plus en 
#'profondeur' dans le graphe.

def find(parent,k):  
    if parent[k]==k:
        return k
    return find(parent, parent[k])

#Fonction permettant d'unir deux sous-arbres. On attribue à chaque noeud un rang: cela permet de déterminer
#si, lorsque l'on doit unir deux sous arbres, on garde le noeud du premier ou du deuxième sous arbre 
# comme origine de l'arbre résultant. Les noeuds à l'origine des sous arbresont le rang le plus élevé.
# Le rang symbolise donc ici la profondeur d'un noeud dans un arbre.

def union(parent, rank, i,j): 
    root_i=find(parent,i)
    root_j=find(parent,j)
    if root_i!=root_j:
        if root_i<root_j:
            parent[root_i]=root_j
        else:
            parent[root_j]=root_i
            if rank[root_i]==rank[root_j]:
                rank[root_i]+=1

'''On écrit alors les fonctions nécessaires pour renvoyer un arbre couvrant de puissance minimale. 
'''

#Fonction permettant d'obtenir la liste des arêtes par ordre de puissance croissant

def sort_by_power(g):
    L_sort_by_power=sorted(g.edges,key=lambda x:x[2])
    return L_sort_by_power

#Algorithme inspiré de l'algorithme de Kruskal

def kruskal(g):
    parent={}
    rank={}
    new_g=Graph([])
    makeset(parent, rank, g.nb_nodes)
    g_sorted=sort_by_power(g)
    for k in range (len(g_sorted)):
        n1=g_sorted[k][0]
        n2=g_sorted[k][1]
        if find(parent,n1)!=find(parent,n2):
            new_g.add_edge(n1,n2,g_sorted[k][2])
            union(parent,rank,n1,n2)
    return new_g
    

'''
Question 14

Il s'agit de créer deux dictionnaires, l'un contenant les parents (ie le noeud qui précède dans l'arbre)
de chaque noeud, l'autre leur profondeur (ie le nombre d'arête qui les sépare du premier noeud que l'on
considère, idéalement la racine). 
L'idée est ensuite de partir du noeud le plus profond dans le graphe, de remonter jusqu'à atteindre la
même profondeur que le second noeud dans le graphe, et si ces deux derniers sont différents, on remonte
simultanément dans l'arbre jusqu'à ce que les deux parcours se rejoignent.
'''

def dictionnaries(g):
    nodes=g.nodes
    parents=dict(((node, (-1,-1))for node in nodes)) #dictionnaire des parents
    depths=dict(((node,0)for node in nodes)) #dictionnaire des profondeurs de chaque noeud
    parents[nodes[0]]=(nodes[0],0) #on choisit le premier noeud de l'arbre en guise de racine
    '''
    On définit une sous fonction dico qui permet de construire, par un parcours en profondeur récursif, les 
    dictionnaires parents et depths. Pour chaque noeud, l'on parcourt ses voisins pour lesquels on met à 
    jour le parent et la profondeur et l'on rappelle la fonction sur chaque voisin. 
    '''
    def dico(g,node,depths2):
        for edge in g.graph[node]:
            ngb= edge[0]
            if parents[ngb]==(-1,-1):
                parents[ngb]=(node,edge[1])
                depths[ngb]=depths2
                dico(g,ngb,depths2+1)
    dico(g,nodes[0],1)
    return parents, depths

'''
Complexité de l'algorithme:
On note n le nombre de noeud de l'arbre.
Dans cet algorithme on parcourt chaque noeud du graphe, jusqu'à ce qu'ils soient tous marqués. Pour chacun 
des noeuds, on exécute une boucle for sur l'ensemble des voisins des noeuds. On en déduit une complexité 
en O(n*avg(nb_ngb)) où avg(nb_ngb) est le nombre moyen de voisins par noeud du graphe. S'agissant d'un arbre
il est raisonnable de considérer que avg(ng_ngb) est petit, on peut donc approximer la complexité par O(n)
'''
'''
On écrit à présent une fonction retournant la puissance nécessaire pour effectuer le trajet dans l'arbre.
On met en oeuvre la méthode expliquée au début de la question. (NB: s'agissant d'un arbre, le trajet entre
deux noeuds est unique, la puissance renvoyée correspond donc bien ici à la puissance nécessaire pour effectuer
le trajet.)
'''
def new_get_power(parents, depths, src,dest): 
    power=0
    depth_src=depths[src]
    depth_dest=depths[dest]
    n1=0
    n2=0
    if depth_src>depth_dest:
        n1=src
        n2=dest
    else:
        n1=dest
        n2=src
    res=depths[n2]
    while depths[n1]!=res: #tant que les profondeurs des noeuds dans le graphe sont différentes
        power=max(parents[n1][1],power) #on met à jour la puissance car l'on est en train de parcourir le chemin entre les noeuds
        n1=parents[n1][0] #on remonte dans l'arbre 
    while n1!=n2: #lorsque les profondeurs sont égales mais que les noeuds sont différents
        power=max(parents[n1][1],power) #mise à jour de la puissance
        power=max(parents[n2][1],power) #mise à jour de la puissance
        n1=parents[n1][0] #on remonte simultanément dans le graphe
        n2=parents[n2][0]
    return power


'''
Dans cette fonction, on parcourt au maximum l'ensemble des noeuds de l'arbre, à travers les boucles while.
De plus, on effectue des opérations bornées au sein des boucles. D'où une complexité dans le pire des cas en 
O(n). 
Pour déterminer la puissance minimale pour parcourir un chemin au sein d'un graphe, on effectue successivement
les fonctions dictionnaries et new_get_power, ayant chacune dans le meilleur des cas une complexité en 
O(n). D'où une complexité linéaire en O(n) pour déterminer la puissance minimale d'un trajet. 
'''
'''
Question 15
'''

def test_time(filename1, filename2, filename3):
    g= graph_from_file(filename2)
    with open(filename1, "r") as file:
        nb_trajet=list(map(int, file.readline().split()))[0]
        tot_time=0
        start=time.perf_counter()
        new_g=kruskal(g)
        parents, depths=dictionnaries(new_g)
        for _ in range (nb_trajet):
            src, dest, cost=list(map(int, file.readline().split()))
            pow=new_get_power(parents,depths, src, dest)
        stop=time.perf_counter()
        tot_time+=(stop-start)
    return tot_time

def write_in_routes(filename1, filename2, filename3):
    g= graph_from_file(filename2)
    with open(filename1, "r") as file:
        otherfile=open(filename3,"r+")
        nb_trajet=list(map(int, file.readline().split()))[0]
        new_g=kruskal(g)
        parents, depths=dictionnaries(new_g)
        for _ in range (nb_trajet):
            src, dest, cost=list(map(int, file.readline().split()))
            pow=new_get_power(parents,depths, src, dest)
            if len(otherfile.readlines())!=nb_trajet:
                    otherfile.write(str(pow)+"\n")
        otherfile.close()

'''
Temps d'exécution (tot_time) obtenus pour les différents fichiers routes:
routes.1= 0.0003954999992856756 s
routes.2=1.4070429999992484 s
routes.3=18.02433879999444 s
routes.4=18.71622809999826 s
routes.5=5.997746599998209 s
routes.6=24.45902919999935 s
routes.7=23.63708449999831 s
routes.8=19.877801799993904 s
routes.9=25.81786640000064 s
routes.10=
'''
print(write_in_routes(filename2_1,filename2_2,filename2_3))
#print(test_time(filename1_1,filename1_2,filename1_3))
#print(test_time(filename2_1,filename2_2,filename2_3))
#print(test_time(filename3_1,filename3_2,filename3_3))
#print(test_time(filename4_1,filename4_2,filename4_3))
#print(test_time(filename5_1,filename5_2,filename5_3))
#print(test_time(filename6_1,filename6_2,filename6_3))
#print(test_time(filename7_1,filename7_2,filename7_3))
#print(test_time(filename8_1,filename8_2,filename8_3))
#print(test_time(filename9_1,filename9_2,filename9_3))
#print(test_time(filename10_1,filename10_2,filename10_3))