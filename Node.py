

# =============================================================================
# ********************************* import libraries*****************************
# =============================================================================

import numpy as np
import itertools
import random as rand
import time
from pylab import pi
import math
from scipy import special as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# =============================================================================
# ***************************** Global Initilizations**************************
# =============================================================================

#--------------Signal Processing-------------------------
snr_dB = -9
snr = pow(10,(snr_dB/10)) #Linear value of SNR 
fc = 2*pow(10,9)
L = 1000 #Number of samples
N0 = 1/snr 
#           Generating 0 and 1 with equal probability for BPSK
mes = np.random.randint(0,2, L)
#           BPSK modulation
s = 2*(mes)-1
pf = np.arange(0, 1, 0.05)# probability of false alarm 
all_nodes_pd = np.arange(0, 1, 0.05)
pd = np.arange(0, 1, 0.05)# probability of detection
Cpd = np.arange(0, 1, 0.05)# Cooperative probability of detection
thresh = [None] * len(pf) # the threshold
Round = 0 #number of Rounds ==> monte carlo number 
Consensus_iters = 0
#--------------Deep walk initilizations-------------------------
cnames = {
'darkblue':             '#00008B',
'darkgreen':            '#006400',
'deeppink':             '#FF1493'}
dims = 3
n_runs = 1
step_n = 120
step_set = [-1, 0 ,1]
runs = np.arange(n_runs)
step_shape = (step_n,dims)
distance = 0
positions = []
#common steps between all the nodes:
steps = [[ 1,  1,  0],[ 0,  0,  1],[ 1,  1,  0],[ 1,  0,  1],[ 1,  1,  1],[ 0,  1, -1],[ 1,  0,  1],[ 1,  0,  1],[ 0, -1,  1],[ 0,  0,  0],[-1,  1,  0],[-1, -1,  0],[ 0,  0,  0],[ 0, -1, -1],[ 0,  1, -1],[-1, -1,  0],[-1, -1,  0],[-1, -1,  0],[-1,  0,  1],[-1, -1,  1],[ 1,  1,  0],[ 0, -1,  1],[ 1, -1,  1],[ 0, -1, -1],[-1,  0, -1],[-1,  1, -1],[ 1,  1,  0],[ 0,  0,  0],[-1, -1,  0],[-1, -1,  1], [-1,  1,  0],[ 1,  1, -1],[-1, -1,  0],[-1,  0,  1],[ 0,  1, -1],[-1,  0,  0],[-1, -1, -1],[ 0, -1, -1],[ 1,  1, -1],[-1,  0,  0],[-1, -1,  0],[-1, -1,  0],[-1,  0,  0],[ 1,  1,  0],[ 0, -1, -1],[ 0,  0, -1],[-1,  0, -1],[ 0,  0, -1],[ 0,  0, -1],[-1,  1,  1],[ 0,  1, -1],[ 0,  1,  0],[ 1,  0,  0],[ 0, -1,  1],[-1,  1,  1],[ 1,  0, -1],[-1,  1,  0],[-1, -1, -1],[ 1,  1, -1],[ 1,  0,  1],[-1,  1,  1],[ 1,  0, -1],[ 1,  0,  0],[ 0,  0, -1],[ 1,  0, -1],[-1, -1, -1],[ 0,  0, -1],[-1,  0, -1],[ 1,  1, -1],[ 0,  0,  0],[-1,  0, -1],[-1,  0, -1],[ 0,  0,  0],[ 1,  1, -1],[ 0,  1,  0],[ 1,  1,  0],[-1,  0, -1],[ 1, -1,  0],[ 1, -1,  0],[ 1,  1,  0],[ 0,  0,  0],[ 0,  1,  0],[ 0,  1, -1],[ 0, -1,  0],[ 0, -1, -1],[-1,  1,  1],[ 0, -1,  1],[ 1,  1, -1],[-1,  0,  0],[ 1,  0, -1],[ 1,  0,  1],[ 1,  1, -1],[ 1,  0,  0],[-1,  1,  0],[-1,  1,  1],[ 1,  0,  1],[ 0,  0, -1],[-1,  1,  1],[ 1,  1,  0],[ 0,  1,  1],[ 1,  1,  0],[-1,  1,  1],[ 1,  1,  0],[ 0,  0,  1],[-1, -1,  0],[ 0,  1,  0],[ 0, -1,  0],[-1,  1,  0],[-1,  1,  0],[ 1,  1,  1],[-1,  1, -1],[-1, -1,  0],[-1,  0,  0],[ 1,  1,  0],
 [-1,  0,  0],[ 0,  0, -1],[-1,  0,  1],[ 0, -1, -1],[-1,  1,  1],[ 1,  0,  0]]
# =============================================================================
# ************************** Deep Walk*********************************
# =============================================================================
def Deep_walk():
    position = []
    for i, col in zip(runs, cnames):
        # Simulate steps in 3D
        origin = np.random.randint(low=0,high=30,size=(1,dims))
        path = np.concatenate([origin, steps]).cumsum(0)
        start = path[:1]
        stop = path[-1:]
        position.append(stop[0])
    distance = Compute_distance_from_PU(position)
    position_format = np.asarray(position, dtype=np.float32)
    return position
#--------------compute the distance-------------------------
def Compute_distance_from_PU(position): 
    x = tuple(map(tuple, position))[0]
    y =  (-24,30,0)
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance
# =============================================================================
# ************************** Hiarchical Clustering *********************************
# =============================================================================
def hiarchical_clustering(number_of_nodes,positions,number_clusters):    
   cluster = AgglomerativeClustering(n_clusters=number_clusters, affinity='euclidean', linkage='ward')
   cluster.fit_predict(positions)
   labels = cluster.labels_

   clustering = {k: [] for k in range(number_clusters)}
   for i in range(0,len(labels)):
       clustering[labels[i]].append(i)
   return clustering

# =============================================================================
# ************************** Kmeans Clustering *********************************
# =============================================================================
def Kmeans_clustering(number_of_nodes,positions, number_clusters):
   km = KMeans(n_clusters=number_clusters)
   km.fit(positions)
   labels = km.labels_ 
   clustering = {k: [] for k in range(number_clusters)}
   print("labels are: ",labels)
   for i in range(0,len(labels)):
       clustering[labels[i]].append(i)
   return clustering
# =============================================================================
# ************************** generate the signals *********************************
# =============================================================================
def generate_Pu_signal(distance, attempt):
    #AWGN noise with mean 0 and variance 1
    h = np.random.normal(0,0.5,L)+np.random.normal(0,0.5,L)  
    n = np.random.normal(0,1,L) 
    y = math.sqrt(snr)*abs(h)*s+n
    
    return y
# =============================================================================
# ************************** Node class *********************************
# =============================================================================
class Node():
  def __init__(self,number_of_nodes):  
    self.number_of_nodes = number_of_nodes
  def start(self):   
      # Extract the positions from the Deepwalk
      for i in range(0,self.number_of_nodes):
                  position = Deep_walk()
                  positions.append(position[0])
      # get the Clusters from Hiarchical clustering 
      clustering = hiarchical_clustering(self.number_of_nodes,positions,4)
      print("\n the hirachical clusters are: ",clustering)
      # get the clusters from K-means clustering
      kmeans_clustering = Kmeans_clustering(self.number_of_nodes,positions,4)
      print("\n the kmeans clusters are: ",kmeans_clustering)
      # get the distances from the PU 
      distances = mapping_distances(positions)
      # probability of detection mapping initilizations 
      local_pd = {k: [] for k in range(10)}
      all_nodes_pd = np.arange(0, 1, 0.05)
      Hc_pd = {k: [] for k in range(10)}
      kmeans_pd = {k: [] for k in range(10)}
      #looping arround the Pf values 
      for m in range(0,len(pf)): 
          #rounds inilisializations 
          detect = 0
          all_nodes_detect = 0
          Round = 0
          decisions_mapping = {k: [] for k in range(10)}
          decisions_HC_mapping = {k: [] for k in range(10)}
          decisions_kmeans_mapping = {k: [] for k in range(10)}
          cluster_inter_data = dict()
          kmeans_cluster_inter_data = dict()
          consensus_iter = 0
          cons_iter = []
          cons_static = []
          cons_kmeans_static = []
          # get the threshs
          threshs = mapping_thresh(pf[m],distances)
          print(threshs)
          #get the attempt probability 
          attempt = rand.uniform(0, 1)
          # starting the montecarlo simulation 
          while Round < 10000:
              # generate the local statistic tests 
              local_statistics = Local_Statistics_mapping(self.number_of_nodes,distances, attempt)
              all_nodes_Statistics_test = sum(list(local_statistics.values()))/self.number_of_nodes
              # compute all nodes average thresh 
              thresh = get_moyenn_thresh(threshs)
              # get the decisions for local sensing
              for k,v in local_statistics.items():
                  if v > threshs[k]:
                      decisions_mapping[k].append(1)
                  else:
                      decisions_mapping[k].append(0)
              if all_nodes_Statistics_test > thresh:
                   all_nodes_detect += 1 
              # generate the hiarchical intra-clustering data
              cluster_intra_data = dict()
              for k,v in clustering.items():
                  lst = []
                  for r,s in local_statistics.items():
                      for i in range(0,len(v)):
                          lst.append(local_statistics[int(v[i])])
                      cluster_intra_data[k]= sum(lst)/len(v)
                      lst.clear() 
              # generate the K-means intra-clustering data
              kmeans_cluster_intra_data = dict()
              for k,v in kmeans_clustering.items():
                 lst = []
                 for r,s in local_statistics.items():
                     for i in range(0,len(v)):
                         lst.append(local_statistics[int(v[i])])
                     kmeans_cluster_intra_data[k]= sum(lst)/len(v)
                     lst.clear()
              # generate the hiarchical inter-clustering data
              #hirachical clustering consensus
              sigma = 0
              if consensus_iter == 0:
                  cluster_inter_data = cluster_intra_data
              else:
                  for k,v in cluster_inter_data.items():
                      resultset = [value for key, value in cluster_inter_data.items() if key not in ([k])]
                      for i in range(0,len(resultset)):
                          sigma = sigma + (resultset[i]-cluster_inter_data[k])
                      cluster_inter_data[k] = cluster_inter_data[k] + (1/(len(resultset)+1))* sigma
                      sigma = 0
              # generate the k-means inter-clustering data
              #k-means clustering consensus
              kmeans_sigma = 0
              if consensus_iter == 0:
                  kmeans_cluster_inter_data = kmeans_cluster_intra_data
              else:
                  for k,v in kmeans_cluster_inter_data.items():
                      resultsetkm = [value for key, value in kmeans_cluster_inter_data.items() if key not in ([k])]
                      for i in range(0,len(resultsetkm)):
                          kmeans_sigma = kmeans_sigma + (resultsetkm[i]-kmeans_cluster_inter_data[k])
                      kmeans_cluster_inter_data[k] = kmeans_cluster_inter_data[k] + (1/(len(resultsetkm)+1))* kmeans_sigma
                      kmeans_sigma = 0
              #hiarchical decision mapping
              Hc_mapping = dict()
              for k,v in clustering.items():
                  for i in range(0,len(v)):
                      Hc_mapping[int(v[i])]= cluster_inter_data[k]   
              for k,v in Hc_mapping.items():
                  if v > threshs[k]:
                      decisions_HC_mapping[k].append(1)
                  else:
                      decisions_HC_mapping[k].append(0)
              #kmeans decision mapping
              kmeans_mapping = dict()
              for k,v in kmeans_clustering.items():
                  for i in range(0,len(v)):
                      kmeans_mapping[v[i]]= kmeans_cluster_inter_data[k]
              for k,v in kmeans_mapping.items():
                  if v> threshs[k]:
                      decisions_kmeans_mapping[k].append(1)
                  else:
                      decisions_kmeans_mapping[k].append(0) 
#******************************** consensus evaluation *****************************************                      
              # consensus plots data
              cons_iter.append(consensus_iter)
              cons_static.append(cluster_inter_data[1])
              cons_kmeans_static.append(kmeans_cluster_inter_data[1])
              consensus_iter += 1
              Round += 1
          # consensus plotting    
          cons_iter = cons_iter[:50]
          cons_static = cons_static[:50]
          cons_kmeans_static = cons_kmeans_static[:50]
          plt.title(" HC Convergence")
          plt.xticks(np.arange(10,30,step=2))
          start = 10
          end = 30
          plt.plot(cons_iter[start:end],cons_static[start:end],  linestyle='dashdot', marker='+', color='red', markerfacecolor='red')
          plt.xlabel("Iteration Step")
          plt.ylabel("Statistic Test (Yi)");
          plt.show()
          plt.pause(0.05)
          plt.title(" km Convergence")
          plt.xticks(np.arange(10,30,step=2))
          start = 10
          end = 30
          plt.plot(cons_iter[start:end],cons_kmeans_static[start:end],  linestyle='dashdot', marker='+', color='black', markerfacecolor='black')
          plt.xlabel("Iteration Step")
          plt.ylabel("Statistic Test (Yi)");
          plt.show()
#******************************** pd mapping *****************************************          
          # probability of detection calculation 
          #local results for Pd
          for k,v in decisions_mapping.items(): 
             local_pd[k].append(sum(v)/Round)  
          #all nodes participation results for Pd   
          all_nodes_pd[m] = all_nodes_detect/Round 
          #hiarchical results for Pd
          for k,v in decisions_HC_mapping.items(): 
             Hc_pd[k].append(sum(v)/Round) 
          #k-means results for Pd
          for k,v in decisions_kmeans_mapping.items(): 
             kmeans_pd[k].append(sum(v)/Round) 
             
      hierchical_pd_average = list()
      for i in range(0,len(pf)):
          liste = list()
          for k,v in Hc_pd.items():              
              liste.append(v[i])
          hierchical_pd_average.append(sum(liste)/len(liste))
      kmeans_pd_average = list()
      for i in range(0,len(pf)):
          liste = list()
          for k,v in kmeans_pd.items():              
              liste.append(v[i])
          kmeans_pd_average.append(sum(liste)/len(liste))          
#******************************** pf vs pd plotting *****************************************
      # final nodes Evaluation results plotting 
      plt.figure(2)
      ax = plt.axes()
      ax.set_xticks(np.arange(0,1,step=0.1))
      ax.set_yticks(np.arange(0,1,step=0.05))
      ax.plot(pf,kmeans_pd_average, linestyle='dashed', color='black',marker=">",label='kmeans clustering probability of detection')
      ax.plot(pf,hierchical_pd_average, linestyle='solid', color='red',marker=">", label='Hiarchical clustering probability of detection')
      ax.plot(pf,local_pd[1], linestyle='dashed', color='blue',marker=">", label='local probability of detection')
      ax.plot(pf,all_nodes_pd,  linestyle='solid', color='orange',marker=">", label='cooperative probability of detection')
      ax.set_title('Pf Vs Pd with SNR = -9 ' )
      ax.margins(x=0,y=0)
      ax.grid(True)
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
      ax.set_xlabel("Probability of false alarm (Pf)")
      ax.set_ylabel("Probability of detection (Pd)")
      plt.show()
      plt.pause(0.05)
          
          
              
     
     
     
         
     
     
     
     

# =============================================================================
# ************************** Annex functions *********************************
# =============================================================================

def removeDuplicates(listofElements):
    uniqueList = []
    for elem in listofElements:
        if elem not in uniqueList:
            uniqueList.append(elem)

    return uniqueList
def mapping_distances(positions):
    distances = dict()
    lis = []
    for i in range(0,len(positions)):
        lis.append(positions[i].tolist())
        distances[i] = Compute_distance_from_PU(lis)
        lis.clear()
    return distances
def mapping_thresh(pf,distances):
    thresh = dict()
    for k,v in distances.items():
        val = 1-2*pf
        thresh[k] = 2*sp.gammaincinv(L/2,1-pf)/L
    return thresh    
def Local_Statistics_mapping(nodes,distances, attempt):
    signal_mapping = dict()
    for k,v in distances.items():
        y = generate_Pu_signal(v,attempt)
        energy = pow(abs(y),2)
        # generate the statistic_test
        Statistic_test = np.sum(energy)*(1/L)
        signal_mapping[k] = Statistic_test
    return signal_mapping
def generate_local_decisions(local_statistics_mapping,thresh,node_number):
    for k,v in local_statistics_mapping.items(): 
        if v > thresh:
            decisions_mapping[k].append(1)
        else:
            decisions_mapping[k].append(0)
    return decisions_mapping
def get_moyenn_thresh(threshs):
    thresh = list()
    for k,v in threshs.items():
        thresh.append(v)
    moy = sum(thresh)/len(thresh)
    return moy
        
