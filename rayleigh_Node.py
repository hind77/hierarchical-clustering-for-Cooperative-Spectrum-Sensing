#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:34:42 2020

@author: hind
"""
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
import pdb

# =============================================================================
# ***************************** Global Initilizations**************************
# =============================================================================

#--------------Signal Processing-------------------------
snr_dB = -5
snr = pow(10,(snr_dB/10)) #Linear value of SNR 
fc = 2*pow(10,9)
L = 1000 #Number of samples
N0 = 1/snr 
#Generating 0 and 1 with equal probability for BPSK
mes = np.random.randint(0,2, L)
#BPSK modulation
s = 2*(mes)-1
pf = np.arange(0, 1, 0.05)# probability of false alarm 
all_nodes_pd = np.arange(0, 1, 0.05)
all_nodes_simulation_pd = np.arange(0, 1, 0.05)
all_nodes_simulation_pf = np.arange(0, 1, 0.05)
pd = np.arange(0, 1, 0.05)# probability of detection
Cpd = np.arange(0, 1, 0.05)# Cooperative probability of detection

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
   for i in range(0,len(labels)):
       clustering[labels[i]].append(i)
   return clustering
# =============================================================================
# ************************** generate the signals *********************************
# =============================================================================
def generate_Pu_signal(distance, attempt):
    #AWGN noise with mean 0 and variance 1
#    Generating Rayleigh channel coefficient
    h = np.random.normal(0,0.5,L)+np.random.normal(0,0.5,L)  
    n = np.random.normal(0,1,L)   
    
    if attempt> 0.5:
        y = math.sqrt(snr)*abs(h)*s+n
    else:
        y = n    
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
      clustering = hiarchical_clustering(self.number_of_nodes,positions,3)
      print("\n the hirachical clusters are: ",clustering)
      # get the clusters from K-means clustering
      kmeans_clustering = Kmeans_clustering(self.number_of_nodes,positions,3)
      print("\n the kmeans clusters are: ",kmeans_clustering)
      # get the distances from the PU 
      distances = mapping_distances(positions)
      # probability of detection mapping initilizations 
      local_simulation_pd = {k: [] for k in range(10)}
      local_simulation_pf = {k: [] for k in range(10)}
      coop_simulation_pd = {k: [] for k in range(10)}
      coop_simulation_pf = {k: [] for k in range(10)}
      Hc_simulation_pd = {k: [] for k in range(10)}
      Hc_simulation_pf = {k: [] for k in range(10)}
      kmeans_simulation_pd = {k: [] for k in range(10)}
      kmeans_simulation_pf = {k: [] for k in range(10)}
      #looping arround the Pf values 
      for m in range(0,len(pf)): 
          #rounds inilisializations          
          all_nodes_detect = 0
          all_nodes_false_alarm = 0
          active_rounds = 0
          passive_rounds = 0
          Round = 0
          local_pf_simulation_mapping = {k: [] for k in range(10)}
          coop_pf_simulation_mapping = {k: [] for k in range(10)}
          HC_pf_simulation_mapping = {k: [] for k in range(10)}
          kmeans_pf_simulation_mapping = {k: [] for k in range(10)}          
          local_pd_simulation_mapping = {k: [] for k in range(10)}
          coop_pd_simulation_mapping = {k: [] for k in range(10)}
          HC_pd_simulation_mapping = {k: [] for k in range(10)}
          kmeans_pd_simulation_mapping = {k: [] for k in range(10)}          
          cluster_inter_data = dict()
          kmeans_cluster_inter_data = dict()
          cooperative_consensus = dict()
          consensus_Hc_iter = 0
          consensus_coop_iter = 0
          consensus_kmeans_iter = 0
          cons_Hc_iter = []
          cons_kmeans_iter = []
          cons_coop_iter = []
          cons_static = []
          cons_kmeans_static = []
          cons_coop_static =[]
          thresh_CFAR = ((math.sqrt(2)*sp.erfcinv(2*pf[m])*math.sqrt(1/L))+1)
          
          print("this is the issue thresh", thresh_CFAR)
          # get the threshs
          threshs = mapping_thresh(pf[m],distances,thresh_CFAR)

          #get the attempt probability 
          attempt = rand.uniform(0, 1)
          # starting the montecarlo simulation 
          while Round < 10000:
#              compute the attempt proba
              attempt = rand.uniform(0, 1)
              if attempt>0.5:
                  active_rounds = active_rounds + 1
              else:
                  passive_rounds = passive_rounds + 1
                  
              # generate the local statistic tests 
              local_statistics = Local_Statistics_mapping(self.number_of_nodes,distances, attempt) 
              # generate the consensus results for all nodes participation
              sigma_all_nodes = 0             
              for k,v in local_statistics.items():                      
                      cooperative_consensus[k] = local_statistics[k]
              while len(set(cooperative_consensus.values())) != 1:
                  for k,v in cooperative_consensus.items(): 
                      
                      resultsetcoop = [value for key, value in cooperative_consensus.items() if key not in ([k])]
                      for i in range(0,len(resultsetcoop)):
                          sigma_all_nodes = sigma_all_nodes + (resultsetcoop[i]-cooperative_consensus[k])  
                      cooperative_consensus[k] = cooperative_consensus[k] + (1/(len(resultsetcoop)+1))* sigma_all_nodes
                      sigma_all_nodes = 0
                  consensus_coop_iter = consensus_coop_iter + 1 
                  cons_coop_iter.append(consensus_coop_iter)
                  cons_coop_static.append(cooperative_consensus[1])
                  
              # get the decisions for local sensing
              for k,v in local_statistics.items():
                  if v > threshs[k]:
                      if attempt > 0.5:
                          local_pd_simulation_mapping[k].append(1)
                      else:
                          local_pf_simulation_mapping[k].append(1)
                  else:
                      local_pd_simulation_mapping[k].append(0)


              # get the decisions for coop sensing
              for k,v in cooperative_consensus.items():
                  if v > threshs[k]:
                      if attempt > 0.5:
                          coop_pd_simulation_mapping[k].append(1)
                      else:
                          coop_pf_simulation_mapping[k].append(1)
                  else:
                      coop_pd_simulation_mapping[k].append(0)
                            
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
              cluster_inter_data = cluster_intra_data
              while len(set(cluster_inter_data.values())) != 1:
                  for k,v in cluster_inter_data.items():
                      resultset = [value for key, value in cluster_inter_data.items() if key not in ([k])]
                      for i in range(0,len(resultset)):
                          sigma = sigma + (resultset[i]-cluster_inter_data[k])
                      cluster_inter_data[k] = cluster_inter_data[k] + (1/(len(resultset)+1))* sigma
                      sigma = 0
                  consensus_Hc_iter = consensus_Hc_iter + 1 
                         
              # generate the k-means inter-clustering data
              #k-means clustering consensus
              kmeans_sigma = 0
              kmeans_cluster_inter_data = kmeans_cluster_intra_data
              while len(set(kmeans_cluster_inter_data.values())) != 1: 
                  for k,v in kmeans_cluster_inter_data.items():
                      resultsetkm = [value for key, value in kmeans_cluster_inter_data.items() if key not in ([k])]
                      for i in range(0,len(resultsetkm)):
                          kmeans_sigma = kmeans_sigma + (resultsetkm[i]-kmeans_cluster_inter_data[k])
                      kmeans_cluster_inter_data[k] = kmeans_cluster_inter_data[k] + (1/(len(resultsetkm)+1))* kmeans_sigma
                      kmeans_sigma = 0
                  consensus_kmeans_iter = consensus_kmeans_iter + 1 
                  cons_kmeans_iter.append(consensus_kmeans_iter)
                     
              #hiarchical decision mapping
              Hc_mapping = dict()
              for k,v in clustering.items():
                  for i in range(0,len(v)):
                      Hc_mapping[int(v[i])]= cluster_inter_data[k]   
              for k,v in Hc_mapping.items():
                  if v > threshs[k]:
                      if attempt >0.5:
                          HC_pd_simulation_mapping[k].append(1)
                      else:
                          HC_pd_simulation_mapping[k].append(0)
                          
                  else:
                      HC_pd_simulation_mapping[k].append(0)
                      
              #kmeans decision mapping
              kmeans_mapping = dict()
              for k,v in kmeans_clustering.items():
                  for i in range(0,len(v)):
                      kmeans_mapping[v[i]]= kmeans_cluster_inter_data[k]
              for k,v in kmeans_mapping.items():
                  if v> threshs[k]:
                      if attempt > 0.5:
                          kmeans_pd_simulation_mapping[k].append(1)
                      else:
                          kmeans_pf_simulation_mapping[k].append(1)                          
                  else:
                      kmeans_pd_simulation_mapping[k].append(0) 
                      
              Round += 1
#********************************  single pd mapping *****************************************          

#        #hiarchical results for Pd & pf
          for k,v in HC_pd_simulation_mapping.items():
              Hc_simulation_pd[k].append(sum(v)/active_rounds)
          for k,v in HC_pf_simulation_mapping.items():
              Hc_simulation_pf[k].append(sum(v)/passive_rounds)
#        #kmeans results for Pd & pf
          for k,v in kmeans_pd_simulation_mapping.items():
              kmeans_simulation_pd[k].append(sum(v)/active_rounds)
          for k,v in kmeans_pf_simulation_mapping.items():
              kmeans_simulation_pf[k].append(sum(v)/passive_rounds) 
#        coop results for Pd & pf     
          for k,v in coop_pd_simulation_mapping.items(): 
             coop_simulation_pd[k].append(sum(v)/active_rounds)
          for k,v in coop_pf_simulation_mapping.items(): 
             coop_simulation_pf[k].append(sum(v)/passive_rounds)            
          #local results for Pd & pf
          for k,v in local_pd_simulation_mapping.items(): 
             local_simulation_pd[k].append(sum(v)/active_rounds)
          for k,v in local_pf_simulation_mapping.items(): 
             local_simulation_pf[k].append(sum(v)/passive_rounds) 
      
      
#********************************  simulated averaged pd and pf mapping *****************************************          

          # hierarchical averaged results
      hierchical_simulated_pd_average = list()

      for i in range(0,len(pf)):
          liste = list()
          for k,v in Hc_simulation_pd.items():              
              liste.append(v[i])
          hierchical_simulated_pd_average.append(sum(liste)/len(liste))

      hierchical_simulated_pf_average = list()
      
      for i in range(0,len(pf)):
          liste = list()
          for k,v in Hc_simulation_pf.items():
              
              liste.append(v[i])
          hierchical_simulated_pf_average.append(sum(liste)/len(liste))          
    

          # cooperative all nodes averaged results
      coop_simulated_pd_average = list()
      for i in range(0,len(pf)):
          liste = list()
          for k,v in coop_simulation_pd.items():              
              liste.append(v[i])
          coop_simulated_pd_average.append(sum(liste)/len(liste))
   
      coop_simulated_pf_average = list()
      
      for i in range(0,len(pf)):
          liste = list()
          for k,v in coop_simulation_pf.items():
              
              liste.append(v[i])
          coop_simulated_pf_average.append(sum(liste)/len(liste)) 

          
          #k-means results for Pd
      kmeans_simulated_pd_average = list()
      
      for i in range(0,len(pf)):
          liste = list()
          for k,v in kmeans_simulation_pd.items():
              
              liste.append(v[i])
          kmeans_simulated_pd_average.append(sum(liste)/len(liste))
          
      kmeans_simulated_pf_average = list()
      
      for i in range(0,len(pf)):
          liste = list()
          for k,v in kmeans_simulation_pf.items():
              
              liste.append(v[i])
          kmeans_simulated_pf_average.append(sum(liste)/len(liste))
         
      
#******************************** pf vs pd plotting *****************************************
      # final nodes Evaluation results plotting       
      plt.figure(2)
      ax = plt.axes()
      ax.set_xticks(np.arange(0,1,step=0.1))
      ax.set_yticks(np.arange(0,1,step=0.05))
      ax.plot(pf,kmeans_simulated_pd_average, linestyle='solid', color='black',marker=">",label='kmeans clustering probability of detection')
      ax.plot(pf,hierchical_simulated_pd_average, linestyle='dashed', color='red',marker=">",label='Hiarchical clustering probability of detection')
      ax.plot(pf,local_simulation_pd[1], linestyle='dashed', color='blue',marker=">",label='local probability of detection')
      ax.plot(pf,coop_simulated_pd_average,  linestyle='dashed', color='orange',marker=">",label='cooperative probability of detection')
      ax.set_title('Pf Vs Pd using Adaptive Threshold')
      ax.margins(x=0,y=0)
      ax.grid(True)
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
      ax.set_xlabel("Probability of false alarm (Pf)")
      ax.set_ylabel("Probability of detection (Pd)")
      plt.show()
      plt.pause(0.05)

#      plt.figure(3)
#      ax = plt.axes()
#      ax.set_xticks(np.arange(0,1,step=0.1))
#      ax.set_yticks(np.arange(0,1,step=0.05))
##      ax.plot(pf,kmeans_simulated_pf_average, linestyle='dashed', color='black',marker=">")
##      ax.plot(pf,hierchical_simulated_pf_average, linestyle='solid', color='red',marker=">")
#      ax.plot(pf,local_simulation_pf[1], linestyle='dashed', color='blue',marker=">",label='local simulated probability false alarm')
#      ax.plot(pf,all_nodes_simulation_pf,  linestyle='solid', color='orange',marker=">",label='cooperative simulated probability of false alarm')
#      ax.set_title('Pf Vs simulated pf ' )
#      ax.margins(x=0,y=0)
#      ax.grid(True)
##      black_patch = mpatches.Patch(color='black', label='kmeans clustering probability of detection')
##      red_patch = mpatches.Patch(color='red', label='Hiarchical clustering probability of detection')
#      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#      ax.set_xlabel("Probability of false alarm (Pf)")
#      ax.set_ylabel("Simulated Probability of false alarm (S-Pf)")
#      plt.show()
#      plt.pause(0.05)
#          
#          
              
     
     
     
         
     
     
     
     

# =============================================================================
# ************************** Annex functions *********************************
# =============================================================================


def mapping_distances(positions):
    distances = dict()
    lis = []
    for i in range(0,len(positions)):
        lis.append(positions[i].tolist())
        distances[i] = Compute_distance_from_PU(lis)
        lis.clear()
    return distances

def mapping_thresh(pf,distances,thresh_CFAR):
    thresh = dict()
    for k,v in distances.items():
        thresh[k] = gsection(compute_PDray, thresh_CFAR*0.9, thresh_CFAR  , thresh_CFAR*1.1 , tol = 1e-9)
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

def compute_PDray(th):
    
    first_part = 0.5*sp.erfc((th*math.sqrt(L)-math.sqrt(L))/math.sqrt(2))
    second_part = math.exp((1/2*L*pow(th,2)*pow(snr,2))-((th-1)/th*snr))
    third_part = 0.5*sp.erfc((math.sqrt(1/L)*(1/th*snr)-th*math.sqrt(L)+math.sqrt(L))/math.sqrt(2))
    return first_part+second_part*third_part

def gsection(compute_PDray, thresh_lower, thresh, thresh_upper, tol = 1e-9):
    gr1 = 1 + (1 + np.sqrt(5))/2
    fl = compute_PDray(thresh_lower)
    fr = compute_PDray(thresh_upper)
    fm = compute_PDray(thresh)
    while ((thresh_upper - thresh_lower) > tol):
        if ((thresh_upper - thresh) > (thresh - thresh_lower)):
            y = thresh + (thresh_upper - thresh)/gr1
            fy = compute_PDray(y)
            if (fy >= fm):
                thresh_lower = thresh
                fl = fm
                thresh = y
                fm = fy
            else:
                thresh_upper = y
                fr = fy
        else:
            y = thresh - (thresh - thresh_lower)/gr1
            fy = compute_PDray(y)
            if (fy >= fm):
                thresh_upper = thresh
                fr = fm
                thresh = y
                fm = fy
            else:
                thresh_lower = y
                fl = fy
    return(thresh)
        
