#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:24:16 2020

@author: hind
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:34:08 2020

@author: hind
"""

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
snr_dB = np.arange(-15, -7, 1) # the snr in dB range
snr = pow(10,(snr_dB/10)) #Linear value of SNR 
L = 1000 #Number of samples
mes = np.random.randint(0,2, L) #Generating 0 and 1 with equal probability for BPSK
s = 2*(mes)-1 #BPSK modulation
snr_data = {k: [] for k in range(0,len(snr))} #SNR variation results storage
snr_values = {k: [] for k in range(0,len(snr))} #SNR dB values storage
pf = np.arange(0, 1, 0.05)# probability of false alarm 
all_nodes_simulation_pd = np.arange(0, 1, 0.05)
all_nodes_simulation_pf = np.arange(0, 1, 0.05)
pd = np.arange(0, 1, 0.05)# probability of detection
thresh = [None] * len(pf) # the threshold



# =============================================================================
# ************************** generate the signals *********************************
# =============================================================================
def generate_Pu_signal(attempt,snr):
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
      #looping arround the Pf values
      for x in range(0,len(snr)): 
          coop_simulation_pd = {k: [] for k in range(10)}
          coop_simulation_pf = {k: [] for k in range(10)}
          snr_values[x] = snr_dB[x]
          print(x)
          for m in range(0,len(pf)): 
              #rounds inilisializations          
              active_rounds = 0
              passive_rounds = 0
              Round = 0
              coop_pf_simulation_mapping = {k: [] for k in range(10)}
              coop_pd_simulation_mapping = {k: [] for k in range(10)}      
              cluster_inter_data = dict()
              kmeans_cluster_inter_data = dict()
              cooperative_consensus = dict()
              consensus_coop_iter = 0
              cons_coop_iter = []
              # get the threshs
              threshs = mapping_thresh(pf[m],self.number_of_nodes)    
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
                  local_statistics = Local_Statistics_mapping(self.number_of_nodes,attempt,snr[x])
                  
                  # all nodes participation with consensus                  
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
      
                  # get the decisions for coop sensing
                  for k,v in cooperative_consensus.items():
                      if v > threshs[k]:
                          if attempt > 0.5:
                              coop_pd_simulation_mapping[k].append(1)
                          else:
                              coop_pf_simulation_mapping[k].append(1)
                      else:
                          coop_pd_simulation_mapping[k].append(0)
                  Round += 1

    #********************************  single pd mapping *****************************************          
    

              for k,v in coop_pd_simulation_mapping.items(): 
                 coop_simulation_pd[k].append(sum(v)/active_rounds)
              for k,v in coop_pf_simulation_mapping.items(): 
                 coop_simulation_pf[k].append(sum(v)/passive_rounds)            
          
    #********************************  simulated cooperative pd and pf mapping *****************************************          
              # probability of detection calculation    
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
          print("\n oooooooooooooooooooooooooooooooooooooooo",x)              
          snr_data[x].append(coop_simulated_pd_average)
          print(snr_data)

          
    #******************************** pf vs pd plotting *****************************************
          # final nodes Evaluation results plotting 
      ax = plt.axes()    
      for k,v in snr_data.items():
        ax.plot(pf,v[0], linestyle='dashed',label=snr_values[k])
      ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    
      ax.set_xticks(np.arange(0,1,step=0.1))
      ax.set_yticks(np.arange(0,1,step=0.05))
      ax.margins(x=0,y=0)         
      ax.set_title("ROC curve for SNR varaiations")
      ax.set_xlabel("probability of false alarm")
      ax.set_ylabel("probability of detection")
      ax.grid(True)
      plt.show()
                 
# =============================================================================
# ************************** Annex functions *********************************
# =============================================================================

def mapping_thresh(pf,number_of_nodes):
    thresh = dict()
    for k in range(0,number_of_nodes):
        thresh[k] = 2*sp.gammaincinv(L/2,1-pf)/L
    return thresh 
   
def Local_Statistics_mapping(nodes,attempt,snr):
    signal_mapping = dict()
    for k in range(0,nodes):
        y = generate_Pu_signal(attempt,snr)
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

        
