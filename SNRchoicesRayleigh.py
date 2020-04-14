#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:23:02 2020

@author: hind
"""


#import libararies
import numpy as np
from scipy import special as sp
import math
import matplotlib.pyplot as plt
from pylab import pi

# =============================================================================
# --------------------------Initilizations------------------------------------
# =============================================================================
snr_dB = np.arange(-15, 5, 1)
snr = pow(10,(snr_dB/10)) #Linear value of SNR 
fc = 2*pow(10,8)
L = 1000 #Number of samples
s = np.ones(L) 
pf = np.arange(0, 1, 0.01)# probability of false alarm 
pd = np.arange(0, 1, 0.01)  # probability of detection
pmd = [None] * len(snr_dB) # probability of miss detection
L = 1000 #Number of samples
snr_data = {k: [] for k in range(0,len(snr))}
snr_values = {k: [] for k in range(0,len(snr))}
thresh = [None] * len(pf) # the threshold
colors = ['blue','green','red','cyan','magenta','yellow','black']

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
# --------------------------signal Processing------------------------------------
# ============================================================================= 
distance = Compute_distance_from_PU(Deep_walk())
for i in range(0,len(snr)):
    snr_values[i]= snr_dB[i]
    for m in range(0,len(pf)):
        detect = 0
        distance = Compute_distance_from_PU(Deep_walk())
        for k in range(0,10000):#Number of Monte Carlo Simulations
            
#           Generating 0 and 1 with equal probability for BPSK
            mes = np.random.randint(0,2, L)
#           BPSK modulation
            s = 2*(mes)-1
#           Generating Rayleigh channel coefficient
            h = np.random.normal(0,0.5,L)+np.random.normal(0,0.5,L)  
            n = np.random.normal(0,1,L)   
            y = math.sqrt(snr[i])*abs(h)*s+n
            energy = pow(abs(y),2)#Energy of received signal over L samples
            Statistic_test = np.sum(energy)*(1/L) #statistic test
#            print("this is Statistic",Statistic_test)
        
#            val = 1-2*pf[m]
#            thresh[m] = (((math.sqrt(2)*sp.erfinv(val))/ math.sqrt(L))+1) 
            # thresh rayleigh version
            #thresh
#            print("this is pf",pf[m])
            thresh[m] = 2*sp.gammaincinv(L/2,1-pf[m])/L
#            print("this is thresh",thresh[m])
            if(Statistic_test >= thresh[m]):
                detect += 1
        pd[m] = detect/k
        snr_data[i].append(pd[m])
        
#    print("this is snr",snr[i])
#    print("this is thresh",thresh)    
#    print("this pd",pd)
#    print(snr_data)


#--------------------------------------------------------    
# simulation plots
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
#    
#    plt.plot(snr_dB,pd)
