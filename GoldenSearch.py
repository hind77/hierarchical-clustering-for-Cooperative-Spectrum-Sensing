#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:14:04 2020

@author: hind
"""
#libraries

import numpy as np
import itertools
import random as rand
import time
from pylab import pi
import math
from scipy import special as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import factorial
import decimal

#initialistion 
N= 25000
pfa = np.arange(0.1, 1, 0.05)
N0 = 1

#snr_dB = np.arange(-15, 5, 1)
snrDB = -15
snr = pow(10,(snrDB/10)) #Linear value of SNR 
pd_ray = np.arange(0.1, 1, 0.05)# probability of detection of rayleigh
pd_AWGN = np.arange(0.1, 1, 0.05)# probability of detection of rayleigh

def Compute_thresh_CFAR(pfa):
    thresh = ((math.sqrt(2)*sp.erfcinv(2*pfa)*math.sqrt(1/N))+1)
    return round(thresh,30)
    
def compute_PDray(thresh,srnAV):
    
    first_part = 0.5*sp.erfc((thresh*math.sqrt(N)-math.sqrt(N))/math.sqrt(2))
    second_part = math.exp((1/2*N*pow(thresh,2)*pow(srnAV,2))-((thresh-1)/thresh*srnAV))
    third_part = 0.5*sp.erfc((math.sqrt(1/N)*(1/thresh*srnAV)-thresh*math.sqrt(N)+math.sqrt(N))/math.sqrt(2))
    return first_part+second_part*third_part

for m in range(0,len(pfa)):
    thresh = Compute_thresh_CFAR(pfa[m])
    pd_ray[m]= compute_PDray(thresh,snr)
 
print("this is pd_ray",pd_ray)

plt.figure(2)
ax = plt.axes()
ax.set_xticks(np.arange(0.1,1,step=0.1))
#ax.set_yticks(np.arange(0.1,1,step=0.05))
ax.plot(pfa,pd_ray, linestyle='dashed', color='red',marker=">",label='pd_ray')
ax.set_title('Pf Vs Pd ' )
ax.margins(x=0,y=0)
ax.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xlabel("Probability of false alarm (Pf)")
ax.set_ylabel("Probability of detection (Pd)")
plt.show()













    
    
    


