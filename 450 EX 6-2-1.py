# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 18:20:28 2023
    in class monday 



@author: treshur 
"""

""" Example_6-2-1.py """
import matplotlib.pyplot as plt
import numpy as np
from math import exp, cos, sin, log
from math import pi, sqrt, exp, cos,sin
from cmath import exp

NN = 2000
dt = 0.01
TT = np.arange(0,dt*NN,dt)
time1 = np.zeros(NN)
time2 = np.zeros(NN)

""" Matrix definitions """
A = np.matrix('0 1; -2 -3')
B = np.matrix('0; 1')
x = np.matrix('0;0')
f = np.ones(NN)

x[0] = 1
x[1] = 2

""" Begin simulation """

nsteps = NN

for i in range(nsteps):
    time1[i] = x[0]
    time2[i] = x[1]
    
    x = x + dt*A*x + dt*B*f[i]

plt.subplot(2,1,1)
plt.plot(TT,time1,'k',label='x1')
plt.plot(TT,time2,'k-.',label='x2')
plt.legend()
plt.grid()
plt.axis([0,4,-1,2])
plt.savefig('Example-6-2-1.png',dpi=300)

