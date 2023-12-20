# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:03:33 2023

@author: treshur in class monday 
"""

""" Sim_springs.py """
import matplotlib.pyplot as plt
import numpy as np
from math import exp, cos, sin, log
from math import pi, sqrt, exp, cos,sin
from cmath import exp
NN = 10000
dt = 0.01
TT = np.arange(0,dt*NN,dt)
time0 = np.zeros(NN)
time1 = np.zeros(NN)
time2 = np.zeros(NN)
time3 = np.zeros(NN)
""" Same as problem 6.1.4 """
k1 = 0.01
k2 = 0.02
B1 = 3
B2 = 1
m1 = 1
m2 = 5
""" Matrix definitions """
A = np.matrix(' 0 1.000 0 0 ; 0 0 0 0 ; 0 0 0 1; 0 0 0 0')
A[1,0] = -k1/m1
A[1,1] = -B1/m1
A[1,2] = k1/m1
A[1,3] = B1/m1
A[3,0] = k1/m2
A[3,1] = B1/m2
A[3,2] = -(k1+k2)/m2
A[3,3] = -(B1+B2)/m2
B1 = np.matrix('0.000; 0; 0; 0')
B3 = np.matrix('0.000; 0; 0; 0')
B1[1] = 1/m1
B3[3] = 1/m2

x = np.matrix('0;0;0;0')
x[0] = 1
x[2] = -2

""" Forcing functions """
F1 = np.zeros(NN)
F2 = np.zeros(NN)

for n in range(1000,NN):
    F1[n] = .1
for n in range(3000,NN):
    F2[n] = -.1
    
plt.subplot(211)
plt.title('Sim_springs')
plt.plot(TT,F1,'k',label='F1')
plt.plot(TT,F2,'k--',label='F2')
plt.legend()
plt.axis([0,50,-.15,.15])
plt.ylabel('Forces')
plt.grid()
plt.savefig('Input.png',dpi=300)
plt.show()

""" Begin simulation """
nsteps = NN
for i in range(nsteps):
    x = x + dt*A*x + dt*B1*F1[i] + dt*B3*F2[i]
time0[i] = x[0]
time1[i] = x[1]
time2[i] = x[2]
time3[i] = x[3]
plt.subplot(211)
plt.plot(TT,time0,'k',label='x0')
plt.plot(TT,time2,'k-.',label='x2')
plt.ylabel('Positions')
plt.legend()
plt.grid()
plt.axis([0,50,-5,5])
plt.subplot(212)
plt.plot(TT,time1,'k',label='x1')
plt.plot(TT,time3,'k-.',label='x3')
plt.ylabel('Speeds')
plt.xlabel('sec')
plt.legend(loc='upper left')
plt.axis([0,50,-.2,.2])
plt.grid()
plt.savefig('Sim_springs.png',dpi=300)
plt.show()
