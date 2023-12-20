# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:32:34 2023
EXAM 1 
@author: Treshur Allen
"""

import matplotlib.pyplot as plt
import numpy as np
from math import exp, cos, sin, log
from math import pi, sqrt, exp, cos,sin
from cmath import exp
#from scipy import linalg 

'''DEFINING STEP FUNCTION'''

def u(t):
    u = np.zeros(t.shape) #init
    for i in range(len(t)):
        if t[i] <= 0:
            u[i] = 0
        else:
            u[i] = 1
    return u
        

NN = 10000
dt = 0.01
TT = np.arange(0,dt*NN,dt)
time1 = np.zeros(NN)
time2 = np.zeros(NN)
time3 = np.zeros(NN)
time4 = np.zeros(NN)

""" Matrix definitions """
A = np.matrix('-20 0 0 0; 0 0 1 0; 400 -400 -10 0; 0 -20 0 -5')
B1 = np.matrix('0; 0; -20; 0')
B2 = np.matrix('0; 0; 0; 1')
#B = np.matrix('0 0; 0 0; -20 0; 0 1')
C = np.matrix('0 0 0 1')
D = np.matrix('0')
x = np.matrix('0; 0; 0; 0')
w = np.matrix('0; 0; 0; 0')

"""Input Functions"""

f = 5*(u(TT) - u(TT-1))
#f = np.ones(NN)
g = 5*(u(TT-1)-u(TT-2))
#g = np.ones(NN)


""" Simulation finite difference """

nsteps = NN

for i in range(nsteps):
    time1[i] = x[0]
    time2[i] = x[1]
    time3[i] = x[2]
    time4[i] = x[3]
    x = x + dt*A*x + dt*B1*f[i] + dt*B2*g[i]
    y = dt*C*x
    
"""Simulation State Transition"""

dt = 0.01
TT = np.arange(0,dt*NN,dt)
wtime1 = np.zeros(NN)
wtime2 = np.zeros(NN)
wtime3 = np.zeros(NN)
wtime4 = np.zeros(NN)
nsteps = NN


I4 = np.eye(4)
A2 = A*A
A3 = A*A2
A4 = A2*A2
F = ( I4 + A*dt + 0.5*A2*(dt**2) + (1/6)*A3*(dt**3) + (1/24)*A4*(dt**4) )
Ainv = np.linalg.inv(A)
G1 = (F-I4)*Ainv*B1
G2 = (F-I4)*Ainv*B2


for i in range(nsteps):
    wtime1[i] = w[0]
    wtime2[i] = w[1]
    wtime3[i] = w[2]
    wtime4[i] = w[3]
    
    w = F*w + G1*f[i] + G2*f[i]
#my plots 

plt.figure(figsize = (10,10))

textstr = 'dt = 0.01'
#plot inputs
plt.subplot(5,1,1) 
plt.plot(TT,f, color = 'red', label='f(t)')
plt.plot(TT,g, color = 'blue', label='g(t)')
plt.ylabel('Input Funtions')
plt.legend()
plt.grid()
plt.axis([0,2.5,0,5.1])
plt.title('ECE 450 EXAM 1: Set 1 dt = 0.01', fontsize = 20)

#finite sim
plt.subplot(5,1,2) 
plt.plot(TT,time1,color = 'blue',label='x1')
plt.plot(TT,time2,color = 'red',label='x2')
plt.plot(TT,time3,color = 'green',label='x3')
#plt.plot(TT,time4,color = 'purple',label='x4 = y(t) = r(t)')
plt.text(1.5,2, textstr, fontsize=14)
plt.ylabel('Finite Simulation')
plt.legend()
plt.grid()
plt.axis([0,2.5,-4.2,4.2])

#State Transition
plt.subplot(5,1,3) 
plt.plot(TT,wtime1,color = 'blue',label='x1')
plt.plot(TT,wtime2,color = 'red',label='x2')
plt.plot(TT,wtime3,color = 'green',label='x3')
#plt.plot(TT,wtime4,color = 'purple',label='x4 = y(t) = r(t)')
plt.text(1.5, 2, textstr, fontsize=14)
plt.ylabel('State Transition')
plt.legend()
plt.grid()
plt.axis([0,2.5,-4.2,4.2])

   

#plt.figure(figsize = (10,10))
#Output Finite
plt.subplot(5,1,4) 
plt.plot(TT,time4,color = 'purple',label='x4 = y(t) = r(t)')
plt.text(1.5,2, textstr, fontsize=14)
plt.ylabel('Finite Output')
plt.legend()
plt.grid()
plt.axis([0,2.5,-4.2,4.2])  
    
    
#Output Transition
plt.subplot(5,1,5) 
plt.plot(TT,wtime4,color = 'purple',label='x4 = y(t) = r(t)')
plt.text(1.5,2, textstr, fontsize=14)
plt.ylabel('Transition Output')
plt.legend()
plt.grid()
plt.axis([0,2.5,-4.2,4.2])   
plt.show()      
    
    
    
    
    
    
    