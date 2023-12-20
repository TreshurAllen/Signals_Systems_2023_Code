# -*- coding: utf-8 -*-
"""
Created on Wed 9 6 18:20:28 2023
    in class wednesday  



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
wtime1 = np.zeros(NN)
wtime2 = np.zeros(NN)

""" Matrix definitions """
A = np.matrix('0 1; -2 -3')
B = np.matrix('0; 1')
x = np.matrix('0;0')
w = np.matrix('0;0')
f = np.ones(NN)

x[0] = 0
x[1] = 1

nsteps = NN

for i in range(nsteps):
    time1[i] = x[0]
    time2[i] = x[1]
    
    x = x + dt*A*x + dt*B*f[i]



"""  def for new state transition sim  """
I2 = np.eye(2)
A2 = A*A
A3 = A*A2
A4 = A2*A2
A5 = A*A4
A6 = A4*A2
F = ( I2 + A*dt + 0.5*A2*(dt**2) + (1/6)*A3*(dt**3) + (1/24)*A4*(dt**4)
     + (1/120)*A5*(dt**5) + (1/720)*A6*(dt**6) )
Ainv = np.linalg.inv(A)
G = (F-I2)*Ainv*B

""" State transition sim    """

nsteps = NN

for i in range(nsteps):
    wtime1[i] = w[0]
    wtime2[i] = w[1]
    
    w = F*w + G*F[i]
    
    
plt.subplot(2,1,1)
plt.title('example old')
plt.plot(TT,time1,'k',label='x1')
plt.plot(TT,time2,'k-.',label='x2')
plt.legend()
plt.grid()
#plt.axis([0,8,-0.5,0.5])
#plt.savefig('Example-6-2-1.png',dpi=300)
    
plt.subplot(2,1,2)
plt.title('state t example')
plt.plot(TT,wtime1,'k',label='w1')
plt.plot(TT,wtime2,'k-.',label='w2')
plt.legend()
plt.grid()
#plt.axis([0,8,-0.5,0.5])
#plt.savefig('Example-6-2-1.png',dpi=300)
plt.show()













