# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:34:39 2023
CHAPTER 7 ANALOG FILTERS 
BUTTERWORTH LPF
@author: tresh
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
from math import log10

  #PASSBAND  
Hp = 0.8
wp = 100

  #STOPBAND  
Hs = 0.2
ws = 70

 #Find n.py 
Hp2 = (Hp**2)
Np = (log10( 1/Hp2 - 1 )) / (2*log10(wp) )
print('np = ', Np)  

Hs2 = (Hs**2)
Ns = (log10(1/Hs2 - 1)) / (2*log10(ws)) 
print('ns = ', Ns)

n = 5
phi = 180/n
a = 1 #gain

n1 = [0,1]
d1 = [1,Ns]

n2 = [0,1]
d2 = [1, wp]

#n3 = [0,0,1]
#d3 = [1, 1.93, 1]

#n12 = a*np.convolve(n1,n2)
#d12 = np.convolve(d1,d2)

#n23 = a*np.convolve(n12,n3)
#d23 = np.convolve(d12,d3)

""" Two poles """
num = n1
den = d1

system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system)

plt.subplot(211)
plt.semilogx(w, 10**(0.05*Hmag), 'k') #plot amplitude, not dB
plt.title('HW8')
plt.axis([0.1, 10, 0, 1.7])
plt.yticks([0, 0.1, 0.5, 0.707, 1])
plt.grid(which='both')
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('|H|')
plt.xticks([.1,.8,1.2,10])
plt.show()

dt = 0.001
NN = 50000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)
A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

""" The Sinusodal input """
omega = 1
for n in range(NN):
    f[n] = sin(omega*n*dt)
    
plt.subplot(211)
plt.plot(TT, f, 'k')
plt.yticks([-1,0,1])
plt.axis([0, NN*dt,-1,1])
plt.grid()
plt.ylabel('f')

for m in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[m]
    y[m] = C.dot(x) + D*f[m]
    
plt.subplot(212)
plt.plot(TT, y, 'k')
plt.axis([0, NN*dt,-1,1])
plt.yticks([-1,-.707,0,.707,1])
plt.text(10,.5,'omega = {}'.format(round(omega,1)),fontsize=12)
plt.grid()
plt.xlabel('T (sec)')
plt.ylabel('y')
plt.show()


