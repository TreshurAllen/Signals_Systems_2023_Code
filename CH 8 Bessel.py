# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:34:39 2023
 
BUTTERWORTH LPF
@author: tresh
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt, log10
from control import margin
from control import tf

""" Bessel """
a = 1


n = 2
d2 = [1,3*a,3*a**2]
K = d2[2]
n2 = [0,0,K]
#d2 = [1,3,3] 

n = 3
d3 = [1,6,15,15]
K = d3[3] 
n3 = [0,0,0,K]
#d3 = [1,6,15,15]

""" 4th order 
n = 4
d4 = [1,10*a,45*a**2,105*a**3,105*a**4]
K4 = d4[4]
n4 = [0,0,0,0,K4]
"""

num = n3
den = d3



""" calculations """

Hp = .9
Hs = .1

P = 20*log10(Hp)
S = 20*log10(Hs)

#we want wp = 1 so ws becomes ws/wp
# -6 dB = 50% attenuation 

wp = 2
ws = 1

system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system)


""" MAG  """
plt.subplot(211) 
plt.semilogx(w, Hmag, 'k')
plt.title('Bessel_Filter')
plt.axis([0, 100, -40, 0])
plt.grid(which='both')
#plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('Mag')
#plt.yticks([0,-6,-20])

""" PHASE """
plt.subplot(212) 
plt.plot(w, Hphase, 'k') #rather than semi use plot
plt.axis([0,40,-270,0])
plt.grid(which='both')
#plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('Phase')
plt.yticks([-180,-90,0])
plt.show()

dt = 0.003
NN = 5000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)
A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

""" The Sinusodal input """
omega = 30 # freq
for n in range(NN):
    #f[n] = sin(omega*(n-500)*dt)
    aaa = ((n-200)/4)**2 #gousian 
    f[n] = cos(omega*(n-200)*dt)*exp(-aaa*dt)
        
plt.subplot(211)
plt.plot(TT, f, 'k')
plt.yticks([-1,0,1])
plt.axis([0, 2,-1,1])
plt.grid()
plt.ylabel('f')

for m in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[m]
    y[m] = C.dot(x) + D*f[m]
    
plt.subplot(212)
plt.plot(TT, y, 'k')
plt.axis([0,2,-1,1])
plt.yticks([-1,-.707,0,.707,1])
#plt.text(10,.5,'omega = {}'.format(round(omega,1)),fontsize=12)
plt.grid()
plt.xlabel('T (sec)')
plt.ylabel('y')
plt.show()


