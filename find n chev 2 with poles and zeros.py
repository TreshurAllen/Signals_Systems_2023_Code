# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:59:25 2023
find n Chev inverse => chev 2
@author: tresh
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
from math import log10

""" Passband """
Hp = .9
wp = 400

""" Stopband """
Hs = .1
ws = 200

""" Epsolan calc, bc elipce"""
eps = np.sqrt( (Hs**2/(1 - Hs**2)) )
print('eps = ', eps)

"""N calc
aaa = np.sqrt( (1/eps**2)*(1/(1-Hp**2)) )
bbb = np.arccosh(aaa)
ccc = (1./np.arccosh(1/wp))

pre_n = bbb*ccc

"""
n = 3
print('n = ', n)

""" Calculations """ 

alpha = (1/eps) + (np.sqrt(1+(1/eps**2)))
print('alpha = ', alpha)
a = 0.5*(alpha**(1/n) - alpha**(-1/n))
b = 0.5*(alpha**(1/n) + alpha**(-1/n))
print('a = ', a, 'b = ', b)


""" Finding Poles -- Q """

theta = 60*pi/180
s1 = a*np.cos(theta) + 1j*b*np.sin(theta)
Q1 = 1/s1
Q1c = np.conjugate(Q1)
print('q = ', Q1c)
d1 = [1, np.real(Q1+Q1c), np.real(Q1*Q1c) ]
d2 = [1,1/a]

den = np.convolve(d1,d2)


""" Finding Zeros -- P """

omega1 = 1/np.cos(pi/6) # pi / 3 = 60deg
#omega2 = 1/np.cos(3*pi/6) # 2pi / 3 = 120deg

n1 = [1, 0, omega1**2]
#n2 = [1, 0, omega2**2]
"""  Constant K  """

#K = d1[2]*d2[2]/(n1[2]*n2[2])
K = den[2]/(n1[2])
num = K*np.convolve(n1, 1)
"""
num = [1, 0, 40000]

d1 = [1.17, 200, 0]
d2 = [.394,110.4,40000]
den = np.convolve(d1,d2)
"""
K = 1
w = np.linspace(.01, 10, 1000)
system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system, n=1000)

""" Mag """

plt.subplot(211)
plt.plot(w,10**(0.05*Hmag), 'k')
plt.title('Exam 3')
plt.axis([.1,2,0,1.4])
plt.yticks([0,.126,.5,1,1.2])
plt.xticks([.1,.5,.71,1,1.25,2])
plt.grid(which='both')
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('|H|')
plt.show()

""" PHASE 
plt.subplot(212) 
plt.plot(w, Hphase, 'k') #rather than semi use plot
plt.axis([0,1.7,-270,0])
plt.grid(which='both')
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('Phase')
plt.yticks([-180,-90,0])
plt.show()
"""
""" sinusoidal """

dt = 0.003
NN = 5000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)
A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

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

    
    
    
    
    
    
    
    
"""  the final convolution on prob 8.4.1 from pics on 10/25 
        the highest order from that convolution is how you get K """
