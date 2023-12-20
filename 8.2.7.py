# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:34:39 2023
problem 8.2.6
BUTTERWORTH LPF
@author: tresh
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt, log10
from control import margin
from control import tf


""" Two poles """
W = 1  #cut off freq
#num = [0, 0, 1]
#den = [1, 1.41, 1]
num = [0,W**2]
den = [1,1.41*W,W**2]

system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system)

plt.subplot(311)
plt.semilogx(w, 10**(0.05*Hmag), 'k') #plot amplitude, not dB
plt.title('ex 8.2.6')
plt.axis([0.1, 10, 0, 1])
plt.yticks([0, 0.1, 0.5, 0.707, 1])
plt.grid(which='both')
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('|H|')
plt.xticks([.1,.8,1.2,10])

""" Four Pole """
wc = 10     #center freq
num = [0,0, W**2,0,0]
den = [1,1.41*W, (2*wc**2+W**2),1.41*W*wc**2, wc**4]

w = np.linspace(100,1000,1000)
system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system, n=1000)

plt.subplot(312)
plt.plot(w,10**(0.05*Hmag), 'k')
plt.axis([.2,2,0,1.2])
plt.yticks([0,.7,1])
plt.xticks([wc-0.5*W,wc+0/5*W])
plt.grid(which= 'both')
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('|H|')
plt.show

dt = 0.002
NN = 20000 #for ex 8.2.7 change to 50000
#to make an accurate guess, you look at your frequency you need 
#at least 10 points per period of frequency so if you have 10khz
# you need dt to give you a division of 10
#the nn is just making sure you have enough points to see on the 
# x axis
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)

A,B,C,D = sig.tf2ss(num,den)
x = np.zeros(np.shape(B))

""" The Sinusodal input """
omega = .75
for n in range(NN):
    f[n] = sin(omega*n*dt)

for m in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[m]
    y[m] = C.dot(x) + D*f[m]
    
    
plt.subplot(313)
plt.plot(TT, f, 'k', label='F_in')
plt.plot(TT, y, 'k--', label='Y_out')
plt.axis([0, NN*dt,-1,1])
plt.yticks([-1,-.35,-.71,0,.35, .71, 1])
plt.text(.1,.5,'omega = {}'.format(round(omega,1)),fontsize=12)
plt.grid()
plt.xlabel('T (sec)')
plt.show()

""" pass band we are going to use the 3dB point therefore its 
pass band = 200
stop band = deltw = 600
so 
W = 200      //scale down to 1 so// = 1
ws = 600    //scale acordingly // =   3
then using the find n code we get that n = 2
"""
