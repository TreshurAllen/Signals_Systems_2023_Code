# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:32:29 2023

@author: tresh
"""

# My_bode.py.  
# This is the Python version

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
#from math import pi, sqrt, exp, cos,sin
#from cmath import exp


n1 = [ 0, 0, 200]     # Example 9.1.1
d1 = [1, 400, 40000]

#n1 = [ 0,0,1]        # 2-pole Butterworth
#d1 = [1,1.41,1]
#n1 = [0,1,0]
#d1 = [1,1,1]

#n12 = np.convolve(n1,n2)
#d12 = np.convolve(d1,d2)

num = n1
den = d1

w = np.linspace(.1,100,num=100)
system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system,100)
#gm, pm, wg, wp = margin(Hmag,Hphase,w)
# wp  freq for phase margin at gain crossover (gain = 1)
# pm phase maring

plt.subplot(211)
plt.semilogx(w,Hmag,'k')
#plt.axis([450, 600, -20, 10])
#plt.yticks([-20,-3,0])
#.xticks([wc-25,wc+25])
plt.ylabel('|H| dB')
plt.title('My_bode')
plt.grid(which='both')

for n in range(100):
    if Hphase[n] > 180:
        Hphase[n] = Hphase[n] - 360
#    if Hphase[n] < 0:
#        Hphase[n] = Hphase[n] + 360

plt.subplot(212)
plt.semilogx(w,Hphase,'k')
#plt.axis([ 450, 600, -90,90])
#plt.xticks([1,3,10,30,100,1000])
#.yticks([-90,0,90])
plt.xlabel('$\omega$ (rad/s)',fontsize=12)
plt.ylabel('/H (degrees)')
#plt.text(3,-120,'pm = {}'.format(round(pm,0)),fontsize=12)
plt.grid(which='both')
plt.savefig('H_bode.png',dpi=300)
plt.show()

