# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:12:28 2023

EXAMPLE 7.7.2
in class
@author: tresh
"""

""" Bode_comp.py.  """

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt, log10
from control import margin
from control import tf

K = 1  #gain
n1 = [0, 0, 400]
d1 = [1, 10, 400]


#Glag
gn2 = [1, 2]
gd2 = [1, 0.04]
K = 1

"""
# First Gc
phim = (pi/180)*10
alpha = (1 + sin(phim)) / (1 - sin(phim))
dbshift = -10*log10(alpha)
wm = 8
wz = wm/sqrt(alpha)
wp = wm*sqrt(alpha)

n2 = [1, wz]
d2 = [1, wp]
K = (wp/wz)
"""
n12 = K*np.convolve(n1,gn2)
d12 = np.convolve(d1,gd2)

"""
# Second Gc
n3 = [1,4]
d3 = [1,40]
n123 = np.convolve(n12,n3)
d123 = np.convolve(d12,d3)
"""

#----UNCOMPENSATED-----#
#num = n1
#den = d1


#----COMPENSATED----#
num = n12
den = d12

w = np.linspace(0.1,100,num=1000)
system = sig.lti(num,den)
w, Hmag, Hphase = sig.bode(system,w)
gm, pm, wg, wp = margin(Hmag,Hphase,w)
# wp  freq for phase margin at gain crossover (gain = 1)
# pm phase maring

plt.subplot(211)
plt.semilogx(w,Hmag,'k')
plt.semilogx(w,Hmag,'k')
plt.axis([ .1, 1e2, -60, 20])
#plt.xticks([1,10,30,100,1000])
plt.ylabel('|H| dB',size = 12)
plt.text(.3,-40,'$\omega$p = {}'.format(round(wp,1)),fontsize=12)
plt.title('Bode Comp')
plt.grid(which='both')

for n in range(100):
    if Hphase[n] > 0:
        Hphase[n] = Hphase[n] - 360
        
""" Bode comp plot """
plt.subplot(212)
plt.semilogx(w,Hphase,'k')
#plt.axis([ .1, 1e2, -180,0])
plt.yticks([-180,-90,0])


plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('Phase (degrees)',size=12)
plt.text(.3,-150,'pm = {}'.format(round(pm,0)),fontsize=12)
plt.grid(which='both')
#plt.savefig('H_bode.png',dpi=300)
plt.show()

""" Time portion """

dt = 0.05
NN = 500
TT = np.arange(0,NN*dt,dt)
step = np.zeros(NN)
ramp = np.zeros(NN)
parabola = np.zeros(NN)
errS = np.zeros(NN)
errR = np.zeros(NN)
errP = np.zeros(NN)

for i in range(NN):
    step[i] = 1.0
    ramp[i] = (dt*i)
    parabola[i] = (dt*i)**(2)
    
denCL = np.add(num,den)

t1, y1, x1 = sig.lsim((num,denCL),step,TT)
t2, y2, x2 = sig.lsim((num,denCL),ramp,TT)
t3, y3, x3 = sig.lsim((num,denCL),parabola,TT)

for i in range(NN):
    errS[i] = step[i] - y1[i]
    errR[i] = ramp[i] - y2[i]
    errP[i] = parabola[i]  - y3[i]    

""" plots"""

plt.subplot(321)
plt.plot(TT,y1,'k--',label='y1(t)')
plt.plot(TT,step,'k',label='u(t)')
#plt.axis([0,2,0,1.5])
plt.ylabel('step')
plt.xlabel('t (sec)')
#plt.yticks([0,.9,1.1,1.5])
plt.legend()
plt.grid()

plt.subplot(322)
plt.plot(TT,errS,'k',label='error')
plt.legend()
#plt.axis([0,2,-.5,1])
#plt.yticks([0,0.02,.05,.1])
plt.grid()
plt.savefig('position.png')
plt.show()

plt.subplot(321)
plt.plot(TT,y2,'k--',label='y2(t)')
plt.plot(TT,ramp,'k',label='r(t)')
plt.xlabel('t (sec)')
plt.ylabel('ramp')
plt.legend()
plt.grid()

plt.subplot(322)
plt.plot(TT,errR,'k',label='error')
plt.legend(loc=4)
plt.xlabel('t (sec)')
#plt.axis([0,2,0,.2])
plt.yticks([0,.05,.2])
plt.grid()
plt.savefig('velocity.png')
plt.show()

plt.subplot(321)
plt.plot(TT,y3,'k--',label='y3(t)')
plt.plot(TT,parabola,'k',label='parab(t)')
plt.xlabel('t (sec)')
plt.ylabel('parabola')
plt.legend()
plt.grid()

plt.subplot(322)
plt.plot(TT,errP,'k',label='error')
plt.xlabel('t (sec)')
plt.legend()
plt.grid()
plt.savefig('time_plots.png',dpi=300)
plt.show()
