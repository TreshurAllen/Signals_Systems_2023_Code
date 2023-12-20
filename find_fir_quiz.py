""" Find_FIR.py """

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt

NN = 100
N2 = int(NN/2)
x = np.zeros(NN)
y = np.zeros(NN)

dt = .05
TT = np.linspace(0,dt*(NN-1),NN)
DF = 1/(dt*NN)
FF = np.linspace(0,DF*(NN-1),NN)

f1 = 1
f2 = 2
f3 = 3
f4 = 4
freq1 = 2*pi*f1
freq2 = 2*pi*f2
freq3 = 2*pi*f3
freq4 = 2*pi*f4

#These are the spikes in the top right graph

x = 2*np.sin(freq1*TT) + 2*np.sin(freq2*TT) + 2*np.sin(freq3*TT) + 2*np.sin(freq4*TT) 
           
plt.subplot(321)
plt.plot(TT,x,'k')
plt.axis([0,NN*dt,-2.5,2.5])
#plt.text(5,-15,'$\phi$ = {}'.format(round(14,3)),fontsize=12)
plt.title('Find_FIR')
plt.ylabel('a). x[k]')
plt.xlabel('T (sec)')
plt.grid()

X = (1/NN)*np.fft.fft(x)

""" Create the filter   """
H = np.zeros(NN)

""" Rectangular Low pass 
for n in range(8):
    H[n] = 1
"""
""" Low pass """
#for n in range(4):
#    H[n] = 1
#for n in range(4,16):
#    H[n] =  exp(-.5*((n-4)/4)**2)

""" Band pass     """
for n in range(10):    
    H[n] =  exp(-2*((n-10)/3)**2)
for n in range(10,15):    
    H[n] =  1
for n in range(15,25):    
    H[n] =  exp(-2*((n-15)/3)**2) 
#for n in range(N2):    
#    H[n] =  exp(-.5*((n-16)/3)**2)

""" High pass """    
#for n in range(15):
#    H[n] =  exp(-.5*((n-15)/4)**2)    
#for n in range(15,N2+2):
#    H[n] = 1
    
""" Reflect the positive frequencies to the right side """
for n in range(1,N2-1):
    H[NN-n] = H[n]  
    
Y = H*X   

plt.subplot(322)
plt.plot(FF,abs(X),'k',label='X')
plt.plot(FF,H,'k--',label='H')
plt.legend(loc='upper right')
plt.ylabel('b). H(w),X(w)')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.axis([0,5,0,1.1])
#plt.xticks([20,80,150])

h = np.fft.ifft(H)

plt.subplot(323)
plt.plot(h.real,'k')
plt.xlabel('k')
plt.ylabel('c). h[k]')
plt.grid()
plt.axis([0,NN,-.1,.2])

M = 9
hh = np.zeros(NN)

""" Move the filter to the left side """
for n in range(M):
    hh[n+M] = h[n].real
    hh[M-n] = hh[M+n]

plt.subplot(324)
plt.plot(hh,'ok')
plt.axis([0 ,2*M,-.3,.5])
plt.xlabel('k')
plt.ylabel('d). hh[k]')
plt.grid()

""" Convolve hh and x """

yy=np.convolve(hh,x)

y = np.zeros(NN)
for n in range(NN):
    y[n] = yy[n+M]

plt.subplot(325)
plt.plot(TT,y,'k')
plt.ylabel('e). y[k]')
plt.xlabel('T (sec)')
plt.grid()

Y = (1/NN)*np.fft.fft(y)

plt.subplot(326)
plt.plot(FF,abs(Y),'k')
plt.axis([0,5,-.1,1.2])
#plt.xticks([0,20,80,150])
plt.yticks([.15,.5, .7, 1])
plt.ylabel('f). Y[w]')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.savefig('f.png',dpi=300)
plt.show()

#print(hh)

""" for example 1 old exam - given specificatins 
step 1 scale problem will be 1 & 3
if corner is at -3dB ==> butterworth 
.1 micro sec becomes dt = .05

given the hs
residue function does partial fractions
take the filter and write a time domaine and convolve
when you have something like below 100 and above 400 stopband those are when they shut off


no bilinear or backward rectangular 


"""