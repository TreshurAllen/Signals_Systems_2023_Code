""" Find_FIR.py """

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt

NN = 100
N2 = int(NN/2)
x = np.zeros(NN)
y = np.zeros(NN)

dt = .002
TT = np.linspace(0,dt*(NN-1),NN)
DF = 1/(dt*NN)
FF = np.linspace(0,DF*(NN-1),NN)

f1 = 100
"""f2 = 100
f3 = 180
f4 = 300"""
freq1 = 2*pi*f1
"""freq2 = 2*pi*f2
freq3 = 2*pi*f3 
freq4 = 2*pi*f4"""

#These are the spikes in the top right graph

x = .5*np.sin(freq1*TT) 

for n in range(NN):
    x[n] = np.sin(2*pi*100*n*dt) + 0.5*np.random.normal(0,1)
           
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

""" Band pass """    

for n in range(19):    
    H[n] =  exp(-10*((n-19)/3)**2)
for n in range(19,21):    
    H[n] =  1    
for n in range(21,30):    
    H[n] =  exp(-10*((n-21)/3)**2)

#for n in range(60,90):    
#    H[n] =  exp(-.5*((n-60)/3)**2)

"""
for n in range(3):    
    H[n] =  exp(-.5*((n-3)/3)**2)
for n in range(3,8):    
    H[n] =  1
for n in range(8,50):    
    H[n] =  exp(-.5*((n-36)/3)**2)
"""

""" High pass     
for n in range(60):
    H[n] =  exp(-.5*((n-60)/4)**2)    
for n in range(60,N2+2):
    H[n] = 1
"""
""" Reflect the positive frequencies to the right side """
for n in range(1,N2-1):
    H[NN-n] = H[n]  
    
Y = H*X   

plt.subplot(322)
plt.plot(FF,abs(X),'k',label='X')
plt.plot(FF,H,'k--',label='H')
#plt.legend(loc='upper right')
plt.ylabel('b). H(w),X(w)')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.axis([0,200,0,1.1])
plt.xticks([20,100,180])

h = np.fft.ifft(H)

plt.subplot(323)
plt.plot(h.real,'k')
plt.xlabel('k')
plt.ylabel('c). h[k]')
plt.grid()
plt.axis([0,NN,-.1,.2])

M = 7
hh = np.zeros(NN)

""" Move the filter to the left side """
for n in range(M):
    hh[n+M] = h[n].real
    hh[M-n] = hh[M+n]

plt.subplot(324)
plt.plot(hh,'ok')
plt.axis([0 ,2*M,-.3,1])
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
plt.axis([0,200,-.1,.5])
plt.xticks([20,100,180])
plt.yticks([.15,.3,.5,])
plt.ylabel('f). Y[w]')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.savefig('f.png',dpi=300)
plt.show()



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