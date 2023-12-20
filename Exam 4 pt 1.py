import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
import cmath

NN = 100000
phi = np.linspace(0,2*pi,NN)
dt = .01
z = np.zeros(NN, dtype = np.complex)
H = np.zeros(NN, dtype = np.complex)

for n in range(0,NN):
    z = cmath.exp(1j*phi[n])
    a = 0.5
    w = np.sqrt(27.5)
    H[n] = 0.01*(z**2 -0.989052*z)/(z**2-1.98756*z+0.99005)
   

phi_w0 = round(5*dt*(180/pi),4)
omega = phi_w0/(dt*(180/pi))

plt.figure ( figsize = (9 , 6) )
plt.subplot(211)
plt.semilogx((180/pi)*phi,20*np.log10(H),'k')
plt.axis([1,10, -30, 5])
plt.ylabel('|H| dB')
plt.axvline(phi_w0,color='k')
# plt.text(1,0,'$\omega$ = {}'.format(round(omega,2)),fontsize=12)
plt.title('Bode Plot of H(z)')
plt.grid(which='both')
aaa = np.angle(H)


plt.subplot(212)
plt.semilogx((180/pi)*phi,(180/pi)*aaa,'k')
plt.ylabel('/H (degrees)')
plt.axis([.1,100, -180,90])
plt.yticks([-135,-90,-45,0,45])
plt.axvline(phi_w0,color='k')
plt.text(1,-70,'$\phi$ = {}'.format(round(phi_w0,2)),fontsize=12)
plt.grid(which='both')
plt.xlabel('$\phi$ (degrees)')
plt.show()


#----- Time Simulation -------------------------------------------------------#

# SimTime = 400
# dt = 0.00001
# TT = np.arange(0,NN*dt,dt)

# y = np.zeros(NN)
# x = np.zeros(NN)


# for k in range(NN):
#     x[k] = np.sin(200*dt*k) + np.sin(5000*dt*k) + np.sin(10000*dt*k)
#     # y[k] = 1.921*y[k-1] -0.922*y[k-2] + 0.001*x[k-1]
#     y[k] = 1.98756*y[k-1] - 0.99*y[k-2] + 0.01*x[k] - 0.00989*x[k-1]
    
    
# plt.figure ( figsize = (9 , 6) )
# plt.subplot(211)
# plt.plot(TT,x, 'g', label ='x(t)')
# plt.title('Time Domain Response')
# plt.ylabel('x(t)')
# plt.axis([0,0.1,-2,2])
# plt.grid()


# plt.subplot(212)
# plt.plot(TT, y, 'b', label = 'y(t)')
# # plt.text(8,1,'$\omega_1$ = 0.5', fontsize = 12)
# # plt.text(20,1,'$\omega_2$ = 5', fontsize = 12)
# plt.ylabel('y(t)')
# plt.xlabel('Time [s]')
# plt.axis([0,0.1,-2,2])
# plt.grid()
