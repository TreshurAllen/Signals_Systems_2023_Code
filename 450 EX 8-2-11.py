# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:59:25 2023
find n.py 
BANDPASS EX 8-2-11
@author: tresh
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
from math import log10

wc = 305

wp = (1100-910)/wc

Hp2 = (.85**2)
Np = (log10( 1/Hp2 - 1 )) / (2*log10(wp) )
print('np = ', Np)  

ws = (1480-675) /wc
Hs2 = (.35**2)
Ns = (log10(1/Hs2 - 1)) / (2*log10(ws)) 
print('ns = ', Ns)
