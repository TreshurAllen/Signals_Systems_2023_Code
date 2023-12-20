# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:59:25 2023
find n.py
@author: tresh
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
from math import log10

wc = 190

wp = (1080-920)/wc

Hp2 = (.8**2)
Np = (log10( 1/Hp2 - 1 )) / (2*log10(wp) )
print('np = ', Np)  

ws = (1300-700) /wc
Hs2 = (.15**2)
Ns = (log10(1/Hs2 - 1)) / (2*log10(ws)) 
print('ns = ', Ns)
