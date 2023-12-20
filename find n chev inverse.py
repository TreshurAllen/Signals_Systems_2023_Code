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
wp = 2

""" Stopband """
Hs = .1
ws = 1

""" Epsolan calc, bc elipce"""
eps = np.sqrt( ((Hs**2)/(1 - Hs**2)) )
print('eps = ', eps)

"""N calc"""
aaa = np.sqrt( (1/eps**2)*(1/(1-Hp**2)) )
bbb = np.arccosh(aaa)
ccc = (1/(np.arccosh(1/wp)))

n = bbb*ccc

print('n = ', n)