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

""" Passband """
wp = 400
ws = 200

Hp = .9
eps = np.sqrt( (1/Hp**2) -1 )
print('eps = ', eps)

""" Stopband """
Hs = .1

aaa = np.sqrt( (1/Hs**2) - 1 )
bbb = np.arccosh(aaa/eps)
ccc = (1./np.arccosh(ws))

n = bbb*ccc

print('n = ', n)