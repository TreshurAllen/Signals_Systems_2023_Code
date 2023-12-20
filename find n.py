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
Hp = .85
wph = 5000

""" Stopband """
Hs = .1
wsh = 200

#guess the shift freq in page 111
#we are looking for the wc that will make np&ns equal enough
wc = 339

ws = 1/(wsh/wc)
wp = 1/(wph/wc)

Hp2 = (Hp**2)
Np = (log10( 1/Hp2 - 1 )) / (2*log10(wp) )
print('np = ', Np)  

Hs2 = (Hs**2)
Ns = (log10(1/Hs2 - 1)) / (2*log10(ws)) 
print('ns = ', Ns)
