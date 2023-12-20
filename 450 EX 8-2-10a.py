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

wc =  130

wp = (500-400)/wc

Hp2 = (.9**2)
Np = (log10( 1/Hp2 - 1 )) / (2*log10(wp) )
print('np = ', Np)  

ws = (600 - 300) /wc
Hs2 = (.1**2)
Ns = (log10(1/Hs2 - 1)) / (2*log10(ws)) 
print('ns = ', Ns)


""" METHODOLOGY:
    
    when looking at the range of wp = 500-400 = 100
    and                         ws = 600 - 300 = 300
    so our bounds should be some where between 100 and 300 
    so I first guessed 200 then work twords the same number
"""