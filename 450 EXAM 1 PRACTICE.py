# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:36:26 2023
EXAM 1 PRACTICE AND REFERENCE
@author: tresh
"""

import matplotlib.pyplot as plt
import numpy as np
from math import exp, cos, sin, log
from math import pi, sqrt, exp, cos,sin
from cmath import exp

NN = 2000
dt = 0.01
TT = np.arange(0,dt*NN,dt)
time1 = np.zeros(NN)
time2 = np.zeros(NN)

""" Matrix definitions """
A = np.matrix('0 1 0 0 0 0; -10 -5 0 0 0 0; 0 0 0 1 0 0; 0 1 -2 -7 0 0; 0 0 0 0 0 1; -3 0 0 -2 -1 0')
B1 = np.matrix('0; 1; 0; 0; 0; 0')
B2 = np.matrix('0; 0; 0; 0; 0; 1')
x = np.matrix('0;0;0;0;0;0')
f = np.ones(NN)
'''LOOK AT IF THIS IS THE STEP FUNCTION^ '''
x[0] = 1
x[1] = 2