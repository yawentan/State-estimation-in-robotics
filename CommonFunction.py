# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:20:29 2020

@author: Administrator
"""


import numpy as np

def GuassFunction(mu,sigma,x):
    return 1/np.sqrt(2 * np.pi*sigma)*np.exp(-(x-mu)**2/(2*sigma))