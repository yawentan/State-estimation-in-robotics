# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:16:00 2020

@author: Administrator
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.figure(dpi=240)


def PlotGuassFunction(mu,sigma,title,LS,xlim):
    x = np.arange(xlim[0], xlim[1], 0.1)
    plt.plot(x, 1/np.sqrt(2 * np.pi*sigma)*np.exp(-(x-mu)**2/(2*sigma)),linewidth=1,ls=LS)
    plt.axis()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('p')
    #plt.savefig('./images/figure_2_4.png')
    #plt.show()