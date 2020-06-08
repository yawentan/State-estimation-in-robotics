# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:48:56 2020

@author: Administrator
@description:
    求解高斯先验经过非线性变化后的概率密度方法对比
    1.Monte Carlo
    2.Linearization
    3.Unscented(无迹) or SigmaPoint
    MC采样出来的均值和方差雨Unscented相同
    本MC采样结果用直方图表现出来，即是实际的p(y)分布
    而2，3的p(y)均是高斯函数
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from PlotFunction import PlotGuassFunction

MCSampleNumber = 1000000
x_prior_mu = 5
x_prior_sigma = (3/2)**2

"""
MonteCarlo采样方法，得到的p(y)
"""
def MonteCarlo(x_prior_mu,x_prior_sigma,MCSampleNumber):
    #MC
    #采用多个xi传入g(x)算出p(y)的均值和方差
    x = np.random.normal(x_prior_mu,math.sqrt(x_prior_sigma),MCSampleNumber)
    #print(np.mean(x),np.var(x))
    y = G(x)
    y_mu = np.mean(y)
    y_sigama = np.var(y)
    y_true_mu = x_prior_mu**2+x_prior_sigma
    y_true_sigma = 4*x_prior_mu**2*x_prior_sigma+2*x_prior_sigma**2
    plt.hist(y,80,density='True')
    PlotGuassFunction(y_mu,y_sigama,'MC','--',[0,80])
    plt.axis()
    plt.title("MC p(y)")
    plt.xlabel('y')
    plt.ylabel('p(y)')
    print("MC采样均值：{}采样方差：{}".format(y_mu,y_sigama))
    print("MC真实均值：{}真实方差：{}".format(y_true_mu,y_true_sigma))

def Linearization(x_prior_mu,x_prior_sigma,MCSampleNumber):
    x = np.random.normal(x_prior_mu,math.sqrt(x_prior_sigma),MCSampleNumber)
    #print(np.mean(x),np.var(x))
    y = 2*x_prior_mu*x-x_prior_mu**2
    y_mu = np.mean(y)
    y_sigama = np.var(y)
    y_true_mu = x_prior_mu**2
    y_true_sigma = 4*x_prior_mu**2*x_prior_sigma
    plt.axis()
    plt.title("Linearization p(y)")
    PlotGuassFunction(y_mu,y_sigama,'LinearGuass','-.',[0,80])
    plt.xlabel('y')
    plt.ylabel('p(y)')
    print("线性化采样均值：{}采样方差：{}".format(y_mu,y_sigama))
    print("线性化真实均值：{}真实方差：{}".format(y_true_mu,y_true_sigma))

def SigmaPoint(x_prior_mu,x_prior_sigma,Sigma,MCSampleNumber):
    x0 = x_prior_mu
    x1 = x_prior_mu+np.sqrt((1+Sigma)*x_prior_sigma)
    x2 = x_prior_mu-np.sqrt((1+Sigma)*x_prior_sigma)
    y0 = G(x0)
    y1 = G(x1)
    y2 = G(x2)
    y_mu = 1/(1+Sigma)*(Sigma*y0+(y1+y2)/2)
    y_sigama = 1/(1+Sigma)*(Sigma*(y0-y_mu)**2 + ((y1-y_mu)**2+(y2-y_mu)**2)/2)
    y_true_mu = x_prior_mu**2+x_prior_sigma
    y_true_sigma = 4*x_prior_mu**2*x_prior_sigma+Sigma*x_prior_sigma**2
    PlotGuassFunction(y_mu,y_sigama,'SigmaPoint','--',[0,80])
    print("SP采样均值：{}采样方差：{}".format(y_mu,y_sigama))
    print("SP真实均值：{}真实方差：{}".format(y_true_mu,y_true_sigma))
    
    
def G(x):
    return x**2

if __name__ == '__main__':
    MonteCarlo(x_prior_mu,x_prior_sigma,MCSampleNumber)
    Linearization(x_prior_mu,x_prior_sigma,MCSampleNumber)
    SigmaPoint(x_prior_mu,x_prior_sigma,2,MCSampleNumber)
    plt.legend(["MC","Linearization",'SigmaPoint',"MC采样分布直方图"],loc='upper right')
    plt.title('PDF分布图')
    plt.savefig('./images/figure_FZ_1.png')
    plt.savefig('./images/figure_FZ_1.eps')