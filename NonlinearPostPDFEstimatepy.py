# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:52:33 2020

@author: Administrator
@description:
    1.EKF
    3.SPKF
    
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import *
import matplotlib
from PlotFunction import PlotGuassFunction
from CommonFunction import GuassFunction
from EstimateAlgorithm import EKFUpdate
from EstimateAlgorithm import IEKFUpdate
from EstimateAlgorithm import SigmaPointKFUpdate
from EstimateAlgorithm import ISigmaPointKFUpdate

matplotlib.use('Agg')

MCSampleNumber = 1000000
x_prior = 20
P_prior = 9
R = 0.09
x_true = 26
f = 400
b = 0.1
y_meas = f*b/x_true-0.6


def g(x):
    return f*b/x

if __name__ == "__main__":
    Gap = 0.0001
    xlim = [15,35]
    x = np.arange(xlim[0], xlim[1], Gap)
    px = GuassFunction(x_prior,P_prior,x)
    py_x = GuassFunction(g(x),R,y_meas)
    px_y = px*py_x
    px_y = px_y/sum(px_y*Gap)
    x_map = np.argmax((px_y))+(xlim[0])/Gap
    x_map*=Gap
    x_mean = sum(px_y*x*Gap)
    print("最大后验：{},后验均值：{}".format(x_map,x_mean))
    
    
    G = -f*b/(x_prior**2)
    Xekf_post,Pekf_post = EKFUpdate(P_prior,G,R,x_prior,y_meas,g(x_prior),x_prior)
    print("EKF估计后验：{}".format(Xekf_post))
    
    Xiekf_post,Piekf_post = IEKFUpdate(P_prior,G,R,x_prior,y_meas,g(x_prior),0.001)
    print("IEKF估计后验：{}".format(Xiekf_post))
    
    Xspkf_post,Pspkf_post = SigmaPointKFUpdate(P_prior,R,x_prior,y_meas,2,x_prior)
    print("SPKF估计后验：{}".format(Xspkf_post[0][0]))
    
    Xispkf_post,Pispkf_post = ISigmaPointKFUpdate(P_prior,R,x_prior,y_meas,2,0.001)
    print("ISPKF估计后验：{}".format(Xispkf_post[0][0]))
    
    
    
    #画图
    plt.plot(x,px_y)
    #PlotGuassFunction(x_prior,P_prior,"prior",'--',xlim)
    #PlotGuassFunction(Xekf_post[0,0],Pekf_post[0,0],'EKF','-.',xlim)
    PlotGuassFunction(Xiekf_post[0,0],Piekf_post[0,0],'IEKF','-',xlim)
    PlotGuassFunction(Xspkf_post[0,0],Pspkf_post[0,0],'ISPEKF','-.',xlim)
    PlotGuassFunction(Xispkf_post[0,0],Pispkf_post[0,0],'ISPEKF','--',xlim)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.title('概率分布')
    #plt.grid()
    plt.plot([x_map,x_map],[0,0.2])
    plt.plot([x_mean,x_mean],[0,0.2],'--')
    #plt.legend(["p(x|y)","p(x)","EKF",'IEKF','最大后验','后验均值'],loc='upper right')
    plt.legend(["p(x|y)","IEKF",'SPKF','ISPKF','最大后验','后验均值'],loc='upper right')
    plt.savefig('./images/figure_FZ_SPKF.png')
    plt.savefig('./images/figure_FZ_SPKF.eps')
