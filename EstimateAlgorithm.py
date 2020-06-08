# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:26:09 2020

@author: Administrator
"""



import numpy as np
from numpy import *
import scipy.linalg
from PlotFunction import PlotGuassFunction


f = 400
b = 0.1
def g(x):
    return f*b/x

"""
Input
    Xk_prior:先验均值
    P_prior:先验方差
    R:测量方差
    kesai:参数
SigmaPoint采样来得到：
    Mu_y
    Sigma_yy
    Sigma_xy
    Sigma_xx
"""
def SigmaPointSample(Xk_prior,P_prior,R,kesai):
    #P_prior = mat(P_prior)
    #R = mat(R)
    Mu_z = [Xk_prior,0]
    Sigma_zz = diag([P_prior,R])
    #Cholesky分解，L为小三角矩阵
    L = scipy.linalg.cholesky(Sigma_zz,True,overwrite_a=True)
    dimL,_ = L.shape
    #采样点
    z = [0]*(2*dimL+1)
    xk = [0]*(2*dimL+1)
    nk = [0]*(2*dimL+1)
    yk = [0]*(2*dimL+1)
    z[0] = Mu_z
    N=1
    xk[0] = z[0][0:N]
    nk[0] = z[0][N:]
    yk[0] = g(xk[0][0])+nk[0][0]
    for i in range(1,dimL+1):
        z[i] = Mu_z+np.sqrt(dimL+kesai)*L[:,i-1]
        z[dimL+i] = Mu_z-np.sqrt(dimL+kesai)*L[:,i-1]
        xk[i] = z[i][0:N]
        xk[dimL+i] = z[dimL+i][0:N]
        nk[i] = z[i][N:]
        nk[dimL+i] = z[dimL+i][N:]
        yk[i] = g(xk[i][0])+nk[i][0]
        yk[dimL+i] = g(xk[dimL+i][0])+nk[dimL+i][0]
        
    #求Mu_y,Sigma_yy,Sigma_xy
    Mu_y = 0
    Sigma_yy = 0
    Sigma_xy = 0
    Sigma_xx = 0
    for i in range(2*dimL+1):
        if i == 0:
            alpha = (kesai)/(dimL+kesai)
        else:
            alpha = 1/(2*(dimL+kesai))
        Mu_y += alpha*yk[i]
    for i in range(2*dimL+1):
        if i == 0:
            alpha = (kesai)/(dimL+kesai)
        else:
            alpha = 1/(2*(dimL+kesai))
        Sigma_yy += alpha*(yk[i]-Mu_y)*(yk[i]-Mu_y)
        Sigma_xy += alpha*(xk[i][0]-Xk_prior)*(yk[i]-Mu_y)
        Sigma_xx += alpha*(xk[i][0]-Xk_prior)*(xk[i][0]-Xk_prior)
    return Mu_y,Sigma_xy,Sigma_yy,Sigma_xx
        


def SigmaPointKFUpdate(Pk_prior,Rk,Xk_prior,yk_meas,kesai,Xop_k):
    from numpy import mat
    muy_k,Sigmaxy_k,Sigmayy_k,Sigmaxx_k = SigmaPointSample(Xop_k,Pk_prior,Rk,kesai)
    Sigmaxy_k = mat(Sigmaxy_k)
    Sigmayy_k = mat(Sigmayy_k)
    Sigmaxx_k = mat(Sigmaxx_k)
    muy_k = mat(muy_k)
    Kk = Sigmaxy_k*Sigmayy_k.I
    Pk_post = Sigmaxx_k-Kk*Sigmaxy_k.T
    #Pk_post = (1-Kk*Sigmaxy_k.TSigmaxx_k*Sigmaxx_k.I)*Sigmaxx_k
    Xk_post = Xk_prior+Kk*(yk_meas-muy_k- Sigmaxy_k.T*Sigmaxx_k.I*(Xk_prior-Xop_k))
    return Xk_post,Pk_post
    
def ISigmaPointKFUpdate(Pk_prior,Rk,Xk_prior,yk_meas,kesai,Epsilon):
    Xop_k = mat(Xk_prior)
    while 1:
        Xk_post,Pk_post= SigmaPointKFUpdate(Pk_prior,Rk,Xk_prior,yk_meas,kesai,Xop_k)
        #print(Xop_k,Xk_post,Xk_post-Xop_k)
        #print(Xop_k,Xk_post[0][0])
        if abs(Xk_post[0][0]-Xop_k[0][0]) < Epsilon:
            break
        Xop_k = Xk_post
    return Xk_post,Pk_post

if __name__ == '__main__':
    P_prior = 9
    R = 0.09
    Xk_prior = 20
    P_prior = 9
    kesai =2
    x_true = 26
    yk_meas = f*b/x_true-0.6
    Xspkf,Pspkf =ISigmaPointKFUpdate(P_prior,R,Xk_prior,yk_meas,2,0.0001)
    PlotGuassFunction(Xspkf[0,0],Pspkf[0,0],"Pspkf")
    print("ok")

"""
InputTyp:mat,np.array
OutputType:mat

Input:Pk_prior,Gk,Rk,Xk_prior,yk_meas,yk_prior,Xop_k
Output:Pk_post,Xk_post
"""
def EKFUpdate(Pk_prior,Gk,Rk,Xk_prior,yk_meas,yk_prior,Xop_k):
    from numpy import mat
    Pk_prior = mat(Pk_prior)
    Gk = mat(Gk)
    Rk = mat(Rk)
    Xk_prior = mat(Xk_prior)
    yk_meas = mat(yk_meas)
    yk_prior = mat(yk_prior)
    
    Kk = Pk_prior*Gk.T*(Gk*Pk_prior*Gk.T+Rk).I
    Pk_post = (1-Kk*Gk)*Pk_prior
    Xk_post = Xk_prior+Kk*(yk_meas-yk_prior-Gk*(Xk_prior-Xop_k))
    #print("Xk_post",Kk*(yk_meas-yk_prior-Gk*(Xk_prior-Xop_k)),yk_prior+Gk*(Xk_prior-Xop_k))
    return Xk_post,Pk_post

def IEKFUpdate(Pk_prior,Gk,Rk,Xk_prior,yk_meas,yk_prior,Epsilon):
    Xop_k = Xk_prior
    while 1:
        Xk_post,Pk_post= EKFUpdate(Pk_prior,Gk,Rk,Xk_prior,yk_meas,yk_prior,Xop_k)
        #print(Xop_k,Xk_post,Xk_post-Xop_k)
        if abs(Xk_post-Xop_k) <Epsilon:
            break
        Xop_k = Xk_post
        yk_prior = g(Xop_k)
        Gk = -f*b/(Xop_k**2)
    return Xk_post,Pk_post