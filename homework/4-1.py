# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:43:01 2018

@author: Guo xiao
"""
import numpy as np
#import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import numpy as np  
from numpy.linalg import cholesky  

sampleNo = 1000; 
mu = 3  
sigma = 0.1  
'''
np.random.seed(0)  
s = np.random.normal(mu, sigma, sampleNo )  
plt.subplot(141)  
plt.hist(s, 30, normed=True)  
  
np.random.seed(0)  
s = sigma * np.random.randn(sampleNo ) + mu  
plt.subplot(142)  
plt.hist(s, 30, normed=True)  
  
np.random.seed(0)  
s = sigma * np.random.standard_normal(sampleNo ) + mu  
plt.subplot(143)  
plt.hist(s, 30, normed=True)  
'''
# 二维正态分布  
#mu = np.array([[0, 0]]) 
mu = np.array([0, 0])  
Sigma1 = np.array([[1, 0.8], [0.8, 1]])
Sigma2 = np.array([[1, -0.8], [-0.8, 1]])  
#R = cholesky(Sigma)  
#nom2=np.random.randn(sampleNo, 2)#x,y不相关
#print np.dot(nom2,R)
#s = np.dot(nom2, R) + mu
#print s
s1 = np.random.multivariate_normal(mu,Sigma1,sampleNo) #
s2 = np.random.multivariate_normal(mu,Sigma2,sampleNo)
#s = nom2@R + mu  
#print s
#plt.subplot(144)  
# 注意绘制的是散点图，而不是直方图  
plt.plot(s1[:,0],s1[:,1],'+')
plt.plot(s2[:,0],s2[:,1],'.k')
#plt.plot(nom2[:,0],nom2[:,1],'.k')  
plt.show() 

