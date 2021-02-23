# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:43:01 2018

@author: Guo xiao
"""
import numpy as np
#from matplotlib import *
from pylab import *
from math import *

def fm(m):
    return sqrt(8/pi)*exp(-8*(m-15)**2)
    
def dm(m):
    return 10**((m+1)/5.)
    
def fd(d):
    return 10/(log(10)*d)*sqrt(2/pi)*exp(-8*(5*log10(d)-16)**2)
    
mm=14
mM=16
M=np.linspace(mm,mM,num=200)
#print dm(mm)
#print dm(mM)
D=np.linspace(dm(mm),dm(mM),num=200)
FM=map(fm,M)
FD=map(fd,D)

figure(figsize=[8,6])
plot(M,FM,linewidth=2,color='blue')
title('The distribution function of $m$',fontsize=20)
xlabel('$m$')
ylabel('$f(m)$')
xlim(mm,mM)
ylim(0,)
savefig('1-2fm.png')

figure(figsize=[8,6])
plot(D,FD,linewidth=2,color='red')
title('The distribution function of $d$',fontsize=20)
xlabel('$d$(pc)')
ylabel('$f(d)$')
xlim(dm(mm),dm(mM))
ylim(0,)
savefig('1-2fd.png')
show()
#print FD
