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



def loadData(filename):
    '''
    Read fits data
    '''
    tchfits = fits.open(filename)
    tgas = tchfits[1].data
    return tgas

         
def drawHRdiagram(Teff,MK):
    '''
    draw Teff-MK (HR) diagram
    '''
    Tgrid = np.linspace(3500,9000,num=50)
    Mgrid = np.linspace(-10,10,num=100)
    HRD,xedges,yedges = np.histogram2d(MK,Teff,bins=[Mgrid,Tgrid])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.imshow(HRD,extent=[yedges[0],yedges[-1],xedges[-1],xedges[0]])
    ax.contour(HRD,20,extent=[yedges[0],yedges[-1],xedges[0],xedges[-1]])    
    ax.set_aspect('auto')
    ax.set_xlim([8500,3600])
    ax.set_ylim([10,-10])
    plt.xlabel(r'$T_{\rm eff}$')
    plt.ylabel(r'$M_K$')
    plt.savefig('HRc.png')
    fig.show()
    
    
filename = 'LAMOST_TGAS_astrostat_demo.fits'
tgas = loadData(filename)
#V0 = 238 #km/s
#velsun1 = [9.58,10.52,7.01] #Tian et al. 2015
#Rsun = 8340 #Reid 2014 in parsec
#X,Y,Z = spatialCoords(tgas.l_tgas,tgas.b_tgas,tgas.dist_tgas)
#R = np.sqrt(X**2+Y**2)
#Phi = np.arctan(Y/X)
#ind = tgas.dist_error_tgas<0.3
#drawSpatialDist(X[ind],Y[ind],Z[ind])
#U, V, W, Uerr, Verr, Werr,VR,VPHI,VRerr,VPHIerr,VZ,VZerr = velocity(tgas,V0,velsun1,Rsun)
#ind = (tgas.dist_error_tgas<0.3) * (np.abs(Z)<300)
#drawInplaneVelDist(VR[ind],VPHI[ind],tgas.feh[ind])

data=tgas[(tgas.teff>0)&(tgas.MK_tgas>-100)]
print len(tgas.teff)

teff=data.teff
MK_tgas=data.MK_tgas
#print teff
#print MK_tgas
drawHRdiagram(tgas.teff,tgas.MK_tgas)
#print len(MK_tgas)
#print len(teff)

n=len(teff)
print n

num=1000#
T_min=np.min(teff)
T_max=np.max(teff)
M_min=np.min(MK_tgas)
M_max=np.max(MK_tgas)


plt.figure(figsize=[8,8])
plt.plot(teff,MK_tgas,'.',markersize=1)
#plt.scatter(teff,MK_tgas,s=1,c=teff)
plt.title('HR diagram',fontsize=20)
plt.xlabel(r'$T_{\rm eff}$')
plt.ylabel(r'$M_K$')
plt.xlim(T_max,T_min)
plt.ylim(M_max,M_min)
plt.savefig('HR.png')

plt.figure()
fig, ax = plt.subplots()
n1, bins = np.histogram(teff, num)
n1=n1/float(len(teff))/(bins[1]-bins[0])
#plt.plot(T,f_T)
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n1
# we need a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it
patch = patches.PathPatch(barpath,facecolor='red', edgecolor='green')
ax.add_patch(patch)
ax.set_title(r'$f(T_{\rm eff})$',fontsize=20)
ax.set_xlabel(r'$T_{\rm eff}$')
ax.set_ylabel(r'$f(T_{\rm eff})$')
#plt.xlim([8500,3600])
ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), 1.01*top.max())
plt.savefig('f_T.png')

plt.figure()
fig, ax = plt.subplots()
n2, bins = np.histogram(MK_tgas, num)
n2=n2/float(len(MK_tgas))/(bins[1]-bins[0])
#plt.plot(T,f_T)
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n2
# we need a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it
patch = patches.PathPatch(barpath,facecolor='blue', edgecolor='yellow')
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), 1.01*top.max())
ax.set_title(r'$f(M_K)$',fontsize=20)
ax.set_xlabel(r'$M_K$')
ax.set_ylabel(r'$f(M_K)$')
plt.savefig('f_M.png')

#conditional density
#-2<MK<0
data1=data[(data.MK_tgas<0)&(data.MK_tgas>-2)]
plt.figure()
fig, ax = plt.subplots()
n1, bins = np.histogram(data1.teff, num)
n1=n1/float(len(data1.teff))/(bins[1]-bins[0])
#plt.plot(T,f_T)
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n1
# we need a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it
patch = patches.PathPatch(barpath,facecolor='red', edgecolor='green')
ax.add_patch(patch)
ax.set_title(r'$f(T_{\rm eff}|-2<M_K<0)$',fontsize=20)
ax.set_xlabel(r'$T_{\rm eff}$')
ax.set_ylabel(r'$f(T_{\rm eff}|-2<M_K<0)$')
#plt.xlim([8500,3600])
ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), 1.01*top.max())
plt.savefig('f_Tc.png')


data2=data[(data.MK_tgas>2.5)&(data.teff<4700)&(data.teff>4300)]
plt.figure()
fig, ax = plt.subplots()
n2, bins = np.histogram(data2.MK_tgas, num)
n2=n2/float(len(data2.MK_tgas))/(bins[1]-bins[0])
#plt.plot(T,f_T)
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n2
# we need a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it
patch = patches.PathPatch(barpath,facecolor='blue', edgecolor='yellow')
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), 1.01*top.max())
ax.set_title(r'$f(M_K|M_K>2.5,4300<T_{\rm eff}<4700)$',fontsize=20)
ax.set_xlabel(r'$M_K$')
ax.set_ylabel(r'$f(M_K|M_K>2.5,4300<T_{\rm eff}<4700)$')
plt.savefig('f_Mc.png')
plt.show()
