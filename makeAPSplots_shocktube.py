# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:40:44 2012

@author: sousae
"""

from numpy import *
from pylab import *
import matplotlib.font_manager as fm


gm = 1.4

data1 = loadtxt('shocktube.dat')

lbls = ['p', r'$\rho$','u','e']
prop = fm.FontProperties(size=9.5)

val = data1.shape

r = zeros(val[0])
u = zeros(val[0])
e = zeros(val[0])

r_inf = 1.0
P_inf = 0.75
Mach = 0.0
a_inf = sqrt(gm*P_inf/r_inf)

for i in range(0,val[0]):
    
    if data1[i,0]<5.0:
        
        r[i]  = 1.0
        u[i]= 0.0
        e[i]= 2.5
        
    else:
        
        r[i]  = 4.0
        u[i]= 0.0
        e[i]= 10.0
        
r = r/r_inf
u = u/a_inf
e = e/(r_inf*a_inf*a_inf)
p = (gm -1.)*(e-0.5*r*u*u)   

subplot(2,2,1)
plot(data1[:,0],data1[:,1], 'b'), ylabel('p')
plot(data1[:,0], p,'--r')

subplot(2,2,2)
plot(data1[:,0],data1[:,2], 'b'), ylabel(r'$\rho$')
plot(data1[:,0], r,'--r')

subplot(2,2,3)
plot(data1[:,0],data1[:,3], 'b'), ylabel('u')
plot(data1[:,0], u,'--r')

subplot(2,2,4)
plot(data1[:,0],data1[:,4], 'b'), ylabel('e')
plot(data1[:,0], e,'--r')

legend(('$t=0.085$', '$t = 0.0$'), 0, prop=prop)    
filename = 'shockTubePlot.png'
savefig(filename, dpi=200)