# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:28:55 2012

@author: sousae
"""


from numpy import *
from pylab import *
import matplotlib.font_manager as fm


font = {'fontsize'   : 20}

# 
data1 = loadtxt('data_200_1.050_.dat')
data2 = loadtxt('data_200_1.155_.dat')
data3 = loadtxt('data_200_1.260_.dat')

#
data4 = loadtxt('data_1.050_.dat')
data5 = loadtxt('data_100_1.050_.dat')

lbls = ['p', r'$\rho$','u','e']
prop = fm.FontProperties(size=20)


plot(data1[:,0],data1[:,1], 'b',linewidth=2), ylabel(lbls[1],font), ylim(0.2, 1.3), xlabel('x',font)
plot(data2[:,0],data2[:,1], 'r',linewidth=2), ylabel(lbls[1])
plot(data3[:,0],data3[:,1], 'k',linewidth=2), ylabel(lbls[1])
legend((r'$p_e=1.0$',r'$p_e=1.1$',r'$p_e=1.2$'), 0, prop=prop)
plot([5.6, 5.6],[0.2, 1.3],'k--',linewidth=2)
yticks(fontsize=18)
xticks(fontsize=18)
filename = 'nozzlePlots1.png'
savefig(filename, dpi=200)
clf()

plot(data1[:,0],data1[:,1], 'b',linewidth=2), ylabel(lbls[1],font), xlabel('x',font)
plot(data5[:,0],data5[:,1], 'k',linewidth=2), ylabel(lbls[1],font)
plot(data4[:,0],data4[:,1], 'r',linewidth=2), ylabel(lbls[1],font)
legend((r'$200$',r'$100$',r'$50$'), 0, prop=prop)
yticks(fontsize=18)
xticks(fontsize=18)
filename = 'nozzlePlots2.png'
savefig(filename, dpi=200)
clf()

nx = 200
grid = linspace(0.0,10,nx)
A = zeros(nx)
for i in range(0,nx):
    A[i] = 1.398 + 0.347*tanh(0.8*grid[i]-4.0)
    
plot(grid, A,linewidth=2), ylabel('Area',font), xlabel('x',font)
yticks(fontsize=18)
xticks(fontsize=18)
text(0.1,1.75,r'$A=1.398+0.347\tanh(0.8x-4.0)$', fontsize=20)
#legend(r'$1.398+0.347\tanh(0.8*x-4.0)$', 2, prop=prop)
filename = 'nozzlePlots3.png'
savefig(filename, dpi=200)
clf()