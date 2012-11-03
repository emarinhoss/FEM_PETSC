# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:28:55 2012

@author: sousae
"""


from numpy import *
from pylab import *
import matplotlib.font_manager as fm


# ============= normalized parameters
gm = 1.4
r_inf = 1.0
P_inf = 0.75
a_inf = sqrt(gm*P_inf/r_inf)
tau = 10./a_inf

font = {'fontsize'   : 20}

# 
data1 = loadtxt('data_200_1.155_.dat')
data2 = loadtxt('data_200_1.260_.dat')
data3 = loadtxt('data_200_1.365_.dat')

#
data4 = loadtxt('data_050_1.155_.dat')
data5 = loadtxt('data_100_1.155_.dat')

lbls = ['p', r'$\rho$','u','e']
prop = fm.FontProperties(size=18)


plot(data1[:,0],data1[:,2], 'b',linewidth=2), ylabel(lbls[1],font), ylim(0.4, 1.5), xlabel('x',font)
plot(data2[:,0],data2[:,2], 'r',linewidth=2), ylabel(lbls[1])
plot(data3[:,0],data3[:,2], 'k',linewidth=2), ylabel(lbls[1])
legend((r'$p_e/p_c=0.5945$',r'$p_e/p_c=0.6486$',r'$p_e/p_c=0.7026$'), 3, prop=prop)
plot([6.37, 6.37],[0.4, 1.5],'b--',linewidth=2)
plot([5.61, 5.61],[0.4, 1.5],'r--',linewidth=2)
plot([5.16, 5.16],[0.4, 1.5],'k--',linewidth=2)
yticks(fontsize=18)
xticks(fontsize=18)
filename = 'nozzlePlots1.png'
savefig(filename, dpi=200)
clf()

plot(data1[:,0],data1[:,2], 'b',linewidth=2), ylabel(lbls[1],font), xlabel('x',font), ylim(0.4, 1.5)
plot(data5[:,0],data5[:,2], 'k',linewidth=2), ylabel(lbls[1],font)
plot(data4[:,0],data4[:,2], 'r',linewidth=2), ylabel(lbls[1],font)
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