# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:16:10 2012

1d advection test case using FEM modal implentation with scipy

@author: eder
"""

from math import *
from numpy import *
from pylab import *
from scipy.linalg import block_diag
from auxiliary_funcs import *

# ============ variables ============ #
nx = 25	# grid nodes
a  = 1.0	# advection speed
L  = 5.0	# domain half-size
x0 = 0.   # gaussian peak location

# ============ Options ============ #
order = 2   # quadrature order
wi, xi = gaussl(order)

# ============ grid ============ #
grid = linspace(-L,L,nx)
dx = grid[2]-grid[1]

# ============ timestep ============ #
cfl = 0.1
dt  = abs(cfl*dx/a)
ndt = 20        # number of cycles (timesteps)

x1 = grid[0]
x2 = grid[1]

m11 = 0.0
m12 = 0.0
m21 = 0.0
m22 = 0.0

k11 = 0.0
k12 = 0.0
k21 = 0.0
k22 = 0.0
for i in range(0,order):
        x = xi[i]*dx/2+0.5*(x1+x2)
        
        # Mass matrix
        m11 += wi[i]                        # Diagonal 1
        m22 += wi[i]*((2*x-x2-x1)/dx)**2    # Diagonal 2
        m21 += wi[i]*(2*x-x1-x2)/dx         # Off-diagonal
        m12 += wi[i]*(2*x-x1-x2)/dx         # Off-diagonal
        
        # stiffness matrix
        k22 += wi[i]*(2*(2*x-x1-x2)/dx**2)  # Diagonal 2
        k21 += wi[i]*2/dx                   # Off-diagonal
        
K = array([[k11,k12],[k21,k22]])
K = a*K
M = array([[m11,m12],[m21,m22]])
M = a*M

KK = [K]*(2*nx)
block_diag( *KK)

MM = [M]*(2*nx)
block_diag( *MM)

# ============ Initialize ============ #
u_n = zeros((2*nx,1))
u_n1 = u_n
plt = u_n
for i in range(0,nx-1):
    u1 = exp(-(grid[i]-x0)**2)
    u2 = exp(-(grid[i+1]-x0)**2)
    x1 = grid[i]
    x2 = grid[i+1]
    #u_n.setValue(i, exp(-grid2[i]*grid2[i]),PETSc.InsertMode.INSERT)
    u_n[2*i] = (x2*u1-x1*u2)/dx
    u_n[2*i+1] = (u2-u1)/dx
    
for t in range(0,ndt):
    for k in range(0,nx):
        plt[2*k] = u_n[2*k]-u_n[2*k+1]
        plt[2*k+1] = u_n[2*k]+u_n[2*k+1]
    subplot(1,2,1)
    plot(plt.array)
    subplot(1,2,2)
    plot(u_n.array)
    filename = 'scipy_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    b = (MM+KK)*u_n
    ksp.solve(b, u_n1)
    u_n = u_n1