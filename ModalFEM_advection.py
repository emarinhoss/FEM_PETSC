# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:57:26 2012

1d advection test case using FEM modal implentation with PETSc

@author: eder
"""

from math import *
from numpy import *
from pylab import *

import petsc4py, sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
from auxiliary_funcs import *

# ============ variables ============ #
nx = 25	# grid nodes
a  = 1.0	# advection speed
L  = 5.0	# domain half-size
x0 = 0.   # gaussian peak location

# ============ Options ============ #
order = 3   # quadrature order
wi, xi = gaussl(order)

# ============ grid ============ #
grid = linspace(-L,L,nx)
mid  = zeros(nx)
dx = grid[2]-grid[1]

# ============ timestep ============ #
cfl = 0.5
dt  = abs(cfl*dx/a)
ndt = 20        # number of cycles (timesteps)
# ============ Create matrices ============ #
# Stiffness matrix
K = PETSc.Mat().createDense([2*(nx-1), 2*(nx-1)])

# Mass matrix
M = PETSc.Mat().createDense([2*(nx-1), 2*(nx-1)])

# ============ Create vectors ============ #
u_n  = PETSc.Vec().createSeq(2*(nx-1))
u_n1 = PETSc.Vec().createSeq(2*(nx-1))
b1   = PETSc.Vec().createSeq(2*(nx-1))
b2   = PETSc.Vec().createSeq(2*(nx-1))
b    = PETSc.Vec().createSeq(2*(nx-1))
plt  = PETSc.Vec().createSeq(2*(nx-1))

# ============ Initialize ============ #
for i in range(0,nx-1):
    #u1 = exp(-(grid[i]-x0)**2)
    #u2 = exp(-(grid[i+1]-x0)**2)
    u1 = 1.0
    u2 = 1.0
    u_n.setValue(2*i, 0.5*(u1+u2))
    u_n.setValue(2*i+1, 0.5*(u2-u1))
    
for k in range(0, nx-1):
    x1 = grid[k]
    x2 = grid[k+1]
    for i in range(0,order):
        x = 0.5*(xi[i]*dx+(x1+x2))
        
        # Mass matrix
        M.setValue(2*k, 2*k, wi[i]*0.5*dx,PETSc.InsertMode.ADD) 
        M.setValue(2*k+1, 2*k+1, wi[i]*0.5*dx*((2.0*x-x2-x1)/dx)**2,PETSc.InsertMode.ADD)
        M.setValue(2*k+1, 2*k, wi[i]*0.5*dx*(2.0*x-x2-x1)/dx,PETSc.InsertMode.ADD)
        M.setValue(2*k, 2*k+1, wi[i]*0.5*dx*(2.0*x-x2-x1)/dx,PETSc.InsertMode.ADD)
        
        # stiffness matrix
        #K.setValue(2*k, 2*k, wi[i]*0.5*dx,PETSc.InsertMode.ADD)
        #K.setValue(2*k, 2*k+1, wi[i]*0.5*dx,PETSc.InsertMode.ADD)
        K.setValue(2*k+1, 2*k+1, wi[i]*0.5*dx*2.0*(2.0*x-x1-x2)/dx**2,PETSc.InsertMode.ADD)
        K.setValue(2*k+1, 2*k, wi[i]*0.5*dx*2.0/dx,PETSc.InsertMode.ADD)
    
    # cell boundaries matching condition    
#    K.setValue(2*k+1, 2*k, 1.0/(a*dt),PETSc.InsertMode.ADD)
#    K.setValue(2*k+1, 2*k+1, 1.0/(a*dt),PETSc.InsertMode.ADD)
#    K.setValue(2*k+1, 2*k+2, -1.0/(a*dt),PETSc.InsertMode.ADD)
#    K.setValue(2*k+1, 2*k+3, 1.0/(a*dt),PETSc.InsertMode.ADD)

# Flux at first element
#K.setValue(0, 0, -1.0,PETSc.InsertMode.ADD)
#K.setValue(1, 1, -1.0,PETSc.InsertMode.ADD)
#K.setValue(1, 0, -1.0,PETSc.InsertMode.ADD)
#K.setValue(0, 1, -1.0,PETSc.InsertMode.ADD)

# Flux at last element
#K.setValue(2*(nx-1), 2*(nx-1), 1.0,PETSc.InsertMode.ADD)
#K.setValue(2*(nx-1)+1, 2*(nx-1)+1, 1.0,PETSc.InsertMode.ADD)
#K.setValue(2*(nx-1)+1, 2*(nx-1), -1.0,PETSc.InsertMode.ADD)
#K.setValue(2*(nx-1), 2*(nx-1)+1, -1.0,PETSc.InsertMode.ADD)

# Make matrices useable.
M.assemblyBegin()
M.assemblyEnd()
K.assemblyBegin()
K.assemblyEnd()
K.scale(a*dt)

# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with: ', ksp.getType()

for t in range(0,ndt):
    for k in range(0,nx-1):
        plt[2*k] = u_n[2*k]-u_n[2*k+1]
        plt[2*k+1] = u_n[2*k]+u_n[2*k+1]
    subplot(1,2,1)
    plot(plt.array)
    subplot(1,2,2)
    plot(u_n.array)
    filename = 'modalAdv_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    K.mult(u_n,b1)
    M.mult(u_n,b2)
    b = b1 + b2
    ksp.solve(b, u_n1)
    u_n = u_n1

for k in range(0,nx-1):
        plt[2*k] = u_n[2*k]-u_n[2*k+1]
        plt[2*k+1] = u_n[2*k]+u_n[2*k+1]
subplot(1,2,1)
plot(plt.array)
subplot(1,2,2)
plot(u_n.array)
filename = 'modalAdv_' + str('%03d' % ndt) + '_.png'
savefig(filename, dpi=200)
clf()
