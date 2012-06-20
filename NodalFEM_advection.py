# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:42:13 2012

1d advection test case using FEM nodal implentation with PETSc

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
nx = 200	# grid nodes
a  = 1.0	# advection speed
L  = 5.0	# domain half-size
x0 = 0.   # gaussian peak location

# ============ Options ============ #
order = 4   # quadrature order
wi, xi = gaussl(order)

# ============ grid ============ #
grid = linspace(-L,L,nx)
dx = grid[2]-grid[1]

# ============ timestep ============ #
cfl = 1.0
dt  = abs(cfl*dx/a)
ndt = 40        # number of cycles (timesteps)
# ============ Create matrices ============ #
# Stiffness matrix
K = PETSc.Mat().createAIJ([nx, nx], nnz=3)

# Mass matrix
M = PETSc.Mat().createAIJ([nx, nx], nnz=3)

# ============ Create vectors ============ #
u_n  = PETSc.Vec().createSeq(nx)
u_n1 = PETSc.Vec().createSeq(nx)
b1   = PETSc.Vec().createSeq(nx)
b2   = PETSc.Vec().createSeq(nx)
b    = PETSc.Vec().createSeq(nx)
# ============ Initialize ============ #
for i in range(0,nx):
    #if (grid[i] > -4 and grid[i] < -3):
        #u_n.setValue(i, 1.0)
    #else:
        #u_n.setValue(i, 0.0)
    u_n.setValue(i, exp(-(grid[i]-x0)**2))
    
# ============ Populate Matrices ============ #

for k in range(0, nx-1):
    x1 = grid[k]
    x2 = grid[k+1]
    for i in range(0,order):
        x = 0.5*(xi[i]*dx+(x1+x2))
        
        # Mass matrix
        M.setValue(k, k, wi[i]*dx*0.5*(x2-x)**2,PETSc.InsertMode.ADD) # Diagonal 1
        M.setValue(k+1, k+1, wi[i]*dx*0.5*(x-x1)**2,PETSc.InsertMode.ADD) # Diagonal 2
        M.setValue(k+1, k, wi[i]*dx*0.5*(x-x1)*(x2-x),PETSc.InsertMode.ADD) # Off-diagonal.
        M.setValue(k, k+1, wi[i]*dx*0.5*(x-x1)*(x2-x),PETSc.InsertMode.ADD) # Off-diagonal.
        
        # stiffness matrix
        K.setValue(k, k, wi[i]*dx*0.5*(x-x2),PETSc.InsertMode.ADD) # Diagonal 1
        K.setValue(k+1, k+1, wi[i]*dx*0.5*(x-x1),PETSc.InsertMode.ADD) # Diagonal 2
        K.setValue(k, k+1, wi[i]*dx*0.5*(x1-x),PETSc.InsertMode.ADD) # Off-diagonal.
        K.setValue(k+1, k, wi[i]*dx*0.5*(x2-x),PETSc.InsertMode.ADD) # Off-diagonal.
        
# set periodic BC's
#x1u = grid[nx-2]
#x2u = grid[nx-1]
#x1l = grid[0]
#x2l = grid[1]
#for i in range(0,order):
#    xu = 0.5*(xi[i]*dx+(x2u+x1u))
#    xl = 0.5*(xi[i]*dx+(x2l+x1l))
#    
#    #mass matrix
#    M.setValue(0, 0, wi[i]*dx*0.5*(xu-x1u)**2,PETSc.InsertMode.ADD)
#    M.setValue(nx-1, nx-1, wi[i]*dx*0.5*(x2l-xl)**2,PETSc.InsertMode.ADD)
#    M.setValue(0, nx-1, wi[i]*dx*0.5*(xu-x1u)*(x2u-xu),PETSc.InsertMode.ADD)
#    M.setValue(nx-1, 0, wi[i]*dx*0.5*(xl-x1l)*(x2l-xl),PETSc.InsertMode.ADD)
#
#    # Stiffness matrix    
#    K.setValue(0, 0, wi[i]*dx*0.5*(xu-x1u),PETSc.InsertMode.ADD)
#    K.setValue(nx-1, nx-1, wi[i]*dx*0.5*(xl-x2l),PETSc.InsertMode.ADD)
#    K.setValue(0, nx-1, wi[i]*dx*0.5*(x1u-xu),PETSc.InsertMode.ADD)
#    K.setValue(nx-1, 0, wi[i]*dx*0.5*(xl-x2l),PETSc.InsertMode.ADD)

# Make matrices useable.
M.assemblyBegin()
M.assemblyEnd()
M.scale(1.0/(dx*dx))
K.assemblyBegin()
K.assemblyEnd()
K.scale(a*dt/(dx*dx))

# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

for t in range(0,ndt):
    
    plot(grid,u_n.array)
    #axis((-L,L,-0.05,1.5))
    filename = 'nodalAdv_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    K.mult(u_n,b1)
    M.mult(u_n,b2)
    b = b1 + b2
    b.setValue(0, dt*a*u_n[0], PETSc.InsertMode.ADD)
    b.setValue(nx-1, -dt*a*u_n[nx-1], PETSc.InsertMode.ADD)
    ksp.solve(b, u_n1)
    u_n = u_n1

plot(grid,u_n.array)
#axis((-L,L,-0.05,1.5))
filename = 'nodalAdv_' + str('%03d' % ndt) + '_.png'
savefig(filename, dpi=200)
clf()
