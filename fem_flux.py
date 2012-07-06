# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:56:32 2012

1d advection test case using FEM nodal implentation with PETSc.
Using explicit flux jacobian implementation.

@author: sousae
"""

from math import *
from numpy import *
from pylab import *

import petsc4py, sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
from auxiliary_funcs import *

# ============ variables ============ #
nx = 301	# grid nodes
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
cfl = 1.0
dt  = abs(cfl*dx/a)
ndt = 20        # number of cycles (timesteps)
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
    u_n.setValue(i, exp(-0.4*(grid[i]-x0)**2))
    
# ============ Populate Matrices ============ #

x1 = grid[0]
x2 = grid[1]
M11 = 0.
M22 = 0.
M12 = 0.
K11 = 0.
K22 = 0.
for i in range(0,order):
    x = 0.5*(xi[i]*dx+(x1+x2))     
    # Mass matrix
    M11 += wi[i]*dx*0.5*(x2-x)**2
    M22 += wi[i]*dx*0.5*(x-x1)**2
    M12 += wi[i]*dx*0.5*(x-x1)*(x2-x)
    # Stiffness Matrix
    K11 += wi[i]*dx*0.5*(x-x2)
    K22 += wi[i]*dx*0.5*(x-x1)
    
for k in range(0, nx-1):  
    M.setValue(k, k, M11, PETSc.InsertMode.ADD) # Diagonal 1
    M.setValue(k+1, k+1, M22, PETSc.InsertMode.ADD) # Diagonal 2
    M.setValue(k+1, k, M12, PETSc.InsertMode.ADD) # Off-diagonal.
    M.setValue(k, k+1, M12, PETSc.InsertMode.ADD) # Off-diagonal.
    # stiffness matrix
    K.setValue(k, k, K11, PETSc.InsertMode.ADD) # Diagonal 1
    K.setValue(k+1, k+1, K22, PETSc.InsertMode.ADD) # Diagonal 2
    K.setValue(k, k+1, -K11, PETSc.InsertMode.ADD) # Off-diagonal.
    K.setValue(k+1, k, -K22, PETSc.InsertMode.ADD) # Off-diagonal.    
    
# Make matrices useable.
M.assemblyBegin()
M.assemblyEnd()
M.scale(1.0/(dx*dx))
K.assemblyBegin()
K.assemblyEnd()
K.scale(-a*dt/(dx*dx))

# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

for t in range(0,ndt):
    
    plot(grid,u_n.array)
    axis((-L,L,-0.2,1.2))
    filename = 'fem_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    K.mult(u_n,b1)
    M.mult(u_n,b2)
    b = b1 + b2
    ksp.solve(b, u_n1)
    u_n = u_n1

plot(grid,u_n.array)
axis((-L,L,-0.2,1.2))
filename = 'fem_' + str('%03d' % ndt) + '_.png'
savefig(filename, dpi=200)
clf()