# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:03:53 2012

1d advection test case using FEM nodal implentation with PETSc

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

x1 = grid[0]
x2 = grid[1]
for i in range(0,order):
    x = 0.5*(xi[i]*dx+(x1+x2))     
    # Mass matrix
    M11 = wi[i]*dx*0.5*(x2-x)**2
    M22 = wi[i]*dx*0.5*(x-x1)**2
    M12 = wi[i]*dx*0.5*(x-x1)*(x2-x)
    # Stiffness Matrix
    K11 = wi[i]*dx*0.5*(x-x2)
    K22 = wi[i]*dx*0.5*(x-x1)


for k in range(0, nx-1):  
    M.setValue(k, k, M11, PETSc.InsertMode.ADD) # Diagonal 1
    M.setValue(k+1, k+1, M22, PETSc.InsertMode.ADD) # Diagonal 2
    M.setValue(k+1, k, M12, PETSc.InsertMode.ADD) # Off-diagonal.
    M.setValue(k, k+1, M12, PETSc.InsertMode.ADD) # Off-diagonal.
    # stiffness matrix
    K.setValue(k, k, K11, PETSc.InsertMode.ADD) # Diagonal 1
    K.setValue(k+1, k+1, K22, PETSc.InsertMode.ADD) # Diagonal 2
    K.setValue(k, k+1, -K22, PETSc.InsertMode.ADD) # Off-diagonal.
    K.setValue(k+1, k, -K11, PETSc.InsertMode.ADD) # Off-diagonal.
        
# set periodic BC's
M.setValue(0, 0, M22, PETSc.InsertMode.ADD)
M.setValue(nx-1, nx-1, M11, PETSc.InsertMode.ADD)
M.setValue(0, nx-1, M12, PETSc.InsertMode.ADD)
M.setValue(nx-1, 0, M12, PETSc.InsertMode.ADD)  
K.setValue(0, 0, K22, PETSc.InsertMode.ADD)
K.setValue(nx-1, nx-1, K11, PETSc.InsertMode.ADD)
K.setValue(0, nx-1, -K22, PETSc.InsertMode.ADD)
K.setValue(nx-1, 0, -K11, PETSc.InsertMode.ADD)

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