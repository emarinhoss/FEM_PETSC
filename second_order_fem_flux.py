# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:56:32 2012

1d advection test case using FEM nodal implentation with PETSc.
Using explicit flux jacobian implementation and second order elements.

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
nx = 101	# grid nodes
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
x3 = grid[2]
Phi1 = 0.
Phi2 = 0.
Phi3 = 0.
dPhi1 = 0.
dPhi2 = 0.
dPhi3 = 0.

for i in range(0,order):
    x = xi[i]*dx+x2
    
    Phi1 += wi[i]*dx*(x*x-x*(x2+x3)+x2*x3)/(dx*dx)
    Phi2 += wi[i]*dx*(x*x-x*(x3+x1)+x3*x1)/(-dx*dx)
    Phi3 += wi[i]*dx*(x*x-x*(x2+x1)+x2*x1)/(dx*dx)
    
    dPhi1 += wi[i]*dx*(2*x-x2-x3)/(dx*dx)
    dPhi2 += wi[i]*dx*(2*x-x1-x3)/(-dx*dx)
    dPhi3 += wi[i]*dx*(2*x-x2-x1)/(dx*dx)
    
for k in range(0, (nx-1)/2):
    # ======================
    # Mass matrix
    # ======================
    # 1st row
    M.setValue(2*k, 2*k, Phi1*Phi1, PETSc.InsertMode.ADD) 
    M.setValue(2*k, 2*k+1, Phi1*Phi2, PETSc.InsertMode.ADD) 
    M.setValue(2*k, 2*k+2, Phi1*Phi3, PETSc.InsertMode.ADD)
    # 2nd row
    M.setValue(2*k+1, 2*k, Phi2*Phi1, PETSc.InsertMode.ADD) 
    M.setValue(2*k+1, 2*k+1, Phi2*Phi2, PETSc.InsertMode.ADD) 
    M.setValue(2*k+1, 2*k+2, Phi2*Phi3, PETSc.InsertMode.ADD)
    # 3rd row
    M.setValue(2*k+2, 2*k, Phi3*Phi1, PETSc.InsertMode.ADD) 
    M.setValue(2*k+2, 2*k+1, Phi3*Phi2, PETSc.InsertMode.ADD) 
    M.setValue(2*k+2, 2*k+2, Phi3*Phi3, PETSc.InsertMode.ADD)
    # ======================
    # Stiffness matrix
    # ======================
    # 1st row
    K.setValue(2*k, 2*k, Phi1*dPhi1, PETSc.InsertMode.ADD) 
    K.setValue(2*k, 2*k+1, Phi1*dPhi2, PETSc.InsertMode.ADD) 
    K.setValue(2*k, 2*k+2, Phi1*dPhi3, PETSc.InsertMode.ADD)
    # 2nd row
    K.setValue(2*k+1, 2*k, Phi2*dPhi1, PETSc.InsertMode.ADD) 
    K.setValue(2*k+1, 2*k+1, Phi2*dPhi2, PETSc.InsertMode.ADD) 
    K.setValue(2*k+1, 2*k+2, Phi2*dPhi3, PETSc.InsertMode.ADD)
    # 3rd row
    K.setValue(2*k+2, 2*k, Phi3*dPhi1, PETSc.InsertMode.ADD) 
    K.setValue(2*k+2, 2*k+1, Phi3*dPhi2, PETSc.InsertMode.ADD) 
    K.setValue(2*k+2, 2*k+2, Phi3*dPhi3, PETSc.InsertMode.ADD)

    
# Make matrices useable.
M.assemblyBegin()
M.assemblyEnd()
K.assemblyBegin()
K.assemblyEnd()
K.scale(-a*dt)

# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

for t in range(0,ndt):
    
    plot(grid,u_n.array)
    axis((-L,L,-0.2,1.2))
    filename = 'sec_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    K.mult(u_n,b1)
    M.mult(u_n,b2)
    b = b1 + b2
    ksp.solve(b, u_n1)
    u_n = u_n1

plot(grid,u_n.array)
axis((-L,L,-0.2,1.2))
filename = 'sec_' + str('%03d' % ndt) + '_.png'
savefig(filename, dpi=200)
clf()
