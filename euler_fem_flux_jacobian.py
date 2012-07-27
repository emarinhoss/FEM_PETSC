# -*- coding: utf-8 -*-
"""
Created on Tue Jul  25 17:26:31 2012

1d euler eqn using FEM nodal implentation with PETSc.
Using explicit flux jacobian implementation and first order elements.

@author: sousae
"""


def dataforplot(comp, vec):
    qw = size(vec)
    
    q1 = zeros(qw/comp)
    q2 = zeros(qw/comp)
    q3 = zeros(qw/comp)
        
    for i in range(0,qw/comp):
        q1[i] = vec[i*comp]
        q2[i] = vec[i*comp+1]
        q3[i] = vec[i*comp+2]
    
    return q1, q2, q3

from math import *
from numpy import *
from pylab import *

import petsc4py, sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
from auxiliary_funcs import *

# ============ variables grid ============ #
gm = 1.4

# ============ variables grid ============ #
nx   = 101	# grid nodes
L    = 5.0	# domain half-size
x0   = 0.   # gaussian peak location
grid = linspace(-L,L,nx)
dx   = grid[2]-grid[1]

# ============ GL quadrature integration ============ #
order = 2   # order
wi, xi = gaussl(order)

# ============ timestep ============ #
#cfl = 1.0
dt  = 0.1
ndt = 3        # number of cycles (timesteps)

# ============ Create matrices ============ #
N  = 1		# interpolation function order
V  = 3		# number of components per node
blk= (N+1)*V# block size

# Stiffness matrix
K = PETSc.Mat().createAIJ([nx*V, nx*V], nnz=3)
# Mass matrix
M = PETSc.Mat().createAIJ([nx*V, nx*V], nnz=3)
# Flux jacobian
dF = PETSc.Mat().createDense([nx*V, nx*V])

# ============ Create vectors ============ #
u_n  = PETSc.Vec().createSeq(nx*V)
u_n1 = PETSc.Vec().createSeq(nx*V)
b1   = PETSc.Vec().createSeq(nx*V)
b2   = PETSc.Vec().createSeq(nx*V)
b    = PETSc.Vec().createSeq(nx*V)

# ============ Populate Matrices ============ #

x1 = grid[0]
x2 = grid[1]

M11 = 0.
M12 = 0.
M22 = 0.

K11 = 0.
K12 = 0.
K21 = 0.
K22 = 0.

for i in range(0,order):
    x = xi[i]*dx+x2
    
    Phi1 = (x2-x)/dx
    Phi2 = (x-x1)/dx
    
    dPhi1 = -1./dx
    dPhi2 =  1./dx
    
    M11 += wi[i]*dx*0.5*Phi1*Phi1
    M12 += wi[i]*dx*0.5*Phi1*Phi2
    M22 += wi[i]*dx*0.5*Phi2*Phi2

    
    K11 += wi[i]*dx*0.5*Phi1*dPhi1
    K12 += wi[i]*dx*0.5*Phi1*dPhi2
    K21 += wi[i]*dx*0.5*Phi2*dPhi1
    K22 += wi[i]*dx*0.5*Phi2*dPhi2
        
for k in range(0, (nx-1)):
	for m in range(0, V):
		# ======================
		# Mass matrix
		# ======================
		M.setValue(k*V+m, k*V+m,   M11, PETSc.InsertMode.ADD)
		M.setValue(k*V+V+m, k*V+m,   M12, PETSc.InsertMode.ADD)
		M.setValue(k*V+m, k*V+V+m,   M12, PETSc.InsertMode.ADD)
		M.setValue(k*V+V+m, k*V+V+m,   M22, PETSc.InsertMode.ADD)
		
		# ======================
		# Stiffness matrix
		# ======================
		K.setValue(k*V+m, k*V+m,   K11, PETSc.InsertMode.ADD)
		K.setValue(k*V+V+m, k*V+m, K21, PETSc.InsertMode.ADD)
		K.setValue(k*V+m, k*V+V+m, K12, PETSc.InsertMode.ADD)
		K.setValue(k*V+V+m, k*V+V+m, K22, PETSc.InsertMode.ADD)
    
# Make matrices useable.
M.assemblyBegin()
M.assemblyEnd()
K.assemblyBegin()
K.assemblyEnd()
K.scale(-dt)

# ============ Initialize ============ #
for i in range(0,nx):
    if (grid[i] < 1.5):
        q1 = 1.0
        q2 = 0.0
        q3 = 2.5
    else:
        q1 = 0.125
        q2 = 0.0
        q3 = 0.25
    u_n.setValue(i*V, q1)
    u_n.setValue(i*V+1, q2)
    u_n.setValue(i*V+2, q3)
    #val = 1.+exp(-0.4*(grid[i]-x0)**2)
    #for l in range(0,V):
		#u_n.setValue(i*V+l, val)
		


# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

for t in range(0,ndt):
    
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, (nx-1)):
		dF.setValue(k*V, k*V+1, 1., PETSc.InsertMode.ADD)
		dF.setValue(k*V+1, k*V, 0.5*(gm-3.)*(u_n[k*V+1]/u_n[k*V])**2, PETSc.InsertMode.ADD)
		dF.setValue(k*V+1, k*V+1, (3.-gm)*u_n[k*V+1]/u_n[k*V], PETSc.InsertMode.ADD)
		dF.setValue(k*V+1, k*V+2, (gm-1), PETSc.InsertMode.ADD)
		dF.setValue(k*V+2, k*V, -gm*u_n[k*V+1]*u_n[k*V+2]/u_n[k*V]+0.5*(u_n[k*V+1]/u_n[k*V])**3, PETSc.InsertMode.ADD)
		dF.setValue(k*V+2, k*V+1, gm*u_n[k*V+2]/u_n[k*V]-1.5*(gm-1.)*(u_n[k*V+1]/u_n[k*V])**2, PETSc.InsertMode.ADD)
		dF.setValue(k*V+2, k*V+2, gm*u_n[k*V+1]/u_n[k*V], PETSc.InsertMode.ADD)
		
		dF.setValue(k*V+V, k*V+V+1, 1., PETSc.InsertMode.ADD)
		dF.setValue(k*V+V+1, k*V+V, 0.5*(gm-3.)*(u_n[k*V+1]/u_n[k*V])**2, PETSc.InsertMode.ADD)
		dF.setValue(k*V+V+1, k*V+V+1, (3.-gm)*u_n[k*V+1]/u_n[k*V], PETSc.InsertMode.ADD)
		dF.setValue(k*V+V+1, k*V+V+2, (gm-1), PETSc.InsertMode.ADD)
		dF.setValue(k*V+V+2, k*V+V, -gm*u_n[k*V+1]*u_n[k*V+2]/u_n[k*V]+0.5*(u_n[k*V+1]/u_n[k*V])**3, PETSc.InsertMode.ADD)
		dF.setValue(k*V+V+2, k*V+V+1, gm*u_n[k*V+2]/u_n[k*V]-1.5*(gm-1.)*(u_n[k*V+1]/u_n[k*V])**2, PETSc.InsertMode.ADD)
		dF.setValue(k*V+V+2, k*V+V+2, gm*u_n[k*V+1]/u_n[k*V], PETSc.InsertMode.ADD)
		
    dF.assemblyBegin()
    dF.assemblyEnd()
    S = dot(dF, K)
    a,b,c = dataforplot(V,u_n)
    subplot(131), plot(grid,a)#, axis((-L,L,0.0,3.0))
    subplot(132), plot(grid,b)#, axis((-L,L,0.0,3.0))
    subplot(133), plot(grid,c)#, axis((-L,L,0.0,3.0))
    filename = 'euler_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    S.mult(u_n,b1)
    M.mult(u_n,b2)
    b = b1 + b2
    ksp.solve(b, u_n1)
    u_n = u_n1


a,b,c = dataforplot(V,u_n)
subplot(131), plot(grid,a)#, axis((-L,L,0.0,3.0))
subplot(132), plot(grid,b)#, axis((-L,L,0.0,3.0))
subplot(133), plot(grid,c)#, axis((-L,L,0.0,3.0))
filename = 'euler_' + str('%03d' % ndt) + '_.png'
savefig(filename, dpi=200)
clf()