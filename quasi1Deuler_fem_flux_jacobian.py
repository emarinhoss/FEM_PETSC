"""
Created on Thu Jul 26 12:37:31 2012

Quasi-1d euler eqn using FEM nodal implentation with PETSc.
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

def lowerBC(gm, A, vec):
    #V2 = vec[4]/vec[3]/A[2]
    #V3 = vec[7]/vec[6]/A[3]
    
    vec[0] = 1.0*A[0]
    vec[1] = 2*vec[4]-vec[7]
    #vec[2] = 2*vec[5]-vec[8]
    vec[2] = 1.0*A[0]/(gm-1.)+0.5*vec[1]*vec[1]/vec[0]
    
    return vec
    
def upperBC(A, vec):
    N = size(vec)
    N = N - 1
    n = size(A)
    n = n -1
    
    vec[N-2] = (2*vec[N-5]/A[n-1]-vec[N-8]/A[n-2])*A[n]
    vec[N-1] = (2*vec[N-4]/A[n-1]-vec[N-7]/A[n-2])*A[n]
    vec[N]   = (2*vec[N-3]/A[n-1]-vec[N-6]/A[n-2])*A[n]
    
    return vec
    
def timestep(cfl, gm, dx, u):
    n = size(u)
    dt = 1.0e3    
    
    for k in range(0, n/3):
        P = (gm - 1.)*u[3*k+2] - 0.5*u[3*k+1]*u[3*k+1]/u[3*k]
        if P < 0:
            print k , P
        c = sqrt(gm*P/u[3*k])
        v = u[3*k+1]/u[3*k]
        
        dt = min(dt, cfl*dx/(c+v))
        
    return dt


from math import *
from numpy import *
from pylab import *

import petsc4py, sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
from auxiliary_funcs import *

# ============ variables grid ============ #
gm = 1.4    # gas gamma

# ============ variables grid ============ #
nx   = 101	 # grid nodes
L    = 3.0	 # domain size

grid = linspace(0.0,L,nx)
dx   = grid[2]-grid[1]

# ============ GL quadrature integration ============ #
order = 3   # order
wi, xi = gaussl(order)

# ============ timestep ============ #
cfl = 0.5      # cfl condition (for stability)
ndt = 30        # number of cycles (timesteps)

# ============ Create matrices ============ #
N  = 1		# interpolation function order
V  = 3		# number of components per node

# Stiffness matrix
K = PETSc.Mat().createAIJ([nx*V, nx*V], nnz=3)
# Mass matrix
M = PETSc.Mat().createAIJ([nx*V, nx*V], nnz=3)
# Flux jacobian
dF = PETSc.Mat().createDense([nx*V, nx*V])
# Source
#S = zeros([nx*V, nx*V])
S = PETSc.Mat().createAIJ([nx*V, nx*V], nnz=3)

# ============ Create vectors ============ #
A = zeros(nx)
dA = zeros(nx)
u_n  = PETSc.Vec().createSeq(nx*V)
u_n1 = PETSc.Vec().createSeq(nx*V)
b    = PETSc.Vec().createSeq(nx*V)

# ============ Initialize ============ #
for i in range(0,nx):
    A[i] = 1.+2.2*(grid[i]-1.5)**2
    dA[i]= 4.4*(grid[i]-1.5)
    rho = 1.*(1.-0.3146*grid[i])
    T = 1.*(1.-0.2314*grid[i])
    v = 1.*((0.1+1.09*grid[i])*sqrt(T))
    P = rho*T
    e = P/(gm-1.)+0.5*rho*v*v
    
    u_n.setValue(i*V, rho*A[i])
    u_n.setValue(i*V+1, rho*v*A[i])
    u_n.setValue(i*V+2, e*A[i])

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
    # ======================
    # Source matrix
    # ======================    
    S.setValue(k*V+1, k*V+1,   dA[k]*M11, PETSc.InsertMode.ADD)
    S.setValue(k*V+V+1, k*V+1,   dA[k+1]*M12, PETSc.InsertMode.ADD)
    S.setValue(k*V+1, k*V+V+1,   dA[k]*M12, PETSc.InsertMode.ADD)
    S.setValue(k*V+V+1, k*V+V+1,   dA[k+1]*M22, PETSc.InsertMode.ADD)
    
# Make matrices useable.
M.assemblyBegin()
M.assemblyEnd()
K.assemblyBegin()
K.assemblyEnd()
S.assemblyBegin()
S.assemblyEnd()

# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

for t in range(0,ndt):
    
    # ============ Apply BC's ============ #
    u_n = lowerBC(gm,A,u_n)
    u_n = upperBC(A, u_n)
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

    a,d,c = dataforplot(V,u_n)
    subplot(231), plot(grid,a)#, axis((-L,L,0.0,3.0))
    subplot(232), plot(grid,d)#, axis((-L,L,0.0,3.0))
    subplot(233), plot(grid,c)#, axis((-L,L,0.0,3.0))
    subplot(234), plot(grid,a/A,'r')#, axis((-L,L,0.0,3.0))
    subplot(235), plot(grid,d/A,'r')#, axis((-L,L,0.0,3.0))
    subplot(236), plot(grid,c/A,'r')#, axis((-L,L,0.0,3.0))
    
    filename = 'quasi_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    # ============ calculate RHS ============ #
    #dt = timestep(cfl, gm, dx, u_n)
    dt = 0.001
    #print dt
    
    # ============ calculate RHS ============ #
    RHS = M - dt*dot(dF,K) + dt*S
    #LHS = M + dt*dot(dF,K) - dt*S
    RHS.mult(u_n, b)
    #M.mult(u_n, b)
    # ============ Solve ============ #
    #ksp.setOperators(LHS)
    ksp.solve(b, u_n1)
    u_n = u_n1


a,d,c = dataforplot(V,u_n)
subplot(231), plot(grid,a)#, axis((-L,L,0.0,3.0))
subplot(232), plot(grid,d)#, axis((-L,L,0.0,3.0))
subplot(233), plot(grid,c)#, axis((-L,L,0.0,3.0))
subplot(234), plot(grid,a/A,'r')#, axis((-L,L,0.0,3.0))
subplot(235), plot(grid,d/A,'r')#, axis((-L,L,0.0,3.0))
subplot(236), plot(grid,c/A,'r')#, axis((-L,L,0.0,3.0))
filename = 'quasi_' + str('%03d' % ndt) + '_.png'
savefig(filename, dpi=200)
clf()