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
    V2 = vec[4]/vec[3]
    V3 = vec[7]/vec[6]
    
    vec[0] = 1.0*A[0]
    #vec[1] = 0.1*A[0]
    vec[1] = (2*V2-V3)*vec[0]
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
        P = (gm - 1.)*(u[3*k+2] - 0.5*u[3*k+1]*u[3*k+1]/u[3*k])
        if P < 0:
            print k , P
        c = sqrt(gm*P/u[3*k])
        v = u[3*k+1]/u[3*k]
        
        dt = min(dt, cfl*dx/(c+v))
        
    return dt

def fluxJacob(g,q):
    F = array([[0.,1.,0.],[0.5*(g-3.)*(q[1]/q[0])**2,(3.-g)*q[1]/q[0], g-1],[-g*q[1]*q[2]/q[0]**2+0.5*(g-1.)*(q[1]/q[0])**3, g*q[2]/q[1]-1.5*(g-1.)*(q[1]/q[0])**2, g*q[1]/q[0]]])
    
    return F

def computeSource(mt,g,order,u_n,grid):
    wi, xi = gaussl(order)
    n  = size(u_n)
    dx = grid[1]-grid[0]

    src = PETSc.Vec().createSeq(n)
    
    for k in range(0,n/3-1):
        rho1 = u_n[V*k]
        mu1  = u_n[V*k+1]
        e1   = u_n[V*k+2]
        p1   = (g-1.)*(e1-0.5*mu1*mu1/rho1)
        s1   = 0.
        rho2 = u_n[V*k+V]
        mu2  = u_n[V*k+V+1]
        e2   = u_n[V*k+V+2]
        p2   = (g-1.)*(e2-0.5*mu2*mu2/rho2)
        s2   = 0.
        
        for i in range(0,order):
            x    = 0.5*(xi[i]*dx+(grid[k]+grid[k+1]))
            Phi1 = (grid[k+1]-x)/dx
            Phi2 = (x-grid[k])/dx
            dA   = 2*mt*(x-1.5)
            
            s1 += wi[i]*dx*0.5*(Phi1*Phi1*p1 + Phi1*Phi2*p2)*dA
            s2 += wi[i]*dx*0.5*(Phi1*Phi2*p1 + Phi2*Phi2*p2)*dA
            
        src.setValue(k*V+1, s1, PETSc.InsertMode.ADD)
        src.setValue(k*V+V+1, s2, PETSc.InsertMode.ADD)
              
    return src
        
        
        

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
order = 2   # order
wi, xi = gaussl(order)

# ============ timestep ============ #
cfl = 0.1      # cfl condition (for stability)
ndt = 60        # number of cycles (timesteps)
time = 0.0      # current time
# ============ Create matrices ============ #
N  = 1		# interpolation function order
V  = 3		# number of components per node

# Stiffness matrix
K = PETSc.Mat().createAIJ([nx*V, nx*V], nnz=3)
# Mass matrix
M = PETSc.Mat().createAIJ([nx*V, nx*V], nnz=3)
# Flux jacobian
dF = PETSc.Mat().createDense([nx*V, nx*V])

# ============ Create vectors ============ #
A = zeros(nx)
dA = zeros(nx)
u_n  = PETSc.Vec().createSeq(nx*V)
u_n1 = PETSc.Vec().createSeq(nx*V)
b    = PETSc.Vec().createSeq(nx*V)
b1   = PETSc.Vec().createSeq(nx*V)
S    = PETSc.Vec().createSeq(nx*V)

# ============ Initialize ============ #
mt = 0
for i in range(0,nx):
    A[i] = 1.+mt*(grid[i]-1.5)**2
    dA[i]= 2*mt*(grid[i]-1.5)
    rho = 1.0
    T = 1.0
    Mach = 0.1
    v = Mach*sqrt(gm*T)
    #rho = (1.-0.3146*grid[i])    
    #T = (1.-0.2314*grid[i])
    #v = (0.4+1.09*grid[i])*sqrt(T)
    P = rho*T
    #print i , P
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
    x = 0.5*(xi[i]*dx+(x2+x1))
    
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
#    S.setValue(k*V+1, k*V+1,   dA[k]*M11, PETSc.InsertMode.ADD)
#    S.setValue(k*V+V+1, k*V+1,   dA[k+1]*M12, PETSc.InsertMode.ADD)
#    S.setValue(k*V+1, k*V+V+1,   dA[k]*M12, PETSc.InsertMode.ADD)
#    S.setValue(k*V+V+1, k*V+V+1,   dA[k+1]*M22, PETSc.InsertMode.ADD)
    
# Make matrices useable.
M.assemblyBegin()
M.assemblyEnd()
K.assemblyBegin()
K.assemblyEnd()
#S.assemblyBegin()
#S.assemblyEnd()

# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

for t in range(0,ndt):
    print t, time
    # ============ Apply BC's ============ #
    u_n = lowerBC(gm,A,u_n)
    u_n = upperBC(A, u_n)
    # ============ Calculate Flux Jacobian ============ #

    for k in range(0, nx-1):
        if (k == 0) or (k == nx-2):
            val = 1.
        else:
            val = 2.
        
        FluxJ = fluxJacob(gm, u_n[k*V:k*V+V])
        dF.setValues(range(k*V,k*V+V), range(k*V,k*V+V), val*FluxJ, PETSc.InsertMode.INSERT)
		
    dF.assemblyBegin()
    dF.assemblyEnd()
    #print dF[3:6,3:6]

    a,d,c = dataforplot(V,u_n)
    subplot(311), plot(grid,a), ylabel(r'$\rho A$')#, axis((0.,L,0.0,7.0))
    subplot(312), plot(grid,d), ylabel(r'$\rho uA$')#, axis((0.,L,0.0,6.0))
    subplot(313), plot(grid,c), ylabel(r'$eA$'), xlabel(r'$x$')#, axis((0.,L,0.0,20.0))
    
    filename = 'quasi_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    # ============ calculate timestep ============ #
    dt = timestep(cfl, gm, dx, u_n)
    #dt = 0.005
    #print dt
    
    # ============ calculate RHS ============ #
    S = computeSource(mt,gm,order,u_n,grid)
    RHS = M - dt*dot(dF,K)
    #RHS = M - dt*K
    RHS.mult(u_n, b1)
    b = b1 + S
    
    # ============ Solve ============ #
    ksp.solve(b, u_n1)
    u_n = u_n1
    time += dt


a,d,c = dataforplot(V,u_n)
subplot(311), plot(grid,a), ylabel(r'$\rho A$')#, axis((0.,L,0.0,7.0))
subplot(312), plot(grid,d), ylabel(r'$\rho uA$')#, axis((0.,L,0.0,6.0))
subplot(313), plot(grid,c), ylabel(r'$eA$'), xlabel(r'$x$')#, axis((0.,L,0.0,20.0))
filename = 'quasi_' + str('%03d' % ndt) + '_.png'
savefig(filename, dpi=200)
clf()