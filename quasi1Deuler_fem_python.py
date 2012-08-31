# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:49:44 2012

Quasi-1d euler eqn using FEM nodal implentation with Scipy.
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

def lowerBC(A, vec):
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
    
def timestep(u):
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

def fluxJacob(q):
    F = array([[0.,1.,0.],[0.5*(gm-3.)*(q[1]/q[0])**2,(3.-gm)*q[1]/q[0], gm-1],[-gm*q[1]*q[2]/q[0]**2+0.5*(gm-1.)*(q[1]/q[0])**3, gm*q[2]/q[1]-1.5*(gm-1.)*(q[1]/q[0])**2, gm*q[1]/q[0]]])
    
    return F

def computeSource(u_n):
    wi, xi = gaussl(order)
    n  = size(u_n)
    
    src = zeros(size(u_n))
    
    for k in range(0,n/3-1):
        rho1 = u_n[V*k]
        mu1  = u_n[V*k+1]
        e1   = u_n[V*k+2]
        p1   = (gm-1.)*(e1-0.5*mu1*mu1/rho1)
        s1   = 0.
        rho2 = u_n[V*k+V]
        mu2  = u_n[V*k+V+1]
        e2   = u_n[V*k+V+2]
        p2   = (gm-1.)*(e2-0.5*mu2*mu2/rho2)
        s2   = 0.
        
        for i in range(0,order):
            x    = 0.5*(xi[i]*dx+(grid[k]+grid[k+1]))
            Phi1 = (grid[k+1]-x)/dx
            Phi2 = (x-grid[k])/dx
            dA   = 2*mt*(x-1.5)
            
            s1 += wi[i]*dx*0.5*(Phi1*Phi1*p1 + Phi1*Phi2*p2)*dA
            s2 += wi[i]*dx*0.5*(Phi1*Phi2*p1 + Phi2*Phi2*p2)*dA
        
        src[k*V+1] += s1
        src[k*V+V+1] += s2
        #src.setValue(k*V+1, s1, PETSc.InsertMode.ADD)
        #src.setValue(k*V+V+1, s2, PETSc.InsertMode.ADD)
              
    return src
    
def interpolationf(xg, x, Phi, dPhi):
    pts = size(xg)
    
    for k in range(0,pts):
        Phi[k] = 0.0
        dPhi[k]= 0.0
    
    if pts==3:
        Phi[0] = (x-xg[1])*(x-xg[2])/((xg[0]-xg[1])*(xg[0]-xg[2]))
        Phi[1] = (x-xg[0])*(x-xg[2])/((xg[1]-xg[0])*(xg[1]-xg[2]))
        Phi[2] = (x-xg[0])*(x-xg[1])/((xg[2]-xg[1])*(xg[2]-xg[0]))
        
        dPhi[0] = (2*x-(xg[1]+xg[2]))/((xg[0]-xg[1])*(xg[0]-xg[2]))
        dPhi[1] = (2*x-(xg[0]+xg[2]))/((xg[1]-xg[0])*(xg[1]-xg[2]))
        dPhi[2] = (2*x-(xg[1]+xg[0]))/((xg[2]-xg[1])*(xg[2]-xg[0]))
        
    else:
        Phi[0] = (x-xg[1])/(xg[0]-xg[1])
        Phi[1] = (x-xg[0])/(xg[1]-xg[0])
        
        dPhi[0] = 1./(xg[0]-xg[1])
        dPhi[1] = 1./(xg[1]-xg[0])

def plot_solution(i, un):
    a,d,c = dataforplot(V,un)
    subplot(311), plot(grid,a), ylabel(r'$\rho A$')
    subplot(312), plot(grid,d), ylabel(r'$\rho uA$')
    subplot(313), plot(grid,c), ylabel(r'$eA$'), xlabel(r'$x$')
    
    filename = 'py_quasi_' + str('%03d' % i) + '_.png'
    savefig(filename, dpi=200)
    clf()
        
def forwardEuler(dt, un, SOL):
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, nx-1):
        if (k == 0) or (k == nx-2):
            val = 1.
        else:
            val = 2.
        
        FluxJ = fluxJacob(u_n[k*V:k*V+V])
        dF[k*V:k*V+V, k*V:k*V+V] = val*FluxJ
    
    # ============ calculate RHS ============ #
    S = computeSource(un)
    RHS = M - dt*dot(dF,K)
    b = S + dot(RHS,un)
    
    # ============ Solve ============ #
    start = time.clock()
    #lu = scipy.linalg.lu_factor(M)
    #u_n1 = scipy.linalg.lu_solve(lu, b)
    #u_n1 = scipy.linalg.solve(M, b)
    unp1 = dot(SOL, b)
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
    print 'residual', scipy.linalg.norm(dot(M, unp1) - b)/scipy.linalg.norm(M)
    
    return unp1

def leapFrog(dt, un, unm1, SOL):
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, nx-1):
        if (k == 0) or (k == nx-2):
            val = 1.
        else:
            val = 2.
        
        FluxJ = fluxJacob(u_n[k*V:k*V+V])
        dF[k*V:k*V+V, k*V:k*V+V] = val*FluxJ
    
    # ============ calculate RHS ============ #
    S = computeSource(un)
    RHS = dot(dF,K)
    b = dot(M,unm1) + 2.*dt*(S - dot(RHS,un))
    
    # ============ Solve ============ #
    start = time.clock()
    #lu = scipy.linalg.lu_factor(M)
    #u_n1 = scipy.linalg.lu_solve(lu, b)
    #u_n1 = scipy.linalg.solve(M, b)
    unp1 = dot(SOL, b)
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
    print 'residual', scipy.linalg.norm(dot(M, unp1) - b)/scipy.linalg.norm(M)
    
    return unp1
    
from math import *
from numpy import *
from pylab import *
import scipy.linalg
import time

from auxiliary_funcs import *
global order, gm, dx, cfl, mt, grid
global V, nx
# ============ variables grid ============ #
gm = 1.4    # gas gamma

# ============ variables grid ============ #
nelem= 50	 # element number
L    = 3.0	 # domain size

# ============ GL quadrature integration ============ #
order = 2   # order
wi, xi = gaussl(order)

# ============ timestep ============ #
cfl = 0.5      # cfl condition (for stability)
ndt = 60        # number of cycles (timesteps)
tme = 0.0      # current time
# ============ Create matrices ============ #
N  = 1          # interpolation function order
V  = 3          # number of components per node
nx = nelem*N+1  # number of nodes

# ============ create grid ============ #
grid = linspace(0.0,L,nx)


global K, M, dF
# Stiffness matrix
K = zeros((nx*V, nx*V))
# Mass matrix
M = zeros((nx*V, nx*V))
# Flux jacobian
dF= zeros((nx*V, nx*V))

# ============ Create vectors ============ #
A    = zeros(nx)    # area 
u_n  = zeros(nx*V)    # current solution
u_n1 = zeros(nx*V)    # solution at n+1
b    = zeros(nx*V)    # right hand side
S    = zeros(nx*V)    # sources

# ============ Initialize ============ #
mt = 0
for i in range(0,nx):
    A[i] = 1.+mt*(grid[i]-1.5)**2
    rho = 1.0
    T = 1.0
    Mach = 1.0e-1
    v = Mach*sqrt(gm*T)
    #rho = (1.-0.3146*grid[i])    
    #T = (1.-0.2314*grid[i])
    #v = (0.4+1.09*grid[i])*sqrt(T)
    P = rho*T
    #print i , P
    e = P/(gm-1.)+0.5*rho*v*v
    
    u_n[i*V]  = rho*A[i]
    u_n[i*V+1]= rho*v*A[i]
    u_n[i*V+2]= e*A[i]

# ============ Populate Matrices ============ #
blk = (N+1)*V
xval= zeros(N+1)
Phi = zeros(N+1)
dPhi= zeros(N+1)
for l in range(0,N+1):
    xval[l] = grid[l]
    
dx = xval[N]-xval[0]

Me=zeros((blk,blk))
Ke=zeros((blk,blk))

for l in range(0,order):
    x = 0.5*(xi[l]*dx+(xval[N]+xval[0]))
    
    interpolationf(xval, x, Phi, dPhi)
    
    for i in range(0,N+1):
        for j in range(0,N+1):
            for k in range(0,V):
                Me[i*V+k,j*V+k] += wi[l]*0.5*dx*Phi[i]*Phi[j]
                Ke[i*V+k,j*V+k] += wi[l]*0.5*dx*Phi[i]*dPhi[j]
            
        
for k in range(0, nelem):
    M[k*N*V:k*N*V+blk,k*N*V:k*N*V+blk] += Me
    K[k*N*V:k*N*V+blk,k*N*V:k*N*V+blk] += Ke

SOL = scipy.linalg.inv(M)

# ============ Take first a forward Euler step ============ #
# ============ Apply BC's ============ #
u_n = lowerBC(A,u_n)
u_n = upperBC(A, u_n)
# ============ Plot Solution ============ #
plot_solution(0, u_n)
# ============ calculate timestep ============ #
dt = timestep(u_n)
# ============ Advance Solution ============ #
unm1 = u_n
un = forwardEuler(dt, u_n, SOL)

for t in range(1,ndt):
    print '#========================#'
    print t, tme
    
    # ============ Apply BC's ============ #
    un = lowerBC(A,un)
    un = upperBC(A, un)
    # ============ Plot Solution ============ #
    plot_solution(t, un)
    # ============ calculate timestep ============ #
    dt = timestep(un)
    # ============ Advance Solution ============ #
    unp1 = leapFrog(dt, un, unm1, SOL)
    unm1 = un
    un = unp1
    tme += dt

plot_solution(ndt, u_n)