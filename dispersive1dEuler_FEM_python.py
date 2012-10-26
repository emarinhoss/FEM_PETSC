# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:38:51 2012

1d euler eqn with dispersive source term using FEM nodal 
implentation with Scipy. Using implicit flux jacobian 
implementation and first order elements.

@author: sousae
"""

def dataforplot(comp, vec):
    qw = size(vec)
    
    q1 = zeros(qw/comp)
    q2 = zeros(qw/comp)
    q3 = zeros(qw/comp)
    q4 = zeros(qw/comp)
        
    for i in range(0,qw/comp):
        q1[i] = vec[i*comp]
        q2[i] = vec[i*comp+1]
        q3[i] = vec[i*comp+2]
        q3[i] = vec[i*comp+3]
    
    return q1, q2, q3, q4
    
def fluxJacob(q):
    
    F = zeros((4,4))
    
    F[0,1] = 1.
    F[1,0] = 0.5*(gm-1.)*(q[2]/q[0])**2-0.5*(3.-gm)*(q[1]/q[0])**2
    F[1,1] = (3.-gm)*q[1]/q[0]
    F[1,2] = (1.-gm)*q[2]/q[0]
    F[1,3] = gm - 1.
    F[2,0] = -(q[1]*q[2])/q[1]**2
    F[2,1] = q[2]/q[0]
    F[2,2] = q[1]/q[0]
    F[3,0] = (gm - 1.)*((q[1]/q[0])**3+(q[2]/q[0])**2*(q[1]/q[0]))-gm*(q[3]/q[0])*(q[1]/q[0])
    F[3,1] = gm*(q[3]/q[0])-0.5*(gm-1.)*(3*(q[1]/q[0])**2+(q[2]/q[0])**2)
    F[3,2] = -(gm - 1.)*(q[1]/q[0])*(q[2]/q[0])
    F[3,3] = gm*q[1]/q[0]
    
    return F
    
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
    p,pu,pv,e = dataforplot(V,un)
    subplot(411), plot(grid,p), ylabel(r'$\rho$')
    subplot(412), plot(grid,pu), ylabel(r'$\rho u$')
    subplot(413), plot(grid,pv), ylabel(r'$\rho v$')
    subplot(414), plot(grid,e), ylabel(r'$e$'), xlabel(r'$x$')
    
    filename = 'dispersive_' + str('%03d' % i) + '_.png'
    savefig(filename, dpi=200)
    clf()

def ImplicitIntegration(dt, un):
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, nx):
        if (k == 0) or (k == nx-2):
            val = 1.
        else:
            val = 2.
        
        FluxJ = fluxJacob(un[k*V:k*V+V])
        dF[k*V:k*V+V, k*V:k*V+V] = val*FluxJ
        
    b = dot(M,un)
    LHS = M - dt*(Src - dot(dF,K))
    
    start = time.clock()
    unp1 = scipy.linalg.solve(LHS, b)
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
    print 'residual', scipy.linalg.norm(dot(LHS, unp1) - b)/scipy.linalg.norm(LHS)
    return unp1

def periodicBCs(q):
    Num = size(q)
    Num = Num - 1
    
    q[Num]   = q[3]
    q[Num-1] = q[2]
    q[Num-2] = q[1]
    q[Num-3] = q[0]
    
    return q

from math import *
from numpy import *
from pylab import *
import scipy.linalg
import time

from auxiliary_funcs import *
global order, gm, dx, cfl, mt, grid
global V, nx, N

# ============ problem variables ============ #
gm = 1.4    # gas gamma
mas= 1.     # mass
q  = 10.    # charge
Bz = 1.0    # magnetic field
# ============ variables grid ============ #
nelem= 50	 # element number
L    = 1.0	 # domain size
# ============ GL quadrature integration ============ #
order = 3   # order
wi, xi = gaussl(order)
# ============ timestep ============ #
dt = 0.01      # cfl condition (for stability)
ndt = 40       # number of cycles (timesteps)
tme = 0.0      # current time
# ============ Create matrices ============ #
N  = 2          # interpolation function order
V  = 4          # number of components per node
nx = nelem*N+1  # number of nodes

# ============ create grid ============ #
grid = linspace(0.0,L,nx)

global K, M, dF, Src
# Stiffness matrix
K  = zeros((nx*V, nx*V))
# Mass matrix
M  = zeros((nx*V, nx*V))
# Flux jacobian
dF = zeros((nx*V, nx*V))
# Source jacobian
Src= zeros((nx*V, nx*V))

# ============ Create vectors ============ #
un = zeros(nx*V)    # current solution
b  = zeros(nx*V)    # right hand side

# ============ Initialize ============ #
for i in range(0,nx):
    rho = 1.0
    u = 1.0
    v = 1.0
    P = 1.0
    e = P/(gm-1.)+0.5*rho*(u*u+v*v)
    
    un[i*V]  = rho
    un[i*V+1]= rho*u
    un[i*V+2]= rho*v
    un[i*V+3]= e

# ============ Populate Matrices ============ #
blk = (N+1)*V
xval= zeros(N+1)
Phi = zeros(N+1)
dPhi= zeros(N+1)
for l in range(0,N+1):
    xval[l] = grid[l]
    
dx = xval[N]-xval[0]

# ============ Populate Matrices ============ #
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
                #Se[i*V+k,j*V+k] += wi[l]*0.5*dx*Phi[i]*Phi[j]
            
Se = array([[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]])
        
for k in range(0, nelem):
    M[k*N*V:k*N*V+blk,k*N*V:k*N*V+blk] += Me
    K[k*N*V:k*N*V+blk,k*N*V:k*N*V+blk] += Ke

# ============ Apply periodic BCs ============ #
M[0:4,0:4] += Me[blk-V:blk,blk-V:blk]
M[(nx-1)*V:nx*V,(nx-1)*V:nx*V] += Me[0:4,0:4]
M[0:4,(nx-1)*V:nx*V] += Me[blk-V:blk,0:4]
M[(nx-1)*V:nx*V,0:4] += Me[0:4,blk-V:blk]

K[0:4,0:4] += Ke[blk-V:blk,blk-V:blk]
K[(nx-1)*V:nx*V,(nx-1)*V:nx*V] += Ke[0:4,0:4]
K[0:4,(nx-1)*V:nx*V] += Ke[blk-V:blk,0:4]
K[(nx-1)*V:nx*V,0:4] += Ke[0:4,blk-V:blk]


# ============ Advance in time ============ #    
for t in range(0,ndt):
    print '#========================#'
    print t, tme
    
    # ============ Plot Solution ============ #
    plot_solution(t, un)
    
    # ============ Advance Solution ============ #
    unp1 = ImplicitIntegration(dt, un)

    un = unp1
    tme += dt

plot_solution(ndt, un)