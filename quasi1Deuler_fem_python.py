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
    #V2 = vec[4]/vec[3]
    #V3 = vec[7]/vec[6]
    
    vec[0] = (1.0*A[0])
    vec[1] = (Mach*a_inf*A[0])
    vec[2] = (0.75*A[0]/(gm-1.)+0.5*vec[1]*vec[1]/vec[0])    
    
#    vec[0] += (-M21)*(1.0*A[0])
#    vec[1] += (-M21)*(Mach*a_inf*A[0])
#    vec[2] += (-M21)*(0.75*A[0]/(gm-1.)+0.5*vec[1]*vec[1]/vec[0])
    
    return vec
    
def upperBC(A, vec):
    N = size(vec)
    N = N - 1
    n = size(A)
    n = n -1
    
    #vec[N-2] = vec[N-5]/A[n-1]*A[n]
    #vec[N-1] = vec[N-4]/A[n-1]*A[n]
    #vec[N-2] = (2*vec[N-5]/A[n-1]-vec[N-8]/A[n-2])*A[n]
    #vec[N-1] = (2*vec[N-4]/A[n-1]-vec[N-7]/A[n-2])*A[n]
    #vec[N]   = (2*vec[N-3]/A[n-1]-vec[N-6]/A[n-2])*A[n]
    
    vec[N]    = (P_pr*A[n]/(gm-1.)+0.5*vec[N-1]*vec[N-1]/vec[N-2])
    #vec[N]    += (-M21)*(P_pr*A[n]/(gm-1.)+0.5*vec[N-1]*vec[N-1]/vec[N-2])
    
    #vec[N-2] = vec[N-5]/A[n-1]*A[n]
    #vec[N-1] = vec[N-4]/A[n-1]*A[n]
    #vec[N]   = vec[N-3]/A[n-1]*A[n]
    
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
    F = zeros((V,V))
    
    F[0,1] = 1.
    F[1,0] = 0.5*(gm-3.)*(q[1]/q[0])**2
    F[1,1] = (3.-gm)*q[1]/q[0]
    F[1,2] = gm-1.
    F[2,0] = -gm*q[1]*q[2]/q[0]**2+0.5*(gm-1.)*(q[1]/q[0])**3
    F[2,1] = gm*q[2]/q[1]-1.5*(gm-1.)*(q[1]/q[0])**2
    F[2,2] = gm*q[1]/q[0]
    
    return F
    
def calc_flux(q):
	
	f   = zeros(3)
	
	rho = q[0]	# density
	rhou= q[1]	# momentum
	e   = q[2]	# energy
	
	p   = (gm-1.)*(e - 0.5*rhou*rhou/rho)	# pressure
	u   = rhou/rho	# velocity
	
	f[0] = rhou
	f[1] = rho*u*u + p
	f[2] = (e+p)*u
	
	return f

def computeSource(u_n):
    wi, xi = gaussl(order)
    n  = size(u_n)
    
    val = zeros(size(u_n))
    
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
            dA   = 0.2776*(1.-tanh(0.8*x-4.)*tanh(0.8*x-4.))
            
            s1 += wi[i]*dx*0.5*(Phi1*Phi1*p1 + Phi1*Phi2*p2)*dA
            s2 += wi[i]*dx*0.5*(Phi1*Phi2*p1 + Phi2*Phi2*p2)*dA
        
        val[k*V+1] += s1
        val[k*V+V+1] += s2
              
    return val
    
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

def plot_solution(i, un, tns):
    a,d,c = dataforplot(V,un)
    rho = a/r_inf/A
    u = d/(r_inf*a_inf)/a
    e = c/(r_inf*a_inf*a_inf)/A
    p = (gm-1.)*(e-0.5*rho*u*u)
    
    subplot(221), plot(grid,rho), ylabel(r'$\rho$')#, ylim([0.5, 2.])
    title("at time t=%3.5f" % (tns))
    subplot(222), plot(grid,p), ylabel(r'$p$')#, ylim([0., 2])
    subplot(223), plot(grid,u), ylabel(r'$u$')#, ylim([0., 2])
    subplot(224), plot(grid,e), ylabel(r'$e$')#, xlabel(r'$x$')#, ylim([1., 4.])
    
    filename = 'py_quasi_' + str('%03d' % i) + '_.png'
    savefig(filename, dpi=200)
    clf()
        
def forwardEuler(dt, un, SOL):
    f_n = zeros(size(un))
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, nx):
        #if (k == 0) or (k == nx-2):
        #    val = 1.
        #else:
        #    val = 2.
        
        #FluxJ = fluxJacob(un[k*V:k*V+V])
        f_n[k*V:k*V+V] = calc_flux(un[k*V:k*V+V])
        #dF[k*V:k*V+V, k*V:k*V+V] = val*FluxJ
    
    # ============ calculate RHS ============ #
    S = computeSource(un)
    #RHS = dot(dF,K)
    #b = dt*(S - dot(RHS,un)) + dot(M,un)
    b = dt*(S - dot(K,f_n)) + dot(M,un)
    
    # ============ Solve ============ #
    start = time.clock()
    unp1 = dot(SOL,b)
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
    print 'residual', scipy.linalg.norm(dot(M, unp1) - b)/scipy.linalg.norm(M)
    
    return unp1
    
def ImplicitIntegration(dt, un):
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, nx):
        if (k == 0) or (k == nx-2):
            val = 1.
        else:
            val = 2.
        
        FluxJ = fluxJacob(un[k*V:k*V+V])
        #f_n[k*V:k*V+V] = calc_flux(u_n[k*V:k*V+V])
        dF[k*V:k*V+V, k*V:k*V+V] = val*FluxJ
        
    b = dot(M,un)
    LHS = M + dt*dot(dF,K)
    
    start = time.clock()
    unp1 = scipy.linalg.solve(LHS, b)
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
    print 'residual', scipy.linalg.norm(dot(LHS, unp1) - b)/scipy.linalg.norm(LHS)
    return unp1

def sensitivityMatrix(dt, un, SOL):
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, nx):
        if (k == 0) or (k == nx-2):
            val = 1.
        else:
            val = 2.
        
        FluxJ = fluxJacob(u_n[k*V:k*V+V])
        #f_n[k*V:k*V+V] = calc_flux(u_n[k*V:k*V+V])
        dF[k*V:k*V+V, k*V:k*V+V] = val*FluxJ
        
    RHS  = M - dt*dot(dF,K)
    A    = dot(SOL,RHS)
    unp1 = dot(A,un)
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
    
def RK4(dt, un):    
    # ============ Calculate Flux Jacobian ============ #
    for k in range(0, nx-1):
        if (k == 0) or (k == nx-2):
            val = 1.
        else:
            val = 2.
        
        FluxJ = fluxJacob(un[k*V:k*V+V])
        dF[k*V:k*V+V, k*V:k*V+V] = val*FluxJ    
    
    start = time.clock()
    # RK step 1
    src = calcrk(un)
    us = un + 0.5*dt*src
    # RK step 2
    srct = calcrk(us)
    us = un + 0.5*dt*srct
    # RK step 3
    srcm = calcrk(us)
    us = un + dt*srcm
    srcm = srct+srcm
    # RK step 4
    srct = calcrk(us)    
    
    # perform final update
    unp1 = un + 1./6.*dt*(src+srct+2.*srcm)
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
    #print 'residual', scipy.linalg.norm(dot(M, unp1) - b)/scipy.linalg.norm(M)
    
    return unp1

def calcrk(vec):
    #S = computeSource(vec)
    
    Afk = dot(dF,K)
    
    ftx = S - dot(Afk,vec) + dot(Vv,vec) - Nbc
    # ============ Apply BC's ============ #
    #ftx = lowerBC(A, ftx)
    #ftx = upperBC(A, ftx)
    
    # ============ solve ============ #
    sol = dot(SOL, ftx)
    
    return sol
    
from math import *
from numpy import *
from pylab import *
import scipy.linalg
import time

from auxiliary_funcs import *
global order, gm, dx, cfl, mt, grid
global V, nx, N, M21
global Mach, r_inf, a_inf, P_inf, epsn, P_pr

# ============ variables grid ============ #
gm = 1.4    # gas gamma

# ============ variables grid ============ #
nelem= 100	 # element number
L    = 10.0	 # domain size

# ============ GL quadrature integration ============ #
order = 3   # order
wi, xi = gaussl(order)

# ============ timestep ============ #
cfl = 0.5      # cfl condition (for stability)
ndt = 10    # number of cycles (timesteps)
tme = 0.0      # current time
# ============ Create matrices ============ #
N  = 2          # interpolation function order
V  = 3          # number of components per node
nx = nelem*N+1  # number of nodes
epsn= 4.0e-1    # artificial diffusion parameter
# ============ create grid ============ #
grid = linspace(0.0,L,nx)


global K, M, dF, Vv, Nbc, A, SOL
# Stiffness matrix
K  = zeros((nx*V, nx*V))
# Mass matrix
M  = zeros((nx*V, nx*V))
# Flux jacobian
dF = zeros((nx*V, nx*V))
# Arificial dissipation volume term
Vv = zeros((nx*V, nx*V))


# ============ Create vectors ============ #
A  = zeros(nx)    # area 
un = zeros(nx*V)    # current solution
b  = zeros(nx*V)    # right hand side
S  = zeros(nx*V)    # sources
Nbc= zeros(nx*V)    # natural BC's

# ============ Initialize ============ #
mt = 2.2
r_inf = 1.0
P_inf = 0.75
Mach = 1.25
a_inf = sqrt(gm*P_inf/r_inf)
v = Mach*a_inf
e = P_inf/(gm-1.)+0.5*r_inf*v*v
tau = L/a_inf
Pe = 0.75
P_pr= Pe*r_inf*a_inf*a_inf

for i in range(0,nx):
    A[i] = 1.398 + 0.347*tanh(0.8*grid[i]-4.0)
    
    un[i*V]  = r_inf*A[i]
    un[i*V+1]= r_inf*v*A[i]
    un[i*V+2]= e*A[i]

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
Ve=zeros((blk,blk))

for l in range(0,order):
    x = 0.5*(xi[l]*dx+(xval[N]+xval[0]))
    
    interpolationf(xval, x, Phi, dPhi)
    
    for i in range(0,N+1):
        for j in range(0,N+1):
            for k in range(0,V):
                Me[i*V+k,j*V+k] += wi[l]*0.5*dx*Phi[i]*Phi[j]
                Ke[i*V+k,j*V+k] += wi[l]*0.5*dx*Phi[i]*dPhi[j]
                Ve[i*V+k,j*V+k] += wi[l]*0.5*dx*dPhi[i]*dPhi[j]
            
        
for k in range(0, nelem):
    M[k*N*V:k*N*V+blk,k*N*V:k*N*V+blk] += Me
    K[k*N*V:k*N*V+blk,k*N*V:k*N*V+blk] += Ke
    Vv[k*N*V:k*N*V+blk,k*N*V:k*N*V+blk] += Ve

Vv = -epsn*Vv

## ============ Dirichlet BC's ============ #
M[0,:]=0; M[0,0]=1.0
M[1,:]=0; M[1,1]=1.0
M[2,:]=0; M[2,2]=1.0
M[nx*V-1,:]=0; M[nx*V-1,nx*V-1]=1.0
#M21 = M[0,3]

## ============ Inverse of M ============ #
SOL = scipy.linalg.inv(M)

for t in range(0,ndt):
    print '#========================#'
    print t, tme
    
    # ============ Apply BC's ============ #
    un = lowerBC(A, un)
    un = upperBC(A, un)
    
    # ============ Plot Solution ============ #
    plot_solution(t, un, tme/tau)
    # ============ calculate timestep ============ #
    dt = timestep(un)
    
    # ============ Advance Solution ============ #
    unp1 = RK4(dt, un)

    un = unp1
    tme += dt

plot_solution(ndt, un,tme/tau)
