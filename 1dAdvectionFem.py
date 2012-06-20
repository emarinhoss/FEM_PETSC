from math import *
from numpy import *
from pylab import *

import petsc4py, sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
from auxiliary_funcs import *

# ============ variables ============ #
nx = 100	# grid nodes
a  = 5.0	# advection speed
L  = 5.0	# domain half-size
x0 = 0.   # gaussian peak location

# ============ Options ============ #
order = 2   # quadrature order
wi, xi = gaussl(order)

# ============ grid ============ #
grid = linspace(-L,L,nx)
dx = grid[2]-grid[1]

# ============ timestep ============ #
cfl = 0.5
dt  = abs(cfl*dx/a)
ndt = 10        # number of cycles (timesteps)
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
    u_n.setValue(i, exp(-(grid[i]-x0)**2))
    
# ============ Populate Mass matrix ============ #
a11   = 0.
alast = 0.
diag  = 0.
offdiag = 0.

for i in range(0,order):
    x = xi[i]*dx/2+0.5*(grid[2]+grid[1])
    diag += wi[i]*((x-grid[0])**2)
    offdiag += wi[i]*((grid[0]-x)*(grid[1]-x))
    a11 += wi[i]*(grid[1]-x)**2
    alast += wi[i]*(grid[nx-1]-x)**2

M.setValue(0, 0, a/dx**2*a11)
M.setValue(nx-1, nx-1, a/dx**2*alast)
for k in range(1, nx-1):
    M.setValue(k, k, a/dx**2*diag**2) # Diagonal.
    M.setValue(k-1, k, a/dx**2*offdiag) # Off-diagonal.
    M.setValue(k, k-1, a/dx**2*offdiag) # Off-diagonal.
    
M.setValue(nx-2, nx-1, a/dx**2*offdiag)
M.setValue(nx-1, nx-2, a/dx**2*offdiag)
# set periodic BC's
M.setValue(0, nx-1, a/dx**2*offdiag)
M.setValue(nx-1, 0, a/dx**2*offdiag)
M.assemblyBegin() # Make matrices useable.
M.assemblyEnd()
# ============ Populate Stiffness matrix ============ #
m11   = 0.
mlast = 0.
mdiag = 0.
moffd1= 0.
moffd2= 0.

for i in range(0,order):
    x = xi[i]*dx/2+0.5*(grid[2]+grid[1])
    mdiag += wi[i]*((x-grid[0]))
    moffd1+= wi[i]*(x-grid[1])
    moffd2+= wi[i]*(x-grid[2])
    
K.setValue(0, 0, a/dx**2*dt*moffd1)
K.setValue(nx-2,nx-1, a/dx**2*dt*moffd1)
K.setValue(nx-1, nx-1, a/dx**2*dt*moffd1)
K.setValue(nx-1, nx-2, a/dx**2*dt*moffd2)
for k in range(1,nx-1):
    K.setValue(k, k, 2.*a/dx**2*dt*mdiag) # Diagonal.
    K.setValue(k-1, k, a/dx**2*dt*moffd1) # Off-diagonal.
    K.setValue(k, k-1, a/dx**2*dt*moffd2) # Off-diagonal.

# set periodic BC's
K.setValue(0, nx-1, a/dx**2*moffd2)
K.setValue(nx-1, 0, a/dx**2*moffd1)
K.assemblyBegin() # Make matrices useable.
K.assemblyEnd()

# ============ Initialize ksp solver ============ #
ksp = PETSc.KSP().create()
ksp.setOperators(M)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

for t in range(0,ndt):
    
    plot(grid,u_n.array)
    filename = 'adv_' + str('%03d' % t) + '_.png'
    savefig(filename, dpi=200)
    clf()
    
    K.mult(u_n,b1)
    M.mult(u_n,b2)
    b = b1 + b2
    b.setValue(0, b[0]+dt*a*u_n[0])
    b.setValue(nx-1, b[nx-1]-dt*a*u_n[nx-1])
    ksp.solve(b, u_n1)
    u_n = u_n1
