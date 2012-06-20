import petsc4py, sys

petsc4py.init(sys.argv)
from petsc4py import PETSc

# grid size and spacing
m, n = 32, 32
hx = 1.0/(m-1)
hy = 1.0/(n-1)

# create sparse matrix
A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes([m*n, m*n])
A.setType('aij') # sparse

# precompute values for setting
# diagonal and non-diagonal entries
diagv = 2.0/hx**2 + 2.0/hy**2
offdx = -1.0/hx**2
offdy = -1.0/hy**2

# loop over owned block of rows on this
# processor and insert entry values
Istart, Iend = A.getOwnershipRange()
for I in xrange(Istart, Iend):
	A[I,I] = diagv
	i = I//n # map row number to
	j = I - i*n # grid coordinates
	if i> 0 : J = I-n; A[I,J] = offdx
	if i< m-1: J = I+n; A[I,J] = offdx
	if j> 0 : J = I-1; A[I,J] = offdy
	if j< n-1: J = I+1; A[I,J] = offdy

# communicate off-processor values
# and setup internal data structures
# for performing parallel operations
A.assemblyBegin()
A.assemblyEnd()

# create linear solver
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)

# use conjugate gradients
ksp.setType('cg')

# and incomplete Cholesky
ksp.getPC().setType('icc')

# obtain sol & rhs vectors
x, b = A.getVecs()
x.set(0)
b.set(1)

# and next solve
ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b, x)

try:
	from matplotlib import pylab
except ImportError:
	raise SystemExit("matplotlib not available")

from numpy import mgrid

X, Y = mgrid[0:1:1j*m,0:1:1j*n]
Z = x[...].reshape(m,n)
pylab.figure()
pylab.contourf(X,Y,Z)
pylab.plot(X.ravel(),Y.ravel(),'.k')
pylab.axis('equal')
pylab.colorbar()
pylab.show()
