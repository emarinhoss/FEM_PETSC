from numpy import *
from math import *

EPS = 3.0e-13

def gauleg(x1, x2, x, w, n):
	
	m = (n+1)/2
	xm = 0.5*(x2+x1)
	xl = 0.5*(x2-x1)
	
	for i in range(1,m+1):
		z = cos(pi*(i-0.25)/(n+0.5))
		
		while True:
			p1 = 1.0
			p2 = 0.0
			for j in range(1,n+1):
				p3 = p2;
				p2 = p1;
				p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
			
			pp = n*(z*p1-p2)/(z*z-1.0)
			z1 = z
			z = z1-p1/pp
			
			if (abs(z-z1) > EPS):
				break
				
		x[i] = xm-xl*z
		x[n+1-i] = xm+xl*z
		w[i] = 2.0*xl/((1.0-z*z)*pp*pp)
		w[n+1-i] = w[i]
		
def legendre_p(n, x):
	p0 = 1.0
	p1 = x

	if (n==0): return p0
	if (n==1): return p1
	
	pn = 0.0
	pn1 = p1
	pn2 = p0	# initialize recurrence
	
	for i in range(2, n+1):
		# use recurrence relation to compute P_n
		pn = (x*(2.*i-1.)*pn1 - (i-1.)*pn2)/(1.*i)
		pn2 = pn1
		pn1 = pn
	return pn
	
def legendre_p_d(n, x):
	dp0 = 0
	dp1 = 1
	
	if (n==0): return dp0
	if (n==1): return dp1
	
	dpn = ((n+1.)*x*legendre_p(n,x) - (n+1.)*legendre_p(n+1,x))/(1.-x*x)
	
	return dpn

def gaussl(o):
    if (o==2):
        xi = array([-1./sqrt(3.), 1./sqrt(3.)])
        wi = array([1., 1.])
    elif(o==3):
        xi = array([-sqrt(15.)/5., 0., sqrt(15.)/5.])
        wi = array([5./9., 8./9., 5./9.])
    elif(o==4):
        xi = array([-sqrt((3.+2.*sqrt(6./5.))/7.), -sqrt((3.-2.*sqrt(6./5.))/7.), sqrt((3.-2.*sqrt(6./5.))/7.), sqrt((3.+2.*sqrt(6./5.))/7.)])
        wi = array([(18.-sqrt(30.))/36., (18.+sqrt(30.))/36., (18.+sqrt(30.))/36., (18.-sqrt(30.))/36])
    else:
        xi = array([0.])
        wi = array([2.])
    
    return wi, xi